from __future__ import annotations

import hashlib
import re
from collections import deque
from pathlib import Path
from urllib.parse import urlparse

from .agent import IRPageAgent
from .browser import PlaywrightRenderer
from .http import HttpClient, RobotsDisallowedError, RobotsUnavailableError
from .metadata import TranscriptMetadataAgent, extract_metadata_heuristic
from .models import CandidateLink, CandidatePage, Company, CrawlFailure, CrawlResult, FailureType, TranscriptRecord
from .navigation import discover_navigation_seeds
from .parsing import classify_transcript, docx_text, extract_links, looks_like_js_shell, looks_like_transcript, page_title, pdf_text, visible_text
from .runtime import ProgressReporter, timeout_after
from .search import find_ir_candidates
from .state import CrawlState
from .urls import host, normalize_url, resolve_document_url


TRANSCRIPT_HINTS = ("transcript", "earnings-call", "earnings call", "quarterly-results")
IR_HINTS = ("investor", "/ir", "shareholder", "financial", "events", "earnings", "quarter")
TRANSCRIPT_DOCUMENT_EXTENSIONS = (".pdf", ".docx")


class TranscriptCrawler:
    def __init__(
        self,
        *,
        model: str,
        out_dir: Path,
        max_pages_per_company: int = 40,
        max_depth: int = 3,
        use_playwright: bool = False,
        playwright_mode: str = "off",
        ollama_base_url: str | None = None,
        resume: bool = True,
        review_only: bool = False,
        seed_urls: list[str] | None = None,
        discovery_mode: str = "nav-first",
        include_discovery_guesses: bool = False,
        disable_official_homepage_overrides: bool = False,
        rerank_discovery: bool = False,
        extract_metadata_with_llm: bool = False,
        latest_only: bool = False,
        search_timeout_seconds: float = 30.0,
        llm_timeout_seconds: float = 45.0,
        navigation_llm_max_links: int = 12,
        page_llm_max_links: int = 12,
        llm_text_chars: int = 900,
        progress: ProgressReporter | None = None,
        http: HttpClient | None = None,
    ) -> None:
        self.model = model
        self.ollama_base_url = ollama_base_url
        self.agent = IRPageAgent(
            model,
            base_url=ollama_base_url,
            max_links=page_llm_max_links,
            text_chars=llm_text_chars,
        )
        self.metadata_agent = TranscriptMetadataAgent(model, base_url=ollama_base_url) if extract_metadata_with_llm else None
        self.out_dir = out_dir
        self.max_pages_per_company = max_pages_per_company
        self.max_depth = max_depth
        self.playwright_mode = "auto" if use_playwright and playwright_mode == "off" else playwright_mode
        self.resume = resume
        self.review_only = review_only
        self.seed_urls = seed_urls or []
        self.discovery_mode = discovery_mode
        self.include_discovery_guesses = include_discovery_guesses
        self.disable_official_homepage_overrides = disable_official_homepage_overrides
        self.rerank_discovery = rerank_discovery
        self.latest_only = latest_only
        self.search_timeout_seconds = search_timeout_seconds
        self.llm_timeout_seconds = llm_timeout_seconds
        self.navigation_llm_max_links = navigation_llm_max_links
        self.page_llm_max_links = page_llm_max_links
        self.llm_text_chars = llm_text_chars
        self.progress = progress or ProgressReporter(enabled=False)
        self.http = http or HttpClient()
        self.renderer = PlaywrightRenderer(self.http) if self.playwright_mode != "off" else None

    def crawl_company(self, company: Company) -> CrawlResult:
        self.progress.log(f"{company.symbol}: crawl starting")
        seeds = self._discover_seeds(company)
        seeds = [resolve_document_url(seed) for seed in seeds]
        if not seeds:
            self.progress.log(f"{company.symbol}: crawl skipped; no investor-relations candidates found")
            return CrawlResult(company=company, skipped_reason="No investor-relations candidates found")

        result = CrawlResult(company=company, ir_url=seeds[0])
        self.progress.log(f"{company.symbol}: crawling {len(seeds)} seed URL(s)")
        queue: deque[tuple[str, int]] = deque((seed, 0) for seed in seeds)
        run_visited: set[str] = set()
        allowed_hosts = {host(seed) for seed in seeds}
        state = CrawlState.load(self._state_path(company)) if self.resume else CrawlState(path=self._state_path(company))

        while queue and len(run_visited) < self.max_pages_per_company:
            url, depth = queue.popleft()
            url = resolve_document_url(url)
            normalized = normalize_url(url)
            if normalized in run_visited or depth > self.max_depth:
                continue
            if state.has_visited(url):
                continue
            if host(url) not in allowed_hosts:
                continue

            run_visited.add(normalized)
            try:
                self.progress.log(f"{company.symbol}: fetching depth={depth} {url}")
                response = self.http.get(url)
            except RobotsDisallowedError as exc:
                result.failures.append(self._failure(company, url, "robots_disallowed", exc))
                continue
            except RobotsUnavailableError as exc:
                result.failures.append(self._failure(company, url, "robots_unavailable", exc))
                continue
            except PermissionError as exc:
                result.failures.append(self._failure(company, url, "robots_blocked", exc))
                continue
            except TimeoutError as exc:
                result.failures.append(self._failure(company, url, "timeout", exc))
                continue
            except Exception as exc:
                result.failures.append(self._failure(company, url, "http_error", exc))
                continue
            state.mark_visited(url)

            content_type = response.headers.get("content-type", "").lower()
            if "application/pdf" in content_type or url.lower().endswith(".pdf"):
                self._cache_bytes(company, url, response.content, suffix=".pdf")
                record = self._record_pdf(company, url, response.content, state, result)
                if record:
                    result.transcripts.append(record)
                continue
            if is_docx_response(url, content_type, response.content):
                self._cache_bytes(company, url, response.content, suffix=".docx")
                if not self.review_only:
                    record = self._record_docx(company, url, response.content, state, result)
                    if record:
                        result.transcripts.append(record)
                else:
                    try:
                        text = docx_text(response.content)
                    except Exception as exc:
                        result.failures.append(self._failure(company, url, "parse_error", exc))
                        continue
                    detection = classify_transcript(text, title=url.rstrip("/").split("/")[-1], url=url)
                    result.candidates.append(
                        CandidatePage(
                            company=company,
                            url=url,
                            title=url.rstrip("/").split("/")[-1] or "transcript.docx",
                            depth=depth,
                            heuristic_score=link_score(CandidateLink(url=url, source_url=url, label="docx transcript")),
                            llm_page_type="transcript" if detection.is_transcript else "not_relevant",
                            llm_confidence=1.0 if detection.is_transcript else None,
                            reason=detection.reason,
                        )
                    )
                continue

            html = response.text
            self._cache_text(company, url, html, suffix=".html")
            title = page_title(html)
            text = visible_text(html)
            rendered = False
            if self.renderer and should_render_html(self.playwright_mode, html, text):
                try:
                    self.progress.log(f"{company.symbol}: rendering {url}")
                    html = self.renderer.render_html(url)
                    self._cache_text(company, url, html, suffix=".rendered.html")
                    title = page_title(html)
                    text = visible_text(html)
                    rendered = True
                except Exception as exc:
                    result.failures.append(self._failure(company, url, "playwright_error", exc))
                    rendered = False
            links = extract_links(html, url)
            detection = classify_transcript(text, title=title, url=url)

            if detection.is_transcript:
                if not self.review_only:
                    record = self._record_html(company, url, title, html, text, state, result, rendered=rendered)
                    if record:
                        result.transcripts.append(record)

            try:
                self.progress.log(f"{company.symbol}: asking Ollama to classify page {url}")
                with timeout_after(self.llm_timeout_seconds, f"classifying page {url}"):
                    decision = self.agent.decide(
                        company_name=company.name,
                        ticker=company.symbol,
                        url=url,
                        title=title,
                        text=text,
                        links=self._prioritize_links(links),
                    )
            except Exception as exc:
                self.progress.log(f"{company.symbol}: page LLM skipped ({type(exc).__name__}: {url})")
                result.failures.append(self._failure(company, url, "llm_error", exc))
                decision = None

            candidate = CandidatePage(
                company=company,
                url=url,
                title=title,
                depth=depth,
                heuristic_score=max((link_score(link) for link in links), default=link_score(CandidateLink(url=url, source_url=url))),
                llm_page_type=decision.page_type if decision else None,
                llm_confidence=decision.confidence if decision else None,
                reason=candidate_reason(decision.reason if decision else "", detection.reason),
            )
            result.candidates.append(candidate)

            if decision:
                for link in decision.useful_links:
                    if self._should_follow(link, allowed_hosts, current_url=url):
                        allowed_hosts.add(host(link.url))
                        queue.append((link.url, depth + 1))

            for link in self._heuristic_links(links):
                if self._should_follow(link, allowed_hosts, current_url=url):
                    allowed_hosts.add(host(link.url))
                    queue.append((link.url, depth + 1))

        result.visited_count = len(run_visited)
        if self.latest_only and len(result.transcripts) > 1:
            before = len(result.transcripts)
            result.transcripts = keep_latest_transcripts(result.transcripts)
            self._remove_filtered_transcripts(company, result.transcripts)
            self.progress.log(f"{company.symbol}: latest-only kept {len(result.transcripts)} of {before} transcript(s)")
        self._write_company_index(result)
        self._write_failures(result)
        self._write_candidates(result)
        state.save()
        self.progress.log(
            f"{company.symbol}: crawl finished with {len(result.transcripts)} transcript(s), "
            f"{result.visited_count} page(s) visited"
        )
        return result

    def _discover_seeds(self, company: Company) -> list[str]:
        if self.seed_urls:
            self.progress.log(f"{company.symbol}: using {len(self.seed_urls)} provided seed URL(s)")
            return self.seed_urls
        if self.discovery_mode == "search-first":
            return self._search_seeds(company)

        self.progress.log(f"{company.symbol}: nav-first discovery")
        navigation = discover_navigation_seeds(
            company,
            http=self.http,
            model=self.model,
            ollama_base_url=self.ollama_base_url,
            include_guesses=self.include_discovery_guesses,
            playwright_mode=self.playwright_mode,
            disable_official_homepage_overrides=self.disable_official_homepage_overrides,
            search_timeout_seconds=self.search_timeout_seconds,
            llm_timeout_seconds=self.llm_timeout_seconds,
            navigation_llm_max_links=self.navigation_llm_max_links,
            llm_text_chars=self.llm_text_chars,
            progress=self.progress,
        )
        self._write_navigation_trace(navigation.trace)
        if navigation.seeds:
            self.progress.log(f"{company.symbol}: nav-first discovery produced {len(navigation.seeds)} seed(s)")
            return navigation.seeds
        self.progress.log(f"{company.symbol}: nav-first discovery empty; falling back to search")
        return self._search_seeds(company)

    def _search_seeds(self, company: Company) -> list[str]:
        self.progress.log(f"{company.symbol}: search-first discovery")
        return find_ir_candidates(
            company,
            include_guesses=self.include_discovery_guesses,
            rerank_model=self.model if self.rerank_discovery else None,
            ollama_base_url=self.ollama_base_url,
            search_timeout_seconds=self.search_timeout_seconds,
            llm_timeout_seconds=self.llm_timeout_seconds,
            progress=self.progress,
        )

    def _record_html(
        self,
        company: Company,
        url: str,
        title: str,
        html: str,
        text: str,
        state: CrawlState,
        result: CrawlResult,
        *,
        rendered: bool = False,
    ) -> TranscriptRecord | None:
        if state.has_transcript_url(url):
            return None
        digest = content_hash(text)
        if state.has_content_hash(digest):
            return None
        company_dir = self._company_dir(company)
        stem = artifact_stem(title, url)
        raw_path = company_dir / f"{stem}.html"
        raw_path.write_text(html, encoding="utf-8")
        extracted = self._metadata(title, text, result, url)
        record = TranscriptRecord(
            company=company,
            source_url=url,
            fiscal_period=extracted.fiscal_period,
            call_date=extracted.call_date,
            title=title,
            text=text,
            raw_path=raw_path,
            metadata={
                "format": "html",
                "rendered_with_playwright": str(rendered).lower(),
                "content_hash": digest,
            },
        )
        self._write_record(company, record)
        state.mark_transcript_url(url)
        state.mark_content_hash(digest)
        return record

    def _record_pdf(
        self,
        company: Company,
        url: str,
        content: bytes,
        state: CrawlState,
        result: CrawlResult,
    ) -> TranscriptRecord | None:
        try:
            text = pdf_text(content)
        except Exception as exc:
            result.failures.append(self._failure(company, url, "pdf_error", exc))
            return None
        title = url.rstrip("/").split("/")[-1] or "transcript.pdf"
        if not looks_like_transcript(text, title=title, url=url):
            result.failures.append(self._failure(company, url, "not_transcript"))
            return None
        if state.has_transcript_url(url):
            return None
        digest = content_hash(text)
        if state.has_content_hash(digest):
            return None
        company_dir = self._company_dir(company)
        raw_path = company_dir / f"{artifact_stem(title, url)}.pdf"
        raw_path.write_bytes(content)
        extracted = self._metadata(title, text, result, url)
        record = TranscriptRecord(
            company=company,
            source_url=url,
            fiscal_period=extracted.fiscal_period,
            call_date=extracted.call_date,
            title=title,
            text=text,
            raw_path=raw_path,
            metadata={"format": "pdf", "content_hash": digest},
        )
        self._write_record(company, record)
        state.mark_transcript_url(url)
        state.mark_content_hash(digest)
        return record

    def _record_docx(
        self,
        company: Company,
        url: str,
        content: bytes,
        state: CrawlState,
        result: CrawlResult,
    ) -> TranscriptRecord | None:
        try:
            text = docx_text(content)
        except Exception as exc:
            result.failures.append(self._failure(company, url, "parse_error", exc))
            return None
        title = url.rstrip("/").split("/")[-1] or "transcript.docx"
        if not looks_like_transcript(text, title=title, url=url):
            result.failures.append(self._failure(company, url, "not_transcript"))
            return None
        if state.has_transcript_url(url):
            return None
        digest = content_hash(text)
        if state.has_content_hash(digest):
            return None
        company_dir = self._company_dir(company)
        raw_path = company_dir / f"{artifact_stem(title, url)}.docx"
        raw_path.write_bytes(content)
        extracted = self._metadata(title, text, result, url)
        record = TranscriptRecord(
            company=company,
            source_url=url,
            fiscal_period=extracted.fiscal_period,
            call_date=extracted.call_date,
            title=title,
            text=text,
            raw_path=raw_path,
            metadata={"format": "docx", "content_hash": digest},
        )
        self._write_record(company, record)
        state.mark_transcript_url(url)
        state.mark_content_hash(digest)
        return record

    def _company_dir(self, company: Company) -> Path:
        path = self.out_dir / company.symbol
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _cache_dir(self, company: Company) -> Path:
        path = self._company_dir(company) / "_page_cache"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _cache_text(self, company: Company, url: str, content: str, *, suffix: str) -> Path:
        path = self._cache_dir(company) / f"{artifact_stem('page', url)}{suffix}"
        path.write_text(content, encoding="utf-8")
        return path

    def _cache_bytes(self, company: Company, url: str, content: bytes, *, suffix: str) -> Path:
        path = self._cache_dir(company) / f"{artifact_stem('page', url)}{suffix}"
        path.write_bytes(content)
        return path

    def _write_record(self, company: Company, record: TranscriptRecord) -> None:
        company_dir = self._company_dir(company)
        path = company_dir / f"{artifact_stem(record.title, str(record.source_url))}.json"
        path.write_text(record.model_dump_json(indent=2), encoding="utf-8")

    def _write_company_index(self, result: CrawlResult) -> None:
        path = self._company_dir(result.company) / "_crawl_result.json"
        path.write_text(result.model_dump_json(indent=2), encoding="utf-8")

    def _write_failures(self, result: CrawlResult) -> None:
        path = self._company_dir(result.company) / "_failures.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            for failure in result.failures:
                handle.write(failure.model_dump_json() + "\n")

    def _write_candidates(self, result: CrawlResult) -> None:
        path = self._company_dir(result.company) / "_candidates.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            for candidate in result.candidates:
                handle.write(candidate.model_dump_json() + "\n")

    def _write_navigation_trace(self, trace) -> None:
        path = self._company_dir(trace.company) / "_navigation_trace.json"
        path.write_text(trace.model_dump_json(indent=2), encoding="utf-8")

    def _remove_filtered_transcripts(self, company: Company, kept: list[TranscriptRecord]) -> None:
        company_dir = self._company_dir(company)
        kept_json_names = {
            f"{artifact_stem(record.title, str(record.source_url))}.json"
            for record in kept
        }
        kept_raw_paths = {
            Path(record.raw_path).resolve()
            for record in kept
            if record.raw_path
        }
        for path in company_dir.iterdir():
            if path.name.startswith("_"):
                continue
            if path.suffix == ".json" and path.name not in kept_json_names:
                path.unlink(missing_ok=True)
                continue
            if path.suffix in {".html", ".pdf", ".docx"} and path.resolve() not in kept_raw_paths:
                path.unlink(missing_ok=True)

    def _prioritize_links(self, links: list[CandidateLink]) -> list[CandidateLink]:
        return sorted(links, key=lambda link: link_score(link), reverse=True)[:80]

    def _heuristic_links(self, links: list[CandidateLink]) -> list[CandidateLink]:
        scored_links = [link for link in links if link_score(link) >= 2]
        return sorted(scored_links, key=lambda link: link_score(link), reverse=True)[:30]

    def _should_follow(self, link: CandidateLink, allowed_hosts: set[str], *, current_url: str) -> bool:
        link.url = resolve_document_url(link.url)
        parsed = urlparse(link.url)
        if parsed.scheme not in {"http", "https"}:
            return False
        link_host = host(link.url)
        if link_host not in allowed_hosts and not can_expand_host(link, current_url):
            return False
        return link_score(link) > 0

    def _state_path(self, company: Company) -> Path:
        return self._company_dir(company) / "_crawl_state.json"

    def _failure(
        self,
        company: Company,
        url: str,
        failure_type: FailureType,
        exc: Exception | None = None,
    ) -> CrawlFailure:
        return CrawlFailure(company=company, url=url, failure_type=failure_type, message=str(exc or ""))

    def _metadata(self, title: str, text: str, result: CrawlResult, url: str):
        if self.metadata_agent:
            try:
                return self.metadata_agent.extract(title, text)
            except Exception as exc:
                result.failures.append(self._failure(result.company, url, "llm_error", exc))
        return extract_metadata_heuristic(title, text)


def link_score(link: CandidateLink) -> int:
    haystack = f"{link.url} {link.label}".lower()
    score = 0
    score += sum(3 for hint in TRANSCRIPT_HINTS if hint in haystack)
    score += sum(1 for hint in IR_HINTS if hint in haystack)
    if "transcript" in haystack and urlparse(link.url.lower()).path.endswith(TRANSCRIPT_DOCUMENT_EXTENSIONS):
        score += 12
    if "earnings-call" in haystack or "earnings call" in haystack:
        score += 4
    return score


def can_expand_host(link: CandidateLink, current_url: str) -> bool:
    if host(link.url) == host(current_url):
        return True
    haystack = f"{link.url} {link.label}".lower()
    vendor_markers = (
        "investor",
        "ir.",
        "q4cdn",
        "cdn-dynmedia",
        "microsoftcorp",
        "events",
        "earnings",
        "webcast",
        "quarter",
        "transcript",
    )
    return any(marker in haystack for marker in vendor_markers) and link_score(link) >= 2


def candidate_reason(*parts: str) -> str:
    return "; ".join(part for part in parts if part)


def safe_name(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in ("-", "_", ".") else "_" for char in value)
    return "_".join(part for part in cleaned.split("_") if part) or "untitled"


def artifact_stem(title: str, url: str) -> str:
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]
    return f"{safe_name(title)[:70]}_{digest}"


def content_hash(text: str) -> str:
    normalized = " ".join(text.split()).lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def is_docx_response(url: str, content_type: str, content: bytes) -> bool:
    return (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document" in content_type
        or url.lower().endswith(".docx")
        or content.startswith(b"PK\x03\x04")
    )


def keep_latest_transcripts(records: list[TranscriptRecord]) -> list[TranscriptRecord]:
    if not records:
        return []
    latest = max(transcript_sort_key(record) for record in records)
    return [record for record in records if transcript_sort_key(record) == latest]


def transcript_sort_key(record: TranscriptRecord) -> tuple[int, int, str]:
    haystack = f"{record.fiscal_period or ''} {record.title} {record.source_url}".lower()
    if record.call_date:
        return (record.call_date.year, quarter_from_month(record.call_date.month), str(record.source_url))
    fiscal = fiscal_period_key(haystack)
    if fiscal:
        return (*fiscal, str(record.source_url))
    return (0, 0, str(record.source_url))


def fiscal_period_key(text: str) -> tuple[int, int] | None:
    quarter_words = {"first": 1, "second": 2, "third": 3, "fourth": 4}
    matches: list[tuple[int, int]] = []
    for match in re.finditer(r"\b(?:fy[-\s]?)?(20\d{2})\D{0,20}\bq([1-4])\b", text, re.I):
        matches.append((int(match.group(1)), int(match.group(2))))
    for match in re.finditer(r"\bq([1-4])\D{0,20}\b(?:fy[-\s]?)?(20\d{2})\b", text, re.I):
        matches.append((int(match.group(2)), int(match.group(1))))
    for match in re.finditer(r"\b(first|second|third|fourth) quarter\D{0,20}\b(20\d{2})\b", text, re.I):
        matches.append((int(match.group(2)), quarter_words[match.group(1).lower()]))
    for match in re.finditer(r"\b(20\d{2})\D{0,20}\b(first|second|third|fourth) quarter\b", text, re.I):
        matches.append((int(match.group(1)), quarter_words[match.group(2).lower()]))
    return max(matches) if matches else None


def quarter_from_month(month: int) -> int:
    return ((month - 1) // 3) + 1


def should_render_html(playwright_mode: str, html: str, text: str) -> bool:
    if playwright_mode == "always":
        return True
    if playwright_mode == "auto":
        return looks_like_js_shell(html, text)
    return False
