from __future__ import annotations

import hashlib
from collections import deque
from pathlib import Path
from urllib.parse import urlparse

from .agent import IRPageAgent
from .browser import PlaywrightRenderer
from .http import HttpClient, RobotsDisallowedError, RobotsUnavailableError
from .metadata import TranscriptMetadataAgent, extract_metadata_heuristic
from .models import CandidateLink, CandidatePage, Company, CrawlFailure, CrawlResult, FailureType, TranscriptRecord
from .parsing import extract_links, looks_like_js_shell, looks_like_transcript, page_title, pdf_text, visible_text
from .search import find_ir_candidates
from .state import CrawlState
from .urls import host, normalize_url


TRANSCRIPT_HINTS = ("transcript", "earnings-call", "earnings call", "quarterly-results")
IR_HINTS = ("investor", "/ir", "shareholder", "financial", "events", "earnings", "quarter")


class TranscriptCrawler:
    def __init__(
        self,
        *,
        model: str,
        out_dir: Path,
        max_pages_per_company: int = 40,
        max_depth: int = 3,
        use_playwright: bool = False,
        ollama_base_url: str | None = None,
        resume: bool = True,
        review_only: bool = False,
        seed_urls: list[str] | None = None,
        include_discovery_guesses: bool = False,
        rerank_discovery: bool = False,
        extract_metadata_with_llm: bool = False,
        http: HttpClient | None = None,
    ) -> None:
        self.model = model
        self.ollama_base_url = ollama_base_url
        self.agent = IRPageAgent(model, base_url=ollama_base_url)
        self.metadata_agent = TranscriptMetadataAgent(model, base_url=ollama_base_url) if extract_metadata_with_llm else None
        self.out_dir = out_dir
        self.max_pages_per_company = max_pages_per_company
        self.max_depth = max_depth
        self.resume = resume
        self.review_only = review_only
        self.seed_urls = seed_urls or []
        self.include_discovery_guesses = include_discovery_guesses
        self.rerank_discovery = rerank_discovery
        self.http = http or HttpClient()
        self.renderer = PlaywrightRenderer(self.http) if use_playwright else None

    def crawl_company(self, company: Company) -> CrawlResult:
        seeds = self.seed_urls or find_ir_candidates(
            company,
            include_guesses=self.include_discovery_guesses,
            rerank_model=self.model if self.rerank_discovery else None,
            ollama_base_url=self.ollama_base_url,
        )
        if not seeds:
            return CrawlResult(company=company, skipped_reason="No investor-relations candidates found")

        result = CrawlResult(company=company, ir_url=seeds[0])
        queue: deque[tuple[str, int]] = deque((seed, 0) for seed in seeds)
        run_visited: set[str] = set()
        allowed_hosts = {host(seed) for seed in seeds}
        state = CrawlState.load(self._state_path(company)) if self.resume else CrawlState(path=self._state_path(company))

        while queue and len(run_visited) < self.max_pages_per_company:
            url, depth = queue.popleft()
            normalized = normalize_url(url)
            if normalized in run_visited or depth > self.max_depth:
                continue
            if state.has_visited(url):
                continue
            if host(url) not in allowed_hosts:
                continue

            run_visited.add(normalized)
            try:
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

            html = response.text
            self._cache_text(company, url, html, suffix=".html")
            title = page_title(html)
            text = visible_text(html)
            rendered = False
            if self.renderer and looks_like_js_shell(html, text):
                try:
                    html = self.renderer.render_html(url)
                    self._cache_text(company, url, html, suffix=".rendered.html")
                    title = page_title(html)
                    text = visible_text(html)
                    rendered = True
                except Exception as exc:
                    result.failures.append(self._failure(company, url, "playwright_error", exc))
                    rendered = False
            links = extract_links(html, url)

            if looks_like_transcript(text):
                if not self.review_only:
                    record = self._record_html(company, url, title, html, text, state, result, rendered=rendered)
                    if record:
                        result.transcripts.append(record)

            try:
                decision = self.agent.decide(
                    company_name=company.name,
                    ticker=company.symbol,
                    url=url,
                    title=title,
                    text=text,
                    links=self._prioritize_links(links),
                )
            except Exception as exc:
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
                reason=decision.reason if decision else "",
            )
            result.candidates.append(candidate)

            if decision:
                if decision.page_type == "transcript" and not looks_like_transcript(text):
                    if not self.review_only:
                        record = self._record_html(company, url, title, html, text, state, result, rendered=rendered)
                        if record:
                            result.transcripts.append(record)

                for link in decision.useful_links:
                    if self._should_follow(link, allowed_hosts, current_url=url):
                        allowed_hosts.add(host(link.url))
                        queue.append((link.url, depth + 1))

            for link in self._heuristic_links(links):
                if self._should_follow(link, allowed_hosts, current_url=url):
                    allowed_hosts.add(host(link.url))
                    queue.append((link.url, depth + 1))

        result.visited_count = len(run_visited)
        self._write_company_index(result)
        self._write_failures(result)
        self._write_candidates(result)
        state.save()
        return result

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
        if not looks_like_transcript(text):
            result.failures.append(self._failure(company, url, "not_transcript"))
            return None
        if state.has_transcript_url(url):
            return None
        digest = content_hash(text)
        if state.has_content_hash(digest):
            return None
        company_dir = self._company_dir(company)
        title = url.rstrip("/").split("/")[-1] or "transcript.pdf"
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

    def _prioritize_links(self, links: list[CandidateLink]) -> list[CandidateLink]:
        return sorted(links, key=lambda link: link_score(link), reverse=True)[:80]

    def _heuristic_links(self, links: list[CandidateLink]) -> list[CandidateLink]:
        return [link for link in links if link_score(link) >= 2][:30]

    def _should_follow(self, link: CandidateLink, allowed_hosts: set[str], *, current_url: str) -> bool:
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
    return score


def can_expand_host(link: CandidateLink, current_url: str) -> bool:
    if host(link.url) == host(current_url):
        return True
    haystack = f"{link.url} {link.label}".lower()
    vendor_markers = ("investor", "ir.", "q4cdn", "events", "earnings", "webcast", "quarter")
    return any(marker in haystack for marker in vendor_markers) and link_score(link) >= 2


def safe_name(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in ("-", "_", ".") else "_" for char in value)
    return "_".join(part for part in cleaned.split("_") if part) or "untitled"


def artifact_stem(title: str, url: str) -> str:
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]
    return f"{safe_name(title)[:70]}_{digest}"


def content_hash(text: str) -> str:
    normalized = " ".join(text.split()).lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
