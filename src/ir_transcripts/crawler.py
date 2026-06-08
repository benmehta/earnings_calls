from __future__ import annotations

from collections import deque
from pathlib import Path
from urllib.parse import urlparse

from .agent import IRPageAgent
from .http import HttpClient
from .models import CandidateLink, Company, CrawlResult, TranscriptRecord
from .parsing import extract_links, looks_like_transcript, page_title, pdf_text, visible_text
from .search import find_ir_candidates


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
        http: HttpClient | None = None,
    ) -> None:
        self.agent = IRPageAgent(model)
        self.out_dir = out_dir
        self.max_pages_per_company = max_pages_per_company
        self.max_depth = max_depth
        self.http = http or HttpClient()

    def crawl_company(self, company: Company) -> CrawlResult:
        seeds = find_ir_candidates(company)
        if not seeds:
            return CrawlResult(company=company, skipped_reason="No investor-relations candidates found")

        result = CrawlResult(company=company, ir_url=seeds[0])
        queue: deque[tuple[str, int]] = deque((seed, 0) for seed in seeds)
        visited: set[str] = set()
        allowed_hosts = {urlparse(seed).netloc for seed in seeds}

        while queue and len(visited) < self.max_pages_per_company:
            url, depth = queue.popleft()
            normalized = url.rstrip("/")
            if normalized in visited or depth > self.max_depth:
                continue
            if urlparse(url).netloc not in allowed_hosts:
                continue

            visited.add(normalized)
            try:
                response = self.http.get(url)
            except Exception:
                continue

            content_type = response.headers.get("content-type", "").lower()
            if "application/pdf" in content_type or url.lower().endswith(".pdf"):
                record = self._record_pdf(company, url, response.content)
                if record:
                    result.transcripts.append(record)
                continue

            html = response.text
            title = page_title(html)
            text = visible_text(html)
            links = extract_links(html, url)

            if looks_like_transcript(text):
                result.transcripts.append(self._record_html(company, url, title, html, text))

            try:
                decision = self.agent.decide(
                    company_name=company.name,
                    ticker=company.symbol,
                    url=url,
                    title=title,
                    text=text,
                    links=self._prioritize_links(links),
                )
            except Exception:
                decision = None

            if decision:
                if decision.page_type == "transcript" and not looks_like_transcript(text):
                    result.transcripts.append(self._record_html(company, url, title, html, text))

                for link in decision.useful_links:
                    if self._should_follow(link, allowed_hosts):
                        queue.append((link.url, depth + 1))

            for link in self._heuristic_links(links):
                if self._should_follow(link, allowed_hosts):
                    queue.append((link.url, depth + 1))

        result.visited_count = len(visited)
        self._write_company_index(result)
        return result

    def _record_html(self, company: Company, url: str, title: str, html: str, text: str) -> TranscriptRecord:
        company_dir = self._company_dir(company)
        raw_path = company_dir / f"{safe_name(title)[:80]}.html"
        raw_path.write_text(html, encoding="utf-8")
        record = TranscriptRecord(company=company, source_url=url, title=title, text=text, raw_path=raw_path)
        self._write_record(company, record)
        return record

    def _record_pdf(self, company: Company, url: str, content: bytes) -> TranscriptRecord | None:
        text = pdf_text(content)
        if not looks_like_transcript(text):
            return None
        company_dir = self._company_dir(company)
        title = url.rstrip("/").split("/")[-1] or "transcript.pdf"
        raw_path = company_dir / safe_name(title)
        raw_path.write_bytes(content)
        record = TranscriptRecord(company=company, source_url=url, title=title, text=text, raw_path=raw_path)
        self._write_record(company, record)
        return record

    def _company_dir(self, company: Company) -> Path:
        path = self.out_dir / company.symbol
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _write_record(self, company: Company, record: TranscriptRecord) -> None:
        company_dir = self._company_dir(company)
        path = company_dir / f"{safe_name(record.title)[:80]}.json"
        path.write_text(record.model_dump_json(indent=2), encoding="utf-8")

    def _write_company_index(self, result: CrawlResult) -> None:
        path = self._company_dir(result.company) / "_crawl_result.json"
        path.write_text(result.model_dump_json(indent=2), encoding="utf-8")

    def _prioritize_links(self, links: list[CandidateLink]) -> list[CandidateLink]:
        return sorted(links, key=lambda link: link_score(link), reverse=True)[:80]

    def _heuristic_links(self, links: list[CandidateLink]) -> list[CandidateLink]:
        return [link for link in links if link_score(link) >= 2][:30]

    def _should_follow(self, link: CandidateLink, allowed_hosts: set[str]) -> bool:
        parsed = urlparse(link.url)
        if parsed.scheme not in {"http", "https"}:
            return False
        if parsed.netloc not in allowed_hosts:
            return False
        return link_score(link) > 0


def link_score(link: CandidateLink) -> int:
    haystack = f"{link.url} {link.label}".lower()
    score = 0
    score += sum(3 for hint in TRANSCRIPT_HINTS if hint in haystack)
    score += sum(1 for hint in IR_HINTS if hint in haystack)
    return score


def safe_name(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in ("-", "_", ".") else "_" for char in value)
    return "_".join(part for part in cleaned.split("_") if part) or "untitled"
