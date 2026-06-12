from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

from .models import Company, CompanyMemory, CrawlResult, FailureAnalysis
from .urls import host


MEMORY_FILENAME = "_company_memory.json"


def memory_path(out_dir: Path, company: Company) -> Path:
    return out_dir / company.symbol / MEMORY_FILENAME


def load_company_memory(out_dir: Path, company: Company) -> CompanyMemory:
    path = memory_path(out_dir, company)
    if path.exists():
        return CompanyMemory.model_validate_json(path.read_text(encoding="utf-8"))
    return CompanyMemory(company=company)


def save_company_memory(out_dir: Path, memory: CompanyMemory) -> Path:
    path = memory_path(out_dir, memory.company)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(memory.model_dump_json(indent=2), encoding="utf-8")
    return path


def remember_crawl_result(memory: CompanyMemory, result: CrawlResult, analysis: FailureAnalysis) -> CompanyMemory:
    navigation_memory = memory.navigation_memory
    for url in [result.ir_url, *(candidate.url for candidate in result.candidates)]:
        if url:
            memory.known_ir_urls = append_unique(memory.known_ir_urls, url)
            url_host = host(url)
            if url_host and not low_value_memory_host(url_host):
                memory.official_hosts = append_unique(memory.official_hosts, url_host)
            if is_ir_home_url(url):
                navigation_memory.known_ir_home_urls = append_unique(navigation_memory.known_ir_home_urls, url)
            if is_event_listing_url(url):
                navigation_memory.known_event_listing_urls = append_unique(navigation_memory.known_event_listing_urls, url)
            remember_low_value_url(navigation_memory.low_value_hosts, navigation_memory.low_value_path_terms, url)

    for failure in result.failures:
        memory.rejected_urls = append_unique(memory.rejected_urls, failure.url)
        failure_host = host(failure.url)
        if failure_host and failure.failure_type in {"robots_blocked", "robots_disallowed", "robots_unavailable"}:
            navigation_memory.robots_blocked_hosts = append_unique(navigation_memory.robots_blocked_hosts, failure_host)
        remember_low_value_url(navigation_memory.low_value_hosts, navigation_memory.low_value_path_terms, failure.url)

    for record in result.transcripts:
        source_url = str(record.source_url)
        memory.successful_transcript_urls = append_unique(memory.successful_transcript_urls, source_url)
        navigation_memory.known_transcript_urls = append_unique(navigation_memory.known_transcript_urls, source_url)
        transcript_host = host(source_url)
        if transcript_host:
            navigation_memory.successful_hosts = append_unique(navigation_memory.successful_hosts, transcript_host)
            navigation_memory.preferred_hosts = append_unique(navigation_memory.preferred_hosts, transcript_host)
        path = urlparse(source_url).path
        if path:
            navigation_memory.successful_paths = append_unique(navigation_memory.successful_paths, path)

    memory.failure_summaries.append(analysis)
    for recommendation in analysis.manual_recommendations:
        memory.recommended_manual_actions = append_unique(memory.recommended_manual_actions, recommendation)
    return memory


def append_unique(values: list[str], value: str) -> list[str]:
    if value not in values:
        values.append(value)
    return values


def low_value_memory_host(value: str) -> bool:
    return any(token in value.lower() for token in ("blog.", "youtube.com", "youtu.be"))


def is_ir_home_url(url: str) -> bool:
    path = urlparse(url).path.lower().rstrip("/")
    return path in {"", "/", "/investor", "/investors", "/ir", "/investor/default.aspx", "/home/default.aspx"}


def is_event_listing_url(url: str) -> bool:
    path = urlparse(url).path.lower()
    return any(token in path for token in ("/events", "/earnings", "/financial-results", "/financial-reports")) and "event-details" not in path


def remember_low_value_url(low_value_hosts: list[str], low_value_path_terms: list[str], url: str) -> None:
    url_host = host(url)
    path = urlparse(url).path.lower()
    if url_host and low_value_memory_host(url_host):
        append_unique(low_value_hosts, url_host)
    for token in ("blog", "youtube", "presentation", "webcast", "press-release"):
        if token in f"{url_host or ''} {path}":
            append_unique(low_value_path_terms, token)
