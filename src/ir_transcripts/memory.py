from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

from .models import Company, CompanyMemory, CrawlReflection, CrawlResult, FailureAnalysis, PromptGuidance
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
            if is_event_listing_url(url) and not low_value_memory_path(url):
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


def apply_crawl_reflection(memory: CompanyMemory, reflection: CrawlReflection) -> CompanyMemory:
    navigation_memory = memory.navigation_memory
    reflection_preferred_hosts = {host(url) for url in reflection.preferred_urls if host(url)}
    for url in reflection.preferred_urls:
        memory.known_ir_urls = append_unique(memory.known_ir_urls, url)
        if is_ir_home_url(url):
            navigation_memory.known_ir_home_urls = append_unique(navigation_memory.known_ir_home_urls, url)
        if is_event_listing_url(url) or looks_like_quarterly_detail_url(url):
            navigation_memory.known_event_listing_urls = append_unique(navigation_memory.known_event_listing_urls, url)
    for url in reflection.avoid_urls:
        memory.rejected_urls = append_unique(memory.rejected_urls, url)
        remember_low_value_url(navigation_memory.low_value_hosts, navigation_memory.low_value_path_terms, url)
        avoided_host = host(url)
        if (
            avoided_host
            and avoided_host not in reflection_preferred_hosts
            and avoided_host not in navigation_memory.preferred_hosts
            and avoided_host not in navigation_memory.successful_hosts
            and not low_value_memory_path(url)
        ):
            navigation_memory.low_value_hosts = append_unique(navigation_memory.low_value_hosts, avoided_host)
    for term in reflection.avoid_terms:
        navigation_memory.low_value_path_terms = append_unique(navigation_memory.low_value_path_terms, term)
    if reflection.prompt_guidance or reflection.preferred_terms or reflection.avoid_terms:
        reflected_guidance = reflection.prompt_guidance.model_copy(
            update={
                "priority_terms": merge_unique(
                    reflection.prompt_guidance.priority_terms,
                    reflection.preferred_terms,
                    limit=10,
                ),
                "avoid_terms": merge_unique(
                    reflection.prompt_guidance.avoid_terms,
                    reflection.avoid_terms,
                    limit=10,
                ),
            }
        )
        memory.prompt_guidance = merge_prompt_guidance(memory.prompt_guidance, reflected_guidance)
    return memory


def merge_prompt_guidance(existing: PromptGuidance | None, reflected: PromptGuidance) -> PromptGuidance:
    if existing is None:
        return reflected
    return PromptGuidance(
        priority_terms=merge_unique(reflected.priority_terms, existing.priority_terms, limit=10),
        avoid_terms=merge_unique(reflected.avoid_terms, existing.avoid_terms, limit=10),
        navigation_guidance=reflected.navigation_guidance or existing.navigation_guidance,
        transcript_guidance=reflected.transcript_guidance or existing.transcript_guidance,
        risk_notes=merge_unique(reflected.risk_notes, existing.risk_notes, limit=5),
    )


def merge_unique(primary: list[str], secondary: list[str], *, limit: int) -> list[str]:
    merged: list[str] = []
    for value in [*primary, *secondary]:
        append_unique(merged, value)
        if len(merged) >= limit:
            break
    return merged


def append_unique(values: list[str], value: str) -> list[str]:
    if value not in values:
        values.append(value)
    return values


def low_value_memory_host(value: str) -> bool:
    return any(token in value.lower() for token in ("blog.", "youtube.com", "youtu.be"))


def low_value_memory_path(url: str) -> bool:
    path = urlparse(url).path.lower()
    return any(
        token in path
        for token in (
            "agm",
            "annual-meeting",
            "annual-reports",
            "board-of-directors",
            "governance",
            "japanese/",
            "press-release",
            "proxy",
            "schinese/",
            "shareholders-meeting",
            "shareholders-meetings",
            "zh/",
            "chinese/",
        )
    )


def is_ir_home_url(url: str) -> bool:
    path = urlparse(url).path.lower().rstrip("/")
    return path in {"", "/", "/investor", "/investors", "/ir", "/investor/default.aspx", "/home/default.aspx"}


def is_event_listing_url(url: str) -> bool:
    path = urlparse(url).path.lower()
    return any(token in path for token in ("/events", "/earnings", "/financial-results", "/financial-reports")) and "event-details" not in path


def looks_like_quarterly_detail_url(url: str) -> bool:
    path = urlparse(url).path.lower()
    return "/quarterly-results/" in path and any(f"/q{quarter}" in path for quarter in range(1, 5))


def remember_low_value_url(low_value_hosts: list[str], low_value_path_terms: list[str], url: str) -> None:
    url_host = host(url)
    path = urlparse(url).path.lower()
    if url_host and low_value_memory_host(url_host):
        append_unique(low_value_hosts, url_host)
    for token in (
        "agm",
        "annual-reports",
        "blog",
        "board-of-directors",
        "chinese/",
        "governance",
        "japanese/",
        "presentation",
        "press-release",
        "proxy",
        "schinese/",
        "shareholders-meeting",
        "shareholders-meetings",
        "webcast",
        "youtube",
    ):
        if token in f"{url_host or ''} {path}":
            append_unique(low_value_path_terms, token)
