from __future__ import annotations

import json
from pathlib import Path
from typing import TypeVar
from urllib.parse import urlparse

from pydantic import BaseModel

from .identity import is_weak_identity
from .models import (
    Company,
    CompanyMemory,
    CompanyNavigationMemory,
    CompanyPlaybook,
    CrawlReflection,
    CrawlResult,
    FailureAnalysis,
    PromptGuidance,
)
from .urls import host


MEMORY_FILENAME = "_company_memory.json"
ModelT = TypeVar("ModelT", bound=BaseModel)


def memory_path(out_dir: Path, company: Company) -> Path:
    return out_dir / company.symbol / MEMORY_FILENAME


def load_company_memory(out_dir: Path, company: Company) -> CompanyMemory:
    path = memory_path(out_dir, company)
    if path.exists():
        return CompanyMemory.model_validate_json(path.read_text(encoding="utf-8"))
    return CompanyMemory(company=company)


def save_company_memory(out_dir: Path, memory: CompanyMemory, *, merge: bool = True) -> Path:
    path = memory_path(out_dir, memory.company)
    path.parent.mkdir(parents=True, exist_ok=True)
    persisted = load_company_memory(out_dir, memory.company) if merge and path.exists() else None
    saved_memory = merge_company_memory(persisted, memory) if persisted else memory
    path.write_text(saved_memory.model_dump_json(indent=2), encoding="utf-8")
    return path


def merge_company_memory(existing: CompanyMemory, incoming: CompanyMemory) -> CompanyMemory:
    merged = existing.model_copy(deep=True)
    merged.company = merge_company(existing.company, incoming.company)
    merged.official_hosts = merge_unique_strings(existing.official_hosts, incoming.official_hosts)
    merged.ir_hosts = merge_unique_strings(existing.ir_hosts, incoming.ir_hosts)
    merged.official_homepage_urls = merge_unique_strings(existing.official_homepage_urls, incoming.official_homepage_urls)
    merged.successful_path_urls = merge_unique_strings(existing.successful_path_urls, incoming.successful_path_urls)
    merged.known_ir_urls = merge_unique_strings(existing.known_ir_urls, incoming.known_ir_urls)
    merged.playbook = merge_company_playbook(existing.playbook, incoming.playbook)
    merged.attempted_configs = merge_unique_models(existing.attempted_configs, incoming.attempted_configs)
    merged.failure_summaries = merge_unique_models(existing.failure_summaries, incoming.failure_summaries)
    merged.rejected_urls = merge_unique_strings(existing.rejected_urls, incoming.rejected_urls)
    merged.recommended_manual_actions = merge_unique_strings(
        existing.recommended_manual_actions,
        incoming.recommended_manual_actions,
    )
    merged.successful_transcript_urls = merge_unique_strings(
        existing.successful_transcript_urls,
        incoming.successful_transcript_urls,
    )
    merged.prompt_guidance = merge_saved_prompt_guidance(existing.prompt_guidance, incoming.prompt_guidance)
    merged.navigation_memory = merge_navigation_memory(existing.navigation_memory, incoming.navigation_memory)
    return merged


def merge_company(existing: Company, incoming: Company) -> Company:
    incoming_name = existing.name if is_weak_identity(incoming) and not is_weak_identity(existing) else incoming.name
    return existing.model_copy(
        update={
            "symbol": incoming.symbol or existing.symbol,
            "name": incoming_name or existing.name,
            "sector": incoming.sector or existing.sector,
            "sub_industry": incoming.sub_industry or existing.sub_industry,
            "cik": incoming.cik or existing.cik,
        }
    )


def merge_navigation_memory(
    existing: CompanyNavigationMemory,
    incoming: CompanyNavigationMemory,
) -> CompanyNavigationMemory:
    successful_hosts = merge_unique_strings(existing.successful_hosts, incoming.successful_hosts)
    preferred_hosts = merge_unique_strings(existing.preferred_hosts, incoming.preferred_hosts)
    positive_hosts = {*successful_hosts, *preferred_hosts}
    return CompanyNavigationMemory(
        successful_hosts=successful_hosts,
        successful_paths=merge_unique_strings(existing.successful_paths, incoming.successful_paths),
        preferred_hosts=preferred_hosts,
        low_value_hosts=[
            value
            for value in merge_unique_strings(existing.low_value_hosts, incoming.low_value_hosts)
            if value not in positive_hosts
        ],
        low_value_path_terms=merge_unique_strings(existing.low_value_path_terms, incoming.low_value_path_terms),
        known_ir_home_urls=merge_unique_strings(existing.known_ir_home_urls, incoming.known_ir_home_urls),
        known_event_listing_urls=merge_unique_strings(
            existing.known_event_listing_urls,
            incoming.known_event_listing_urls,
        ),
        known_transcript_urls=merge_unique_strings(existing.known_transcript_urls, incoming.known_transcript_urls),
        robots_blocked_hosts=[
            value
            for value in merge_unique_strings(existing.robots_blocked_hosts, incoming.robots_blocked_hosts)
            if value not in positive_hosts
        ],
    )


def merge_company_playbook(existing: CompanyPlaybook, incoming: CompanyPlaybook) -> CompanyPlaybook:
    return CompanyPlaybook(
        issuer_name=incoming.issuer_name or existing.issuer_name,
        brand_names=merge_unique_strings(incoming.brand_names, existing.brand_names),
        official_homepage_candidates=merge_unique_strings(
            incoming.official_homepage_candidates,
            existing.official_homepage_candidates,
        ),
        preferred_ir_urls=merge_unique_strings(incoming.preferred_ir_urls, existing.preferred_ir_urls),
        avoid_hosts=merge_unique_strings(incoming.avoid_hosts, existing.avoid_hosts),
        avoid_urls=merge_unique_strings(incoming.avoid_urls, existing.avoid_urls),
        planner_prompt=incoming.planner_prompt or existing.planner_prompt,
        homepage_strategy=incoming.homepage_strategy or existing.homepage_strategy,
        ir_strategy=incoming.ir_strategy or existing.ir_strategy,
        transcript_strategy=incoming.transcript_strategy or existing.transcript_strategy,
        avoid_strategy=incoming.avoid_strategy or existing.avoid_strategy,
        confidence=max(existing.confidence, incoming.confidence),
        evidence=merge_unique_strings(incoming.evidence, existing.evidence),
    )


def merge_saved_prompt_guidance(
    existing: PromptGuidance | None,
    incoming: PromptGuidance | None,
) -> PromptGuidance | None:
    if existing is None:
        return incoming
    if incoming is None:
        return existing
    return PromptGuidance(
        run_objective=incoming.run_objective or existing.run_objective,
        strategy=incoming.strategy or existing.strategy,
        priority_terms=merge_unique_strings(incoming.priority_terms, existing.priority_terms),
        avoid_terms=merge_unique_strings(incoming.avoid_terms, existing.avoid_terms),
        homepage_guidance=incoming.homepage_guidance or existing.homepage_guidance,
        search_guidance=incoming.search_guidance or existing.search_guidance,
        navigation_guidance=incoming.navigation_guidance or existing.navigation_guidance,
        transcript_guidance=incoming.transcript_guidance or existing.transcript_guidance,
        agent_guidance=merge_agent_guidance(incoming.agent_guidance, existing.agent_guidance),
        risk_notes=merge_unique_strings(incoming.risk_notes, existing.risk_notes),
    )


def merge_unique_strings(existing: list[str], incoming: list[str]) -> list[str]:
    merged = list(existing)
    for value in incoming:
        append_unique(merged, value)
    return merged


def merge_unique_models(existing: list[ModelT], incoming: list[ModelT]) -> list[ModelT]:
    merged = list(existing)
    seen = {model_identity(value) for value in merged}
    for value in incoming:
        key = model_identity(value)
        if key not in seen:
            merged.append(value)
            seen.add(key)
    return merged


def model_identity(value: BaseModel) -> str:
    return json.dumps(value.model_dump(mode="json"), sort_keys=True)


def remember_crawl_result(memory: CompanyMemory, result: CrawlResult, analysis: FailureAnalysis) -> CompanyMemory:
    navigation_memory = memory.navigation_memory
    if result.verified_company_name and not is_weak_identity(
        Company(symbol=memory.company.symbol, name=result.verified_company_name)
    ):
        memory.company = memory.company.model_copy(update={"name": result.verified_company_name})
        memory.playbook.issuer_name = result.verified_company_name
        memory.playbook.confidence = max(memory.playbook.confidence, 0.75)
        append_unique(memory.playbook.evidence, f"verified homepage company name: {result.verified_company_name}")
    for url in result.verified_homepage_urls:
        memory.official_homepage_urls = append_unique(memory.official_homepage_urls, url)
        memory.playbook.official_homepage_candidates = append_unique(memory.playbook.official_homepage_candidates, url)
        append_unique(memory.playbook.evidence, f"verified official homepage: {url}")
        homepage_host = host(url)
        if homepage_host:
            memory.official_hosts = append_unique(memory.official_hosts, homepage_host)

    if result.transcripts:
        for url in successful_result_urls(result):
            memory.successful_path_urls = append_unique(memory.successful_path_urls, url)
            path_host = host(url)
            if path_host and not low_value_memory_host(path_host):
                memory.official_hosts = append_unique(memory.official_hosts, path_host)
                if url not in result.verified_homepage_urls:
                    memory.ir_hosts = append_unique(memory.ir_hosts, path_host)

    for url in [result.ir_url, *(candidate.url for candidate in result.candidates)]:
        if url:
            if not low_value_memory_url(url):
                memory.known_ir_urls = append_unique(memory.known_ir_urls, url)
            url_host = host(url)
            if url_host and not low_value_memory_host(url_host):
                memory.official_hosts = append_unique(memory.official_hosts, url_host)
                memory.ir_hosts = append_unique(memory.ir_hosts, url_host)
            if is_ir_home_url(url) and not low_value_memory_url(url):
                navigation_memory.known_ir_home_urls = append_unique(navigation_memory.known_ir_home_urls, url)
                memory.playbook.preferred_ir_urls = append_unique(memory.playbook.preferred_ir_urls, url)
            if is_event_listing_url(url) and not low_value_memory_path(url):
                navigation_memory.known_event_listing_urls = append_unique(navigation_memory.known_event_listing_urls, url)
            remember_low_value_url(navigation_memory.low_value_hosts, navigation_memory.low_value_path_terms, url)

    for failure in result.failures:
        memory.rejected_urls = append_unique(memory.rejected_urls, failure.url)
        failure_host = host(failure.url)
        robots_like_failure = failure.failure_type in {"robots_blocked", "robots_disallowed", "robots_unavailable"} or (
            failure.failure_type == "navigation_fetch_failed" and "RobotsUnavailableError" in failure.message
        )
        if failure_host and robots_like_failure:
            navigation_memory.robots_blocked_hosts = append_unique(navigation_memory.robots_blocked_hosts, failure_host)
            navigation_memory.low_value_hosts = append_unique(navigation_memory.low_value_hosts, failure_host)
            memory.playbook.avoid_hosts = append_unique(memory.playbook.avoid_hosts, failure_host)
            memory.playbook.avoid_urls = append_unique(memory.playbook.avoid_urls, failure.url)
        remember_low_value_url(navigation_memory.low_value_hosts, navigation_memory.low_value_path_terms, failure.url)

    for record in result.transcripts:
        source_url = str(record.source_url)
        memory.successful_transcript_urls = append_unique(memory.successful_transcript_urls, source_url)
        navigation_memory.known_transcript_urls = append_unique(navigation_memory.known_transcript_urls, source_url)
        transcript_host = host(source_url)
        if transcript_host:
            memory.ir_hosts = append_unique(memory.ir_hosts, transcript_host)
            navigation_memory.successful_hosts = append_unique(navigation_memory.successful_hosts, transcript_host)
            navigation_memory.preferred_hosts = append_unique(navigation_memory.preferred_hosts, transcript_host)
        path = urlparse(source_url).path
        if path:
            navigation_memory.successful_paths = append_unique(navigation_memory.successful_paths, path)

    memory.failure_summaries.append(analysis)
    for recommendation in analysis.manual_recommendations:
        memory.recommended_manual_actions = append_unique(memory.recommended_manual_actions, recommendation)
    return memory


def successful_result_urls(result: CrawlResult) -> list[str]:
    urls: list[str] = []
    for url in result.verified_homepage_urls:
        append_unique(urls, url)
    if result.ir_url:
        append_unique(urls, result.ir_url)
    for candidate in result.candidates:
        append_unique(urls, candidate.url)
    for record in result.transcripts:
        append_unique(urls, str(record.source_url))
    return urls


def apply_crawl_reflection(memory: CompanyMemory, reflection: CrawlReflection) -> CompanyMemory:
    navigation_memory = memory.navigation_memory
    reflection_preferred_hosts = {host(url) for url in reflection.preferred_urls if host(url)}
    for url in reflection.preferred_urls:
        memory.known_ir_urls = append_unique(memory.known_ir_urls, url)
        preferred_host = host(url)
        if preferred_host and not low_value_memory_host(preferred_host):
            memory.ir_hosts = append_unique(memory.ir_hosts, preferred_host)
            memory.official_hosts = append_unique(memory.official_hosts, preferred_host)
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
        run_objective=reflected.run_objective or existing.run_objective,
        strategy=reflected.strategy or existing.strategy,
        priority_terms=merge_unique(reflected.priority_terms, existing.priority_terms, limit=10),
        avoid_terms=merge_unique(reflected.avoid_terms, existing.avoid_terms, limit=10),
        homepage_guidance=reflected.homepage_guidance or existing.homepage_guidance,
        search_guidance=reflected.search_guidance or existing.search_guidance,
        navigation_guidance=reflected.navigation_guidance or existing.navigation_guidance,
        transcript_guidance=reflected.transcript_guidance or existing.transcript_guidance,
        agent_guidance=merge_agent_guidance(reflected.agent_guidance, existing.agent_guidance),
        risk_notes=merge_unique(reflected.risk_notes, existing.risk_notes, limit=5),
    )


def merge_agent_guidance(primary: dict[str, str], secondary: dict[str, str]) -> dict[str, str]:
    merged = dict(secondary)
    merged.update(primary)
    return {key: value for key, value in merged.items() if key and value}


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
    low_value_hosts = (
        "blog.",
        "youtube.com",
        "youtu.be",
        "quartr.com",
        "seekingalpha.com",
        "finance.yahoo.com",
        "marketbeat.com",
        "stockanalysis.com",
        "morningstar.com",
        "financialreports.eu",
    )
    return any(token in value.lower() for token in low_value_hosts)


def low_value_memory_url(url: str) -> bool:
    return low_value_memory_host(host(url)) or low_value_memory_path(url)


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
