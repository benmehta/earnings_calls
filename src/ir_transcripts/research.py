from __future__ import annotations

from dataclasses import dataclass

from .agent import OfficialTranscriptResearchAgent, TranscriptResearchJudgeAgent, compact_text
from .http import HttpClient
from .identity import company_display_name
from .models import (
    Company,
    CompanyNavigationMemory,
    FailureType,
    IRDiscoveryCandidate,
    PromptGuidance,
)
from .parsing import extract_links, page_title, visible_text
from .runtime import ProgressReporter, timeout_after
from .search import dedupe, search_ir_candidates
from .urls import host, resolve_document_url


@dataclass
class TranscriptResearchSeedResult:
    seeds: list[str]
    failure_type: FailureType | None = None
    failure_message: str = ""
    failure_urls: list[str] | None = None
    verified_homepage_urls: list[str] | None = None
    verified_company_name: str | None = None


def discover_transcript_research_seeds(
    company: Company,
    *,
    http: HttpClient,
    model: str,
    ollama_base_url: str | None = None,
    search_timeout_seconds: float = 30.0,
    llm_timeout_seconds: float = 45.0,
    prompt_guidance: PromptGuidance | None = None,
    navigation_memory: CompanyNavigationMemory | None = None,
    progress: ProgressReporter | None = None,
    max_search_results: int = 12,
) -> TranscriptResearchSeedResult:
    progress = progress or ProgressReporter(enabled=False)
    progress.log(f"{company.symbol}: official transcript research starting")
    candidates = search_ir_candidates(
        company,
        max_results=max_search_results,
        timeout_seconds=search_timeout_seconds,
        query_model=model,
        ollama_base_url=ollama_base_url,
        llm_timeout_seconds=llm_timeout_seconds,
        prompt_guidance=prompt_guidance,
        navigation_memory=navigation_memory,
        progress=progress,
    )
    if not candidates:
        return TranscriptResearchSeedResult(
            seeds=[],
            failure_type="official_research_unverified",
            failure_message="official transcript research found no search candidates",
        )

    try:
        with timeout_after(llm_timeout_seconds, f"researching official transcript candidates for {company.symbol}"):
            proposal = OfficialTranscriptResearchAgent(model, base_url=ollama_base_url).research(
                company=company,
                candidates=candidates,
                guidance=prompt_guidance,
                navigation_memory=navigation_memory,
            )
    except Exception as exc:
        return TranscriptResearchSeedResult(
            seeds=[],
            failure_type="official_research_unverified",
            failure_message=f"official transcript research failed: {type(exc).__name__}: {exc}",
        )

    proposed_urls = dedupe(
        [
            *proposal.official_homepage_urls,
            *proposal.official_ir_urls,
            *proposal.transcript_candidate_urls,
        ]
    )
    if not proposed_urls:
        return TranscriptResearchSeedResult(
            seeds=[],
            failure_type="official_research_unverified",
            failure_message=f"official transcript research proposed no usable URLs: {proposal.reason}",
        )

    evidence = research_judge_evidence(company=company, candidates=candidates, urls=proposed_urls, http=http, progress=progress)
    try:
        with timeout_after(llm_timeout_seconds, f"judging official transcript research for {company.symbol}"):
            judgment = TranscriptResearchJudgeAgent(model, base_url=ollama_base_url).judge(
                company=company,
                proposal=proposal,
                evidence=evidence,
                guidance=prompt_guidance,
            )
    except Exception as exc:
        return TranscriptResearchSeedResult(
            seeds=[],
            failure_type="official_research_unverified",
            failure_message=f"official transcript research judge failed: {type(exc).__name__}: {exc}",
            failure_urls=proposed_urls,
        )

    accepted = dedupe(
        [
            *judgment.accepted_transcript_urls,
            *judgment.accepted_ir_urls,
            *judgment.accepted_homepage_urls,
        ]
    )
    if not judgment.accepted or judgment.confidence < 0.55 or not accepted:
        return TranscriptResearchSeedResult(
            seeds=[],
            failure_type="official_research_unverified",
            failure_message=judgment.retry_guidance or judgment.reason or "official transcript research was not verified",
            failure_urls=proposed_urls,
            verified_company_name=judgment.official_company_name or proposal.issuer_name,
        )

    progress.log(f"{company.symbol}: official transcript research accepted {len(accepted)} seed URL(s)")
    return TranscriptResearchSeedResult(
        seeds=[resolve_document_url(url) for url in accepted],
        verified_homepage_urls=judgment.accepted_homepage_urls,
        verified_company_name=judgment.official_company_name or proposal.issuer_name,
    )


def research_judge_evidence(
    *,
    company: Company,
    candidates: list[IRDiscoveryCandidate],
    urls: list[str],
    http: HttpClient,
    progress: ProgressReporter,
) -> dict:
    candidate_by_url = {candidate.url: candidate for candidate in candidates}
    fetched = []
    for url in urls[:8]:
        item = {
            "url": url,
            "host": host(url),
            "search_result": candidate_by_url.get(url).model_dump(mode="json") if url in candidate_by_url else None,
        }
        try:
            progress.log(f"{company.symbol}: fetching research evidence {url}")
            response = http.get(url)
        except Exception as exc:
            item["fetch_error"] = f"{type(exc).__name__}: {exc}"
            fetched.append(item)
            continue
        content_type = response.headers.get("content-type", "")
        item["content_type"] = content_type
        if "html" in content_type.lower() or not content_type:
            html = response.text
            links = extract_links(html, url)
            item.update(
                {
                    "title": page_title(html),
                    "text": compact_text(visible_text(html), 1200),
                    "links": [
                        {"url": link.url, "label": compact_text(link.label, 120)}
                        for link in links[:25]
                    ],
                }
            )
        else:
            item["title"] = url.rstrip("/").split("/")[-1]
            item["text"] = f"Fetched non-HTML content for {company_display_name(company)} ({company.symbol})."
        fetched.append(item)
    return {
        "company": company.model_dump(mode="json"),
        "search_candidates": [candidate.model_dump(mode="json") for candidate in candidates[:20]],
        "fetched": fetched,
    }
