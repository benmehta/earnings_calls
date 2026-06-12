from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, HttpUrl


class Company(BaseModel):
    symbol: str
    name: str
    sector: str | None = None
    sub_industry: str | None = None
    cik: str | None = None


class CandidateLink(BaseModel):
    url: str
    label: str = ""
    source_url: str
    reason: str = ""


class IRDiscoveryCandidate(BaseModel):
    url: str
    title: str = ""
    snippet: str = ""
    source: Literal["curated", "search", "deterministic"] = "search"
    score: int = 0
    reasons: list[str] = Field(default_factory=list)


class IRDiscoverySelection(BaseModel):
    url: str
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = ""


class IRDiscoveryDecision(BaseModel):
    selections: list[IRDiscoverySelection] = Field(default_factory=list)


class NavigationDecision(BaseModel):
    chosen_urls: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = ""
    stop_reason: str | None = None


class NavigationValidationResult(BaseModel):
    is_valid: bool = False
    accepted_urls: list[str] = Field(default_factory=list)
    rejected_urls: list[str] = Field(default_factory=list)
    reason: str = ""
    repair_guidance: str | None = None


class NavigationStep(BaseModel):
    current_url: str
    title: str = ""
    chosen_urls: list[str] = Field(default_factory=list)
    rejected_urls: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = ""
    stop_reason: str | None = None


class NavigationTrace(BaseModel):
    company: Company
    steps: list[NavigationStep] = Field(default_factory=list)


class PromptGuidance(BaseModel):
    priority_terms: list[str] = Field(default_factory=list)
    avoid_terms: list[str] = Field(default_factory=list)
    navigation_guidance: str = ""
    transcript_guidance: str = ""
    risk_notes: list[str] = Field(default_factory=list)


class PageDecision(BaseModel):
    page_type: Literal[
        "transcript",
        "earnings_event",
        "press_release",
        "filings",
        "ir_index",
        "not_relevant",
    ]
    confidence: float = Field(ge=0.0, le=1.0)
    useful_links: list[CandidateLink] = Field(default_factory=list)
    reason: str = ""


class PageDecisionDraft(BaseModel):
    page_type: Literal[
        "transcript",
        "earnings_event",
        "press_release",
        "filings",
        "ir_index",
        "not_relevant",
    ]
    confidence: float = Field(ge=0.0, le=1.0)
    useful_urls: list[str] = Field(default_factory=list)
    reason: str = ""


class TranscriptRecord(BaseModel):
    company: Company
    source_url: HttpUrl | str
    fiscal_period: str | None = None
    call_date: date | None = None
    title: str
    text: str
    raw_path: Path | None = None
    metadata: dict[str, str] = Field(default_factory=dict)


FailureType = Literal[
    "robots_blocked",
    "robots_disallowed",
    "robots_unavailable",
    "timeout",
    "http_error",
    "parse_error",
    "llm_error",
    "pdf_error",
    "playwright_error",
    "not_transcript",
    "unknown_error",
]


class CrawlFailure(BaseModel):
    company: Company
    url: str
    failure_type: FailureType
    message: str = ""


class CandidatePage(BaseModel):
    company: Company
    url: str
    title: str = ""
    depth: int = 0
    heuristic_score: int = 0
    llm_page_type: str | None = None
    llm_confidence: float | None = None
    reason: str = ""


class CrawlResult(BaseModel):
    company: Company
    ir_url: str | None = None
    transcripts: list[TranscriptRecord] = Field(default_factory=list)
    candidates: list[CandidatePage] = Field(default_factory=list)
    failures: list[CrawlFailure] = Field(default_factory=list)
    skipped_reason: str | None = None
    visited_count: int = 0


class CompanyNavigationMemory(BaseModel):
    successful_hosts: list[str] = Field(default_factory=list)
    successful_paths: list[str] = Field(default_factory=list)
    preferred_hosts: list[str] = Field(default_factory=list)
    low_value_hosts: list[str] = Field(default_factory=list)
    low_value_path_terms: list[str] = Field(default_factory=list)
    known_ir_home_urls: list[str] = Field(default_factory=list)
    known_event_listing_urls: list[str] = Field(default_factory=list)
    known_transcript_urls: list[str] = Field(default_factory=list)
    robots_blocked_hosts: list[str] = Field(default_factory=list)


class CrawlAttemptConfig(BaseModel):
    attempt: int = 1
    max_pages_per_company: int = 40
    max_depth: int = 3
    use_playwright: bool = False
    playwright_mode: Literal["off", "auto", "always"] = "off"
    resume: bool = True
    review_only: bool = False
    seed_urls: list[str] = Field(default_factory=list)
    discovery_mode: Literal["nav-first", "search-first"] = "nav-first"
    include_discovery_guesses: bool = False
    disable_official_homepage_overrides: bool = False
    rerank_discovery: bool = False
    extract_metadata_with_llm: bool = False
    latest_only: bool = False
    allow_official_linked_documents_on_robots_unavailable: bool = False
    use_prompt_planner: bool = False
    prompt_guidance: PromptGuidance | None = None
    navigation_memory: CompanyNavigationMemory | None = None
    search_timeout_seconds: float = 30.0
    llm_timeout_seconds: float = 45.0
    navigation_llm_max_links: int = 12
    page_llm_max_links: int = 12
    llm_text_chars: int = 900
    reason: str = "initial"


FailureCategory = Literal[
    "success",
    "wrong_identity",
    "robots_unavailable",
    "render_needed",
    "needs_deeper_crawl",
    "no_useful_links",
    "not_transcript",
    "no_transcript_found",
]


class FailureAnalysis(BaseModel):
    category: FailureCategory
    summary: str
    retryable: bool = False
    evidence_urls: list[str] = Field(default_factory=list)
    manual_recommendations: list[str] = Field(default_factory=list)


SupervisorActionType = Literal[
    "finish",
    "identity_correction",
    "search_first",
    "playwright_retry",
    "deeper_crawl",
    "manual_review",
]


class SupervisorAction(BaseModel):
    action_type: SupervisorActionType
    reason: str
    next_company: Company | None = None
    next_config: CrawlAttemptConfig | None = None
    manual_recommendations: list[str] = Field(default_factory=list)


class CompanyMemory(BaseModel):
    company: Company
    official_hosts: list[str] = Field(default_factory=list)
    known_ir_urls: list[str] = Field(default_factory=list)
    attempted_configs: list[CrawlAttemptConfig] = Field(default_factory=list)
    failure_summaries: list[FailureAnalysis] = Field(default_factory=list)
    rejected_urls: list[str] = Field(default_factory=list)
    recommended_manual_actions: list[str] = Field(default_factory=list)
    successful_transcript_urls: list[str] = Field(default_factory=list)
    prompt_guidance: PromptGuidance | None = None
    navigation_memory: CompanyNavigationMemory = Field(default_factory=CompanyNavigationMemory)


class SupervisorRunResult(BaseModel):
    company: Company
    final_company: Company
    attempts: list[CrawlAttemptConfig] = Field(default_factory=list)
    analyses: list[FailureAnalysis] = Field(default_factory=list)
    actions: list[SupervisorAction] = Field(default_factory=list)
    result: CrawlResult | None = None
    memory_path: Path | None = None
    status: Literal["success", "partial", "manual_review", "failed"] = "failed"
