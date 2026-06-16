from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, HttpUrl
from pydantic import field_validator


class Company(BaseModel):
    symbol: str
    name: str | None = None
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

    @field_validator("confidence", mode="before")
    @classmethod
    def coerce_confidence(cls, value):
        if isinstance(value, (int, float)) and value > 1:
            return value / 100
        return value


class IRDiscoveryDecision(BaseModel):
    selections: list[IRDiscoverySelection] = Field(default_factory=list)


class HomepagePrediction(BaseModel):
    homepage_urls: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = ""

    @field_validator("confidence", mode="before")
    @classmethod
    def coerce_confidence(cls, value):
        if isinstance(value, (int, float)) and value > 1:
            return value / 100
        return value


class HomepageValidationDecision(BaseModel):
    is_official: bool = False
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    official_company_name: str | None = None
    linked_ir_urls: list[str] = Field(default_factory=list)
    reason: str = ""

    @field_validator("confidence", mode="before")
    @classmethod
    def coerce_confidence(cls, value):
        if isinstance(value, (int, float)) and value > 1:
            return value / 100
        return value


class DocumentLinkTriageDecision(BaseModel):
    is_priority_transcript_document: bool = False
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    document_type: Literal[
        "earnings_call_transcript",
        "earnings_release",
        "presentation",
        "webcast",
        "filing",
        "other",
    ] = "other"
    reason: str = ""

    @field_validator("confidence", mode="before")
    @classmethod
    def coerce_confidence(cls, value):
        if isinstance(value, (int, float)) and value > 1:
            return value / 100
        return value


class TranscriptEvidenceDecision(BaseModel):
    is_transcript: bool = False
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence: list[str] = Field(default_factory=list)
    rejection_reason: str = "transcript_rejected_missing_evidence"

    @field_validator("confidence", mode="before")
    @classmethod
    def coerce_confidence(cls, value):
        if isinstance(value, (int, float)) and value > 1:
            return value / 100
        return value


class LinkTriageSelection(BaseModel):
    url: str
    priority: int = Field(default=0, ge=0, le=100)
    should_follow: bool = False
    reason: str = ""


class LinkBatchTriageDecision(BaseModel):
    selections: list[LinkTriageSelection] = Field(default_factory=list)


class LatestTranscriptSelectionDecision(BaseModel):
    selected_urls: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = ""

    @field_validator("selected_urls", mode="before")
    @classmethod
    def coerce_selected_urls(cls, value):
        if not isinstance(value, list):
            return value
        urls = []
        for item in value:
            if isinstance(item, str):
                urls.append(item)
            elif isinstance(item, dict) and isinstance(item.get("url"), str):
                urls.append(item["url"])
        return urls

    @field_validator("confidence", mode="before")
    @classmethod
    def coerce_confidence(cls, value):
        if isinstance(value, (int, float)) and value > 1:
            return value / 100
        return value


class TranscriptDocumentRankingDecision(BaseModel):
    ordered_urls: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = ""

    @field_validator("ordered_urls", mode="before")
    @classmethod
    def coerce_ordered_urls(cls, value):
        if not isinstance(value, list):
            return value
        urls = []
        for item in value:
            if isinstance(item, str):
                urls.append(item)
            elif isinstance(item, dict) and isinstance(item.get("url"), str):
                urls.append(item["url"])
        return urls

    @field_validator("confidence", mode="before")
    @classmethod
    def coerce_confidence(cls, value):
        if isinstance(value, (int, float)) and value > 1:
            return value / 100
        return value


class NavigationCandidateSelection(BaseModel):
    url: str
    priority: int = Field(default=0, ge=0, le=100)
    should_follow: bool = False
    reason: str = ""


class NavigationCandidateRankingDecision(BaseModel):
    selections: list[NavigationCandidateSelection] = Field(default_factory=list)


class NavigationPageContextDecision(BaseModel):
    page_context: Literal["homepage", "ir", "event_listing", "transcript_link"] = "homepage"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = ""

    @field_validator("confidence", mode="before")
    @classmethod
    def coerce_confidence(cls, value):
        if isinstance(value, (int, float)) and value > 1:
            return value / 100
        return value


class RenderedPageRecoveryDecision(BaseModel):
    should_recover: bool = False
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    actions: list[
        Literal[
            "wait_for_dynamic_content",
            "select_latest_option",
            "expand_disclosure_controls",
        ]
    ] = Field(default_factory=list)
    target_ids: list[str] = Field(default_factory=list)
    reason: str = ""

    @field_validator("confidence", mode="before")
    @classmethod
    def coerce_confidence(cls, value):
        if isinstance(value, (int, float)) and value > 1:
            return value / 100
        return value


class SearchQueryPlan(BaseModel):
    queries: list[str] = Field(default_factory=list)
    reason: str = ""

    @field_validator("queries", mode="before")
    @classmethod
    def coerce_queries(cls, value):
        if not isinstance(value, list):
            return value
        queries = []
        for item in value:
            if isinstance(item, str):
                queries.append(item)
            elif isinstance(item, dict) and isinstance(item.get("query"), str):
                queries.append(item["query"])
        return queries


class SearchCandidateSelection(BaseModel):
    url: str
    priority: int = Field(default=0, ge=0, le=100)
    is_official_candidate: bool = False
    reason: str = ""


class SearchCandidateRankingDecision(BaseModel):
    selections: list[SearchCandidateSelection] = Field(default_factory=list)


class TranscriptResearchProposal(BaseModel):
    issuer_name: str | None = None
    official_homepage_urls: list[str] = Field(default_factory=list)
    official_ir_urls: list[str] = Field(default_factory=list)
    transcript_candidate_urls: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = ""
    evidence: list[str] = Field(default_factory=list)

    @field_validator("confidence", mode="before")
    @classmethod
    def coerce_confidence(cls, value):
        if isinstance(value, (int, float)) and value > 1:
            return value / 100
        return value


class TranscriptResearchJudgment(BaseModel):
    accepted: bool = False
    accepted_homepage_urls: list[str] = Field(default_factory=list)
    accepted_ir_urls: list[str] = Field(default_factory=list)
    accepted_transcript_urls: list[str] = Field(default_factory=list)
    official_company_name: str | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = ""
    retry_guidance: str = ""

    @field_validator("confidence", mode="before")
    @classmethod
    def coerce_confidence(cls, value):
        if isinstance(value, (int, float)) and value > 1:
            return value / 100
        return value


class CrawlNavigatorDecision(BaseModel):
    page_type: Literal[
        "transcript",
        "earnings_event",
        "press_release",
        "filings",
        "ir_index",
        "not_relevant",
    ] = "not_relevant"
    chosen_urls: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = ""
    stop_reason: str | None = None

    @field_validator("page_type", mode="before")
    @classmethod
    def normalize_page_type(cls, value):
        return normalize_agent_page_type(value)

    @field_validator("chosen_urls", mode="before")
    @classmethod
    def coerce_chosen_urls(cls, value):
        if not isinstance(value, list):
            return value
        urls = []
        for item in value:
            if isinstance(item, str):
                urls.append(item)
            elif isinstance(item, dict) and isinstance(item.get("url"), str):
                urls.append(item["url"])
        return urls

    @field_validator("confidence", mode="before")
    @classmethod
    def coerce_confidence(cls, value):
        if isinstance(value, (int, float)) and value > 1:
            return value / 100
        return value


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


class NavigationCandidateTrace(BaseModel):
    url: str
    label: str = ""
    context: str = ""
    score: int = 0


class NavigationStep(BaseModel):
    current_url: str
    title: str = ""
    candidate_urls: list[str] = Field(default_factory=list)
    candidate_links: list[NavigationCandidateTrace] = Field(default_factory=list)
    raw_link_count: int = 0
    chosen_urls: list[str] = Field(default_factory=list)
    rejected_urls: list[str] = Field(default_factory=list)
    render_strategy: str | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = ""
    stop_reason: str | None = None


class NavigationTrace(BaseModel):
    company: Company
    steps: list[NavigationStep] = Field(default_factory=list)


class PromptGuidance(BaseModel):
    run_objective: str = ""
    strategy: str = ""
    priority_terms: list[str] = Field(default_factory=list)
    avoid_terms: list[str] = Field(default_factory=list)
    homepage_guidance: str = ""
    search_guidance: str = ""
    navigation_guidance: str = ""
    transcript_guidance: str = ""
    agent_guidance: dict[str, str] = Field(default_factory=dict)
    risk_notes: list[str] = Field(default_factory=list)

    @field_validator("priority_terms", "avoid_terms", "risk_notes", mode="before")
    @classmethod
    def coerce_string_list(cls, value):
        if isinstance(value, str):
            return [value]
        return value


class CrawlReflection(BaseModel):
    preferred_urls: list[str] = Field(default_factory=list)
    preferred_terms: list[str] = Field(default_factory=list)
    avoid_urls: list[str] = Field(default_factory=list)
    avoid_terms: list[str] = Field(default_factory=list)
    prompt_guidance: PromptGuidance = Field(default_factory=PromptGuidance)
    reason: str = ""


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

    @field_validator("page_type", mode="before")
    @classmethod
    def normalize_page_type(cls, value):
        return normalize_agent_page_type(value)

    @field_validator("useful_urls", mode="before")
    @classmethod
    def coerce_useful_urls(cls, value):
        if not isinstance(value, list):
            return value
        urls = []
        for item in value:
            if isinstance(item, str):
                urls.append(item)
            elif isinstance(item, dict) and isinstance(item.get("url"), str):
                urls.append(item["url"])
        return urls


def normalize_agent_page_type(value):
    if not isinstance(value, str):
        return value
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "transcript_link": "ir_index",
        "transcript_links": "ir_index",
        "transcript_document_link": "ir_index",
        "document_link": "ir_index",
        "document_links": "ir_index",
        "quarterly_results": "ir_index",
        "financial_results": "ir_index",
        "results_page": "ir_index",
        "investor_relations": "ir_index",
        "ir_page": "ir_index",
        "event": "earnings_event",
        "event_detail": "earnings_event",
        "earnings_call": "earnings_event",
        "earnings_results": "earnings_event",
        "earnings_release": "press_release",
        "press_release_page": "press_release",
        "sec_filings": "filings",
        "sec_filing": "filings",
        "filing": "filings",
        "irrelevant": "not_relevant",
        "not relevant": "not_relevant",
    }
    return aliases.get(normalized, normalized)


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
    "identity_low_confidence",
    "homepage_unverified",
    "official_research_unverified",
    "navigation_fetch_failed",
    "navigation_render_failed",
    "navigation_llm_failed",
    "page_fetch_failed",
    "page_render_failed",
    "page_classification_failed",
    "document_parse_failed",
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
    verified_homepage_urls: list[str] = Field(default_factory=list)
    verified_company_name: str | None = None
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


class CompanyPlaybook(BaseModel):
    issuer_name: str | None = None
    brand_names: list[str] = Field(default_factory=list)
    official_homepage_candidates: list[str] = Field(default_factory=list)
    preferred_ir_urls: list[str] = Field(default_factory=list)
    avoid_hosts: list[str] = Field(default_factory=list)
    avoid_urls: list[str] = Field(default_factory=list)
    planner_prompt: str = ""
    homepage_strategy: str = ""
    ir_strategy: str = ""
    transcript_strategy: str = ""
    avoid_strategy: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence: list[str] = Field(default_factory=list)

    @field_validator("confidence", mode="before")
    @classmethod
    def coerce_confidence(cls, value):
        if isinstance(value, (int, float)) and value > 1:
            return value / 100
        return value


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
    disable_predictive_identity: bool = False
    use_research_agent: bool = False
    rerank_discovery: bool = False
    extract_metadata_with_llm: bool = False
    latest_only: bool = False
    allow_official_linked_documents_on_robots_unavailable: bool = False
    use_prompt_planner: bool = False
    disable_memory: bool = False
    no_memory_write: bool = False
    prompt_guidance: PromptGuidance | None = None
    navigation_memory: CompanyNavigationMemory | None = None
    identity_name_hint: str | None = None
    homepage_candidates: list[str] = Field(default_factory=list)
    search_timeout_seconds: float = 30.0
    llm_timeout_seconds: float = 45.0
    navigation_llm_max_links: int = 12
    page_llm_max_links: int = 12
    llm_text_chars: int = 900
    reason: str = "initial"


FailureCategory = Literal[
    "success",
    "wrong_identity",
    "identity_low_confidence",
    "homepage_unverified",
    "official_research_unverified",
    "navigation_step_failed",
    "crawl_step_failed",
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


class OrchestrationAnalysisDecision(BaseModel):
    category: FailureCategory
    summary: str
    retryable: bool = False
    evidence_urls: list[str] = Field(default_factory=list)
    manual_recommendations: list[str] = Field(default_factory=list)


class RetryPlanningDecision(BaseModel):
    action_type: str
    reason: str
    config_updates: dict[str, object] = Field(default_factory=dict)
    manual_recommendations: list[str] = Field(default_factory=list)


SupervisorActionType = Literal[
    "finish",
    "identity_correction",
    "identity_retry",
    "step_retry",
    "retry",
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
    playbook: CompanyPlaybook = Field(default_factory=CompanyPlaybook)
    official_hosts: list[str] = Field(default_factory=list)
    ir_hosts: list[str] = Field(default_factory=list)
    official_homepage_urls: list[str] = Field(default_factory=list)
    successful_path_urls: list[str] = Field(default_factory=list)
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
