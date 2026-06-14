from __future__ import annotations

import json
import re
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from .agent_prompts import load_agent_prompt
from .identity import company_display_name
from .models import (
    CandidateLink,
    Company,
    CompanyMemory,
    CompanyNavigationMemory,
    CompanyPlaybook,
    CrawlNavigatorDecision,
    CrawlReflection,
    DocumentLinkTriageDecision,
    HomepagePrediction,
    HomepageValidationDecision,
    IRDiscoveryCandidate,
    IRDiscoveryDecision,
    LinkBatchTriageDecision,
    NavigationDecision,
    NavigationCandidateRankingDecision,
    NavigationPageContextDecision,
    NavigationValidationResult,
    OrchestrationAnalysisDecision,
    PageDecision,
    PageDecisionDraft,
    PromptGuidance,
    RetryPlanningDecision,
    SearchCandidateRankingDecision,
    SearchQueryPlan,
    TranscriptResearchJudgment,
    TranscriptResearchProposal,
    TranscriptEvidenceDecision,
)


NavigationAgentKind = Literal["homepage", "ir_section", "event_listing", "transcript_link"]
LINKS_JSON_CHAR_BUDGET = 800
IR_PAGE_PROMPT = load_agent_prompt("ir_page")
CRAWL_NAVIGATOR_PROMPT = load_agent_prompt("crawl_navigator")
IR_DISCOVERY_PROMPT = load_agent_prompt("ir_discovery")
HOMEPAGE_PREDICTION_PROMPT = load_agent_prompt("homepage_prediction")
HOMEPAGE_VALIDATION_PROMPT = load_agent_prompt("homepage_validation")
DOCUMENT_LINK_TRIAGE_PROMPT = load_agent_prompt("document_link_triage")
TRANSCRIPT_EVIDENCE_PROMPT = load_agent_prompt("transcript_evidence")
LINK_BATCH_TRIAGE_PROMPT = load_agent_prompt("link_batch_triage")
NAVIGATION_CANDIDATE_RANKING_PROMPT = load_agent_prompt("navigation_candidate_ranking")
NAVIGATION_PAGE_CONTEXT_PROMPT = load_agent_prompt("navigation_page_context")
MEMORY_GUIDANCE_PROMPT = load_agent_prompt("memory_guidance")
COMPANY_PLAYBOOK_PROMPT = load_agent_prompt("company_playbook")
SEARCH_QUERY_PLANNER_PROMPT = load_agent_prompt("search_query_planner")
SEARCH_CANDIDATE_RANKING_PROMPT = load_agent_prompt("search_candidate_ranking")
TRANSCRIPT_RESEARCH_PROMPT = load_agent_prompt("transcript_research")
TRANSCRIPT_RESEARCH_JUDGE_PROMPT = load_agent_prompt("transcript_research_judge")
ORCHESTRATION_ANALYSIS_PROMPT = load_agent_prompt("orchestration_analysis")
RETRY_PLANNING_PROMPT = load_agent_prompt("retry_planning")
PROMPT_PLANNER_PROMPT = load_agent_prompt("prompt_planner")
CRAWL_REFLECTION_PROMPT = load_agent_prompt("crawl_reflection")
LINK_SELECTION_PROMPT = load_agent_prompt("link_selection")
NAVIGATION_VALIDATION_PROMPT = load_agent_prompt("navigation_validation")
HOMEPAGE_NAV_PROMPT = load_agent_prompt("homepage_nav")
IR_SECTION_PROMPT = load_agent_prompt("ir_section")
EVENT_LISTING_PROMPT = load_agent_prompt("event_listing")
TRANSCRIPT_LINK_PROMPT = load_agent_prompt("transcript_link")


def build_llm(
    model: str,
    temperature: float = 0.0,
    base_url: str | None = None,
    *,
    json_mode: bool = False,
) -> ChatOllama:
    kwargs = {"model": model, "temperature": temperature}
    if base_url:
        kwargs["base_url"] = base_url
    if json_mode:
        kwargs["format"] = "json"
    return ChatOllama(**kwargs)


class CrawlNavigatorAgent:
    """Unified page classifier and link navigator for the live crawler path."""

    SYSTEM_PROMPT = CRAWL_NAVIGATOR_PROMPT.system_prompt
    HUMAN_PROMPT = CRAWL_NAVIGATOR_PROMPT.human_prompt or ""

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        *,
        max_links: int = 40,
        text_chars: int = 900,
        guidance: PromptGuidance | None = None,
    ) -> None:
        self.max_links = max_links
        self.text_chars = text_chars
        self.guidance = guidance
        self.chain = (
            ChatPromptTemplate.from_messages(
                [
                    ("system", self.SYSTEM_PROMPT),
                    ("human", self.HUMAN_PROMPT),
                ]
            )
            | build_llm(model, base_url=base_url, json_mode=True)
        )

    def decide(
        self,
        *,
        company_name: str,
        ticker: str,
        url: str,
        title: str,
        text: str,
        links: list[CandidateLink],
        page_context: str = "",
    ) -> CrawlNavigatorDecision:
        compact_links = compact_candidate_links(links, max_links=self.max_links)
        message = self.chain.invoke(
            {
                "company_name": company_name,
                "ticker": ticker,
                "url": url,
                "title": title,
                "page_context": page_context or "unknown",
                "guidance": guidance_for_agent(getattr(self, "guidance", None), "navigation", agent_name="CrawlNavigatorAgent"),
                "text": compact_text(text, self.text_chars),
                "links_json": json.dumps(compact_links, ensure_ascii=True),
            }
        )
        decision = CrawlNavigatorDecision.model_validate(extract_json_object(message.content))
        candidate_urls = {link.url for link in links}
        decision.chosen_urls = [url for url in decision.chosen_urls if url in candidate_urls]
        return decision

    def decide_page(self, **kwargs) -> PageDecision:
        decision = self.decide(**kwargs)
        links_by_url = {link.url: link for link in kwargs["links"]}
        return PageDecision(
            page_type=decision.page_type,
            confidence=decision.confidence,
            useful_links=[links_by_url[url] for url in decision.chosen_urls if url in links_by_url],
            reason=decision.reason,
        )

    def decide_navigation(self, **kwargs) -> NavigationDecision:
        decision = self.decide(**kwargs)
        return NavigationDecision(
            chosen_urls=decision.chosen_urls,
            confidence=decision.confidence,
            reason=decision.reason,
            stop_reason=decision.stop_reason,
        )


class IRPageAgent:
    """Small LangChain/Ollama page classifier used by the crawler."""

    SYSTEM_PROMPT = IR_PAGE_PROMPT.system_prompt
    HUMAN_PROMPT = IR_PAGE_PROMPT.human_prompt or ""

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        *,
        max_links: int = 12,
        text_chars: int = 900,
        guidance: PromptGuidance | None = None,
    ) -> None:
        self.max_links = max_links
        self.text_chars = text_chars
        self.guidance = guidance
        self.chain = (
            ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        self.SYSTEM_PROMPT,
                    ),
                    (
                        "human",
                        self.HUMAN_PROMPT,
                    ),
                ]
            )
            | build_llm(model, base_url=base_url, json_mode=True)
        )

    def decide(
        self,
        *,
        company_name: str,
        ticker: str,
        url: str,
        title: str,
        text: str,
        links: list[CandidateLink],
    ) -> PageDecision:
        compact_links = compact_candidate_links(links, max_links=self.max_links)
        message = self.chain.invoke(
            {
                "company_name": company_name,
                "ticker": ticker,
                "url": url,
                "title": title,
                "guidance": guidance_for_agent(getattr(self, "guidance", None), "page", agent_name="IRPageAgent"),
                "text": compact_text(text, self.text_chars),
                "links_json": json.dumps(compact_links, ensure_ascii=True),
            }
        )
        draft = PageDecisionDraft.model_validate(extract_json_object(message.content))
        links_by_url = {link.url: link for link in links}
        useful_links = [
            links_by_url[useful_url]
            for useful_url in draft.useful_urls
            if useful_url in links_by_url
        ]
        return PageDecision(
            page_type=draft.page_type,
            confidence=draft.confidence,
            useful_links=useful_links,
            reason=draft.reason,
        )


class IRDiscoveryAgent:
    """Local Ollama reranker for search-derived investor-relations candidates."""

    SYSTEM_PROMPT = IR_DISCOVERY_PROMPT.system_prompt
    HUMAN_PROMPT = IR_DISCOVERY_PROMPT.human_prompt or ""

    def __init__(self, model: str, base_url: str | None = None) -> None:
        self.parser = PydanticOutputParser(pydantic_object=IRDiscoveryDecision)
        self.chain = (
            ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        self.SYSTEM_PROMPT,
                    ),
                    (
                        "human",
                        self.HUMAN_PROMPT,
                    ),
                ]
            )
            | build_llm(model, base_url=base_url)
            | self.parser
        )

    def rerank(
        self,
        *,
        company_name: str,
        ticker: str,
        candidates: list[IRDiscoveryCandidate],
        limit: int,
        guidance: PromptGuidance | None = None,
        navigation_memory: CompanyNavigationMemory | None = None,
    ) -> IRDiscoveryDecision:
        compact = [
            {
                "url": candidate.url,
                "title": candidate.title,
                "snippet": candidate.snippet[:500],
                "source": candidate.source,
                "score": candidate.score,
                "reasons": candidate.reasons[:6],
            }
            for candidate in candidates[:25]
        ]
        return self.chain.invoke(
            {
                "company_name": company_name,
                "ticker": ticker,
                "guidance": guidance_for_agent(guidance, "search", agent_name="IRDiscoveryAgent"),
                "memory_summary": discovery_memory_summary(navigation_memory),
                "candidates_json": json.dumps(compact, ensure_ascii=True),
                "limit": limit,
                "format_instructions": self.parser.get_format_instructions(),
            }
        )


class HomepagePredictionAgent:
    """Predicts official company homepages from a ticker before discovery."""

    SYSTEM_PROMPT = HOMEPAGE_PREDICTION_PROMPT.system_prompt
    HUMAN_PROMPT = HOMEPAGE_PREDICTION_PROMPT.human_prompt or ""

    def __init__(self, model: str, base_url: str | None = None) -> None:
        self.chain = (
            ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        self.SYSTEM_PROMPT,
                    ),
                    (
                        "human",
                        self.HUMAN_PROMPT,
                    ),
                ]
            )
            | build_llm(model, base_url=base_url, json_mode=True)
        )

    def predict(self, *, ticker: str, name_hint: str | None = None) -> HomepagePrediction:
        message = self.chain.invoke(
            {
                "ticker": ticker,
                "name_hint": name_hint or "None.",
            }
        )
        return HomepagePrediction.model_validate(extract_json_object(message.content))


class HomepageValidationAgent:
    """Validates fetched homepage evidence and extracts official IR links."""

    SYSTEM_PROMPT = HOMEPAGE_VALIDATION_PROMPT.system_prompt
    HUMAN_PROMPT = HOMEPAGE_VALIDATION_PROMPT.human_prompt or ""

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        *,
        max_links: int = 16,
        text_chars: int = 1200,
    ) -> None:
        self.max_links = max_links
        self.text_chars = text_chars
        self.chain = (
            ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        self.SYSTEM_PROMPT,
                    ),
                    (
                        "human",
                        self.HUMAN_PROMPT,
                    ),
                ]
            )
            | build_llm(model, base_url=base_url, json_mode=True)
        )

    def validate(
        self,
        *,
        ticker: str,
        name_hint: str | None,
        url: str,
        title: str,
        text: str,
        links: list[CandidateLink],
    ) -> HomepageValidationDecision:
        compact_links = [
            {"url": link.url, "label": compact_text(link.label, 80), "context": compact_text(link.reason, 80)}
            for link in links[: self.max_links]
        ]
        message = self.chain.invoke(
            {
                "ticker": ticker,
                "name_hint": name_hint or "None.",
                "url": url,
                "title": title,
                "text": compact_text(text, self.text_chars),
                "links_json": json.dumps(compact_links, ensure_ascii=True),
            }
        )
        decision = HomepageValidationDecision.model_validate(extract_json_object(message.content))
        candidate_urls = {link.url for link in links}
        decision.linked_ir_urls = [
            url
            for url in decision.linked_ir_urls
            if url in candidate_urls
        ]
        return decision


class DocumentLinkTriageAgent:
    """Classifies document-like links before queue promotion."""

    SYSTEM_PROMPT = DOCUMENT_LINK_TRIAGE_PROMPT.system_prompt
    HUMAN_PROMPT = DOCUMENT_LINK_TRIAGE_PROMPT.human_prompt or ""

    def __init__(self, model: str, base_url: str | None = None) -> None:
        self.chain = (
            ChatPromptTemplate.from_messages(
                [
                    ("system", self.SYSTEM_PROMPT),
                    ("human", self.HUMAN_PROMPT),
                ]
            )
            | build_llm(model, base_url=base_url, json_mode=True)
        )

    def triage(
        self,
        *,
        company_name: str,
        ticker: str,
        source_url: str,
        source_title: str,
        link: CandidateLink,
    ) -> DocumentLinkTriageDecision:
        message = self.chain.invoke(
            {
                "company_name": company_name,
                "ticker": ticker,
                "source_url": source_url,
                "source_title": source_title,
                "document_url": link.url,
                "label": compact_text(link.label, 240),
                "context": compact_text(link.reason, 240),
            }
        )
        return DocumentLinkTriageDecision.model_validate(extract_json_object(message.content))


class TranscriptEvidenceAgent:
    """High-precision transcript evidence classifier for the final save gate."""

    SYSTEM_PROMPT = TRANSCRIPT_EVIDENCE_PROMPT.system_prompt
    HUMAN_PROMPT = TRANSCRIPT_EVIDENCE_PROMPT.human_prompt or ""

    def __init__(self, model: str, base_url: str | None = None, *, text_chars: int = 5000) -> None:
        self.text_chars = text_chars
        self.chain = (
            ChatPromptTemplate.from_messages(
                [
                    ("system", self.SYSTEM_PROMPT),
                    ("human", self.HUMAN_PROMPT),
                ]
            )
            | build_llm(model, base_url=base_url, json_mode=True)
        )

    def classify(
        self,
        *,
        company_name: str,
        ticker: str,
        url: str,
        title: str,
        text: str,
    ) -> TranscriptEvidenceDecision:
        message = self.chain.invoke(
            {
                "company_name": company_name,
                "ticker": ticker,
                "url": url,
                "title": title,
                "text": compact_text(text, self.text_chars),
            }
        )
        return TranscriptEvidenceDecision.model_validate(extract_json_object(message.content))


class LinkBatchTriageAgent:
    """Ranks extracted page links for crawl follow-up in one LLM call."""

    SYSTEM_PROMPT = LINK_BATCH_TRIAGE_PROMPT.system_prompt
    HUMAN_PROMPT = LINK_BATCH_TRIAGE_PROMPT.human_prompt or ""

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        *,
        max_links: int = 80,
        guidance: PromptGuidance | None = None,
    ) -> None:
        self.max_links = max_links
        self.guidance = guidance
        self.chain = (
            ChatPromptTemplate.from_messages(
                [
                    ("system", self.SYSTEM_PROMPT),
                    ("human", self.HUMAN_PROMPT),
                ]
            )
            | build_llm(model, base_url=base_url, json_mode=True)
        )

    def triage(
        self,
        *,
        company_name: str,
        ticker: str,
        url: str,
        title: str,
        links: list[CandidateLink],
    ) -> LinkBatchTriageDecision:
        compact_links = compact_candidate_links(links, max_links=self.max_links)
        message = self.chain.invoke(
            {
                "company_name": company_name,
                "ticker": ticker,
                "url": url,
                "title": title,
                "guidance": guidance_for_agent(getattr(self, "guidance", None), "page", agent_name="LinkBatchTriageAgent"),
                "links_json": json.dumps(compact_links, ensure_ascii=True),
            }
        )
        decision = LinkBatchTriageDecision.model_validate(extract_json_object(message.content))
        candidate_urls = {link.url for link in links}
        decision.selections = [selection for selection in decision.selections if selection.url in candidate_urls]
        return decision


class NavigationCandidateRankingAgent:
    """Ranks navigation candidate links before specialist selection."""

    SYSTEM_PROMPT = NAVIGATION_CANDIDATE_RANKING_PROMPT.system_prompt
    HUMAN_PROMPT = NAVIGATION_CANDIDATE_RANKING_PROMPT.human_prompt or ""

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        *,
        max_links: int = 40,
        guidance: PromptGuidance | None = None,
    ) -> None:
        self.max_links = max_links
        self.guidance = guidance
        self.chain = (
            ChatPromptTemplate.from_messages(
                [
                    ("system", self.SYSTEM_PROMPT),
                    ("human", self.HUMAN_PROMPT),
                ]
            )
            | build_llm(model, base_url=base_url, json_mode=True)
        )

    def rank(
        self,
        *,
        company_name: str,
        ticker: str,
        url: str,
        title: str,
        page_context: str,
        links: list[CandidateLink],
    ) -> NavigationCandidateRankingDecision:
        compact_links = compact_candidate_links(links, max_links=self.max_links)
        message = self.chain.invoke(
            {
                "company_name": company_name,
                "ticker": ticker,
                "url": url,
                "title": title,
                "page_context": page_context,
                "guidance": guidance_for_agent(getattr(self, "guidance", None), "navigation", agent_name="NavigationCandidateRankingAgent"),
                "links_json": json.dumps(compact_links, ensure_ascii=True),
            }
        )
        decision = NavigationCandidateRankingDecision.model_validate(extract_json_object(message.content))
        candidate_urls = {link.url for link in links}
        decision.selections = [selection for selection in decision.selections if selection.url in candidate_urls]
        return decision


class NavigationPageContextAgent:
    """Classifies the navigation role of a fetched page."""

    SYSTEM_PROMPT = NAVIGATION_PAGE_CONTEXT_PROMPT.system_prompt
    HUMAN_PROMPT = NAVIGATION_PAGE_CONTEXT_PROMPT.human_prompt or ""

    def __init__(self, model: str, base_url: str | None = None, *, text_chars: int = 900) -> None:
        self.text_chars = text_chars
        self.chain = (
            ChatPromptTemplate.from_messages(
                [
                    ("system", self.SYSTEM_PROMPT),
                    ("human", self.HUMAN_PROMPT),
                ]
            )
            | build_llm(model, base_url=base_url, json_mode=True)
        )

    def classify(
        self,
        *,
        company_name: str,
        ticker: str,
        url: str,
        title: str,
        text: str,
    ) -> NavigationPageContextDecision:
        message = self.chain.invoke(
            {
                "company_name": company_name,
                "ticker": ticker,
                "url": url,
                "title": title,
                "text": compact_text(text, self.text_chars),
            }
        )
        return NavigationPageContextDecision.model_validate(extract_json_object(message.content))


class CompanyPlaybookAgent:
    """Builds a ticker-specific advisory playbook stored in company memory."""

    SYSTEM_PROMPT = COMPANY_PLAYBOOK_PROMPT.system_prompt
    HUMAN_PROMPT = COMPANY_PLAYBOOK_PROMPT.human_prompt or ""

    def __init__(self, model: str, base_url: str | None = None, *, text_chars: int = 1800) -> None:
        self.text_chars = text_chars
        self.chain = (
            ChatPromptTemplate.from_messages(
                [
                    ("system", self.SYSTEM_PROMPT),
                    ("human", self.HUMAN_PROMPT),
                ]
            )
            | build_llm(model, base_url=base_url, json_mode=True)
        )

    def build(self, *, company: Company, memory: CompanyMemory) -> CompanyPlaybook:
        memory_json = compact_text(
            json.dumps(memory.model_dump(mode="json"), ensure_ascii=True),
            self.text_chars,
        )
        message = self.chain.invoke(
            {
                "company": f"{company_display_name(company)} ({company.symbol})",
                "memory_json": memory_json,
            }
        )
        return sanitize_company_playbook(
            CompanyPlaybook.model_validate(extract_json_object(message.content)),
        )


class MemoryGuidanceAgent:
    """Converts company memory into advisory prompt guidance."""

    SYSTEM_PROMPT = MEMORY_GUIDANCE_PROMPT.system_prompt
    HUMAN_PROMPT = MEMORY_GUIDANCE_PROMPT.human_prompt or ""

    def __init__(self, model: str, base_url: str | None = None, *, text_chars: int = 1400) -> None:
        self.text_chars = text_chars
        self.chain = (
            ChatPromptTemplate.from_messages(
                [
                    ("system", self.SYSTEM_PROMPT),
                    ("human", self.HUMAN_PROMPT),
                ]
            )
            | build_llm(model, base_url=base_url, json_mode=True)
        )

    def guide(self, *, company: Company, memory: CompanyMemory) -> PromptGuidance:
        memory_json = compact_text(
            json.dumps(memory.model_dump(mode="json"), ensure_ascii=True),
            self.text_chars,
        )
        message = self.chain.invoke(
            {
                "company": f"{company_display_name(company)} ({company.symbol})",
                "memory_json": memory_json,
            }
        )
        return sanitize_prompt_guidance(
            PromptGuidance.model_validate(extract_json_object(message.content)),
            company=company,
        )


class SearchQueryPlannerAgent:
    """Plans discovery search queries from identity and memory context."""

    SYSTEM_PROMPT = SEARCH_QUERY_PLANNER_PROMPT.system_prompt
    HUMAN_PROMPT = SEARCH_QUERY_PLANNER_PROMPT.human_prompt or ""

    def __init__(self, model: str, base_url: str | None = None, *, text_chars: int = 1000) -> None:
        self.text_chars = text_chars
        self.chain = (
            ChatPromptTemplate.from_messages(
                [
                    ("system", self.SYSTEM_PROMPT),
                    ("human", self.HUMAN_PROMPT),
                ]
            )
            | build_llm(model, base_url=base_url, json_mode=True)
        )

    def plan(
        self,
        *,
        company: Company,
        guidance: PromptGuidance | None = None,
        navigation_memory: CompanyNavigationMemory | None = None,
        limit: int = 4,
    ) -> SearchQueryPlan:
        message = self.chain.invoke(
            {
                "company_name": company_display_name(company),
                "ticker": company.symbol,
                "guidance": guidance_for_agent(guidance, "search", agent_name="SearchQueryPlannerAgent"),
                "memory_summary": discovery_memory_summary(navigation_memory),
                "limit": limit,
            }
        )
        plan = SearchQueryPlan.model_validate(extract_json_object(message.content))
        plan.queries = sanitize_search_queries(plan.queries, limit=limit, ticker=company.symbol)
        return plan


class SearchCandidateRankingAgent:
    """Ranks search result candidates before deterministic fallback scoring is used."""

    SYSTEM_PROMPT = SEARCH_CANDIDATE_RANKING_PROMPT.system_prompt
    HUMAN_PROMPT = SEARCH_CANDIDATE_RANKING_PROMPT.human_prompt or ""

    def __init__(self, model: str, base_url: str | None = None, *, max_candidates: int = 25) -> None:
        self.max_candidates = max_candidates
        self.chain = (
            ChatPromptTemplate.from_messages(
                [
                    ("system", self.SYSTEM_PROMPT),
                    ("human", self.HUMAN_PROMPT),
                ]
            )
            | build_llm(model, base_url=base_url, json_mode=True)
        )

    def rank(
        self,
        *,
        company: Company,
        candidates: list[IRDiscoveryCandidate],
        guidance: PromptGuidance | None = None,
        navigation_memory: CompanyNavigationMemory | None = None,
    ) -> SearchCandidateRankingDecision:
        compact = [
            {
                "url": candidate.url,
                "title": compact_text(candidate.title, 120),
                "snippet": compact_text(candidate.snippet, 240),
                "source": candidate.source,
                "fallback_score": candidate.score,
                "fallback_reasons": candidate.reasons[:5],
            }
            for candidate in candidates[: self.max_candidates]
        ]
        message = self.chain.invoke(
            {
                "company_name": company_display_name(company),
                "ticker": company.symbol,
                "guidance": guidance_for_agent(guidance, "search", agent_name="SearchCandidateRankingAgent"),
                "memory_summary": discovery_memory_summary(navigation_memory),
                "candidates_json": json.dumps(compact, ensure_ascii=True),
            }
        )
        decision = SearchCandidateRankingDecision.model_validate(extract_json_object(message.content))
        candidate_urls = {candidate.url for candidate in candidates}
        decision.selections = [selection for selection in decision.selections if selection.url in candidate_urls]
        return decision


class OfficialTranscriptResearchAgent:
    """Proposes official-source transcript discovery targets from search evidence."""

    SYSTEM_PROMPT = TRANSCRIPT_RESEARCH_PROMPT.system_prompt
    HUMAN_PROMPT = TRANSCRIPT_RESEARCH_PROMPT.human_prompt or ""

    def __init__(self, model: str, base_url: str | None = None, *, max_candidates: int = 30) -> None:
        self.max_candidates = max_candidates
        self.chain = (
            ChatPromptTemplate.from_messages(
                [
                    ("system", self.SYSTEM_PROMPT),
                    ("human", self.HUMAN_PROMPT),
                ]
            )
            | build_llm(model, base_url=base_url, json_mode=True)
        )

    def research(
        self,
        *,
        company: Company,
        candidates: list[IRDiscoveryCandidate],
        guidance: PromptGuidance | None = None,
        navigation_memory: CompanyNavigationMemory | None = None,
    ) -> TranscriptResearchProposal:
        compact = [
            {
                "url": candidate.url,
                "title": compact_text(candidate.title, 140),
                "snippet": compact_text(candidate.snippet, 320),
                "source": candidate.source,
                "score": candidate.score,
                "reasons": candidate.reasons[:5],
            }
            for candidate in candidates[: self.max_candidates]
        ]
        message = self.chain.invoke(
            {
                "company_name": company_display_name(company),
                "ticker": company.symbol,
                "guidance": guidance_for_agent(guidance, "search", agent_name="OfficialTranscriptResearchAgent"),
                "memory_summary": discovery_memory_summary(navigation_memory),
                "candidates_json": json.dumps(compact, ensure_ascii=True),
            }
        )
        proposal = TranscriptResearchProposal.model_validate(extract_json_object(message.content))
        allowed_urls = {candidate.url for candidate in candidates}
        proposal.official_homepage_urls = [url for url in proposal.official_homepage_urls if url in allowed_urls]
        proposal.official_ir_urls = [url for url in proposal.official_ir_urls if url in allowed_urls]
        proposal.transcript_candidate_urls = [url for url in proposal.transcript_candidate_urls if url in allowed_urls]
        return proposal


class TranscriptResearchJudgeAgent:
    """Judges whether a research proposal has enough official-source evidence."""

    SYSTEM_PROMPT = TRANSCRIPT_RESEARCH_JUDGE_PROMPT.system_prompt
    HUMAN_PROMPT = TRANSCRIPT_RESEARCH_JUDGE_PROMPT.human_prompt or ""

    def __init__(self, model: str, base_url: str | None = None, *, text_chars: int = 4000) -> None:
        self.text_chars = text_chars
        self.chain = (
            ChatPromptTemplate.from_messages(
                [
                    ("system", self.SYSTEM_PROMPT),
                    ("human", self.HUMAN_PROMPT),
                ]
            )
            | build_llm(model, base_url=base_url, json_mode=True)
        )

    def judge(
        self,
        *,
        company: Company,
        proposal: TranscriptResearchProposal,
        evidence: dict,
        guidance: PromptGuidance | None = None,
    ) -> TranscriptResearchJudgment:
        message = self.chain.invoke(
            {
                "company_name": company_display_name(company),
                "ticker": company.symbol,
                "guidance": guidance_for_agent(guidance, "search", agent_name="TranscriptResearchJudgeAgent"),
                "proposal_json": proposal.model_dump_json(),
                "evidence_json": compact_text(json.dumps(evidence, ensure_ascii=True), self.text_chars),
            }
        )
        judgment = TranscriptResearchJudgment.model_validate(extract_json_object(message.content))
        proposed_urls = {
            *proposal.official_homepage_urls,
            *proposal.official_ir_urls,
            *proposal.transcript_candidate_urls,
        }
        judgment.accepted_homepage_urls = [url for url in judgment.accepted_homepage_urls if url in proposed_urls]
        judgment.accepted_ir_urls = [url for url in judgment.accepted_ir_urls if url in proposed_urls]
        judgment.accepted_transcript_urls = [url for url in judgment.accepted_transcript_urls if url in proposed_urls]
        if not (judgment.accepted_homepage_urls or judgment.accepted_ir_urls or judgment.accepted_transcript_urls):
            judgment.accepted = False
        return judgment


class OrchestrationAnalysisAgent:
    """Classifies crawl outcomes for supervisor retries."""

    SYSTEM_PROMPT = ORCHESTRATION_ANALYSIS_PROMPT.system_prompt
    HUMAN_PROMPT = ORCHESTRATION_ANALYSIS_PROMPT.human_prompt or ""

    def __init__(self, model: str, base_url: str | None = None, *, text_chars: int = 2200) -> None:
        self.text_chars = text_chars
        self.chain = (
            ChatPromptTemplate.from_messages(
                [
                    ("system", self.SYSTEM_PROMPT),
                    ("human", self.HUMAN_PROMPT),
                ]
            )
            | build_llm(model, base_url=base_url, json_mode=True)
        )

    def analyze(self, *, company: Company, evidence: dict) -> OrchestrationAnalysisDecision:
        message = self.chain.invoke(
            {
                "company": f"{company_display_name(company)} ({company.symbol})",
                "evidence_json": compact_text(json.dumps(evidence, ensure_ascii=True), self.text_chars),
            }
        )
        return OrchestrationAnalysisDecision.model_validate(extract_json_object(message.content))


class RetryPlanningAgent:
    """Plans the next supervisor action from a failure analysis and current config."""

    SYSTEM_PROMPT = RETRY_PLANNING_PROMPT.system_prompt
    HUMAN_PROMPT = RETRY_PLANNING_PROMPT.human_prompt or ""

    def __init__(self, model: str, base_url: str | None = None, *, text_chars: int = 1600) -> None:
        self.text_chars = text_chars
        self.chain = (
            ChatPromptTemplate.from_messages(
                [
                    ("system", self.SYSTEM_PROMPT),
                    ("human", self.HUMAN_PROMPT),
                ]
            )
            | build_llm(model, base_url=base_url, json_mode=True)
        )

    def plan(self, *, company: Company, evidence: dict) -> RetryPlanningDecision:
        message = self.chain.invoke(
            {
                "company": f"{company_display_name(company)} ({company.symbol})",
                "evidence_json": compact_text(json.dumps(evidence, ensure_ascii=True), self.text_chars),
            }
        )
        return RetryPlanningDecision.model_validate(extract_json_object(message.content))


class PromptPlannerAgent:
    """Produces non-authoritative prompt guidance from company memory."""

    SYSTEM_PROMPT = PROMPT_PLANNER_PROMPT.system_prompt
    HUMAN_PROMPT = PROMPT_PLANNER_PROMPT.human_prompt or ""

    def __init__(self, model: str, base_url: str | None = None, *, text_chars: int = 1200) -> None:
        self.text_chars = text_chars
        self.chain = (
            ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        self.SYSTEM_PROMPT,
                    ),
                    (
                        "human",
                        self.HUMAN_PROMPT,
                    ),
                ]
            )
            | build_llm(model, base_url=base_url, json_mode=True)
        )

    def plan(self, *, company: Company, memory: CompanyMemory) -> PromptGuidance:
        memory_json = compact_text(
            json.dumps(memory.model_dump(mode="json"), ensure_ascii=True),
            self.text_chars,
        )
        message = self.chain.invoke(
            {
                "company": f"{company_display_name(company)} ({company.symbol})",
                "memory_json": memory_json,
            }
        )
        return sanitize_prompt_guidance(
            PromptGuidance.model_validate(extract_json_object(message.content)),
            company=company,
        )


class CrawlReflectionAgent:
    """Learns advisory navigation guidance from a failed supervised attempt."""

    SYSTEM_PROMPT = CRAWL_REFLECTION_PROMPT.system_prompt
    HUMAN_PROMPT = CRAWL_REFLECTION_PROMPT.human_prompt or ""

    def __init__(self, model: str, base_url: str | None = None, *, text_chars: int = 1800) -> None:
        self.text_chars = text_chars
        self.chain = (
            ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        self.SYSTEM_PROMPT,
                    ),
                    (
                        "human",
                        self.HUMAN_PROMPT,
                    ),
                ]
            )
            | build_llm(model, base_url=base_url, json_mode=True)
        )

    def reflect(self, *, company: Company, evidence: dict) -> CrawlReflection:
        evidence_json = compact_text(json.dumps(evidence, ensure_ascii=True), self.text_chars)
        message = self.chain.invoke(
            {
                "company": f"{company_display_name(company)} ({company.symbol})",
                "evidence_json": evidence_json,
            }
        )
        return sanitize_crawl_reflection(CrawlReflection.model_validate(extract_json_object(message.content)))


class LinkSelectionAgent:
    """Small specialist agent that chooses links for one navigation task."""

    HUMAN_PROMPT = LINK_SELECTION_PROMPT.human_prompt or ""

    def __init__(
        self,
        *,
        model: str,
        system_prompt: str,
        base_url: str | None = None,
        max_links: int = 12,
        text_chars: int = 800,
        guidance: PromptGuidance | None = None,
    ) -> None:
        self.max_links = max_links
        self.text_chars = text_chars
        self.guidance = guidance
        self.chain = (
            ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    ("human", self.HUMAN_PROMPT),
                ]
            )
            | build_llm(model, base_url=base_url, json_mode=True)
        )

    def decide(
        self,
        *,
        company_name: str,
        ticker: str,
        url: str,
        title: str,
        text: str,
        links: list[CandidateLink],
        repair_guidance: str | None = None,
    ) -> NavigationDecision:
        compact_links = compact_candidate_links(links, max_links=self.max_links)
        message = self.chain.invoke(
            {
                "company_name": company_name,
                "ticker": ticker,
                "url": url,
                "title": title,
                "guidance": combine_guidance(
                    guidance_for_agent(getattr(self, "guidance", None), "navigation", agent_name=self.__class__.__name__),
                    repair_guidance,
                ),
                "text": compact_text(text, self.text_chars),
                "links_json": json.dumps(compact_links, ensure_ascii=True),
            }
        )
        draft = NavigationDecision.model_validate(extract_json_object(message.content))
        urls_by_candidate = {link.url for link in links}
        draft.chosen_urls = [url for url in draft.chosen_urls if url in urls_by_candidate]
        return draft


class NavigationValidationAgent:
    """Soft semantic validator for specialist navigation choices."""

    SYSTEM_PROMPT = NAVIGATION_VALIDATION_PROMPT.system_prompt
    HUMAN_PROMPT = NAVIGATION_VALIDATION_PROMPT.human_prompt or ""

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        *,
        max_links: int = 12,
        guidance: PromptGuidance | None = None,
    ) -> None:
        self.max_links = max_links
        self.guidance = guidance
        self.chain = (
            ChatPromptTemplate.from_messages(
                [
                    ("system", self.SYSTEM_PROMPT),
                    ("human", self.HUMAN_PROMPT),
                ]
            )
            | build_llm(model, base_url=base_url, json_mode=True)
        )

    def validate(
        self,
        *,
        kind: NavigationAgentKind,
        company_name: str,
        ticker: str,
        url: str,
        title: str,
        links: list[CandidateLink],
        decision: NavigationDecision,
    ) -> NavigationValidationResult:
        compact_links = compact_candidate_links(links, max_links=self.max_links)
        message = self.chain.invoke(
            {
                "kind": kind,
                "company_name": company_name,
                "ticker": ticker,
                "url": url,
                "title": title,
                "guidance": guidance_for_agent(getattr(self, "guidance", None), "navigation", agent_name="NavigationValidationAgent"),
                "chosen_urls_json": json.dumps(decision.chosen_urls, ensure_ascii=True),
                "links_json": json.dumps(compact_links, ensure_ascii=True),
            }
        )
        result = NavigationValidationResult.model_validate(extract_json_object(message.content))
        candidate_urls = {link.url for link in links}
        result.accepted_urls = [url for url in result.accepted_urls if url in candidate_urls]
        result.rejected_urls = [url for url in result.rejected_urls if url in candidate_urls]
        if result.is_valid and not result.accepted_urls:
            result.accepted_urls = [url for url in decision.chosen_urls if url in candidate_urls]
        result.is_valid = bool(result.is_valid and result.accepted_urls)
        return result


class HomepageNavAgent(LinkSelectionAgent):
    SYSTEM_PROMPT = HOMEPAGE_NAV_PROMPT.system_prompt

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        *,
        max_links: int = 12,
        text_chars: int = 800,
        guidance: PromptGuidance | None = None,
    ) -> None:
        super().__init__(
            model=model,
            system_prompt=self.SYSTEM_PROMPT,
            base_url=base_url,
            max_links=max_links,
            text_chars=text_chars,
            guidance=guidance,
        )


class IRSectionAgent(LinkSelectionAgent):
    SYSTEM_PROMPT = IR_SECTION_PROMPT.system_prompt

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        *,
        max_links: int = 12,
        text_chars: int = 800,
        guidance: PromptGuidance | None = None,
    ) -> None:
        super().__init__(
            model=model,
            system_prompt=self.SYSTEM_PROMPT,
            base_url=base_url,
            max_links=max_links,
            text_chars=text_chars,
            guidance=guidance,
        )


class EventListingAgent(LinkSelectionAgent):
    SYSTEM_PROMPT = EVENT_LISTING_PROMPT.system_prompt

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        *,
        max_links: int = 12,
        text_chars: int = 800,
        guidance: PromptGuidance | None = None,
    ) -> None:
        super().__init__(
            model=model,
            system_prompt=self.SYSTEM_PROMPT,
            base_url=base_url,
            max_links=max_links,
            text_chars=text_chars,
            guidance=guidance,
        )


class TranscriptLinkAgent(LinkSelectionAgent):
    SYSTEM_PROMPT = TRANSCRIPT_LINK_PROMPT.system_prompt

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        *,
        max_links: int = 12,
        text_chars: int = 800,
        guidance: PromptGuidance | None = None,
    ) -> None:
        super().__init__(
            model=model,
            system_prompt=self.SYSTEM_PROMPT,
            base_url=base_url,
            max_links=max_links,
            text_chars=text_chars,
            guidance=guidance,
        )


class IRNavigationAgent:
    """Routes navigation decisions to focused local Ollama link-selection agents."""

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        *,
        max_links: int = 12,
        text_chars: int = 800,
        guidance: PromptGuidance | None = None,
        confidence_floor: float = 0.55,
    ) -> None:
        self.guidance = guidance
        self.confidence_floor = confidence_floor
        self.agents: dict[NavigationAgentKind, LinkSelectionAgent] = {
            "homepage": HomepageNavAgent(model, base_url=base_url, max_links=max_links, text_chars=text_chars, guidance=guidance),
            "ir_section": IRSectionAgent(model, base_url=base_url, max_links=max_links, text_chars=text_chars, guidance=guidance),
            "event_listing": EventListingAgent(model, base_url=base_url, max_links=max_links, text_chars=text_chars, guidance=guidance),
            "transcript_link": TranscriptLinkAgent(model, base_url=base_url, max_links=max_links, text_chars=text_chars, guidance=guidance),
        }
        self.validator = NavigationValidationAgent(
            model,
            base_url=base_url,
            max_links=max_links,
            guidance=guidance,
        )
        self.graph = self._build_graph()

    def decide(
        self,
        *,
        company_name: str,
        ticker: str,
        url: str,
        title: str,
        text: str,
        links: list[CandidateLink],
        page_context: str,
    ) -> NavigationDecision:
        kind = self.select_agent_kind(url=url, title=title, text=text, links=links, page_context=page_context)
        if not hasattr(self, "graph"):
            decision = self.agents[kind].decide(
                company_name=company_name,
                ticker=ticker,
                url=url,
                title=title,
                text=text,
                links=links,
            )
            decision.reason = f"{kind}: {decision.reason}" if decision.reason else kind
            return decision
        state: NavigationDecisionState = {
            "kind": kind,
            "company_name": company_name,
            "ticker": ticker,
            "url": url,
            "title": title,
            "text": text,
            "links": links,
            "page_context": page_context,
            "repair_used": False,
        }
        final_state = self.graph.invoke(state)
        decision = final_state.get("final_decision") or NavigationDecision(
            chosen_urls=[],
            confidence=0.0,
            reason=f"{kind}: navigation_decision_failed",
            stop_reason="navigation_decision_failed",
        )
        decision.reason = f"{kind}: {decision.reason}" if not decision.reason.startswith(f"{kind}:") else decision.reason
        return decision

    def _build_graph(self):
        graph = StateGraph(NavigationDecisionState)
        graph.add_node("call_specialist", self._call_specialist_node)
        graph.add_node("hard_validate", self._hard_validate_node)
        graph.add_node("semantic_validate", self._semantic_validate_node)
        graph.add_node("repair", self._repair_node)
        graph.add_node("fallback", self._fallback_node)
        graph.add_edge(START, "call_specialist")
        graph.add_edge("call_specialist", "hard_validate")
        graph.add_conditional_edges(
            "hard_validate",
            self._after_hard_validate,
            {"semantic": "semantic_validate", "repair": "repair", "fallback": "fallback"},
        )
        graph.add_conditional_edges(
            "semantic_validate",
            self._after_semantic_validate,
            {"accept": END, "repair": "repair", "fallback": "fallback"},
        )
        graph.add_edge("repair", "call_specialist")
        graph.add_edge("fallback", END)
        return graph.compile()

    def _call_specialist_node(self, state: "NavigationDecisionState") -> "NavigationDecisionState":
        kind = state["kind"]
        decision = self.agents[kind].decide(
            company_name=state["company_name"],
            ticker=state["ticker"],
            url=state["url"],
            title=state["title"],
            text=state["text"],
            links=state["links"],
            repair_guidance=state.get("repair_guidance"),
        )
        state["decision"] = decision
        return state

    def _hard_validate_node(self, state: "NavigationDecisionState") -> "NavigationDecisionState":
        result = hard_validate_navigation_decision(
            kind=state["kind"],
            decision=state["decision"],
            links=state["links"],
            guidance=self.guidance,
            confidence_floor=self.confidence_floor,
        )
        state["hard_validation"] = result
        return state

    def _semantic_validate_node(self, state: "NavigationDecisionState") -> "NavigationDecisionState":
        try:
            result = self.validator.validate(
                kind=state["kind"],
                company_name=state["company_name"],
                ticker=state["ticker"],
                url=state["url"],
                title=state["title"],
                links=state["links"],
                decision=state["decision"],
            )
        except Exception:
            result = state["hard_validation"]
        state["semantic_validation"] = result
        if result.is_valid:
            state["final_decision"] = decision_from_validation(
                state["decision"],
                result,
                reason_prefix="agent_validated",
            )
        return state

    def _repair_node(self, state: "NavigationDecisionState") -> "NavigationDecisionState":
        validation = state.get("semantic_validation") or state.get("hard_validation")
        state["repair_guidance"] = compact_text(
            validation.repair_guidance if validation and validation.repair_guidance else (
                f"Previous decision rejected: {validation.reason if validation else 'invalid choice'}. "
                "Choose only links that fit the specialist task and run guidance."
            ),
            300,
        )
        state["repair_used"] = True
        state.pop("decision", None)
        state.pop("hard_validation", None)
        state.pop("semantic_validation", None)
        return state

    def _fallback_node(self, state: "NavigationDecisionState") -> "NavigationDecisionState":
        links = fallback_links_for_kind(state["kind"], state["links"], self.guidance)
        reason = "heuristic_repair_after_agent" if links else "agent_repair_failed"
        state["final_decision"] = NavigationDecision(
            chosen_urls=[link.url for link in links],
            confidence=0.0,
            reason=reason,
            stop_reason=None if links else "agent_repair_failed",
        )
        return state

    def _after_hard_validate(self, state: "NavigationDecisionState") -> str:
        validation = state["hard_validation"]
        if validation.is_valid:
            return "semantic"
        if not state.get("repair_used"):
            return "repair"
        return "fallback"

    def _after_semantic_validate(self, state: "NavigationDecisionState") -> str:
        validation = state["semantic_validation"]
        if validation.is_valid:
            return "accept"
        if not state.get("repair_used"):
            return "repair"
        return "fallback"

    def select_agent_kind(
        self,
        *,
        url: str,
        title: str,
        text: str,
        links: list[CandidateLink],
        page_context: str,
    ) -> NavigationAgentKind:
        return navigation_agent_kind(url=url, title=title, text=text, links=links, page_context=page_context)


def navigation_agent_kind(
    *,
    url: str,
    title: str,
    text: str,
    links: list[CandidateLink],
    page_context: str,
) -> NavigationAgentKind:
    haystack = f"{url} {title} {text[:500]}".lower()
    link_haystack = " ".join(f"{link.url} {link.label} {link.reason}" for link in links).lower()
    if page_context == "homepage":
        return "homepage"
    if any(token in haystack for token in ("event-details", "earnings-call", "earnings call")):
        return "transcript_link"
    if any(token in link_haystack for token in ("event-details", "earnings-call", "earnings call")):
        return "event_listing"
    if any(token in haystack for token in ("events", "earnings", "quarterly results", "financial reports")):
        return "event_listing"
    return "ir_section"


class NavigationDecisionState(TypedDict, total=False):
    kind: NavigationAgentKind
    company_name: str
    ticker: str
    url: str
    title: str
    text: str
    links: list[CandidateLink]
    page_context: str
    decision: NavigationDecision
    hard_validation: NavigationValidationResult
    semantic_validation: NavigationValidationResult
    repair_guidance: str
    repair_used: bool
    final_decision: NavigationDecision


def hard_validate_navigation_decision(
    *,
    kind: NavigationAgentKind,
    decision: NavigationDecision,
    links: list[CandidateLink],
    guidance: PromptGuidance | None,
    confidence_floor: float = 0.55,
) -> NavigationValidationResult:
    candidate_by_url = {link.url: link for link in links}
    accepted: list[str] = []
    rejected: list[str] = []
    for chosen_url in decision.chosen_urls[:3]:
        link = candidate_by_url.get(chosen_url)
        if not link:
            rejected.append(chosen_url)
            continue
        if link_matches_avoid_terms(link, guidance) and better_non_avoided_links(kind, links, guidance):
            rejected.append(chosen_url)
            continue
        if not link_fits_navigation_task(kind, link) and better_task_links(kind, links):
            rejected.append(chosen_url)
            continue
        accepted.append(chosen_url)

    if decision.confidence < confidence_floor and not accepted:
        return NavigationValidationResult(
            is_valid=False,
            rejected_urls=rejected,
            reason="agent_rejected_low_confidence",
            repair_guidance=f"Previous decision confidence {decision.confidence:.2f} was below {confidence_floor:.2f}. Choose a stronger task-fitting link.",
        )
    if accepted:
        reason = "agent_partially_validated" if rejected else "agent_hard_validated"
        return NavigationValidationResult(
            is_valid=True,
            accepted_urls=accepted,
            rejected_urls=rejected,
            reason=reason,
        )
    reason = "agent_rejected_wrong_task" if rejected else "agent_rejected_empty"
    return NavigationValidationResult(
        is_valid=False,
        rejected_urls=rejected,
        reason=reason,
        repair_guidance=f"Previous decision rejected: {reason}. Choose links that match the {kind} task.",
    )


def decision_from_validation(
    decision: NavigationDecision,
    validation: NavigationValidationResult,
    *,
    reason_prefix: str,
) -> NavigationDecision:
    return NavigationDecision(
        chosen_urls=validation.accepted_urls,
        confidence=decision.confidence,
        reason=f"{reason_prefix}: {validation.reason}",
        stop_reason=decision.stop_reason,
    )


def link_matches_avoid_terms(link: CandidateLink, guidance: PromptGuidance | None) -> bool:
    haystack = f"{link.url} {link.label} {link.reason}".lower()
    if link_matches_intrinsic_avoid_terms(link):
        return True
    if not guidance:
        return False
    return any(term.lower() in haystack for term in guidance.avoid_terms)


def link_matches_intrinsic_avoid_terms(link: CandidateLink) -> bool:
    haystack = f"{link.url} {link.label} {link.reason}".lower()
    return any(
        token in haystack
        for token in (
            "blog.",
            "blog/",
            "youtube.com",
            "youtu.be",
            "presentation",
            "webcast-only",
            "/chinese/",
            "/schinese/",
            "/tchinese/",
            "/japanese/",
            "/korean/",
            "/zh/",
            "/zh-cn/",
            "/zh-tw/",
            "/ja/",
            "/ko/",
        )
    )


def better_non_avoided_links(
    kind: NavigationAgentKind,
    links: list[CandidateLink],
    guidance: PromptGuidance | None,
) -> list[CandidateLink]:
    return [link for link in links if not link_matches_avoid_terms(link, guidance) and link_fits_navigation_task(kind, link)]


def better_task_links(kind: NavigationAgentKind, links: list[CandidateLink]) -> list[CandidateLink]:
    return [link for link in links if link_fits_navigation_task(kind, link)]


def link_fits_navigation_task(kind: NavigationAgentKind, link: CandidateLink) -> bool:
    haystack = f"{link.url} {link.label} {link.reason}".lower()
    task_terms: dict[NavigationAgentKind, tuple[str, ...]] = {
        "homepage": ("investor", "ir", "shareholder"),
        "ir_section": ("earnings", "events", "financial reports", "quarterly results", "results"),
        "event_listing": ("event-details", "earnings-call", "earnings call", "quarterly earnings"),
        "transcript_link": ("transcript", ".pdf", ".docx", "q&a", "prepared remarks"),
    }
    return any(term in haystack for term in task_terms[kind])


def fallback_links_for_kind(
    kind: NavigationAgentKind,
    links: list[CandidateLink],
    guidance: PromptGuidance | None,
    *,
    limit: int = 2,
) -> list[CandidateLink]:
    candidates = [
        link for link in links
        if link_fits_navigation_task(kind, link) and not link_matches_avoid_terms(link, guidance)
    ]
    if not candidates:
        return []
    return candidates[:limit]


def extract_json_object(content: str) -> dict:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", content, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in model output: {content[:200]}")
    return json.loads(match.group(0))


def compact_text(value: str, limit: int) -> str:
    return " ".join(value.split())[:limit]


def compact_candidate_links(
    links: list[CandidateLink],
    *,
    max_links: int,
    char_budget: int = LINKS_JSON_CHAR_BUDGET,
) -> list[dict[str, str]]:
    compact_links: list[dict[str, str]] = []
    for link in links[:max_links]:
        item = {
            "url": link.url,
            "label": compact_text(link.label, 80),
            "context": compact_text(link.reason, 80),
        }
        candidate = [*compact_links, item]
        if compact_links and len(json.dumps(candidate, ensure_ascii=True)) > char_budget:
            break
        compact_links.append(item)
    return compact_links


def guidance_for_agent(
    guidance: PromptGuidance | None,
    mode: Literal["homepage", "navigation", "page", "search"],
    *,
    agent_name: str | None = None,
) -> str:
    if not guidance:
        return "None."
    parts = []
    if guidance.run_objective:
        parts.append(f"Objective: {guidance.run_objective}")
    if guidance.strategy:
        parts.append(f"Run strategy: {guidance.strategy}")
    if guidance.priority_terms:
        parts.append(f"Prefer: {', '.join(guidance.priority_terms[:8])}.")
    if guidance.avoid_terms:
        parts.append(f"Avoid: {', '.join(guidance.avoid_terms[:8])}.")
    if mode == "homepage":
        text = guidance.homepage_guidance or guidance.navigation_guidance
    elif mode == "search":
        text = guidance.search_guidance or guidance.navigation_guidance
    elif mode == "page":
        text = guidance.transcript_guidance
    else:
        text = guidance.navigation_guidance
    if text:
        parts.append(text)
    if agent_name and guidance.agent_guidance.get(agent_name):
        parts.append(f"{agent_name}: {guidance.agent_guidance[agent_name]}")
    if guidance.risk_notes:
        parts.append(f"Risks: {'; '.join(guidance.risk_notes[:3])}.")
    return compact_text(" ".join(parts), 900) or "None."


def combine_guidance(base_guidance: str, repair_guidance: str | None) -> str:
    if not repair_guidance:
        return base_guidance
    if base_guidance == "None.":
        return compact_text(repair_guidance, 500)
    return compact_text(f"{base_guidance} Repair: {repair_guidance}", 500)


def discovery_memory_summary(navigation_memory: CompanyNavigationMemory | None) -> str:
    if not navigation_memory:
        return "None."
    parts = []
    if navigation_memory.preferred_hosts:
        parts.append(f"Prefer hosts: {', '.join(navigation_memory.preferred_hosts[:5])}.")
    if navigation_memory.successful_hosts:
        parts.append(f"Prior successful hosts: {', '.join(navigation_memory.successful_hosts[:5])}.")
    if navigation_memory.known_ir_home_urls:
        parts.append(f"Known IR homes: {', '.join(navigation_memory.known_ir_home_urls[:3])}.")
    if navigation_memory.known_event_listing_urls:
        parts.append(f"Known earnings/results pages: {', '.join(navigation_memory.known_event_listing_urls[:3])}.")
    if navigation_memory.known_transcript_urls:
        parts.append(f"Known transcript URLs: {', '.join(navigation_memory.known_transcript_urls[:3])}.")
    if navigation_memory.low_value_hosts:
        parts.append(f"Avoid low-value hosts: {', '.join(navigation_memory.low_value_hosts[:5])}.")
    if navigation_memory.low_value_path_terms:
        parts.append(f"Avoid path terms: {', '.join(navigation_memory.low_value_path_terms[:8])}.")
    if navigation_memory.robots_blocked_hosts:
        parts.append(f"Robots-risk hosts: {', '.join(navigation_memory.robots_blocked_hosts[:5])}.")
    return compact_text(" ".join(parts), 900) or "None."


def sanitize_prompt_guidance(guidance: PromptGuidance, *, company: Company | None = None) -> PromptGuidance:
    banned = ("ignore robots", "disable robots", "fail open", "third-party transcript")
    risk_notes = [
        note for note in guidance.risk_notes[:5]
        if not any(token in note.lower() for token in banned)
    ]
    banned_exact_terms = prompt_guidance_banned_exact_terms(company)
    return PromptGuidance(
        run_objective="" if not safe_reflection_text(guidance.run_objective, banned) else compact_text(guidance.run_objective, 220),
        strategy="" if not safe_reflection_text(guidance.strategy, banned) else compact_text(guidance.strategy, 360),
        priority_terms=[
            compact_text(term, 80)
            for term in guidance.priority_terms[:10]
            if safe_prompt_guidance_term(term, banned, banned_exact_terms)
        ],
        avoid_terms=[
            compact_text(term, 80)
            for term in guidance.avoid_terms[:10]
            if safe_prompt_guidance_term(term, banned, banned_exact_terms)
        ],
        homepage_guidance="" if not safe_reflection_text(guidance.homepage_guidance, banned) else compact_text(guidance.homepage_guidance, 280),
        search_guidance="" if not safe_reflection_text(guidance.search_guidance, banned) else compact_text(guidance.search_guidance, 280),
        navigation_guidance="" if not safe_reflection_text(guidance.navigation_guidance, banned) else compact_text(guidance.navigation_guidance, 280),
        transcript_guidance="" if not safe_reflection_text(guidance.transcript_guidance, banned) else compact_text(guidance.transcript_guidance, 280),
        agent_guidance={
            compact_text(name, 80): compact_text(text, 240)
            for name, text in list(guidance.agent_guidance.items())[:12]
            if safe_reflection_text(name, banned) and safe_reflection_text(text, banned)
        },
        risk_notes=risk_notes,
    )


def sanitize_company_playbook(playbook: CompanyPlaybook) -> CompanyPlaybook:
    banned = ("ignore robots", "disable robots", "fail open", "third-party transcript", "seeking alpha", "quartr")
    return CompanyPlaybook(
        issuer_name=compact_text(playbook.issuer_name or "", 120) or None,
        brand_names=[
            compact_text(value, 80)
            for value in playbook.brand_names[:8]
            if safe_reflection_text(value, banned)
        ],
        official_homepage_candidates=[
            value
            for value in playbook.official_homepage_candidates[:8]
            if safe_reflection_text(value, banned) and value.startswith(("http://", "https://"))
        ],
        preferred_ir_urls=[
            value
            for value in playbook.preferred_ir_urls[:12]
            if safe_reflection_text(value, banned) and value.startswith(("http://", "https://"))
        ],
        avoid_hosts=[
            compact_text(value, 120)
            for value in playbook.avoid_hosts[:12]
            if safe_reflection_text(value, banned)
        ],
        avoid_urls=[
            value
            for value in playbook.avoid_urls[:12]
            if safe_reflection_text(value, banned) and value.startswith(("http://", "https://"))
        ],
        planner_prompt="" if not safe_reflection_text(playbook.planner_prompt, banned) else compact_text(playbook.planner_prompt, 360),
        homepage_strategy="" if not safe_reflection_text(playbook.homepage_strategy, banned) else compact_text(playbook.homepage_strategy, 280),
        ir_strategy="" if not safe_reflection_text(playbook.ir_strategy, banned) else compact_text(playbook.ir_strategy, 280),
        transcript_strategy="" if not safe_reflection_text(playbook.transcript_strategy, banned) else compact_text(playbook.transcript_strategy, 280),
        avoid_strategy="" if not safe_reflection_text(playbook.avoid_strategy, banned) else compact_text(playbook.avoid_strategy, 280),
        confidence=playbook.confidence,
        evidence=[
            compact_text(value, 120)
            for value in playbook.evidence[:8]
            if safe_reflection_text(value, banned)
        ],
    )


def guidance_from_playbook(playbook: CompanyPlaybook, *, ticker: str) -> PromptGuidance:
    if not playbook_has_content(playbook):
        return PromptGuidance()
    issuer = playbook.issuer_name or ticker
    brands = f" Brands: {', '.join(playbook.brand_names[:4])}." if playbook.brand_names else ""
    return PromptGuidance(
        run_objective=f"Find the latest official quarterly earnings-call transcript for {issuer} ({ticker}).",
        strategy=playbook.planner_prompt,
        priority_terms=[
            value
            for value in [
                *(f"issuer: {issuer}" for _ in [0] if issuer),
                *[f"brand: {brand}" for brand in playbook.brand_names[:4]],
                "official issuer homepage",
                "official investor relations",
            ]
            if value
        ][:10],
        avoid_terms=playbook.avoid_hosts[:6],
        homepage_guidance=f"{playbook.homepage_strategy}{brands}".strip(),
        search_guidance=playbook.ir_strategy,
        navigation_guidance=playbook.ir_strategy,
        transcript_guidance=playbook.transcript_strategy,
        agent_guidance={
            "PromptPlannerAgent": playbook.planner_prompt,
            "HomepagePredictionAgent": playbook.homepage_strategy,
            "SearchQueryPlannerAgent": playbook.ir_strategy,
            "SearchCandidateRankingAgent": playbook.ir_strategy,
            "CrawlNavigatorAgent": playbook.ir_strategy,
        },
        risk_notes=[playbook.avoid_strategy] if playbook.avoid_strategy else [],
    )


def playbook_has_content(playbook: CompanyPlaybook) -> bool:
    return bool(
        playbook.issuer_name
        or playbook.brand_names
        or playbook.official_homepage_candidates
        or playbook.preferred_ir_urls
        or playbook.planner_prompt
    )


def sanitize_crawl_reflection(reflection: CrawlReflection) -> CrawlReflection:
    banned = ("ignore robots", "disable robots", "fail open", "third-party", "quartr", "seeking alpha")
    return CrawlReflection(
        preferred_urls=[url for url in reflection.preferred_urls[:8] if safe_reflection_text(url, banned)],
        preferred_terms=[compact_text(term, 80) for term in reflection.preferred_terms[:10] if safe_reflection_text(term, banned)],
        avoid_urls=[url for url in reflection.avoid_urls[:12] if safe_reflection_text(url, banned)],
        avoid_terms=[compact_text(term, 80) for term in reflection.avoid_terms[:12] if safe_reflection_text(term, banned)],
        prompt_guidance=sanitize_prompt_guidance(reflection.prompt_guidance),
        reason=compact_text(reflection.reason, 240),
    )


def safe_reflection_text(value: str, banned: tuple[str, ...]) -> bool:
    lowered = value.lower()
    placeholders = ("short guidance", "short reason", "term", "ir-related")
    return bool(value.strip()) and lowered not in placeholders and not any(token in lowered for token in banned)


def safe_prompt_guidance_term(value: str, banned: tuple[str, ...], banned_exact_terms: set[str]) -> bool:
    lowered = value.strip().lower()
    return safe_reflection_text(value, banned) and lowered not in banned_exact_terms


def prompt_guidance_banned_exact_terms(company: Company | None) -> set[str]:
    if not company:
        return set()
    terms = {company.symbol.lower()}
    if company.name:
        terms.add(company.name.lower())
    display_name = company_display_name(company)
    if display_name:
        terms.add(display_name.lower())
    return terms


def guidance_from_memory(memory: CompanyMemory) -> PromptGuidance:
    priority_terms: list[str] = []
    avoid_terms: list[str] = []
    risk_notes: list[str] = []
    navigation_memory = memory.navigation_memory

    if navigation_memory.preferred_hosts:
        append_unique_text(priority_terms, "preferred host from company memory")
    if navigation_memory.known_ir_home_urls:
        append_unique_text(priority_terms, "known IR home URL")
    if navigation_memory.known_event_listing_urls:
        append_unique_text(priority_terms, "known event listing URL")
    for url in navigation_memory.known_event_listing_urls:
        lowered = url.lower()
        if "quarterly-results" in lowered:
            append_unique_text(priority_terms, "quarterly results detail pages")
        if "transcript" in lowered:
            append_unique_text(priority_terms, "transcript links")
    if navigation_memory.low_value_hosts:
        append_unique_text(avoid_terms, "previously low-value host")
    for term in navigation_memory.low_value_path_terms:
        append_unique_text(avoid_terms, term)
    if navigation_memory.robots_blocked_hosts:
        append_unique_text(risk_notes, "Do not fetch documents from hosts whose robots.txt could not be verified.")

    for url in [*memory.successful_transcript_urls, *navigation_memory.known_transcript_urls]:
        lowered = url.lower()
        append_unique_text(priority_terms, "successful transcript host")
        append_unique_text(priority_terms, "same host as prior successful transcript")
        for token in ("event-details", "earnings-call", "transcript", "investor/events"):
            if token in lowered:
                append_unique_text(priority_terms, token)

    for url in [*memory.rejected_urls, *memory.known_ir_urls]:
        lowered = url.lower()
        if "blog." in lowered or "blog.google" in lowered:
            append_unique_text(avoid_terms, "blog")
            append_unique_text(avoid_terms, "blog.google")
        if "youtube.com" in lowered or "youtu.be" in lowered:
            append_unique_text(avoid_terms, "youtube")
        if "webcast" in lowered:
            append_unique_text(avoid_terms, "webcast-only")
        if "presentation" in lowered:
            append_unique_text(avoid_terms, "presentation")
        if "robots_unavailable" in lowered:
            append_unique_text(risk_notes, "Do not fetch transcript documents when robots.txt cannot be verified.")

    if memory.successful_transcript_urls:
        append_unique_text(priority_terms, "official investor event-detail pages")
        append_unique_text(priority_terms, "latest earnings-call transcript")

    return PromptGuidance(
        priority_terms=priority_terms[:10],
        avoid_terms=avoid_terms[:10],
        navigation_guidance=(
            "Prefer broad official investor home pages first, then official earnings/events "
            "or event-detail pages. Avoid blog/news or webcast-only paths when official IR "
            "event pages exist."
        ),
        transcript_guidance=(
            "Prefer pages or documents with transcript wording, speaker turns, Q&A, "
            "or earnings-call transcript text."
        ),
        risk_notes=risk_notes[:5],
    )


def sanitize_search_queries(queries: list[str], *, limit: int, ticker: str | None = None) -> list[str]:
    safe: list[str] = []
    banned = (
        "seeking alpha",
        "quartr",
        "marketbeat",
        "stockanalysis",
        "yahoo finance",
        "motley fool",
        "sec.gov",
    )
    for query in queries:
        cleaned = compact_text(" ".join(str(query).split()), 120)
        cleaned = re.sub(r"\bsite:\S+", "", cleaned, flags=re.IGNORECASE)
        cleaned = compact_text(" ".join(cleaned.split()), 120)
        if ticker and ticker.lower() not in cleaned.lower():
            cleaned = compact_text(f"{ticker} {cleaned}", 120)
        lowered = cleaned.lower()
        if not cleaned or any(token in lowered for token in banned):
            continue
        append_unique_text(safe, cleaned)
        if len(safe) >= limit:
            break
    return safe


def merge_prompt_guidance(primary: PromptGuidance, memory_guidance: PromptGuidance) -> PromptGuidance:
    return PromptGuidance(
        run_objective=compact_text(primary.run_objective or memory_guidance.run_objective, 220),
        strategy=compact_text(primary.strategy or memory_guidance.strategy, 360),
        priority_terms=merge_unique(primary.priority_terms, memory_guidance.priority_terms, limit=10),
        avoid_terms=merge_unique(primary.avoid_terms, memory_guidance.avoid_terms, limit=10),
        homepage_guidance=compact_text(
            primary.homepage_guidance or memory_guidance.homepage_guidance,
            280,
        ),
        search_guidance=compact_text(
            primary.search_guidance or memory_guidance.search_guidance,
            280,
        ),
        navigation_guidance=compact_text(
            primary.navigation_guidance or memory_guidance.navigation_guidance,
            280,
        ),
        transcript_guidance=compact_text(
            primary.transcript_guidance or memory_guidance.transcript_guidance,
            280,
        ),
        agent_guidance=merge_agent_guidance(primary.agent_guidance, memory_guidance.agent_guidance),
        risk_notes=merge_unique(primary.risk_notes, memory_guidance.risk_notes, limit=5),
    )


def merge_agent_guidance(primary: dict[str, str], secondary: dict[str, str]) -> dict[str, str]:
    merged = dict(secondary)
    merged.update(primary)
    return {
        compact_text(name, 80): compact_text(text, 240)
        for name, text in list(merged.items())[:12]
        if name and text
    }


def merge_unique(first: list[str], second: list[str], *, limit: int) -> list[str]:
    values: list[str] = []
    for value in [*first, *second]:
        append_unique_text(values, value)
    return values[:limit]


def append_unique_text(values: list[str], value: str) -> None:
    cleaned = compact_text(value, 80)
    if cleaned and cleaned not in values:
        values.append(cleaned)
