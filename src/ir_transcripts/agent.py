from __future__ import annotations

import json
import re
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from .identity import company_display_name
from .models import (
    CandidateLink,
    Company,
    CompanyMemory,
    CompanyNavigationMemory,
    CrawlReflection,
    HomepagePrediction,
    HomepageValidationDecision,
    IRDiscoveryCandidate,
    IRDiscoveryDecision,
    NavigationDecision,
    NavigationValidationResult,
    PageDecision,
    PageDecisionDraft,
    PromptGuidance,
)


NavigationAgentKind = Literal["homepage", "ir_section", "event_listing", "transcript_link"]
LINKS_JSON_CHAR_BUDGET = 800


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


class IRPageAgent:
    """Small LangChain/Ollama page classifier used by the crawler."""

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
                        "Classify one IR page. Return JSON only. "
                        "High precision: transcript only for real earnings-call transcript text.",
                    ),
                    (
                        "human",
                        "{company_name} ({ticker})\n{url}\nTitle: {title}\n"
                        "Run guidance:\n{guidance}\n\n"
                        "Text:\n{text}\n\n"
                        "Links JSON. Put useful URL strings in useful_urls:\n"
                        "{links_json}\n\n"
                        "Return exactly:\n"
                        "{{\"page_type\":\"ir_index\",\"confidence\":0.8,\"useful_urls\":[\"https://example.com/events\"],\"reason\":\"short reason\"}}\n\n"
                        "page_type: transcript, earnings_event, press_release, filings, ir_index, not_relevant.",
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
                "guidance": guidance_for_agent(getattr(self, "guidance", None), "page"),
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

    def __init__(self, model: str, base_url: str | None = None) -> None:
        self.parser = PydanticOutputParser(pydantic_object=IRDiscoveryDecision)
        self.chain = (
            ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "Select official company homepage or investor-relations seed URLs from search results. "
                        "Choose only provided URLs. Prefer the company homepage, official investor-relations "
                        "home, official financial/quarterly results pages, or official vendor-hosted IR pages. "
                        "Reject similarly named but different companies when the title, host, or snippet does "
                        "not match the requested company/ticker. Avoid third-party finance, transcript archive, "
                        "news, SEC, careers, support, and store pages. Return only official seeds.",
                    ),
                    (
                        "human",
                        "Company: {company_name} ({ticker})\n"
                        "Run guidance:\n{guidance}\n\n"
                        "Memory summary:\n{memory_summary}\n\n"
                        "Candidates as JSON:\n{candidates_json}\n\n"
                        "Return up to {limit} URLs in ranked order.\n"
                        "{format_instructions}",
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
                "guidance": guidance_for_agent(guidance, "navigation"),
                "memory_summary": discovery_memory_summary(navigation_memory),
                "candidates_json": json.dumps(compact, ensure_ascii=True),
                "limit": limit,
                "format_instructions": self.parser.get_format_instructions(),
            }
        )


class HomepagePredictionAgent:
    """Predicts official company homepages from a ticker before discovery."""

    def __init__(self, model: str, base_url: str | None = None) -> None:
        self.chain = (
            ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "Predict official public-company homepage URLs from a ticker. "
                        "Return JSON only. Predict homepages only, never investor-relations, "
                        "transcript, SEC, finance portal, news, or third-party URLs. "
                        "If uncertain, return low confidence and few or no URLs.",
                    ),
                    (
                        "human",
                        "Ticker: {ticker}\n"
                        "Memory-backed company name hint: {name_hint}\n\n"
                        "Return exactly:\n"
                        "{{\"homepage_urls\":[\"https://www.example.com\"],"
                        "\"confidence\":0.8,\"reason\":\"short reason\"}}",
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
                        "Validate whether a fetched page is an official company homepage. "
                        "Use only page evidence and provided links. Return JSON only. "
                        "Accept only official home/about/corporate pages. Reject IR pages, "
                        "finance portals, news sites, SEC pages, transcript archives, and unofficial pages. "
                        "If official, extract the concise official company or brand name from page evidence "
                        "(for example Amazon, Microsoft, NVIDIA), not the ticker and not a legal suffix unless needed.",
                    ),
                    (
                        "human",
                        "Ticker: {ticker}\n"
                        "Memory-backed name hint: {name_hint}\n"
                        "URL: {url}\n"
                        "Title: {title}\n"
                        "Text:\n{text}\n\n"
                        "Links JSON:\n{links_json}\n\n"
                        "Return exactly:\n"
                        "{{\"is_official\":true,\"confidence\":0.8,"
                        "\"official_company_name\":\"Example\","
                        "\"linked_ir_urls\":[\"https://example.com/investors\"],"
                        "\"reason\":\"short evidence-based reason\"}}",
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


class PromptPlannerAgent:
    """Produces non-authoritative prompt guidance from company memory."""

    def __init__(self, model: str, base_url: str | None = None, *, text_chars: int = 1200) -> None:
        self.text_chars = text_chars
        self.chain = (
            ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "Create brief advisory guidance for official investor-relations transcript crawler agents. "
                        "Use only the supplied company memory. Do not infer or invent URLs, hosts, company names, "
                        "permissions, or facts not present in memory. Guidance is advisory only: it must not override "
                        "robots.txt, strict transcript detection, official-source requirements, or homepage validation. "
                        "Write reusable navigation and transcript-selection patterns, not broad search queries. "
                        "Priority terms should name concrete page/link patterns such as 'quarterly results', "
                        "'event-detail pages', 'earnings call transcript', or 'speaker turns'. "
                        "Avoid terms should name concrete low-value patterns such as 'blog', 'webcast-only', "
                        "'presentation', 'shareholders meeting', or non-English path variants. "
                        "Do not output the ticker, company name, generic placeholders, or plain words like "
                        "'IR-related' as priority_terms or avoid_terms. "
                        "Do not recommend third-party transcript sources, SEC/finance portals, news sites, "
                        "or any action that ignores, disables, bypasses, or fails open on robots.txt. "
                        "Return JSON only.",
                    ),
                    (
                        "human",
                        "Company: {company}\n"
                        "Memory JSON:\n{memory_json}\n\n"
                        "Return empty arrays/strings when memory is too thin for specific guidance. "
                        "Do not include URLs unless they already appear in memory, and prefer describing URL patterns "
                        "over copying full URLs. "
                        "Return exactly:\n"
                        "{{\"priority_terms\":[\"term\"],\"avoid_terms\":[\"term\"],"
                        "\"navigation_guidance\":\"short guidance\","
                        "\"transcript_guidance\":\"short guidance\","
                        "\"risk_notes\":[\"short note\"]}}",
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
        model_guidance = sanitize_prompt_guidance(
            PromptGuidance.model_validate(extract_json_object(message.content)),
            company=company,
        )
        return merge_prompt_guidance(model_guidance, guidance_from_memory(memory))


class CrawlReflectionAgent:
    """Learns advisory navigation guidance from a failed supervised attempt."""

    def __init__(self, model: str, base_url: str | None = None, *, text_chars: int = 1800) -> None:
        self.text_chars = text_chars
        self.chain = (
            ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "Analyze a failed official investor-relations transcript crawl. "
                        "Return company-specific advisory memory updates for the next attempt. "
                        "Use only evidence in the crawl summary. Do not invent URLs. "
                        "Do not grant permissions, ignore robots.txt, fail open, or recommend third-party transcript sources. "
                        "Return JSON only.",
                    ),
                    (
                        "human",
                        "Company: {company}\n"
                        "Crawl evidence JSON:\n{evidence_json}\n\n"
                        "Return exactly:\n"
                        "{{\"preferred_urls\":[\"https://example.com/earnings\"],"
                        "\"preferred_terms\":[\"quarterly results detail pages\"],"
                        "\"avoid_urls\":[\"https://example.com/shareholders\"],"
                        "\"avoid_terms\":[\"shareholders meeting\"],"
                        "\"prompt_guidance\":{{\"priority_terms\":[\"term\"],\"avoid_terms\":[\"term\"],"
                        "\"navigation_guidance\":\"short guidance\","
                        "\"transcript_guidance\":\"short guidance\","
                        "\"risk_notes\":[\"short note\"]}},"
                        "\"reason\":\"short reason\"}}",
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

    HUMAN_PROMPT = (
        "{company_name} ({ticker})\n{url}\nTitle: {title}\n"
        "Run guidance:\n{guidance}\n\n"
        "Text:\n{text}\n\n"
        "Links JSON:\n{links_json}\n\n"
        "Return exactly: "
        "{{\"chosen_urls\":[\"https://example.com/investor\"],\"confidence\":0.8,"
        "\"reason\":\"short reason\",\"stop_reason\":null}}"
    )

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
                    guidance_for_agent(getattr(self, "guidance", None), "navigation"),
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

    SYSTEM_PROMPT = (
        "Validate a specialist IR navigation choice. Hard policy already removed "
        "invented URLs and disallowed hosts. Decide if the remaining URLs fit the "
        "specialist task. Return JSON only."
    )
    HUMAN_PROMPT = (
        "Specialist: {kind}\n"
        "Company: {company_name} ({ticker})\n"
        "Current URL: {url}\n"
        "Title: {title}\n"
        "Guidance: {guidance}\n"
        "Chosen URLs: {chosen_urls_json}\n"
        "Candidate links JSON:\n{links_json}\n\n"
        "Return exactly: "
        "{{\"is_valid\":true,\"accepted_urls\":[\"https://example.com\"],"
        "\"rejected_urls\":[],\"reason\":\"short reason\","
        "\"repair_guidance\":null}}"
    )

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
                "guidance": guidance_for_agent(getattr(self, "guidance", None), "navigation"),
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
    SYSTEM_PROMPT = (
        "Pick links from an official company homepage toward investor relations. "
        "Use only provided URLs. Prefer Investors/IR, shareholders, financial reports, "
        "earnings, events, webcast, transcript. Footer links matter. Avoid careers, "
        "products, legal, privacy, blogs/news, third-party finance. Return JSON only."
    )

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
    SYSTEM_PROMPT = (
        "You are inside investor relations. Pick section links that move toward "
        "earnings-call transcript materials. Use only provided URLs. Prefer Earnings, "
        "Events, Quarterly Results, Financial Reports, Presentations/Webcasts. Avoid "
        "governance, stock quote, SEC-only, annual meeting, alerts, privacy/legal, "
        "careers, blogs/news unless no better IR links exist. Return JSON only."
    )

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
    SYSTEM_PROMPT = (
        "You are on an earnings/events listing. Pick the latest earnings-call event "
        "or transcript-related event. Use only provided URLs. Prefer current-year or "
        "newest quarter earnings call links, event-details pages, and links mentioning "
        "transcript. Avoid annual meetings, conferences, SEC filings, generic news, "
        "blog posts, YouTube/webcast-only links unless no event page exists. Return JSON only."
    )

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
    SYSTEM_PROMPT = (
        "You are on a specific earnings-call event page. Pick transcript material links. "
        "Use only provided URLs. Prefer links or documents containing transcript, PDF, "
        "DOCX, Q&A, prepared remarks, or earnings-call transcript. Avoid webcast-only, "
        "YouTube, calendar, email alerts, presentations, press releases, blogs/news, "
        "privacy/legal. Return JSON only."
    )

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


def guidance_for_agent(guidance: PromptGuidance | None, mode: Literal["navigation", "page"]) -> str:
    if not guidance:
        return "None."
    parts = []
    if guidance.priority_terms:
        parts.append(f"Prefer: {', '.join(guidance.priority_terms[:8])}.")
    if guidance.avoid_terms:
        parts.append(f"Avoid: {', '.join(guidance.avoid_terms[:8])}.")
    text = guidance.transcript_guidance if mode == "page" else guidance.navigation_guidance
    if text:
        parts.append(text)
    if guidance.risk_notes:
        parts.append(f"Risks: {'; '.join(guidance.risk_notes[:3])}.")
    return compact_text(" ".join(parts), 500) or "None."


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
        navigation_guidance="" if not safe_reflection_text(guidance.navigation_guidance, banned) else compact_text(guidance.navigation_guidance, 280),
        transcript_guidance="" if not safe_reflection_text(guidance.transcript_guidance, banned) else compact_text(guidance.transcript_guidance, 280),
        risk_notes=risk_notes,
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


def merge_prompt_guidance(primary: PromptGuidance, memory_guidance: PromptGuidance) -> PromptGuidance:
    return PromptGuidance(
        priority_terms=merge_unique(primary.priority_terms, memory_guidance.priority_terms, limit=10),
        avoid_terms=merge_unique(primary.avoid_terms, memory_guidance.avoid_terms, limit=10),
        navigation_guidance=compact_text(
            memory_guidance.navigation_guidance or primary.navigation_guidance,
            280,
        ),
        transcript_guidance=compact_text(
            memory_guidance.transcript_guidance or primary.transcript_guidance,
            280,
        ),
        risk_notes=merge_unique(primary.risk_notes, memory_guidance.risk_notes, limit=5),
    )


def merge_unique(first: list[str], second: list[str], *, limit: int) -> list[str]:
    values: list[str] = []
    for value in [*first, *second]:
        append_unique_text(values, value)
    return values[:limit]


def append_unique_text(values: list[str], value: str) -> None:
    cleaned = compact_text(value, 80)
    if cleaned and cleaned not in values:
        values.append(cleaned)
