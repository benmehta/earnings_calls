from __future__ import annotations

import json
import re
from typing import Literal

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from .models import CandidateLink, IRDiscoveryCandidate, IRDiscoveryDecision, NavigationDecision, PageDecision, PageDecisionDraft


NavigationAgentKind = Literal["homepage", "ir_section", "event_listing", "transcript_link"]


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
    ) -> None:
        self.max_links = max_links
        self.text_chars = text_chars
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
        compact_links = [
            {"url": link.url, "label": compact_text(link.label, 80), "context": compact_text(link.reason, 80)}
            for link in links[: self.max_links]
        ]
        message = self.chain.invoke(
            {
                "company_name": company_name,
                "ticker": ticker,
                "url": url,
                "title": title,
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
                        "Select likely official company investor-relations seed URLs from search results. "
                        "Prefer official company-hosted IR pages. Accept vendor-hosted IR pages only when "
                        "the title/snippet strongly indicates they are official for the company. Avoid "
                        "third-party finance, transcript, news, SEC, careers, support, and store pages.",
                    ),
                    (
                        "human",
                        "Company: {company_name} ({ticker})\n"
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
                "candidates_json": json.dumps(compact, ensure_ascii=True),
                "limit": limit,
                "format_instructions": self.parser.get_format_instructions(),
            }
        )


class LinkSelectionAgent:
    """Small specialist agent that chooses links for one navigation task."""

    HUMAN_PROMPT = (
        "{company_name} ({ticker})\n{url}\nTitle: {title}\n"
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
    ) -> None:
        self.max_links = max_links
        self.text_chars = text_chars
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
    ) -> NavigationDecision:
        compact_links = [
            {
                "url": link.url,
                "label": compact_text(link.label, 80),
                "context": compact_text(link.reason, 80),
            }
            for link in links[: self.max_links]
        ]
        message = self.chain.invoke(
            {
                "company_name": company_name,
                "ticker": ticker,
                "url": url,
                "title": title,
                "text": compact_text(text, self.text_chars),
                "links_json": json.dumps(compact_links, ensure_ascii=True),
            }
        )
        draft = NavigationDecision.model_validate(extract_json_object(message.content))
        urls_by_candidate = {link.url for link in links}
        draft.chosen_urls = [url for url in draft.chosen_urls if url in urls_by_candidate]
        return draft


class HomepageNavAgent(LinkSelectionAgent):
    SYSTEM_PROMPT = (
        "Pick links from an official company homepage toward investor relations. "
        "Use only provided URLs. Prefer Investors/IR, shareholders, financial reports, "
        "earnings, events, webcast, transcript. Footer links matter. Avoid careers, "
        "products, legal, privacy, blogs/news, third-party finance. Return JSON only."
    )

    def __init__(self, model: str, base_url: str | None = None, *, max_links: int = 12, text_chars: int = 800) -> None:
        super().__init__(model=model, system_prompt=self.SYSTEM_PROMPT, base_url=base_url, max_links=max_links, text_chars=text_chars)


class IRSectionAgent(LinkSelectionAgent):
    SYSTEM_PROMPT = (
        "You are inside investor relations. Pick section links that move toward "
        "earnings-call transcript materials. Use only provided URLs. Prefer Earnings, "
        "Events, Quarterly Results, Financial Reports, Presentations/Webcasts. Avoid "
        "governance, stock quote, SEC-only, annual meeting, alerts, privacy/legal, "
        "careers, blogs/news unless no better IR links exist. Return JSON only."
    )

    def __init__(self, model: str, base_url: str | None = None, *, max_links: int = 12, text_chars: int = 800) -> None:
        super().__init__(model=model, system_prompt=self.SYSTEM_PROMPT, base_url=base_url, max_links=max_links, text_chars=text_chars)


class EventListingAgent(LinkSelectionAgent):
    SYSTEM_PROMPT = (
        "You are on an earnings/events listing. Pick the latest earnings-call event "
        "or transcript-related event. Use only provided URLs. Prefer current-year or "
        "newest quarter earnings call links, event-details pages, and links mentioning "
        "transcript. Avoid annual meetings, conferences, SEC filings, generic news, "
        "blog posts, YouTube/webcast-only links unless no event page exists. Return JSON only."
    )

    def __init__(self, model: str, base_url: str | None = None, *, max_links: int = 12, text_chars: int = 800) -> None:
        super().__init__(model=model, system_prompt=self.SYSTEM_PROMPT, base_url=base_url, max_links=max_links, text_chars=text_chars)


class TranscriptLinkAgent(LinkSelectionAgent):
    SYSTEM_PROMPT = (
        "You are on a specific earnings-call event page. Pick transcript material links. "
        "Use only provided URLs. Prefer links or documents containing transcript, PDF, "
        "DOCX, Q&A, prepared remarks, or earnings-call transcript. Avoid webcast-only, "
        "YouTube, calendar, email alerts, presentations, press releases, blogs/news, "
        "privacy/legal. Return JSON only."
    )

    def __init__(self, model: str, base_url: str | None = None, *, max_links: int = 12, text_chars: int = 800) -> None:
        super().__init__(model=model, system_prompt=self.SYSTEM_PROMPT, base_url=base_url, max_links=max_links, text_chars=text_chars)


class IRNavigationAgent:
    """Routes navigation decisions to focused local Ollama link-selection agents."""

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        *,
        max_links: int = 12,
        text_chars: int = 800,
    ) -> None:
        self.agents: dict[NavigationAgentKind, LinkSelectionAgent] = {
            "homepage": HomepageNavAgent(model, base_url=base_url, max_links=max_links, text_chars=text_chars),
            "ir_section": IRSectionAgent(model, base_url=base_url, max_links=max_links, text_chars=text_chars),
            "event_listing": EventListingAgent(model, base_url=base_url, max_links=max_links, text_chars=text_chars),
            "transcript_link": TranscriptLinkAgent(model, base_url=base_url, max_links=max_links, text_chars=text_chars),
        }

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
