from __future__ import annotations

import json
import re

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from .models import CandidateLink, IRDiscoveryCandidate, IRDiscoveryDecision, NavigationDecision, PageDecision, PageDecisionDraft


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

    def __init__(self, model: str, base_url: str | None = None) -> None:
        self.chain = (
            ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You classify investor-relations pages for a crawler. "
                        "Prefer high precision. Do not claim a page is a transcript "
                        "unless page text strongly indicates an earnings call transcript. "
                        "Return only one JSON object. Do not return a JSON schema.",
                    ),
                    (
                        "human",
                        "Company: {company_name} ({ticker})\n"
                        "URL: {url}\n"
                        "Page title: {title}\n"
                        "Visible text sample:\n{text}\n\n"
                        "Candidate links as JSON. If a link is useful, return only its URL string in useful_urls:\n"
                        "{links_json}\n\n"
                        "Return exactly this JSON shape:\n"
                        "{{\"page_type\":\"ir_index\",\"confidence\":0.8,\"useful_urls\":[\"https://example.com/events\"],\"reason\":\"short reason\"}}\n\n"
                        "Allowed page_type values: transcript, earnings_event, press_release, filings, ir_index, not_relevant.",
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
            {"url": link.url, "label": link.label, "source_url": link.source_url}
            for link in links[:30]
        ]
        message = self.chain.invoke(
            {
                "company_name": company_name,
                "ticker": ticker,
                "url": url,
                "title": title,
                "text": text[:3000],
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


class IRNavigationAgent:
    """Local Ollama link selector for navigation-first IR discovery."""

    HOMEPAGE_SYSTEM_PROMPT = (
        "You are guiding a polite crawler from an official company homepage toward "
        "the company's investor-relations materials.\n\n"
        "Choose only provided candidate links. Do not invent URLs.\n\n"
        "Prefer links labeled Investors, Investor Relations, Shareholders, Financial Info, "
        "Financial Reports, Quarterly Results, Earnings, Events, Presentations, Webcast, "
        "Transcript, or Results.\n\n"
        "Footer links are important. Many companies put Investors or Investor Relations "
        "in the footer rather than the main navigation.\n\n"
        "Avoid careers, products, support, privacy, legal, store, unrelated blogs/news, "
        "marketing pages, and third-party finance/transcript websites.\n\n"
        "Return only one JSON object."
    )
    IR_SYSTEM_PROMPT = (
        "You are already inside an investor-relations site. Choose links most likely to "
        "lead to earnings call transcript materials or earnings materials.\n\n"
        "Choose only provided candidate links. Do not invent URLs.\n\n"
        "Prefer Financial Info, Financial Reports, Quarterly Results, Earnings Releases, "
        "Events, Presentations, Webcast, Transcript, Results, and News Releases when they "
        "are part of investor relations.\n\n"
        "Avoid governance, stock quote, email alerts, SEC-only pages, annual meeting, "
        "privacy/legal pages, careers, product pages, and unrelated company news unless "
        "there are no better investor-relations materials links.\n\n"
        "Return only one JSON object."
    )
    HUMAN_PROMPT = (
        "Company: {company_name} ({ticker})\n"
        "Current URL: {url}\n"
        "Page title: {title}\n"
        "Visible text sample:\n{text}\n\n"
        "Candidate links as JSON:\n{links_json}\n\n"
        "Return exactly:\n"
        "{{\n"
        "  \"chosen_urls\": [\"https://example.com/investor\"],\n"
        "  \"confidence\": 0.8,\n"
        "  \"reason\": \"short reason\",\n"
        "  \"stop_reason\": null\n"
        "}}"
    )

    def __init__(self, model: str, base_url: str | None = None) -> None:
        self.homepage_chain = self._chain(self.HOMEPAGE_SYSTEM_PROMPT, model, base_url)
        self.ir_chain = self._chain(self.IR_SYSTEM_PROMPT, model, base_url)

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
        compact_links = [
            {
                "url": link.url,
                "label": link.label,
                "source_url": link.source_url,
                "context": link.reason,
            }
            for link in links[:40]
        ]
        chain = self.ir_chain if page_context == "ir" else self.homepage_chain
        message = chain.invoke(
            {
                "company_name": company_name,
                "ticker": ticker,
                "url": url,
                "title": title,
                "text": text[:2500],
                "links_json": json.dumps(compact_links, ensure_ascii=True),
            }
        )
        draft = NavigationDecision.model_validate(extract_json_object(message.content))
        urls_by_candidate = {link.url for link in links}
        draft.chosen_urls = [url for url in draft.chosen_urls if url in urls_by_candidate]
        return draft

    def _chain(self, system_prompt: str, model: str, base_url: str | None):
        return (
            ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    ("human", self.HUMAN_PROMPT),
                ]
            )
            | build_llm(model, base_url=base_url, json_mode=True)
        )


def extract_json_object(content: str) -> dict:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", content, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in model output: {content[:200]}")
    return json.loads(match.group(0))
