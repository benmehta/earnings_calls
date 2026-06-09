from __future__ import annotations

import json

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from .models import CandidateLink, IRDiscoveryCandidate, IRDiscoveryDecision, PageDecision


def build_llm(model: str, temperature: float = 0.0, base_url: str | None = None) -> ChatOllama:
    kwargs = {"model": model, "temperature": temperature}
    if base_url:
        kwargs["base_url"] = base_url
    return ChatOllama(**kwargs)


class IRPageAgent:
    """Small LangChain/Ollama page classifier used by the crawler."""

    def __init__(self, model: str, base_url: str | None = None) -> None:
        self.parser = PydanticOutputParser(pydantic_object=PageDecision)
        self.chain = (
            ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You classify investor-relations pages for a crawler. "
                        "Prefer high precision. Do not claim a page is a transcript "
                        "unless the page text or link labels strongly indicate an "
                        "earnings call transcript or webcast transcript.",
                    ),
                    (
                        "human",
                        "Company: {company_name} ({ticker})\n"
                        "URL: {url}\n"
                        "Page title: {title}\n"
                        "Visible text sample:\n{text}\n\n"
                        "Links as JSON:\n{links_json}\n\n"
                        "{format_instructions}",
                    ),
                ]
            )
            | build_llm(model, base_url=base_url)
            | self.parser
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
            for link in links[:80]
        ]
        return self.chain.invoke(
            {
                "company_name": company_name,
                "ticker": ticker,
                "url": url,
                "title": title,
                "text": text[:9000],
                "links_json": json.dumps(compact_links, ensure_ascii=True),
                "format_instructions": self.parser.get_format_instructions(),
            }
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
