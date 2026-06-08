from __future__ import annotations

import json

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from .models import CandidateLink, PageDecision


def build_llm(model: str, temperature: float = 0.0) -> ChatOllama:
    return ChatOllama(model=model, temperature=temperature)


class IRPageAgent:
    """Small LangChain/Ollama page classifier used by the crawler."""

    def __init__(self, model: str) -> None:
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
            | build_llm(model)
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

