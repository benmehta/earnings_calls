from __future__ import annotations

import re
from datetime import date

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel


class TranscriptMetadata(BaseModel):
    fiscal_period: str | None = None
    call_date: date | None = None


QUARTER_RE = re.compile(r"\b(Q[1-4]|first quarter|second quarter|third quarter|fourth quarter)\b[^.\n]{0,40}\b(20\d{2})\b", re.I)


def extract_metadata_heuristic(title: str, text: str) -> TranscriptMetadata:
    haystack = f"{title}\n{text[:3000]}"
    quarter = QUARTER_RE.search(haystack)
    fiscal_period = None
    if quarter:
        fiscal_period = f"{quarter.group(1).upper()} {quarter.group(2)}"
    return TranscriptMetadata(fiscal_period=fiscal_period)


class TranscriptMetadataAgent:
    def __init__(self, model: str, base_url: str | None = None) -> None:
        self.parser = PydanticOutputParser(pydantic_object=TranscriptMetadata)
        kwargs = {"model": model, "temperature": 0.0}
        if base_url:
            kwargs["base_url"] = base_url
        self.chain = (
            ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "Extract transcript metadata only when strongly evidenced. "
                        "Return null for uncertain fields.",
                    ),
                    (
                        "human",
                        "Title: {title}\nTranscript sample:\n{text}\n\n{format_instructions}",
                    ),
                ]
            )
            | ChatOllama(**kwargs)
            | self.parser
        )

    def extract(self, title: str, text: str) -> TranscriptMetadata:
        return self.chain.invoke(
            {
                "title": title,
                "text": text[:6000],
                "format_instructions": self.parser.get_format_instructions(),
            }
        )
