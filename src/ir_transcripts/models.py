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


class TranscriptRecord(BaseModel):
    company: Company
    source_url: HttpUrl | str
    fiscal_period: str | None = None
    call_date: date | None = None
    title: str
    text: str
    raw_path: Path | None = None
    metadata: dict[str, str] = Field(default_factory=dict)


class CrawlResult(BaseModel):
    company: Company
    ir_url: str | None = None
    transcripts: list[TranscriptRecord] = Field(default_factory=list)
    skipped_reason: str | None = None
    visited_count: int = 0

