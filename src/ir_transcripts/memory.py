from __future__ import annotations

from pathlib import Path

from .models import Company, CompanyMemory, CrawlResult, FailureAnalysis
from .urls import host


MEMORY_FILENAME = "_company_memory.json"


def memory_path(out_dir: Path, company: Company) -> Path:
    return out_dir / company.symbol / MEMORY_FILENAME


def load_company_memory(out_dir: Path, company: Company) -> CompanyMemory:
    path = memory_path(out_dir, company)
    if path.exists():
        return CompanyMemory.model_validate_json(path.read_text(encoding="utf-8"))
    return CompanyMemory(company=company)


def save_company_memory(out_dir: Path, memory: CompanyMemory) -> Path:
    path = memory_path(out_dir, memory.company)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(memory.model_dump_json(indent=2), encoding="utf-8")
    return path


def remember_crawl_result(memory: CompanyMemory, result: CrawlResult, analysis: FailureAnalysis) -> CompanyMemory:
    for url in [result.ir_url, *(candidate.url for candidate in result.candidates)]:
        if url:
            memory.known_ir_urls = append_unique(memory.known_ir_urls, url)
            url_host = host(url)
            if url_host:
                memory.official_hosts = append_unique(memory.official_hosts, url_host)

    for failure in result.failures:
        memory.rejected_urls = append_unique(memory.rejected_urls, failure.url)

    for record in result.transcripts:
        memory.successful_transcript_urls = append_unique(memory.successful_transcript_urls, str(record.source_url))

    memory.failure_summaries.append(analysis)
    for recommendation in analysis.manual_recommendations:
        memory.recommended_manual_actions = append_unique(memory.recommended_manual_actions, recommendation)
    return memory


def append_unique(values: list[str], value: str) -> list[str]:
    if value not in values:
        values.append(value)
    return values
