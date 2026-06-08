from __future__ import annotations

from duckduckgo_search import DDGS

from .models import Company


def find_ir_candidates(company: Company, max_results: int = 6) -> list[str]:
    """Find likely investor-relations pages without needing a paid search API."""
    query = f"{company.name} {company.symbol} investor relations earnings transcripts"
    candidates: list[str] = []

    with DDGS() as ddgs:
        for result in ddgs.text(query, max_results=max_results):
            url = result.get("href") or result.get("url")
            if not url:
                continue
            if any(token in url.lower() for token in ("investor", "/ir", "shareholder", "earnings")):
                candidates.append(url)

    return dedupe(candidates)


def dedupe(urls: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for url in urls:
        normalized = url.rstrip("/")
        if normalized in seen:
            continue
        seen.add(normalized)
        result.append(url)
    return result

