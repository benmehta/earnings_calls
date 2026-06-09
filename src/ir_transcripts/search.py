from __future__ import annotations

from duckduckgo_search import DDGS

from .models import Company


def find_ir_candidates(company: Company, max_results: int = 6) -> list[str]:
    """Find likely investor-relations pages without needing a paid search API."""
    query = f"{company.name} {company.symbol} investor relations earnings transcripts"
    candidates: list[str] = deterministic_ir_candidates(company)

    try:
        with DDGS() as ddgs:
            for result in ddgs.text(query, max_results=max_results):
                url = result.get("href") or result.get("url")
                if not url:
                    continue
                if any(token in url.lower() for token in ("investor", "/ir", "shareholder", "earnings")):
                    candidates.append(url)
    except Exception:
        pass

    return dedupe(candidates)


def deterministic_ir_candidates(company: Company) -> list[str]:
    slug = company_domain_slug(company.name)
    ticker = company.symbol.lower()
    domains = [
        f"{slug}.com",
        f"{ticker}.com",
    ]
    candidates: list[str] = []
    for domain in domains:
        candidates.extend(
            [
                f"https://investors.{domain}",
                f"https://investor.{domain}",
                f"https://ir.{domain}",
                f"https://www.{domain}/investors",
                f"https://www.{domain}/investor-relations",
            ]
        )
    return dedupe(candidates)


def company_domain_slug(name: str) -> str:
    words = []
    stop = {
        "inc",
        "inc.",
        "corp",
        "corp.",
        "corporation",
        "company",
        "co",
        "co.",
        "plc",
        "ltd",
        "limited",
        "class",
    }
    for raw in name.lower().replace("&", "and").split():
        cleaned = "".join(char for char in raw if char.isalnum())
        if cleaned and cleaned not in stop and not cleaned.isdigit():
            words.append(cleaned)
    return "".join(words[:3]) or name.lower().replace(" ", "")


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
