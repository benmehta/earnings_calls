from __future__ import annotations

from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS

from .agent import IRDiscoveryAgent
from .identity import company_display_name, resolve_company_identity_with_overrides
from .models import Company, CompanyNavigationMemory, IRDiscoveryCandidate, PromptGuidance
from .runtime import ProgressReporter, timeout_after
from .urls import normalize_url


CURATED_IR_URLS = {
    "AAPL": ["https://investor.apple.com/investor-relations/default.aspx"],
    "MSFT": ["https://www.microsoft.com/en-us/Investor"],
    "NVDA": ["https://investor.nvidia.com"],
}


def find_ir_candidates(
    company: Company,
    max_results: int = 6,
    *,
    include_guesses: bool = False,
    disable_official_homepage_overrides: bool = False,
    rerank_model: str | None = None,
    ollama_base_url: str | None = None,
    search_timeout_seconds: float = 30.0,
    llm_timeout_seconds: float = 45.0,
    prompt_guidance: PromptGuidance | None = None,
    navigation_memory: CompanyNavigationMemory | None = None,
    progress: ProgressReporter | None = None,
) -> list[str]:
    """Find likely investor-relations pages without needing a paid search API."""
    discovered = discover_ir_candidates(
        company,
        max_results=max_results,
        include_guesses=include_guesses,
        disable_official_homepage_overrides=disable_official_homepage_overrides,
        rerank_model=rerank_model,
        ollama_base_url=ollama_base_url,
        search_timeout_seconds=search_timeout_seconds,
        llm_timeout_seconds=llm_timeout_seconds,
        prompt_guidance=prompt_guidance,
        navigation_memory=navigation_memory,
        progress=progress,
    )
    return [candidate.url for candidate in discovered]


def discover_ir_candidates(
    company: Company,
    max_results: int = 10,
    *,
    include_guesses: bool = False,
    disable_official_homepage_overrides: bool = False,
    rerank_model: str | None = None,
    ollama_base_url: str | None = None,
    search_timeout_seconds: float = 30.0,
    llm_timeout_seconds: float = 45.0,
    prompt_guidance: PromptGuidance | None = None,
    navigation_memory: CompanyNavigationMemory | None = None,
    progress: ProgressReporter | None = None,
) -> list[IRDiscoveryCandidate]:
    company = resolve_company_identity_with_overrides(
        company,
        allow_homepage_overrides=not disable_official_homepage_overrides,
    )
    candidates: list[IRDiscoveryCandidate] = []
    progress = progress or ProgressReporter(enabled=False)

    for url in CURATED_IR_URLS.get(company.symbol.upper(), []):
        candidates.append(
            IRDiscoveryCandidate(
                url=url,
                title=f"{company_display_name(company)} investor relations",
                source="curated",
                score=100,
                reasons=["curated known IR URL"],
            )
        )

    progress.log(f"{company.symbol}: search discovery starting")
    candidates.extend(
        search_ir_candidates(
            company,
            max_results=max_results,
            timeout_seconds=search_timeout_seconds,
            progress=progress,
        )
    )
    progress.log(f"{company.symbol}: search discovery found {len(candidates)} candidate(s)")

    if include_guesses and not candidates:
        for url in deterministic_ir_candidates(company):
            candidates.append(
                score_ir_candidate(
                    url=url,
                    title="",
                    snippet="",
                    company=company,
                    source="deterministic",
                )
            )

    ranked = dedupe_candidates(sorted(candidates, key=lambda item: item.score, reverse=True))
    ranked = apply_discovery_memory(ranked, navigation_memory=navigation_memory)
    if rerank_model and ranked:
        ranked = rerank_ir_candidates(
            company,
            ranked,
            model=rerank_model,
            ollama_base_url=ollama_base_url,
            limit=max_results,
            timeout_seconds=llm_timeout_seconds,
            prompt_guidance=prompt_guidance,
            navigation_memory=navigation_memory,
            progress=progress,
        )
    return apply_discovery_memory(ranked, navigation_memory=navigation_memory)


def rerank_ir_candidates(
    company: Company,
    candidates: list[IRDiscoveryCandidate],
    *,
    model: str,
    ollama_base_url: str | None = None,
    limit: int = 10,
    timeout_seconds: float = 45.0,
    prompt_guidance: PromptGuidance | None = None,
    navigation_memory: CompanyNavigationMemory | None = None,
    progress: ProgressReporter | None = None,
) -> list[IRDiscoveryCandidate]:
    progress = progress or ProgressReporter(enabled=False)
    by_url = {normalize_url(candidate.url): candidate for candidate in candidates}
    try:
        progress.log(f"{company.symbol}: Ollama reranking {len(candidates)} discovery candidate(s)")
        with timeout_after(timeout_seconds, f"reranking {company.symbol} discovery candidates"):
            decision = IRDiscoveryAgent(model, base_url=ollama_base_url).rerank(
                company_name=company_display_name(company),
                ticker=company.symbol,
                candidates=candidates,
                limit=limit,
                guidance=prompt_guidance,
                navigation_memory=navigation_memory,
            )
    except Exception as exc:
        progress.log(f"{company.symbol}: Ollama rerank skipped ({type(exc).__name__}: {exc})")
        return candidates

    reranked: list[IRDiscoveryCandidate] = []
    seen: set[str] = set()
    for selection in decision.selections:
        normalized = normalize_url(selection.url)
        candidate = by_url.get(normalized)
        if not candidate or normalized in seen:
            continue
        candidate.score += int(selection.confidence * 25)
        candidate.reasons = [f"ollama rerank: {selection.reason}"] + candidate.reasons
        reranked.append(candidate)
        seen.add(normalized)

    for candidate in candidates:
        normalized = normalize_url(candidate.url)
        if normalized not in seen:
            reranked.append(candidate)

    return reranked


def apply_discovery_memory(
    candidates: list[IRDiscoveryCandidate],
    *,
    navigation_memory: CompanyNavigationMemory | None = None,
) -> list[IRDiscoveryCandidate]:
    """Reorder discovery candidates using company-specific learned memory."""
    if not navigation_memory:
        return candidates

    rescored = [apply_candidate_memory(candidate, navigation_memory) for candidate in candidates]
    return sorted(rescored, key=lambda candidate: candidate.score, reverse=True)


def apply_candidate_memory(
    candidate: IRDiscoveryCandidate,
    navigation_memory: CompanyNavigationMemory,
) -> IRDiscoveryCandidate:
    adjusted = candidate.model_copy(deep=True)
    parsed = urlparse(adjusted.url)
    destination = parsed.netloc.lower()
    path = parsed.path.lower()
    normalized = normalize_url(adjusted.url)

    memory_rules: list[tuple[bool, int, str]] = [
        (destination in navigation_memory.preferred_hosts, 180, "memory preferred host"),
        (destination in navigation_memory.successful_hosts, 160, "memory successful host"),
        (normalized in {normalize_url(url) for url in navigation_memory.known_transcript_urls}, 220, "memory known transcript URL"),
        (normalized in {normalize_url(url) for url in navigation_memory.known_ir_home_urls}, 140, "memory known IR home URL"),
        (normalized in {normalize_url(url) for url in navigation_memory.known_event_listing_urls}, 90, "memory known event/listing URL"),
        (destination in navigation_memory.low_value_hosts, -220, "memory low-value host"),
        (destination in navigation_memory.robots_blocked_hosts, -50, "memory robots risk host"),
    ]
    for matches, delta, reason in memory_rules:
        if matches and reason not in adjusted.reasons:
            adjusted.score += delta
            append_reason(adjusted, reason)

    low_value_terms = [term.lower() for term in navigation_memory.low_value_path_terms if term.strip()]
    matching_terms = [term for term in low_value_terms if term in path or term in adjusted.url.lower()]
    if matching_terms and not any(reason.startswith("memory avoided path:") for reason in adjusted.reasons):
        adjusted.score -= min(160, 55 * len(matching_terms))
        append_reason(adjusted, f"memory avoided path: {', '.join(matching_terms[:3])}")

    return adjusted


def append_reason(candidate: IRDiscoveryCandidate, reason: str) -> None:
    if reason not in candidate.reasons:
        candidate.reasons.insert(0, reason)


def search_ir_candidates(
    company: Company,
    max_results: int = 10,
    *,
    timeout_seconds: float = 30.0,
    progress: ProgressReporter | None = None,
) -> list[IRDiscoveryCandidate]:
    candidates: list[IRDiscoveryCandidate] = []
    progress = progress or ProgressReporter(enabled=False)

    try:
        with DDGS() as ddgs:
            for query in search_queries(company):
                progress.log(f"{company.symbol}: DuckDuckGo query: {query}")
                try:
                    with timeout_after(timeout_seconds, f"searching DuckDuckGo for {query}"):
                        for result in ddgs.text(query, backend="lite", max_results=max_results):
                            url = result.get("href") or result.get("url")
                            if not url:
                                continue
                            candidates.append(
                                score_ir_candidate(
                                    url=url,
                                    title=result.get("title", ""),
                                    snippet=result.get("body", ""),
                                    company=company,
                                    source="search",
                                )
                            )
                except Exception as exc:
                    progress.log(f"{company.symbol}: DuckDuckGo query skipped ({type(exc).__name__}: {exc})")
                    continue
                if len(candidates) >= max_results:
                    break
    except Exception as exc:
        progress.log(f"{company.symbol}: DuckDuckGo discovery unavailable ({type(exc).__name__}: {exc})")

    if not candidates:
        progress.log(f"{company.symbol}: trying DuckDuckGo lite HTML fallback")
        candidates.extend(
            search_ir_candidates_lite_html(
                company,
                max_results=max_results,
                timeout_seconds=timeout_seconds,
                progress=progress,
            )
        )

    return dedupe_candidates([candidate for candidate in candidates if candidate.score > 0])


def search_ir_candidates_lite_html(
    company: Company,
    max_results: int = 10,
    *,
    timeout_seconds: float = 30.0,
    progress: ProgressReporter | None = None,
) -> list[IRDiscoveryCandidate]:
    candidates: list[IRDiscoveryCandidate] = []
    progress = progress or ProgressReporter(enabled=False)
    session = requests.Session()
    session.headers.update({"User-Agent": "local-ir-discovery/0.1"})

    for query in search_queries(company):
        progress.log(f"{company.symbol}: lite HTML query: {query}")
        try:
            request_timeout = min(timeout_seconds, 20) if timeout_seconds > 0 else 20
            with timeout_after(timeout_seconds, f"searching DuckDuckGo lite HTML for {query}"):
                response = session.post(
                    "https://lite.duckduckgo.com/lite/",
                    data={"q": query},
                    timeout=request_timeout,
                )
                response.raise_for_status()
        except Exception as exc:
            progress.log(f"{company.symbol}: lite HTML query skipped ({type(exc).__name__}: {exc})")
            continue

        soup = BeautifulSoup(response.text, "lxml")
        for anchor in soup.find_all("a", href=True):
            title = anchor.get_text(" ", strip=True)
            url = anchor["href"]
            if not title or not url.startswith("http"):
                continue
            snippet = anchor.find_parent("td").get_text(" ", strip=True) if anchor.find_parent("td") else ""
            candidates.append(
                score_ir_candidate(
                    url=url,
                    title=title,
                    snippet=snippet,
                    company=company,
                    source="search",
                )
            )
            if len(candidates) >= max_results:
                break
        if len(candidates) >= max_results:
            break

    return dedupe_candidates(candidates)


def search_queries(company: Company) -> list[str]:
    if not company.name:
        return [
            f"{company.symbol} investor relations",
            f"{company.symbol} earnings results investor relations",
            f"{company.symbol} company homepage",
            f"{company.symbol} official website",
        ]
    return [
        f"{company.name} investor relations",
        f"{company.name} earnings results investor relations",
        f"{company.symbol} investor relations",
        f"{company.name} earnings call transcript",
    ]


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


def score_ir_candidate(
    *,
    url: str,
    title: str,
    snippet: str,
    company: Company,
    source: IRDiscoveryCandidate.model_fields["source"].annotation,
) -> IRDiscoveryCandidate:
    parsed = urlparse(url)
    haystack = f"{url} {title} {snippet}".lower()
    domain = parsed.netloc.lower()
    score = 0
    reasons: list[str] = []

    company_tokens = company_domain_tokens(company)
    if any(token in domain for token in company_tokens):
        score += 25
        reasons.append("domain matches company")
    official_domain = company_domain_slug(company.name)
    if domain == f"www.{official_domain}.com" or domain.endswith(f".{official_domain}.com"):
        score += 35
        reasons.append("official company domain")
    if company.symbol.lower() in haystack:
        score += 8
        reasons.append("mentions ticker")

    positive = {
        "investor relations": 25,
        "investors": 12,
        "investor": 12,
        "earnings": 10,
        "quarterly results": 10,
        "financial results": 10,
        "events": 6,
        "transcript": 12,
        "shareholder": 6,
    }
    for token, value in positive.items():
        if token in haystack:
            score += value
            reasons.append(f"contains {token}")

    negative = {
        "careers": 30,
        "support": 20,
        "privacy": 20,
        "terms": 20,
        "store": 15,
        "learn": 10,
        "training": 10,
        "news.microsoft.com": 8,
    }
    for token, value in negative.items():
        if token in haystack:
            score -= value
            reasons.append(f"penalized {token}")

    third_party_domains = (
        "fool.com",
        "seekingalpha.com",
        "finance.yahoo.com",
        "financialreports.eu",
        "marketbeat.com",
        "stockanalysis.com",
        "morningstar.com",
        "aol.com",
        "quartr.com",
        "valuesense.io",
        "advfn.com",
        "finviz.com",
        "barchart.com",
        "prnewswire.com",
        "sec.gov",
    )
    if any(domain == third_party or domain.endswith(f".{third_party}") for third_party in third_party_domains):
        score -= 35
        reasons.append("penalized third-party domain")

    if source == "curated":
        score += 50
    elif source == "search":
        score += 10

    return IRDiscoveryCandidate(
        url=url,
        title=title,
        snippet=snippet,
        source=source,
        score=score,
        reasons=reasons,
    )


def company_domain_tokens(company: Company) -> list[str]:
    tokens = [company.symbol.lower()]
    slug = company_domain_slug(company.name)
    if slug:
        tokens.append(slug)
    for word in (company.name or "").lower().split():
        cleaned = "".join(char for char in word if char.isalnum())
        if len(cleaned) > 3:
            tokens.append(cleaned)
    return list(dict.fromkeys(tokens))


def dedupe_candidates(candidates: list[IRDiscoveryCandidate]) -> list[IRDiscoveryCandidate]:
    seen: set[str] = set()
    result: list[IRDiscoveryCandidate] = []
    for candidate in candidates:
        normalized = normalize_url(candidate.url)
        if normalized in seen:
            continue
        seen.add(normalized)
        result.append(candidate)
    return result


def company_domain_slug(name: str | None) -> str:
    if not name:
        return ""
    words = []
    stop = {
        "inc",
        "inc.",
        "com",
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
    normalized = name.lower().replace("&", " and ")
    raw_tokens = []
    current = []
    for char in normalized:
        if char.isalnum():
            current.append(char)
            continue
        if current:
            raw_tokens.append("".join(current))
            current = []
    if current:
        raw_tokens.append("".join(current))

    for raw in raw_tokens:
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
