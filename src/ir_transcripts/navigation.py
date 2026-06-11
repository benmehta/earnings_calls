from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from urllib.parse import urlparse

from .agent import IRNavigationAgent
from .browser import PlaywrightRenderer
from .http import HttpClient
from .models import CandidateLink, Company, NavigationStep, NavigationTrace
from .parsing import extract_links, looks_like_js_shell, page_title, visible_text
from .runtime import ProgressReporter, timeout_after
from .search import company_domain_tokens, company_domain_slug, discover_ir_candidates
from .urls import host, normalize_url, resolve_document_url


@dataclass
class NavigationDiscoveryResult:
    seeds: list[str]
    trace: NavigationTrace


def discover_navigation_seeds(
    company: Company,
    *,
    http: HttpClient,
    model: str,
    ollama_base_url: str | None = None,
    max_steps: int = 4,
    max_links: int = 40,
    include_guesses: bool = False,
    playwright_mode: str = "off",
    disable_official_homepage_overrides: bool = False,
    search_timeout_seconds: float = 30.0,
    llm_timeout_seconds: float = 45.0,
    navigation_llm_max_links: int = 12,
    llm_text_chars: int = 900,
    progress: ProgressReporter | None = None,
) -> NavigationDiscoveryResult:
    progress = progress or ProgressReporter(enabled=False)
    progress.log(f"{company.symbol}: navigation discovery starting")
    starts = navigation_start_urls(
        company,
        include_guesses=include_guesses,
        disable_official_homepage_overrides=disable_official_homepage_overrides,
        search_timeout_seconds=search_timeout_seconds,
        progress=progress,
    )
    trace = NavigationTrace(company=company)
    if not starts:
        progress.log(f"{company.symbol}: navigation discovery has no start URLs")
        return NavigationDiscoveryResult(seeds=[], trace=trace)
    progress.log(f"{company.symbol}: navigation start URL(s): {', '.join(starts[:5])}")

    agent = IRNavigationAgent(
        model,
        base_url=ollama_base_url,
        max_links=navigation_llm_max_links,
        text_chars=llm_text_chars,
    )
    queue: deque[str] = deque(starts)
    visited: set[str] = set()
    allowed_hosts = {host(url) for url in starts}
    discovered: list[str] = []
    renderer = PlaywrightRenderer(http) if playwright_mode != "off" else None

    while queue and len(trace.steps) < max_steps:
        current_url = resolve_document_url(queue.popleft())
        normalized = normalize_url(current_url)
        if normalized in visited or host(current_url) not in allowed_hosts:
            continue
        visited.add(normalized)

        try:
            progress.log(f"{company.symbol}: fetching navigation page {current_url}")
            response = http.get(current_url)
        except Exception as exc:
            progress.log(f"{company.symbol}: navigation fetch skipped ({type(exc).__name__}: {current_url})")
            continue

        html = response.text
        title = page_title(html)
        text = visible_text(html)
        if renderer and should_render_navigation_page(playwright_mode, html, text):
            try:
                progress.log(f"{company.symbol}: rendering navigation page {current_url}")
                html = renderer.render_html(current_url)
                title = page_title(html)
                text = visible_text(html)
            except Exception as exc:
                progress.log(f"{company.symbol}: render skipped ({type(exc).__name__}: {current_url})")
                pass
        page_context = navigation_page_context(current_url, title, text)
        links = navigation_candidate_links(
            extract_links(html, current_url),
            current_url=current_url,
            allowed_hosts=allowed_hosts,
            company=company,
            limit=max_links,
        )
        if not links:
            trace.steps.append(
                NavigationStep(
                    current_url=current_url,
                    title=title,
                    stop_reason="no_useful_links",
                )
            )
            continue

        try:
            agent_kind = agent.select_agent_kind(
                url=current_url,
                title=title,
                text=text,
                links=links,
                page_context=page_context,
            )
            progress.log(
                f"{company.symbol}: asking {agent_kind} agent to choose from {len(links)} navigation link(s)"
            )
            with timeout_after(llm_timeout_seconds, f"choosing navigation links for {current_url}"):
                decision = agent.decide(
                    company_name=company.name,
                    ticker=company.symbol,
                    url=current_url,
                    title=title,
                    text=text,
                    links=links,
                    page_context=page_context,
                )
        except Exception as exc:
            progress.log(f"{company.symbol}: navigation LLM fallback ({type(exc).__name__}: {exc})")
            chosen_urls = [link.url for link in links[:2]]
            confidence = 0.0
            reason = "heuristic fallback"
            stop_reason = None
        else:
            chosen_urls = decision.chosen_urls or [links[0].url]
            confidence = decision.confidence
            reason = decision.reason
            stop_reason = decision.stop_reason

        chosen_urls = sorted(
            chosen_urls,
            key=lambda url: navigation_seed_score(url, "", "", company),
            reverse=True,
        )
        rejected = [link.url for link in links if link.url not in chosen_urls][:10]
        trace.steps.append(
            NavigationStep(
                current_url=current_url,
                title=title,
                chosen_urls=chosen_urls,
                rejected_urls=rejected,
                confidence=confidence,
                reason=reason,
                stop_reason=stop_reason,
            )
        )

        for chosen_url in chosen_urls:
            chosen_host = host(chosen_url)
            if chosen_host not in allowed_hosts:
                allowed_hosts.add(chosen_host)
            if navigation_seed_score(chosen_url, "", "", company) >= 10:
                discovered.append(chosen_url)
            queue.append(chosen_url)

    seeds = rank_discovered_urls(discovered, company)
    progress.log(f"{company.symbol}: navigation discovery found {len(seeds)} seed URL(s)")
    return NavigationDiscoveryResult(seeds=seeds, trace=trace)


def navigation_start_urls(
    company: Company,
    *,
    include_guesses: bool = False,
    disable_official_homepage_overrides: bool = False,
    search_timeout_seconds: float = 30.0,
    progress: ProgressReporter | None = None,
) -> list[str]:
    starts: list[str] = []
    if not disable_official_homepage_overrides:
        starts.extend(official_company_homepages(company))
    slug = company_domain_slug(company.name)
    if slug:
        starts.append(f"https://www.{slug}.com")
    starts.extend(
        candidate.url
        for candidate in discover_ir_candidates(
            company,
            max_results=6,
            include_guesses=include_guesses,
            search_timeout_seconds=search_timeout_seconds,
            progress=progress,
        )
        if candidate.score > 0 and is_company_host(candidate.url, company)
    )
    return dedupe(starts)


def navigation_candidate_links(
    links: list[CandidateLink],
    *,
    current_url: str,
    allowed_hosts: set[str],
    company: Company,
    limit: int,
) -> list[CandidateLink]:
    scored: list[tuple[int, CandidateLink]] = []
    for link in links:
        link.url = resolve_document_url(link.url)
        score = navigation_link_score(link, current_url=current_url, allowed_hosts=allowed_hosts, company=company)
        if score <= 0:
            continue
        link.reason = link.reason or "body"
        scored.append((score, link))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [link for _, link in scored[:limit]]


def navigation_link_score(
    link: CandidateLink,
    *,
    current_url: str,
    allowed_hosts: set[str],
    company: Company,
) -> int:
    parsed = urlparse(link.url)
    if parsed.scheme not in {"http", "https"}:
        return 0
    destination_host = host(link.url)
    if destination_host not in allowed_hosts and not is_same_company_host(link.url, current_url, company):
        return 0

    haystack = f"{link.url} {link.label} {link.reason}".lower()
    score = navigation_seed_score(link.url, link.label, link.reason, company)
    if "footer" in link.reason:
        score += 5
    if "nav" in link.reason or "header" in link.reason:
        score += 4

    negative = (
        "careers",
        "support",
        "privacy",
        "terms",
        "legal",
        "store",
        "products",
        "email alerts",
        "stock quote",
        "governance",
        "annual meeting",
    )
    score -= sum(20 for token in negative if token in haystack)
    return score


def navigation_seed_score(url: str, label: str, context: str, company: Company) -> int:
    haystack = f"{url} {label} {context}".lower()
    positive = {
        "events/event-details": 60,
        "event-details": 55,
        "earnings call": 45,
        "quarterly earnings call": 45,
        "investor relations": 40,
        "investors": 35,
        "investor": 30,
        "shareholders": 20,
        "financial info": 30,
        "financial reports": 35,
        "quarterly results": 30,
        "earnings releases": 30,
        "earnings": 22,
        "events": 16,
        "presentations": 14,
        "webcast": 14,
        "transcript": 25,
        "results": 12,
        "news releases": 10,
    }
    score = sum(value for token, value in positive.items() if token in haystack)
    negative = {
        "contact investor relations": 35,
        "contact": 15,
        "email alerts": 25,
        "stock quote": 25,
        "governance": 20,
        "skip to main content": 50,
        "#maincontent": 50,
        "#main-content": 50,
        "additional information": 20,
        "faqs": 15,
        "home page": 15,
    }
    score -= sum(value for token, value in negative.items() if token in haystack)
    if is_company_host(url, company):
        score += 10
    return score


def navigation_page_context(url: str, title: str, text: str) -> str:
    haystack = f"{url} {title} {text[:1000]}".lower()
    if any(token in haystack for token in ("investor", "financial reports", "earnings", "shareholder")):
        return "ir"
    return "homepage"


def is_same_company_host(url: str, current_url: str, company: Company) -> bool:
    destination = host(url)
    current = host(current_url)
    if not destination:
        return False
    if destination == current:
        return True
    return is_company_host(url, company)


def is_company_host(url: str, company: Company) -> bool:
    destination = host(url)
    if not destination:
        return False
    if destination in official_company_hosts(company):
        return True
    tokens = company_domain_tokens(company)
    if any(token and token in destination for token in tokens):
        return True
    slug = company_domain_slug(company.name)
    return bool(slug and destination.endswith(f".{slug}.com"))


def official_company_homepages(company: Company) -> list[str]:
    match company.symbol.upper():
        case "GOOG" | "GOOGL":
            return ["https://abc.xyz/"]
        case _:
            return []


def official_company_hosts(company: Company) -> set[str]:
    return {host(url) for url in official_company_homepages(company)}


def dedupe(urls: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for url in urls:
        normalized = normalize_url(url)
        if normalized in seen:
            continue
        seen.add(normalized)
        result.append(url)
    return result


def rank_discovered_urls(urls: list[str], company: Company) -> list[str]:
    deduped = dedupe(urls)
    return sorted(
        deduped,
        key=lambda url: navigation_seed_score(url, "", "", company),
        reverse=True,
    )


def should_render_navigation_page(playwright_mode: str, html: str, text: str) -> bool:
    if playwright_mode == "always":
        return True
    if playwright_mode == "auto":
        return looks_like_js_shell(html, text)
    return False
