from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from urllib.parse import urlparse

from .agent import HomepagePredictionAgent, HomepageValidationAgent, IRNavigationAgent
from .browser import PlaywrightRenderer
from .http import HttpClient, RobotsUnavailableError
from .identity import company_display_name, filter_homepage_prediction_urls, is_homepage_candidate_url, is_weak_identity, official_homepage_urls, resolve_company_identity_with_overrides, verify_homepage_content
from .models import CandidateLink, Company, CompanyNavigationMemory, FailureType, NavigationStep, NavigationTrace, PromptGuidance
from .parsing import extract_links, looks_like_js_shell, page_title, visible_text
from .runtime import ProgressReporter, timeout_after
from .search import company_domain_tokens, company_domain_slug, discover_ir_candidates
from .urls import host, normalize_url, resolve_document_url


@dataclass
class NavigationDiscoveryResult:
    seeds: list[str]
    trace: NavigationTrace
    failure_type: FailureType | None = None
    failure_message: str = ""
    failure_urls: list[str] | None = None
    robots_verified_official_urls: list[str] | None = None


@dataclass
class HomepageStartResolution:
    starts: list[str]
    failure_type: FailureType | None = None
    failure_message: str = ""
    failure_urls: list[str] | None = None


PREDICTIVE_HOMEPAGE_CONFIDENCE_FLOOR = 0.55


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
    prompt_guidance: PromptGuidance | None = None,
    navigation_memory: CompanyNavigationMemory | None = None,
    identity_name_hint: str | None = None,
    disable_predictive_identity: bool = False,
    progress: ProgressReporter | None = None,
) -> NavigationDiscoveryResult:
    company = resolve_company_identity_with_overrides(
        company,
        allow_homepage_overrides=not disable_official_homepage_overrides,
    )
    progress = progress or ProgressReporter(enabled=False)
    progress.log(f"{company.symbol}: navigation discovery starting")
    trace = NavigationTrace(company=company)
    if is_weak_identity(company) and not disable_predictive_identity:
        homepage_resolution = predict_and_validate_homepage_starts(
            company,
            http=http,
            model=model,
            ollama_base_url=ollama_base_url,
            llm_timeout_seconds=llm_timeout_seconds,
            llm_text_chars=llm_text_chars,
            identity_name_hint=identity_name_hint,
            progress=progress,
        )
        if homepage_resolution.failure_type:
            progress.log(f"{company.symbol}: homepage prediction stopped attempt ({homepage_resolution.failure_type})")
            return NavigationDiscoveryResult(
                seeds=[],
                trace=trace,
                failure_type=homepage_resolution.failure_type,
                failure_message=homepage_resolution.failure_message,
                failure_urls=homepage_resolution.failure_urls or [],
                robots_verified_official_urls=[],
            )
        starts = homepage_resolution.starts
        if navigation_memory:
            starts = dedupe([*starts, *memory_start_urls(navigation_memory)])
    else:
        starts = navigation_start_urls(
            company,
            model=model,
            ollama_base_url=ollama_base_url,
            include_guesses=include_guesses,
            disable_official_homepage_overrides=disable_official_homepage_overrides,
            search_timeout_seconds=search_timeout_seconds,
            llm_timeout_seconds=llm_timeout_seconds,
            prompt_guidance=prompt_guidance,
            navigation_memory=navigation_memory,
            progress=progress,
        )
    if not starts:
        progress.log(f"{company.symbol}: navigation discovery has no start URLs")
        return NavigationDiscoveryResult(seeds=[], trace=trace, robots_verified_official_urls=[])
    progress.log(f"{company.symbol}: navigation start URL(s): {', '.join(starts[:5])}")

    agent = IRNavigationAgent(
        model,
        base_url=ollama_base_url,
        max_links=navigation_llm_max_links,
        text_chars=llm_text_chars,
        guidance=prompt_guidance,
    )
    queue: deque[str] = deque(starts)
    visited: set[str] = set()
    allowed_hosts = {host(url) for url in starts}
    robots_verified_official_urls: list[str] = []
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
            robots_verified_official_urls.append(current_url)
        except RobotsUnavailableError as exc:
            if not can_delegate_unavailable_ir_subdomain_robots(
                current_url,
                verified_official_urls=robots_verified_official_urls,
            ):
                message = f"navigation fetch failed ({type(exc).__name__}: {exc})"
                progress.log(f"{company.symbol}: {message}: {current_url}")
                return NavigationDiscoveryResult(
                    seeds=[],
                    trace=trace,
                    failure_type="navigation_fetch_failed",
                    failure_message=message,
                    failure_urls=[current_url],
                    robots_verified_official_urls=robots_verified_official_urls,
                )
            progress.log(
                f"{company.symbol}: fetching official IR subdomain despite unavailable robots.txt {current_url}"
            )
            response = http.get_without_robots_check(current_url)
        except Exception as exc:
            message = f"navigation fetch failed ({type(exc).__name__}: {exc})"
            progress.log(f"{company.symbol}: {message}: {current_url}")
            return NavigationDiscoveryResult(
                seeds=[],
                trace=trace,
                failure_type="navigation_fetch_failed",
                failure_message=message,
                failure_urls=[current_url],
                robots_verified_official_urls=robots_verified_official_urls,
            )

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
                message = f"navigation render failed ({type(exc).__name__}: {exc})"
                progress.log(f"{company.symbol}: {message}: {current_url}")
                return NavigationDiscoveryResult(
                    seeds=[],
                    trace=trace,
                    failure_type="navigation_render_failed",
                    failure_message=message,
                    failure_urls=[current_url],
                    robots_verified_official_urls=robots_verified_official_urls,
                )
        page_context = navigation_page_context(current_url, title, text)
        raw_links = extract_links(html, current_url)
        verification = verify_homepage_content(
            company,
            url=current_url,
            title=title,
            text=text,
            links=raw_links,
        )
        if verification.is_official and verification.linked_ir_urls:
            progress.log(
                f"{company.symbol}: homepage evidence found {len(verification.linked_ir_urls)} official IR link(s)"
            )
            for accepted_host in verification.accepted_hosts:
                allowed_hosts.add(accepted_host)
            for linked_url in verification.linked_ir_urls:
                if is_language_variant_url(linked_url):
                    continue
                allowed_hosts.add(host(linked_url))
                discovered.append(linked_url)
                queue.append(linked_url)

        links = navigation_candidate_links(
            raw_links,
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
                    chosen_urls=list(verification.linked_ir_urls),
                    stop_reason="no_useful_links",
                    reason="homepage_content_verified" if verification.linked_ir_urls else "",
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
                    company_name=company_display_name(company),
                    ticker=company.symbol,
                    url=current_url,
                    title=title,
                    text=text,
                    links=links,
                    page_context=page_context,
                )
        except Exception as exc:
            message = f"navigation LLM failed ({type(exc).__name__}: {exc})"
            progress.log(f"{company.symbol}: {message}")
            return NavigationDiscoveryResult(
                seeds=[],
                trace=trace,
                failure_type="navigation_llm_failed",
                failure_message=message,
                failure_urls=[current_url],
                robots_verified_official_urls=robots_verified_official_urls,
            )
        else:
            chosen_urls = decision.chosen_urls or [links[0].url]
            confidence = decision.confidence
            reason = decision.reason
            stop_reason = decision.stop_reason

        chosen_urls = sorted(
            chosen_urls,
            key=lambda url: navigation_choice_score(url, company, navigation_memory=navigation_memory),
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
            if navigation_choice_score(chosen_url, company, navigation_memory=navigation_memory) >= 10:
                discovered.append(chosen_url)
            queue.append(chosen_url)

    seeds = rank_discovered_urls(discovered, company, navigation_memory=navigation_memory)
    progress.log(f"{company.symbol}: navigation discovery found {len(seeds)} seed URL(s)")
    return NavigationDiscoveryResult(
        seeds=seeds,
        trace=trace,
        robots_verified_official_urls=robots_verified_official_urls,
    )


def predict_and_validate_homepage_starts(
    company: Company,
    *,
    http: HttpClient,
    model: str,
    ollama_base_url: str | None = None,
    llm_timeout_seconds: float = 45.0,
    llm_text_chars: int = 900,
    identity_name_hint: str | None = None,
    progress: ProgressReporter | None = None,
) -> HomepageStartResolution:
    progress = progress or ProgressReporter(enabled=False)
    try:
        progress.log(f"{company.symbol}: predicting official homepage candidates")
        with timeout_after(llm_timeout_seconds, f"predicting homepage candidates for {company.symbol}"):
            prediction = HomepagePredictionAgent(model, base_url=ollama_base_url).predict(
                ticker=company.symbol,
                name_hint=identity_name_hint,
            )
    except Exception as exc:
        return HomepageStartResolution(
            starts=[],
            failure_type="identity_low_confidence",
            failure_message=f"homepage prediction failed: {type(exc).__name__}: {exc}",
        )

    homepage_urls = filter_homepage_prediction_urls(prediction.homepage_urls)
    if prediction.confidence < PREDICTIVE_HOMEPAGE_CONFIDENCE_FLOOR:
        return HomepageStartResolution(
            starts=[],
            failure_type="identity_low_confidence",
            failure_message=(
                f"homepage prediction confidence {prediction.confidence:.2f} "
                f"below {PREDICTIVE_HOMEPAGE_CONFIDENCE_FLOOR:.2f}: {prediction.reason}"
            ),
            failure_urls=homepage_urls,
        )
    if not homepage_urls:
        return HomepageStartResolution(
            starts=[],
            failure_type="homepage_unverified",
            failure_message="homepage prediction produced no homepage-only candidate URLs",
            failure_urls=prediction.homepage_urls,
        )

    validator = HomepageValidationAgent(
        model,
        base_url=ollama_base_url,
        text_chars=llm_text_chars,
    )
    verified_starts: list[str] = []
    attempted: list[str] = []
    for url in homepage_urls:
        attempted.append(url)
        if not is_homepage_candidate_url(url):
            continue
        try:
            progress.log(f"{company.symbol}: validating predicted homepage {url}")
            response = http.get(url)
        except Exception as exc:
            progress.log(f"{company.symbol}: homepage validation fetch skipped ({type(exc).__name__}: {url})")
            continue

        html = response.text
        title = page_title(html)
        text = visible_text(html)
        links = extract_links(html, url)
        try:
            with timeout_after(llm_timeout_seconds, f"validating homepage {url}"):
                decision = validator.validate(
                    ticker=company.symbol,
                    name_hint=identity_name_hint,
                    url=url,
                    title=title,
                    text=text,
                    links=links,
                )
        except Exception as exc:
            progress.log(f"{company.symbol}: homepage validator fallback ({type(exc).__name__}: {url})")
            decision = None

        accepted_by_llm = bool(decision and decision.is_official and decision.confidence >= 0.55)
        if not accepted_by_llm:
            continue
        verified_starts.append(url)
        verified_starts.extend(
            linked_url
            for linked_url in (decision.linked_ir_urls if decision else [])
            if not is_language_variant_url(linked_url)
        )

    starts = dedupe(verified_starts)
    if not starts:
        return HomepageStartResolution(
            starts=[],
            failure_type="homepage_unverified",
            failure_message="no predicted homepage could be verified from fetched page evidence",
            failure_urls=attempted,
        )
    return HomepageStartResolution(starts=starts)


def navigation_start_urls(
    company: Company,
    *,
    model: str | None = None,
    ollama_base_url: str | None = None,
    include_guesses: bool = False,
    disable_official_homepage_overrides: bool = False,
    search_timeout_seconds: float = 30.0,
    llm_timeout_seconds: float = 45.0,
    prompt_guidance: PromptGuidance | None = None,
    navigation_memory: CompanyNavigationMemory | None = None,
    progress: ProgressReporter | None = None,
) -> list[str]:
    company = resolve_company_identity_with_overrides(
        company,
        allow_homepage_overrides=not disable_official_homepage_overrides,
    )
    starts: list[str] = []
    if not disable_official_homepage_overrides:
        starts.extend(
            official_company_homepages(
                company,
                allow_homepage_overrides=not disable_official_homepage_overrides,
            )
        )
    slug = company_domain_slug(company.name)
    if slug and not starts:
        starts.append(f"https://www.{slug}.com")
    starts.extend(
        candidate.url
        for candidate in discover_ir_candidates(
            company,
            max_results=6,
            include_guesses=include_guesses,
            disable_official_homepage_overrides=disable_official_homepage_overrides,
            rerank_model=model,
            ollama_base_url=ollama_base_url,
            search_timeout_seconds=search_timeout_seconds,
            llm_timeout_seconds=llm_timeout_seconds,
            prompt_guidance=prompt_guidance,
            navigation_memory=navigation_memory,
            progress=progress,
        )
        if candidate.score > 0 and is_company_host(candidate.url, company) and not is_language_variant_url(candidate.url)
    )
    return rank_navigation_starts(dedupe(starts), company, navigation_memory=navigation_memory)


def memory_start_urls(navigation_memory: CompanyNavigationMemory) -> list[str]:
    return [
        *navigation_memory.known_ir_home_urls,
        *navigation_memory.known_event_listing_urls,
        *navigation_memory.known_transcript_urls,
    ]


IR_SUBDOMAIN_LABELS = {
    "investor",
    "investors",
    "ir",
    "investor-relations",
    "investorrelations",
    "shareholder",
    "shareholders",
}

MULTI_PART_PUBLIC_SUFFIXES = {
    "co.jp",
    "co.uk",
    "com.au",
    "com.br",
    "com.cn",
    "com.hk",
    "com.sg",
    "co.kr",
}


def can_delegate_unavailable_ir_subdomain_robots(
    url: str,
    *,
    verified_official_urls: list[str],
) -> bool:
    target_host = host(url)
    if not is_ir_subdomain_host(target_host):
        return False
    target_domain = registrable_domain(target_host)
    if not target_domain:
        return False
    for verified_url in verified_official_urls:
        verified_host = host(verified_url)
        if verified_host == target_host:
            continue
        if registrable_domain(verified_host) == target_domain and not is_ir_subdomain_host(verified_host):
            return True
    return False


def is_ir_subdomain_host(value: str) -> bool:
    labels = [label for label in value.lower().split(".") if label]
    if len(labels) < 3:
        return False
    return labels[0] in IR_SUBDOMAIN_LABELS


def registrable_domain(value: str) -> str:
    labels = [label for label in value.lower().split(".") if label]
    if len(labels) < 2:
        return value.lower()
    suffix = ".".join(labels[-2:])
    if suffix in MULTI_PART_PUBLIC_SUFFIXES and len(labels) >= 3:
        return ".".join(labels[-3:])
    return suffix


def rank_navigation_starts(
    urls: list[str],
    company: Company,
    *,
    navigation_memory: CompanyNavigationMemory | None = None,
) -> list[str]:
    return sorted(
        urls,
        key=lambda url: navigation_start_score(url, company, navigation_memory=navigation_memory),
        reverse=True,
    )


def navigation_start_score(
    url: str,
    company: Company,
    *,
    navigation_memory: CompanyNavigationMemory | None = None,
) -> int:
    parsed = urlparse(url)
    destination = host(url)
    path = parsed.path.lower().rstrip("/")
    haystack = f"{destination} {path}".lower()
    normalized = normalize_url(url)
    score = 0

    if url in official_company_homepages(company):
        score += 100
    if navigation_memory:
        if normalized in {normalize_url(value) for value in navigation_memory.known_ir_home_urls}:
            score += 160
        if normalized in {normalize_url(value) for value in navigation_memory.known_event_listing_urls}:
            score += 95
        if normalized in {normalize_url(value) for value in navigation_memory.known_transcript_urls}:
            score += 220
        if destination in navigation_memory.preferred_hosts:
            score += 180
        if destination in navigation_memory.successful_hosts:
            score += 160
        if destination in navigation_memory.low_value_hosts:
            score -= 220
        if any(term in path for term in navigation_memory.low_value_path_terms):
            score -= 80
        if destination in navigation_memory.robots_blocked_hosts:
            score -= 50
    if is_company_host(url, company):
        score += 20
    if path in {"", "/", "/investor", "/investors", "/investor/default.aspx", "/home/default.aspx"}:
        score += 45
    elif any(token in path for token in ("/investor", "/investors")):
        score += 20
    if any(token in path for token in ("/earnings", "/events", "/financial-reports", "/financial-info")):
        score += 10

    slug = company_domain_slug(company.name)
    if slug and destination == f"www.{slug}.com":
        score -= 35
    if "q4web.com" in destination:
        score -= 8
    if any(token in haystack for token in ("blog.", "youtube.com", "presentation", "press-release")):
        score -= 35
    if any(token in path for token in ("event-details", "news-details")):
        score -= 15
    return score


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
        if is_language_variant_url(link.url):
            continue
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
    if is_language_variant_url(link.url):
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


def official_company_homepages(
    company: Company,
    *,
    allow_homepage_overrides: bool = True,
) -> list[str]:
    return official_homepage_urls(company, allow_homepage_overrides=allow_homepage_overrides)


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


def rank_discovered_urls(
    urls: list[str],
    company: Company,
    *,
    navigation_memory: CompanyNavigationMemory | None = None,
) -> list[str]:
    deduped = dedupe(urls)
    return sorted(
        deduped,
        key=lambda url: navigation_choice_score(url, company, navigation_memory=navigation_memory),
        reverse=True,
    )


def navigation_choice_score(
    url: str,
    company: Company,
    *,
    navigation_memory: CompanyNavigationMemory | None = None,
) -> int:
    return navigation_seed_score(url, "", "", company) + navigation_memory_score(
        url,
        navigation_memory=navigation_memory,
    )


def navigation_memory_score(
    url: str,
    *,
    navigation_memory: CompanyNavigationMemory | None = None,
) -> int:
    if not navigation_memory:
        return 0
    parsed = urlparse(url)
    destination = host(url)
    path = parsed.path.lower()
    normalized = normalize_url(url)
    score = 0
    if destination in navigation_memory.preferred_hosts:
        score += 180
    if destination in navigation_memory.successful_hosts:
        score += 160
    if normalized in {normalize_url(value) for value in navigation_memory.known_transcript_urls}:
        score += 220
    if normalized in {normalize_url(value) for value in navigation_memory.known_ir_home_urls}:
        score += 140
    if normalized in {normalize_url(value) for value in navigation_memory.known_event_listing_urls}:
        score += 90
    if destination in navigation_memory.low_value_hosts:
        score -= 220
    if destination in navigation_memory.robots_blocked_hosts:
        score -= 50
    low_value_terms = [term.lower() for term in navigation_memory.low_value_path_terms if term.strip()]
    score -= min(160, 55 * sum(1 for term in low_value_terms if term in path or term in url.lower()))
    return score


def is_language_variant_url(url: str) -> bool:
    path = urlparse(url).path.lower()
    variants = (
        "/chinese/",
        "/schinese/",
        "/tchinese/",
        "/japanese/",
        "/korean/",
        "/zh/",
        "/zh-cn/",
        "/zh-tw/",
        "/ja/",
        "/ko/",
    )
    return any(token in path for token in variants)


def should_render_navigation_page(playwright_mode: str, html: str, text: str) -> bool:
    if playwright_mode == "always":
        return True
    if playwright_mode == "auto":
        return looks_like_js_shell(html, text)
    return False
