from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from urllib.parse import urlparse

from .agent import CrawlNavigatorAgent, HomepagePredictionAgent, HomepageValidationAgent, NavigationCandidateRankingAgent, NavigationPageContextAgent, compact_candidate_links
from .browser import BROWSER_COMPATIBLE_USER_AGENT, PlaywrightRenderer
from .http import HttpClient, RobotsUnavailableError
from .identity import company_display_name, curated_homepage_urls, filter_homepage_prediction_urls, is_homepage_candidate_url, is_weak_identity, official_homepage_urls, resolve_company_identity_with_overrides, verify_homepage_content
from .models import CandidateLink, Company, CompanyNavigationMemory, FailureType, NavigationCandidateTrace, NavigationStep, NavigationTrace, PromptGuidance
from .parsing import extract_links, looks_like_js_shell, page_title, visible_text
from .runtime import ProgressReporter, timeout_after
from .search import company_domain_tokens, company_domain_slug
from .urls import host, normalize_url, resolve_document_url

IRNavigationAgent = CrawlNavigatorAgent


@dataclass
class NavigationDiscoveryResult:
    seeds: list[str]
    trace: NavigationTrace
    failure_type: FailureType | None = None
    failure_message: str = ""
    failure_urls: list[str] | None = None
    robots_verified_official_urls: list[str] | None = None
    verified_homepage_urls: list[str] | None = None
    verified_company_name: str | None = None


@dataclass
class HomepageStartResolution:
    starts: list[str]
    failure_type: FailureType | None = None
    failure_message: str = ""
    failure_urls: list[str] | None = None
    verified_homepage_urls: list[str] | None = None
    verified_company_name: str | None = None


PREDICTIVE_HOMEPAGE_CONFIDENCE_FLOOR = 0.55
MAX_HOMEPAGE_EVIDENCE_LINKS = 8
MAX_NAVIGATION_DISCOVERY_SEEDS = 10


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
    homepage_candidates: list[str] | None = None,
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
    verified_homepage_urls: list[str] = []
    verified_company_name: str | None = None
    if should_predict_homepage_identity(company, disable_official_homepage_overrides) and not disable_predictive_identity:
        homepage_resolution = predict_and_validate_homepage_starts(
            company,
            http=http,
            model=model,
            ollama_base_url=ollama_base_url,
            llm_timeout_seconds=llm_timeout_seconds,
            llm_text_chars=llm_text_chars,
            identity_name_hint=identity_name_hint,
            homepage_candidates=homepage_candidates or [],
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
                verified_homepage_urls=homepage_resolution.verified_homepage_urls or [],
                verified_company_name=homepage_resolution.verified_company_name,
            )
        starts = homepage_resolution.starts
        verified_homepage_urls = homepage_resolution.verified_homepage_urls or []
        verified_company_name = homepage_resolution.verified_company_name
        if verified_company_name and is_weak_identity(company):
            company = company.model_copy(update={"name": verified_company_name})
            trace.company = company
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
        return NavigationDiscoveryResult(
            seeds=[],
            trace=trace,
            robots_verified_official_urls=[],
            verified_homepage_urls=verified_homepage_urls,
            verified_company_name=verified_company_name,
        )
    progress.log(f"{company.symbol}: navigation start URL(s): {', '.join(starts[:5])}")

    agent = IRNavigationAgent(
        model,
        base_url=ollama_base_url,
        max_links=navigation_llm_max_links,
        text_chars=llm_text_chars,
        guidance=prompt_guidance,
    )
    ranking_agent = NavigationCandidateRankingAgent(
        model,
        base_url=ollama_base_url,
        max_links=max_links,
        guidance=prompt_guidance,
    )
    page_context_agent = NavigationPageContextAgent(
        model,
        base_url=ollama_base_url,
        text_chars=llm_text_chars,
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

        html: str | None = None
        title = ""
        text = ""
        render_strategy = "http"
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
                    verified_homepage_urls=verified_homepage_urls,
                    verified_company_name=verified_company_name,
                )
            progress.log(
                f"{company.symbol}: fetching official IR subdomain despite unavailable robots.txt {current_url}"
            )
            response = http.get_without_robots_check(current_url)
        except Exception as exc:
            if renderer and is_within_verified_official_domain(
                current_url,
                [*verified_homepage_urls, *robots_verified_official_urls],
            ):
                try:
                    progress.log(f"{company.symbol}: rendering navigation page after HTTP fetch failed {current_url}")
                    html = renderer.render_html(current_url)
                    title = page_title(html)
                    text = visible_text(html)
                    robots_verified_official_urls.append(current_url)
                    render_strategy = "playwright_after_http_error"
                except Exception:
                    html = None
            if html is None:
                message = f"navigation fetch failed ({type(exc).__name__}: {exc})"
                progress.log(f"{company.symbol}: {message}: {current_url}")
                return NavigationDiscoveryResult(
                    seeds=[],
                    trace=trace,
                    failure_type="navigation_fetch_failed",
                    failure_message=message,
                    failure_urls=[current_url],
                    robots_verified_official_urls=robots_verified_official_urls,
                    verified_homepage_urls=verified_homepage_urls,
                    verified_company_name=verified_company_name,
                )

        if html is None:
            html = response.text
            title = page_title(html)
            text = visible_text(html)
        if renderer and should_render_navigation_page(playwright_mode, html, text):
            try:
                progress.log(f"{company.symbol}: rendering navigation page {current_url}")
                html = renderer.render_html(current_url)
                title = page_title(html)
                text = visible_text(html)
                render_strategy = "playwright_crawler_ua"
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
                    verified_homepage_urls=verified_homepage_urls,
                    verified_company_name=verified_company_name,
                )
        raw_links = extract_links(html, current_url)
        verification = verify_homepage_content(
            company,
            url=current_url,
            title=title,
            text=text,
            links=raw_links,
        )
        page_context = classify_navigation_page_context(
            company,
            current_url,
            title,
            text,
            page_context_agent=page_context_agent,
            llm_timeout_seconds=llm_timeout_seconds,
            progress=progress,
        )
        links = navigation_candidate_links(
            raw_links,
            current_url=current_url,
            title=title,
            page_context=page_context,
            allowed_hosts=allowed_hosts,
            company=company,
            limit=max_links,
            ranking_agent=ranking_agent,
            llm_timeout_seconds=llm_timeout_seconds,
            progress=progress,
        )
        if (
            renderer
            and not links
            and should_render_with_browser_user_agent(
                current_url,
                text=text,
                raw_links=raw_links,
                verified_official_urls=[*verified_homepage_urls, *robots_verified_official_urls],
            )
        ):
            try:
                progress.log(f"{company.symbol}: browser-UA rendering navigation page {current_url}")
                html = renderer.render_html_with_user_agent(current_url, BROWSER_COMPATIBLE_USER_AGENT)
                title = page_title(html)
                text = visible_text(html)
                raw_links = extract_links(html, current_url)
                verification = verify_homepage_content(
                    company,
                    url=current_url,
                    title=title,
                    text=text,
                    links=raw_links,
                )
                links = navigation_candidate_links(
                    raw_links,
                    current_url=current_url,
                    title=title,
                    page_context=classify_navigation_page_context(
                        company,
                        current_url,
                        title,
                        text,
                        page_context_agent=page_context_agent,
                        llm_timeout_seconds=llm_timeout_seconds,
                        progress=progress,
                    ),
                    allowed_hosts=allowed_hosts,
                    company=company,
                    limit=max_links,
                    ranking_agent=ranking_agent,
                    llm_timeout_seconds=llm_timeout_seconds,
                    progress=progress,
                )
                render_strategy = "browser_ua_fallback"
            except Exception as exc:
                progress.log(f"{company.symbol}: browser-UA render fallback skipped ({type(exc).__name__}: {current_url})")

        remember_homepage_ir_evidence(
            company=company,
            verification=verification,
            allowed_hosts=allowed_hosts,
            discovered=discovered,
            queue=queue,
            progress=progress,
        )
        if not links:
            trace.steps.append(
                NavigationStep(
                    current_url=current_url,
                    title=title,
                    raw_link_count=len(raw_links),
                    chosen_urls=list(verification.linked_ir_urls),
                    render_strategy=render_strategy,
                    stop_reason="no_useful_links",
                    reason="homepage_content_verified" if verification.linked_ir_urls else "",
                )
            )
            continue

        try:
            sent_link_count = len(compact_candidate_links(links, max_links=navigation_llm_max_links))
            progress.log(
                f"{company.symbol}: asking crawl navigator to choose from "
                f"{sent_link_count} of {len(links)} navigation candidate(s)"
            )
            with timeout_after(llm_timeout_seconds, f"choosing navigation links for {current_url}"):
                decision = decide_navigation_links(
                    agent,
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
            trace.steps.append(
                NavigationStep(
                    current_url=current_url,
                    title=title,
                    candidate_urls=[link.url for link in links],
                    candidate_links=navigation_candidate_trace(
                        links,
                        current_url=current_url,
                        allowed_hosts=allowed_hosts,
                        company=company,
                    ),
                    raw_link_count=len(raw_links),
                    render_strategy=render_strategy,
                    reason=message,
                    stop_reason="navigation_llm_failed",
                )
            )
            return NavigationDiscoveryResult(
                seeds=[],
                trace=trace,
                failure_type="navigation_llm_failed",
                failure_message=message,
                failure_urls=[current_url],
                robots_verified_official_urls=robots_verified_official_urls,
                verified_homepage_urls=verified_homepage_urls,
                verified_company_name=verified_company_name,
            )
        else:
            chosen_urls = decision.chosen_urls or [links[0].url]
            confidence = decision.confidence
            reason = decision.reason
            stop_reason = decision.stop_reason

        chosen_urls = order_chosen_urls(chosen_urls, links)
        rejected = [link.url for link in links if link.url not in chosen_urls][:10]
        trace.steps.append(
            NavigationStep(
                current_url=current_url,
                title=title,
                candidate_urls=[link.url for link in links],
                candidate_links=navigation_candidate_trace(
                    links,
                    current_url=current_url,
                    allowed_hosts=allowed_hosts,
                    company=company,
                ),
                raw_link_count=len(raw_links),
                chosen_urls=chosen_urls,
                rejected_urls=rejected,
                render_strategy=render_strategy,
                confidence=confidence,
                reason=reason,
                stop_reason=stop_reason,
            )
        )

        for chosen_url in chosen_urls:
            chosen_host = host(chosen_url)
            if chosen_host not in allowed_hosts:
                allowed_hosts.add(chosen_host)
            discovered.append(chosen_url)
            queue.append(chosen_url)

    seeds = rank_discovered_urls(discovered, company, navigation_memory=navigation_memory)
    progress.log(f"{company.symbol}: navigation discovery found {len(seeds)} seed URL(s)")
    return NavigationDiscoveryResult(
        seeds=seeds,
        trace=trace,
        robots_verified_official_urls=robots_verified_official_urls,
        verified_homepage_urls=verified_homepage_urls,
        verified_company_name=verified_company_name,
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
    homepage_candidates: list[str] | None = None,
    progress: ProgressReporter | None = None,
) -> HomepageStartResolution:
    progress = progress or ProgressReporter(enabled=False)
    homepage_candidates = homepage_candidates or []
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

    homepage_urls = filter_homepage_prediction_urls([*homepage_candidates, *prediction.homepage_urls])
    if prediction.confidence < PREDICTIVE_HOMEPAGE_CONFIDENCE_FLOOR:
        if homepage_urls and homepage_candidates:
            progress.log(f"{company.symbol}: using playbook homepage candidate(s) despite low prediction confidence")
        else:
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
    verified_homepage_urls: list[str] = []
    verified_company_name: str | None = None
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
        verified_homepage_urls.append(url)
        if decision and decision.official_company_name and not verified_company_name:
            verified_company_name = decision.official_company_name
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
            verified_homepage_urls=verified_homepage_urls,
            verified_company_name=verified_company_name,
        )
    return HomepageStartResolution(
        starts=starts,
        verified_homepage_urls=dedupe(verified_homepage_urls),
        verified_company_name=verified_company_name,
    )


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
    if slug and not starts and not (
        disable_official_homepage_overrides and company.symbol.upper() in {"GOOG", "GOOGL"}
    ):
        starts.append(f"https://www.{slug}.com")
    if navigation_memory:
        starts.extend(memory_start_urls(navigation_memory))
    return rank_navigation_starts(
        [url for url in dedupe(starts) if not memory_excluded_url(url, navigation_memory)],
        company,
        navigation_memory=navigation_memory,
    )


def should_predict_homepage_identity(company: Company, disable_official_homepage_overrides: bool) -> bool:
    return (
        is_weak_identity(company)
        or disable_official_homepage_overrides
        or not curated_homepage_urls(
            company,
            allow_homepage_overrides=not disable_official_homepage_overrides,
        )
    )


def memory_excluded_url(url: str, navigation_memory: CompanyNavigationMemory | None) -> bool:
    if not navigation_memory:
        return False
    destination = host(url)
    return bool(destination and (destination in navigation_memory.low_value_hosts or destination in navigation_memory.robots_blocked_hosts))


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


def should_render_with_browser_user_agent(
    url: str,
    *,
    text: str,
    raw_links: list[CandidateLink],
    verified_official_urls: list[str],
) -> bool:
    if not is_within_verified_official_domain(url, verified_official_urls):
        return False
    return len(text.strip()) < 1_000 or len(raw_links) < 10


def is_within_verified_official_domain(url: str, verified_official_urls: list[str]) -> bool:
    target_domain = registrable_domain(host(url))
    if not target_domain:
        return False
    for verified_url in verified_official_urls:
        verified_domain = registrable_domain(host(verified_url))
        if verified_domain and verified_domain == target_domain:
            return True
    return False


def remember_homepage_ir_evidence(
    *,
    company: Company,
    verification,
    allowed_hosts: set[str],
    discovered: list[str],
    queue: deque[str],
    progress: ProgressReporter,
) -> None:
    if not (verification.is_official and verification.linked_ir_urls):
        return
    progress.log(
        f"{company.symbol}: homepage evidence found {len(verification.linked_ir_urls)} official IR link(s)"
    )
    for accepted_host in verification.accepted_hosts:
        allowed_hosts.add(accepted_host)
    linked_urls = [
        linked_url
        for linked_url in dedupe(list(verification.linked_ir_urls))
        if not is_language_variant_url(linked_url)
        and navigation_choice_score(linked_url, company) > 0
    ]
    linked_urls = sorted(
        linked_urls,
        key=lambda linked_url: navigation_choice_score(linked_url, company),
        reverse=True,
    )[:MAX_HOMEPAGE_EVIDENCE_LINKS]
    for linked_url in linked_urls:
        allowed_hosts.add(host(linked_url))
        discovered.append(linked_url)
        queue.append(linked_url)


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
    if navigation_memory:
        return sorted(
            urls,
            key=lambda url: navigation_memory_score(url, navigation_memory=navigation_memory),
            reverse=True,
        )
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
    title: str = "",
    page_context: str = "",
    allowed_hosts: set[str],
    company: Company,
    limit: int,
    ranking_agent: NavigationCandidateRankingAgent | None = None,
    llm_timeout_seconds: float = 45.0,
    progress: ProgressReporter | None = None,
) -> list[CandidateLink]:
    eligible: list[CandidateLink] = []
    for link in links:
        link.url = resolve_document_url(link.url)
        if not mechanically_eligible_navigation_link(link, current_url=current_url, allowed_hosts=allowed_hosts, company=company):
            continue
        link.reason = link.reason or "body"
        eligible.append(link)
    if ranking_agent and eligible:
        try:
            with timeout_after(llm_timeout_seconds, f"ranking navigation candidates for {current_url}"):
                decision = ranking_agent.rank(
                    company_name=company_display_name(company),
                    ticker=company.symbol,
                    url=current_url,
                    title=title,
                    page_context=page_context or navigation_page_context(current_url, title, ""),
                    links=eligible,
                )
            links_by_url = {link.url: link for link in eligible}
            ranked = [
                (selection.priority, links_by_url[selection.url])
                for selection in decision.selections
                if selection.should_follow and selection.url in links_by_url
            ]
            ranked.sort(key=lambda item: item[0], reverse=True)
            if ranked:
                return [mark_agent_selected(link) for _, link in ranked[:limit]]
            return []
        except Exception as exc:
            if progress:
                progress.log(f"{company.symbol}: navigation candidate ranking failed ({type(exc).__name__}: {current_url})")

    scored: list[tuple[int, CandidateLink]] = [
        (navigation_link_score(link, current_url=current_url, allowed_hosts=allowed_hosts, company=company), link)
        for link in eligible
        if navigation_link_score(link, current_url=current_url, allowed_hosts=allowed_hosts, company=company) > 0
    ]
    scored.sort(key=lambda item: item[0], reverse=True)
    return [link for _, link in scored[:limit]]


def decide_navigation_links(agent, **kwargs) -> "NavigationDecision":
    if hasattr(agent, "decide_navigation"):
        return agent.decide_navigation(**kwargs)
    return agent.decide(**kwargs)


def navigation_candidate_trace(
    links: list[CandidateLink],
    *,
    current_url: str,
    allowed_hosts: set[str],
    company: Company,
) -> list[NavigationCandidateTrace]:
    return [
        NavigationCandidateTrace(
            url=link.url,
            label=link.label,
            context=link.reason,
            score=navigation_link_score(
                link,
                current_url=current_url,
                allowed_hosts=allowed_hosts,
                company=company,
            ),
        )
        for link in links
    ]


def mechanically_eligible_navigation_link(
    link: CandidateLink,
    *,
    current_url: str,
    allowed_hosts: set[str],
    company: Company,
) -> bool:
    parsed = urlparse(link.url)
    if parsed.scheme not in {"http", "https"}:
        return False
    if is_language_variant_url(link.url):
        return False
    destination_host = host(link.url)
    return destination_host in allowed_hosts or is_same_company_host(link.url, current_url, company)


def mark_agent_selected(link: CandidateLink) -> CandidateLink:
    if "[agent-selected]" not in link.reason:
        link.reason = f"[agent-selected] {link.reason}".strip()
    return link


def order_chosen_urls(chosen_urls: list[str], links: list[CandidateLink]) -> list[str]:
    candidate_urls = {link.url for link in links}
    return [url for url in chosen_urls if url in candidate_urls]


def classify_navigation_page_context(
    company: Company,
    url: str,
    title: str,
    text: str,
    *,
    page_context_agent: NavigationPageContextAgent | None = None,
    llm_timeout_seconds: float = 45.0,
    progress: ProgressReporter | None = None,
) -> str:
    if not page_context_agent:
        return navigation_page_context(url, title, text)
    try:
        with timeout_after(llm_timeout_seconds, f"classifying navigation page context {url}"):
            decision = page_context_agent.classify(
                company_name=company_display_name(company),
                ticker=company.symbol,
                url=url,
                title=title,
                text=text,
            )
    except Exception as exc:
        if progress:
            progress.log(f"{company.symbol}: navigation page context failed ({type(exc).__name__}: {url})")
        return "homepage"
    return decision.page_context if decision.confidence >= 0.45 else "homepage"


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
        "transcript": 95,
        "edited transcript": 110,
        "earnings call transcript": 120,
        "quarterly earnings call transcript": 130,
        "events/event-details": 105,
        "event-details": 95,
        "earnings call": 90,
        "quarterly earnings call": 105,
        "q1 earnings": 55,
        "q2 earnings": 55,
        "q3 earnings": 55,
        "q4 earnings": 55,
        "fy earnings": 45,
        "investor relations": 40,
        "investors": 35,
        "investor": 30,
        "shareholders": 20,
        "financial info": 30,
        "financial-info": 35,
        "/financial-info": 40,
        "financial reports": 35,
        "financial-reports": 55,
        "/financial-reports": 65,
        "quarterly results": 30,
        "quarterly-results": 70,
        "/quarterly-results": 80,
        "earnings releases": 30,
        "earnings-releases": 35,
        "earnings": 42,
        "events": 16,
        "presentations": 4,
        "webcast": 14,
        "results": 12,
        "news releases": 10,
    }
    score = sum(value for token, value in positive.items() if token in haystack)
    negative = {
        "contact investor relations": 35,
        "contact": 15,
        "email alerts": 25,
        "email-alert": 35,
        "/email-alert": 45,
        "stock quote": 25,
        "stock-info": 35,
        "/stock-info": 45,
        "governance": 20,
        "/governance": 45,
        "skip to main content": 50,
        "#maincontent": 50,
        "#main-content": 50,
        "additional information": 20,
        "faqs": 15,
        "/faqs": 30,
        "home page": 15,
        "sec filings": 55,
        "/sec-filings": 70,
        "governance": 35,
        "annual meeting": 35,
        "annual-reports": 45,
        "/annual-reports": 55,
        "investor-resources": 20,
        "stock": 30,
        "news releases": 20,
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
        normalized = normalize_navigation_url(url)
        if normalized in seen:
            continue
        seen.add(normalized)
        result.append(url)
    return result


def normalize_navigation_url(url: str) -> str:
    normalized = normalize_url(url)
    parsed = urlparse(normalized)
    path = parsed.path
    if path.lower().endswith("/default.aspx"):
        normalized = parsed._replace(path=path.lower()).geturl()
    return normalized


def rank_discovered_urls(
    urls: list[str],
    company: Company,
    *,
    navigation_memory: CompanyNavigationMemory | None = None,
) -> list[str]:
    deduped = dedupe(urls)
    ranked = sorted(
        deduped,
        key=lambda url: navigation_choice_score(url, company, navigation_memory=navigation_memory),
        reverse=True,
    )
    return ranked[:MAX_NAVIGATION_DISCOVERY_SEEDS]


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
