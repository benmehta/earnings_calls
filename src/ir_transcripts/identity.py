from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse

from .models import CandidateLink, Company
from .urls import host, normalize_url


@dataclass(frozen=True)
class CompanyIdentity:
    company: Company
    aliases: tuple[str, ...] = ()
    homepage_urls: tuple[str, ...] = ()


@dataclass(frozen=True)
class HomepageVerification:
    is_official: bool
    score: int
    linked_ir_urls: tuple[str, ...] = ()
    accepted_hosts: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()


KNOWN_COMPANY_IDENTITIES: dict[str, CompanyIdentity] = {
    "AAPL": CompanyIdentity(
        company=Company(symbol="AAPL", name="Apple"),
        aliases=("Apple", "Apple Inc."),
        homepage_urls=("https://www.apple.com",),
    ),
    "GOOG": CompanyIdentity(
        company=Company(symbol="GOOG", name="Alphabet Google"),
        aliases=("Alphabet", "Google", "Alphabet Inc."),
        homepage_urls=("https://abc.xyz/",),
    ),
    "GOOGL": CompanyIdentity(
        company=Company(symbol="GOOGL", name="Alphabet Google"),
        aliases=("Alphabet", "Google", "Alphabet Inc."),
        homepage_urls=("https://abc.xyz/",),
    ),
    "MSFT": CompanyIdentity(
        company=Company(symbol="MSFT", name="Microsoft"),
        aliases=("Microsoft", "Microsoft Corporation"),
        homepage_urls=("https://www.microsoft.com",),
    ),
    "NVDA": CompanyIdentity(
        company=Company(symbol="NVDA", name="NVIDIA"),
        aliases=("NVIDIA", "NVIDIA Corporation"),
        homepage_urls=("https://www.nvidia.com",),
    ),
    "TSM": CompanyIdentity(
        company=Company(symbol="TSM", name="Taiwan Semiconductor Manufacturing Company"),
        aliases=("TSMC", "Taiwan Semiconductor Manufacturing Company"),
        homepage_urls=("https://www.tsmc.com",),
    ),
}

COMMON_COMPANY_NAMES = {
    symbol: identity.company.name
    for symbol, identity in KNOWN_COMPANY_IDENTITIES.items()
}


def resolve_company_identity(company: Company) -> Company:
    identity = company_identity(company)
    return identity.company


def resolve_company_identity_with_overrides(
    company: Company,
    *,
    allow_homepage_overrides: bool = True,
) -> Company:
    return company_identity(company, allow_homepage_overrides=allow_homepage_overrides).company


def company_identity(
    company: Company,
    *,
    allow_homepage_overrides: bool = True,
) -> CompanyIdentity:
    symbol = company.symbol.upper()
    known = KNOWN_COMPANY_IDENTITIES.get(symbol)
    if known:
        if symbol in {"GOOG", "GOOGL"} and not allow_homepage_overrides:
            aliases = tuple(value for value in (company.name,) if value)
            return CompanyIdentity(company=company, aliases=aliases)
        if not company.name or company.name.upper() in {symbol, "GOOG", "GOOGL", "GOOGLE"}:
            return known
        aliases = tuple(dict.fromkeys((company.name, *known.aliases)))
        return CompanyIdentity(
            company=company.model_copy(update={"name": known.company.name}),
            aliases=aliases,
            homepage_urls=known.homepage_urls,
        )
    return CompanyIdentity(
        company=company,
        aliases=tuple(value for value in (company.name,) if value),
        homepage_urls=tuple(deterministic_homepage_urls(company.name)),
    )


def official_homepage_urls(
    company: Company,
    *,
    allow_homepage_overrides: bool = True,
) -> list[str]:
    identity = company_identity(company, allow_homepage_overrides=allow_homepage_overrides)
    if identity.homepage_urls:
        return dedupe(list(identity.homepage_urls))
    return dedupe([*identity.homepage_urls, *deterministic_homepage_urls(identity.company.name)])


def deterministic_homepage_urls(company_name: str | None) -> list[str]:
    from .search import company_domain_slug

    if not company_name:
        return []
    slug = company_domain_slug(company_name)
    if not slug:
        return []
    return [f"https://www.{slug}.com"]


def verify_homepage_content(
    company: Company,
    *,
    url: str,
    title: str,
    text: str,
    links: list[CandidateLink],
) -> HomepageVerification:
    identity = company_identity(company)
    aliases = identity_aliases(identity)
    current_host = host(url)
    haystack = f"{current_host} {title} {text[:2500]}".lower()
    score = 0
    reasons: list[str] = []

    if host_matches_identity(current_host, identity):
        score += 35
        reasons.append("host matches identity")
    if any(alias.lower() in haystack for alias in aliases):
        score += 30
        reasons.append("page mentions company identity")
    if any(token in haystack for token in ("copyright", "(c)", "©", "all rights reserved")):
        score += 8
        reasons.append("page has corporate footer markers")
    if any(token in haystack for token in ("investor relations", "investors", "about amazon", "about us")):
        score += 10
        reasons.append("page has corporate navigation terms")

    linked_ir_urls = tuple(dedupe([
        link.url
        for link in links
        if link_looks_like_official_ir(link, identity)
    ]))
    if linked_ir_urls:
        score += 35
        reasons.append("page links to official-looking IR URL")

    if any(token in haystack for token in ("unofficial", "fan site", "not affiliated")):
        score -= 60
        reasons.append("page has unofficial-site marker")

    accepted_hosts = [current_host] if current_host and score >= 35 else []
    for linked_url in linked_ir_urls:
        linked_host = host(linked_url)
        if linked_host:
            accepted_hosts.append(linked_host)

    return HomepageVerification(
        is_official=score >= 45,
        score=score,
        linked_ir_urls=linked_ir_urls,
        accepted_hosts=tuple(dedupe(accepted_hosts)),
        reasons=tuple(reasons),
    )


def link_looks_like_official_ir(link: CandidateLink, identity: CompanyIdentity) -> bool:
    haystack = f"{link.url} {link.label} {link.reason}".lower()
    if not any(token in haystack for token in ("investor", "investors", "investor relations", "/ir", "ir.")):
        return False
    return host_matches_identity(host(link.url), identity) or any(
        alias_token in host(link.url)
        for alias_token in identity_host_tokens(identity)
    )


def filter_homepage_prediction_urls(urls: list[str]) -> list[str]:
    return dedupe([url for url in urls if is_homepage_candidate_url(url)])


def is_homepage_candidate_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return False
    destination = parsed.netloc.lower()
    path = parsed.path.lower().rstrip("/")
    haystack = f"{destination} {path}"
    if destination.startswith(("ir.", "investor.", "investors.")):
        return False
    blocked_hosts = (
        "sec.gov",
        "seekingalpha.com",
        "finance.yahoo.com",
        "marketbeat.com",
        "stockanalysis.com",
        "morningstar.com",
        "quartr.com",
        "financialreports.eu",
        "prnewswire.com",
    )
    if any(destination == host_name or destination.endswith(f".{host_name}") for host_name in blocked_hosts):
        return False
    blocked_terms = (
        "investor",
        "investors",
        "ir/",
        "earnings",
        "transcript",
        "sec-filings",
        "financial-results",
        "quarterly-results",
        "press-release",
        "news-release",
    )
    return not any(term in haystack for term in blocked_terms)


def host_matches_identity(destination: str, identity: CompanyIdentity) -> bool:
    if not destination:
        return False
    official_hosts = {host(url) for url in identity.homepage_urls}
    if destination in official_hosts:
        return True
    tokens = identity_host_tokens(identity)
    return any(token and token in destination for token in tokens)


def identity_host_tokens(identity: CompanyIdentity) -> tuple[str, ...]:
    tokens: list[str] = []
    for value in (identity.company.name, *identity.aliases):
        if not value:
            continue
        for token in value.lower().replace("&", " and ").replace(".", " ").split():
            cleaned = "".join(char for char in token if char.isalnum())
            if len(cleaned) > 3 and cleaned not in {"corporation", "company", "inc", "com"}:
                tokens.append(cleaned)
    return tuple(dict.fromkeys(tokens))


def identity_aliases(identity: CompanyIdentity) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value for value in (identity.company.name, *identity.aliases) if value))


def company_display_name(company: Company) -> str:
    return company.name or company.symbol


def is_weak_identity(company: Company) -> bool:
    return not company.name or company.name.upper() == company.symbol.upper()


def dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        key = normalize_url(value)
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result
