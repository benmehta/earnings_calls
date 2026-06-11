from pathlib import Path
from urllib.parse import urljoin

from ir_transcripts.models import Company, NavigationDecision
from ir_transcripts.navigation import (
    discover_navigation_seeds,
    is_company_host,
    navigation_link_score,
    navigation_start_urls,
    rank_discovered_urls,
    navigation_candidate_links,
)
from ir_transcripts.models import IRDiscoveryCandidate
from ir_transcripts.parsing import extract_links


FIXTURES = Path(__file__).parent / "fixtures"


class FakeResponse:
    headers = {"content-type": "text/html"}

    def __init__(self, text: str) -> None:
        self.text = text
        self.content = text.encode("utf-8")


class FakeHttp:
    def __init__(self, pages: dict[str, str]) -> None:
        self.pages = pages

    def get(self, url: str) -> FakeResponse:
        return FakeResponse(self.pages[url])


class FakeNavigationAgent:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def decide(self, *, links, page_context, **kwargs) -> NavigationDecision:
        labels = {link.label.lower(): link.url for link in links}
        if page_context == "homepage":
            chosen = labels.get("investors")
            reason = "footer Investors link"
        elif "financial info" in labels:
            chosen = labels["financial info"]
            reason = "Financial Info inside investor relations"
        else:
            chosen = labels.get("q1 fy27 earnings call transcript") or links[0].url
            reason = "earnings transcript material"
        return NavigationDecision(
            chosen_urls=[chosen] if chosen else [],
            confidence=0.9,
            reason=reason,
        )


def test_navigation_discovery_reaches_nvidia_financial_reports(monkeypatch) -> None:
    pages = {
        "https://www.nvidia.com": (FIXTURES / "navigation_nvidia_home.html").read_text(encoding="utf-8"),
        "https://investor.nvidia.com/home/default.aspx": (
            FIXTURES / "navigation_nvidia_ir_home.html"
        ).read_text(encoding="utf-8"),
        "https://investor.nvidia.com/financial-info/financial-reports/default.aspx": (
            FIXTURES / "navigation_nvidia_financial_reports.html"
        ).read_text(encoding="utf-8"),
    }
    monkeypatch.setattr("ir_transcripts.navigation.IRNavigationAgent", FakeNavigationAgent)
    monkeypatch.setattr("ir_transcripts.navigation.discover_ir_candidates", lambda *args, **kwargs: [])

    result = discover_navigation_seeds(
        Company(symbol="NVDA", name="NVIDIA"),
        http=FakeHttp(pages),  # type: ignore[arg-type]
        model="test-model",
        max_steps=3,
    )

    assert [step.current_url for step in result.trace.steps] == [
        "https://www.nvidia.com",
        "https://investor.nvidia.com/home/default.aspx",
        "https://investor.nvidia.com/financial-info/financial-reports/default.aspx",
    ]
    assert "https://investor.nvidia.com/financial-info/financial-reports/default.aspx" in result.seeds


def test_navigation_rejects_third_party_links_by_default() -> None:
    html = (FIXTURES / "navigation_nvidia_home.html").read_text(encoding="utf-8")
    links = extract_links(html, "https://www.nvidia.com")
    candidates = navigation_candidate_links(
        links,
        current_url="https://www.nvidia.com",
        allowed_hosts={"www.nvidia.com"},
        company=Company(symbol="NVDA", name="NVIDIA"),
        limit=20,
    )

    assert "https://quartr.com/companies/nvidia-corporation_3624" not in [link.url for link in candidates]
    assert "https://investor.nvidia.com/home/default.aspx" in [link.url for link in candidates]


def test_navigation_start_urls_filter_third_party_search_results(monkeypatch) -> None:
    monkeypatch.setattr(
        "ir_transcripts.navigation.discover_ir_candidates",
        lambda *args, **kwargs: [
            IRDiscoveryCandidate(
                url="https://investor.nvidia.com/home/default.aspx",
                title="NVIDIA Investor Relations",
                score=90,
            ),
            IRDiscoveryCandidate(
                url="https://www.gainify.io/stocks/nasdaq/nvda/investor-relations",
                title="NVIDIA investor relations",
                score=80,
            ),
        ],
    )

    starts = navigation_start_urls(Company(symbol="NVDA", name="NVIDIA"))

    assert "https://investor.nvidia.com/home/default.aspx" in starts
    assert "https://www.gainify.io/stocks/nasdaq/nvda/investor-relations" not in starts


def test_navigation_start_urls_use_alphabet_homepage_for_goog(monkeypatch) -> None:
    monkeypatch.setattr("ir_transcripts.navigation.discover_ir_candidates", lambda *args, **kwargs: [])

    starts = navigation_start_urls(Company(symbol="GOOG", name="Alphabet Google"))

    assert starts[0] == "https://abc.xyz/"
    assert is_company_host("https://abc.xyz/investor/", Company(symbol="GOOG", name="Alphabet Google"))


def test_navigation_start_urls_can_disable_alphabet_homepage_override(monkeypatch) -> None:
    monkeypatch.setattr("ir_transcripts.navigation.discover_ir_candidates", lambda *args, **kwargs: [])

    starts = navigation_start_urls(
        Company(symbol="GOOG", name="Alphabet Google"),
        disable_official_homepage_overrides=True,
    )

    assert "https://abc.xyz/" not in starts


def test_discovered_urls_prioritize_financial_reports() -> None:
    ranked = rank_discovered_urls(
        [
            "https://investor.nvidia.com/investor-resources/contact-investor-relations/default.aspx",
            "https://investor.nvidia.com/financial-info/financial-reports/default.aspx",
        ],
        Company(symbol="NVDA", name="NVIDIA"),
    )

    assert ranked[0] == "https://investor.nvidia.com/financial-info/financial-reports/default.aspx"


def test_fixture_relative_financial_info_link_resolves() -> None:
    html = (FIXTURES / "navigation_nvidia_ir_home.html").read_text(encoding="utf-8")
    links = extract_links(html, "https://investor.nvidia.com/home/default.aspx")

    assert urljoin(
        "https://investor.nvidia.com/home/default.aspx",
        "/financial-info/financial-reports/default.aspx",
    ) in [link.url for link in links]


def test_event_detail_links_outrank_page_chrome() -> None:
    company = Company(symbol="GOOG", name="Alphabet Google")
    current_url = "https://abc.xyz/investor/events/default.aspx"
    event = extract_links(
        '<a href="/investor/events/event-details/2026/2026-Q1-Earnings-Call/default.aspx">2026 Q1 Earnings Call</a>',
        current_url,
    )[0]
    investors = extract_links('<a href="/investor/default.aspx">Investors</a>', current_url)[0]
    skip = extract_links('<a href="#maincontent">Skip to main content</a>', current_url)[0]

    assert navigation_link_score(event, current_url=current_url, allowed_hosts={"abc.xyz"}, company=company) > navigation_link_score(
        investors,
        current_url=current_url,
        allowed_hosts={"abc.xyz"},
        company=company,
    )
    assert navigation_link_score(skip, current_url=current_url, allowed_hosts={"abc.xyz"}, company=company) <= 0


def test_navigation_auto_renders_q4_event_listing(monkeypatch) -> None:
    static_events = """
    <html><head><title>Alphabet Investor Relations - Investors - Events</title></head>
    <body>
      <h1>Events & Presentations</h1>
      <div class="evergreen evergreen-event"></div>
      <a href="/investor/default.aspx">Investors</a>
    </body></html>
    """
    rendered_events = """
    <html><head><title>Alphabet Investor Relations - Investors - Events</title></head>
    <body>
      <a href="/investor/events/event-details/2026/2026-Q1-Earnings-Call/default.aspx">2026 Q1 Earnings Call</a>
      <a href="/investor/default.aspx">Investors</a>
    </body></html>
    """
    homepage = """
    <html><head><title>Alphabet</title></head>
    <body><a href="/investor/events/default.aspx">Events</a></body></html>
    """

    class FakeRenderer:
        def __init__(self, http) -> None:
            self.http = http

        def render_html(self, url: str) -> str:
            assert url == "https://abc.xyz/investor/events/default.aspx"
            return rendered_events

    monkeypatch.setattr("ir_transcripts.navigation.IRNavigationAgent", FakeNavigationAgent)
    monkeypatch.setattr("ir_transcripts.navigation.PlaywrightRenderer", FakeRenderer)
    monkeypatch.setattr("ir_transcripts.navigation.discover_ir_candidates", lambda *args, **kwargs: [])

    result = discover_navigation_seeds(
        Company(symbol="GOOG", name="Alphabet Google"),
        http=FakeHttp(
            {
                "https://abc.xyz/": homepage,
                "https://abc.xyz/investor/events/default.aspx": static_events,
            }
        ),  # type: ignore[arg-type]
        model="test-model",
        max_steps=2,
        playwright_mode="auto",
    )

    assert "https://abc.xyz/investor/events/event-details/2026/2026-Q1-Earnings-Call/default.aspx" in result.seeds
