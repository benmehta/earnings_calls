from pathlib import Path
import requests
from urllib.parse import urljoin

from ir_transcripts.http import RobotsUnavailableError
from ir_transcripts.models import CandidateLink, Company, CompanyNavigationMemory, HomepagePrediction, HomepageValidationDecision, NavigationDecision, PromptGuidance
from ir_transcripts.navigation import (
    can_delegate_unavailable_ir_subdomain_robots,
    discover_navigation_seeds,
    is_language_variant_url,
    is_company_host,
    navigation_link_score,
    navigation_start_score,
    navigation_start_urls,
    order_chosen_urls,
    predict_and_validate_homepage_starts,
    rank_discovered_urls,
    rank_navigation_starts,
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

    def get_without_robots_check(self, url: str) -> FakeResponse:
        return self.get(url)


class FakeNavigationAgent:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def select_agent_kind(self, *, page_context, **kwargs) -> str:
        return "homepage" if page_context == "homepage" else "ir_section"

    def decide(self, *, links, page_context, **kwargs) -> NavigationDecision:
        labels = {link.label.lower(): link.url for link in links}
        if page_context == "homepage":
            chosen = labels.get("investors") or labels.get("investor relations")
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
    captured = {}

    def fake_discover(*args, **kwargs):
        captured.update(kwargs)
        return [
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
        ]

    monkeypatch.setattr(
        "ir_transcripts.navigation.discover_ir_candidates",
        fake_discover,
    )

    starts = navigation_start_urls(Company(symbol="NVDA", name="NVIDIA"), model="test-model", ollama_base_url="http://ollama")

    assert "https://investor.nvidia.com/home/default.aspx" in starts
    assert "https://www.gainify.io/stocks/nasdaq/nvda/investor-relations" not in starts
    assert captured["rerank_model"] == "test-model"
    assert captured["ollama_base_url"] == "http://ollama"


def test_navigation_start_urls_pass_memory_and_guidance_to_discovery(monkeypatch) -> None:
    captured = {}
    memory = CompanyNavigationMemory(preferred_hosts=["investor.example.com"])
    guidance = PromptGuidance(priority_terms=["quarterly results"])

    def fake_discover(*args, **kwargs):
        captured.update(kwargs)
        return [
            IRDiscoveryCandidate(
                url="https://investor.example.com",
                title="Example Investor Relations",
                score=80,
            )
        ]

    monkeypatch.setattr("ir_transcripts.navigation.discover_ir_candidates", fake_discover)

    starts = navigation_start_urls(
        Company(symbol="EX", name="Example"),
        model="test-model",
        prompt_guidance=guidance,
        navigation_memory=memory,
    )

    assert starts[0] == "https://investor.example.com"
    assert captured["navigation_memory"] == memory
    assert captured["prompt_guidance"] == guidance


def test_navigation_start_urls_use_alphabet_homepage_for_goog(monkeypatch) -> None:
    monkeypatch.setattr("ir_transcripts.navigation.discover_ir_candidates", lambda *args, **kwargs: [])

    starts = navigation_start_urls(Company(symbol="GOOG", name="Alphabet Google"))

    assert starts[0] == "https://abc.xyz/"
    assert is_company_host("https://abc.xyz/investor/", Company(symbol="GOOG", name="Alphabet Google"))


def test_navigation_start_urls_do_not_hardwire_amzn_homepages(monkeypatch) -> None:
    monkeypatch.setattr("ir_transcripts.navigation.discover_ir_candidates", lambda *args, **kwargs: [])

    starts = navigation_start_urls(Company(symbol="AMZN", name=None))

    assert "https://www.amazon.com" not in starts
    assert "https://ir.aboutamazon.com" not in starts
    assert starts == []


def test_navigation_start_urls_can_disable_alphabet_homepage_override(monkeypatch) -> None:
    monkeypatch.setattr("ir_transcripts.navigation.discover_ir_candidates", lambda *args, **kwargs: [])

    starts = navigation_start_urls(
        Company(symbol="GOOG", name="Alphabet Google"),
        disable_official_homepage_overrides=True,
    )

    assert "https://abc.xyz/" not in starts
    assert "https://www.alphabetgoogle.com" not in starts


def test_navigation_start_urls_exclude_memory_robots_failed_hosts(monkeypatch) -> None:
    monkeypatch.setattr(
        "ir_transcripts.navigation.discover_ir_candidates",
        lambda *args, **kwargs: [
            IRDiscoveryCandidate(url="https://www.alphabetgoogle.com", title="Fake", score=100),
            IRDiscoveryCandidate(url="https://abc.xyz/investor/", title="Alphabet Investor Relations", score=80),
        ],
    )
    memory = CompanyNavigationMemory(
        robots_blocked_hosts=["www.alphabetgoogle.com"],
        low_value_hosts=["www.alphabetgoogle.com"],
    )

    starts = navigation_start_urls(
        Company(symbol="GOOG", name="Alphabet Google"),
        model="test-model",
        disable_official_homepage_overrides=True,
        navigation_memory=memory,
    )

    assert "https://www.alphabetgoogle.com" not in starts
    assert "https://abc.xyz/investor/" in starts


def test_goog_override_disabled_uses_predictive_homepage_before_search(monkeypatch) -> None:
    class FakeHomepagePredictionAgent:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def predict(self, **kwargs):
            return HomepagePrediction(homepage_urls=["https://abc.xyz/"], confidence=0.9, reason="Alphabet homepage")

    class FakeHomepageValidationAgent:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def validate(self, **kwargs):
            return HomepageValidationDecision(
                is_official=True,
                confidence=0.9,
                official_company_name="Alphabet",
                linked_ir_urls=["https://abc.xyz/investor/"],
                reason="official Alphabet evidence",
            )

    pages = {
        "https://abc.xyz/": """
        <html><head><title>Alphabet</title></head>
        <body><a href="/investor/">Investor relations</a></body></html>
        """,
        "https://abc.xyz/investor/": """
        <html><head><title>Alphabet Investor Relations</title></head>
        <body><a href="/investor/events/default.aspx">Events</a></body></html>
        """,
    }
    monkeypatch.setattr("ir_transcripts.navigation.HomepagePredictionAgent", FakeHomepagePredictionAgent)
    monkeypatch.setattr("ir_transcripts.navigation.HomepageValidationAgent", FakeHomepageValidationAgent)
    monkeypatch.setattr("ir_transcripts.navigation.IRNavigationAgent", FakeNavigationAgent)
    monkeypatch.setattr("ir_transcripts.navigation.discover_ir_candidates", lambda *args, **kwargs: [])

    result = discover_navigation_seeds(
        Company(symbol="GOOG", name="Alphabet Google"),
        http=FakeHttp(pages),  # type: ignore[arg-type]
        model="test-model",
        max_steps=1,
        disable_official_homepage_overrides=True,
    )

    assert result.verified_homepage_urls == ["https://abc.xyz/"]
    assert result.verified_company_name == "Alphabet"
    assert result.trace.steps[0].current_url == "https://abc.xyz/"


def test_navigation_start_ranking_prefers_broad_ir_home_over_deep_or_blog_urls() -> None:
    company = Company(symbol="GOOG", name="Alphabet Google")
    ranked = rank_navigation_starts(
        [
            "https://www.alphabetgoogle.com",
            "https://alphabet2025ir.q4web.com/default.aspx",
            "https://abc.xyz/investor/sec-filings/default.aspx",
            "https://blog.google/alphabet/investor-presentation-june-2026/",
            "https://abc.xyz/investor/",
        ],
        company,
    )

    assert ranked[0] == "https://abc.xyz/investor/"
    assert navigation_start_score("https://abc.xyz/investor/", company) > navigation_start_score(
        "https://abc.xyz/investor/sec-filings/default.aspx",
        company,
    )


def test_navigation_start_ranking_uses_structured_memory_generically() -> None:
    company = Company(symbol="EX", name="Example")
    memory = CompanyNavigationMemory(
        preferred_hosts=["ir.example.com"],
        known_ir_home_urls=["https://ir.example.com/investors"],
        low_value_hosts=["blog.example.com"],
    )
    ranked = rank_navigation_starts(
        [
            "https://blog.example.com/earnings-q1",
            "https://ir.example.com/investors/sec-filings",
            "https://ir.example.com/investors",
        ],
        company,
        navigation_memory=memory,
    )

    assert ranked[0] == "https://ir.example.com/investors"


def test_navigation_start_ranking_memory_can_correct_search_order() -> None:
    company = Company(symbol="TSM", name="Taiwan Semiconductor Manufacturing Company")
    memory = CompanyNavigationMemory(
        preferred_hosts=["investor.tsmc.com"],
        successful_hosts=["investor.tsmc.com"],
        low_value_hosts=["www.taiwansemi.com"],
    )
    ranked = rank_navigation_starts(
        [
            "https://www.taiwansemi.com/en/investor-relations/",
            "https://investor.tsmc.com/english",
        ],
        company,
        navigation_memory=memory,
    )

    assert ranked[0] == "https://investor.tsmc.com/english"


def test_discovered_urls_prioritize_financial_reports() -> None:
    ranked = rank_discovered_urls(
        [
            "https://investor.nvidia.com/investor-resources/contact-investor-relations/default.aspx",
            "https://investor.nvidia.com/financial-info/financial-reports/default.aspx",
        ],
        Company(symbol="NVDA", name="NVIDIA"),
    )

    assert ranked[0] == "https://investor.nvidia.com/financial-info/financial-reports/default.aspx"


def test_discovered_urls_use_memory_preferred_host() -> None:
    company = Company(symbol="TSM", name="Taiwan Semiconductor Manufacturing Company")
    memory = CompanyNavigationMemory(preferred_hosts=["investor.tsmc.com"])

    ranked = rank_discovered_urls(
        [
            "https://www.taiwansemi.com/en/investor-relations/financial-reports/",
            "https://investor.tsmc.com/english/quarterly-results",
        ],
        company,
        navigation_memory=memory,
    )

    assert ranked[0] == "https://investor.tsmc.com/english/quarterly-results"


def test_navigation_candidate_links_reject_language_variants() -> None:
    links = extract_links(
        """
        <a href="/english/quarterly-results">Quarterly Results</a>
        <a href="/schinese/quarterly-results">Quarterly Results Chinese</a>
        <a href="/japanese/quarterly-results">Quarterly Results Japanese</a>
        """,
        "https://investor.example.com/english",
    )

    candidates = navigation_candidate_links(
        links,
        current_url="https://investor.example.com/english",
        allowed_hosts={"investor.example.com"},
        company=Company(symbol="EX", name="Example"),
        limit=10,
    )

    assert "https://investor.example.com/english/quarterly-results" in [link.url for link in candidates]
    assert all(not is_language_variant_url(link.url) for link in candidates)


def test_navigation_discovery_uses_homepage_content_to_enqueue_ir_link(monkeypatch) -> None:
    pages = {
        "https://www.example.com": """
        <html><head><title>Example Company</title></head>
        <body>
          <footer>Copyright Example. All rights reserved.</footer>
          <a href="https://ir.example.com">Investor Relations</a>
        </body></html>
        """,
    }
    monkeypatch.setattr("ir_transcripts.navigation.IRNavigationAgent", FakeNavigationAgent)
    monkeypatch.setattr("ir_transcripts.navigation.discover_ir_candidates", lambda *args, **kwargs: [])

    result = discover_navigation_seeds(
        Company(symbol="EX", name="Example"),
        http=FakeHttp(pages),  # type: ignore[arg-type]
        model="test-model",
        max_steps=1,
    )

    assert "https://ir.example.com" in result.seeds


def test_navigation_trace_records_candidates_when_agent_fails(monkeypatch) -> None:
    class FailingNavigationAgent(FakeNavigationAgent):
        def decide(self, **kwargs) -> NavigationDecision:
            raise TimeoutError("local model timed out")

    pages = {
        "https://www.example.com": """
        <html><head><title>Example Company</title></head>
        <body>
          <footer>Copyright Example. All rights reserved.</footer>
          <a href="/investors">Investor Relations</a>
          <a href="/careers">Careers</a>
        </body></html>
        """,
    }
    monkeypatch.setattr("ir_transcripts.navigation.IRNavigationAgent", FailingNavigationAgent)
    monkeypatch.setattr("ir_transcripts.navigation.discover_ir_candidates", lambda *args, **kwargs: [])

    result = discover_navigation_seeds(
        Company(symbol="EX", name="Example"),
        http=FakeHttp(pages),  # type: ignore[arg-type]
        model="test-model",
        max_steps=1,
    )

    assert result.failure_type == "navigation_llm_failed"
    assert result.trace.steps[0].candidate_urls == ["https://www.example.com/investors"]
    assert result.trace.steps[0].candidate_links[0].url == "https://www.example.com/investors"
    assert result.trace.steps[0].candidate_links[0].label == "Investor Relations"
    assert result.trace.steps[0].candidate_links[0].context == "body"
    assert result.trace.steps[0].candidate_links[0].score > 0
    assert result.trace.steps[0].raw_link_count == 2
    assert result.trace.steps[0].stop_reason == "navigation_llm_failed"


def test_navigation_renders_verified_official_domain_after_http_error(monkeypatch) -> None:
    class FailingSubpageHttp(FakeHttp):
        def get(self, url: str) -> FakeResponse:
            if url == "https://www.example.com/investors":
                raise requests.HTTPError("500 Server Error")
            return super().get(url)

    class FakeRenderer:
        def __init__(self, http) -> None:
            self.http = http

        def render_html(self, url: str) -> str:
            assert url == "https://www.example.com/investors"
            return """
            <html><head><title>Example Investor Relations</title></head>
            <body>
              <a href="/events/q1-transcript.pdf">Q1 FY27 Earnings Call Transcript</a>
            </body></html>
            """

    pages = {
        "https://www.example.com": """
        <html><head><title>Example Company</title></head>
        <body>
          <footer>Copyright Example. All rights reserved.</footer>
          <a href="/investors">Investor Relations</a>
        </body></html>
        """,
    }
    monkeypatch.setattr("ir_transcripts.navigation.IRNavigationAgent", FakeNavigationAgent)
    monkeypatch.setattr("ir_transcripts.navigation.PlaywrightRenderer", FakeRenderer)
    monkeypatch.setattr("ir_transcripts.navigation.discover_ir_candidates", lambda *args, **kwargs: [])

    result = discover_navigation_seeds(
        Company(symbol="EX", name="Example"),
        http=FailingSubpageHttp(pages),  # type: ignore[arg-type]
        model="test-model",
        max_steps=2,
        playwright_mode="auto",
    )

    assert result.failure_type is None
    assert result.trace.steps[1].current_url == "https://www.example.com/investors"
    assert result.trace.steps[1].render_strategy == "playwright_after_http_error"
    assert "https://www.example.com/events/q1-transcript.pdf" in result.seeds


def test_navigation_delegates_unavailable_robots_for_official_ir_subdomain(monkeypatch) -> None:
    class DelegatingFakeHttp(FakeHttp):
        def __init__(self, pages: dict[str, str]) -> None:
            super().__init__(pages)
            self.delegated_fetches: list[str] = []

        def get(self, url: str) -> FakeResponse:
            if url == "https://ir.example.com/earnings":
                raise RobotsUnavailableError(f"Could not verify robots.txt for {url}")
            return super().get(url)

        def get_without_robots_check(self, url: str) -> FakeResponse:
            self.delegated_fetches.append(url)
            return FakeResponse(self.pages[url])

    http = DelegatingFakeHttp(
        {
            "https://www.example.com": """
            <html><head><title>Example Company</title></head>
            <body>
              <footer>Copyright Example. All rights reserved.</footer>
              <a href="https://ir.example.com/earnings">Investor Relations</a>
            </body></html>
            """,
            "https://ir.example.com/earnings": """
            <html><head><title>Example Investor Relations - Earnings</title></head>
            <body>
              <a href="https://ir.example.com/q1-transcript.pdf">Q1 FY27 Earnings Call Transcript</a>
            </body></html>
            """,
        }
    )
    monkeypatch.setattr("ir_transcripts.navigation.IRNavigationAgent", FakeNavigationAgent)
    monkeypatch.setattr("ir_transcripts.navigation.discover_ir_candidates", lambda *args, **kwargs: [])

    result = discover_navigation_seeds(
        Company(symbol="EX", name="Example"),
        http=http,  # type: ignore[arg-type]
        model="test-model",
        max_steps=2,
    )

    assert http.delegated_fetches == ["https://ir.example.com/earnings"]
    assert "https://ir.example.com/q1-transcript.pdf" in result.seeds


def test_navigation_does_not_delegate_unavailable_robots_to_unrelated_or_non_ir_hosts() -> None:
    assert can_delegate_unavailable_ir_subdomain_robots(
        "https://investors.example.com/earnings",
        verified_official_urls=["https://www.example.com"],
    )
    assert not can_delegate_unavailable_ir_subdomain_robots(
        "https://investors.other-example.com/earnings",
        verified_official_urls=["https://www.example.com"],
    )
    assert not can_delegate_unavailable_ir_subdomain_robots(
        "https://news.example.com/earnings",
        verified_official_urls=["https://www.example.com"],
    )


def test_navigation_fetch_failure_stops_current_attempt(monkeypatch) -> None:
    monkeypatch.setattr("ir_transcripts.navigation.discover_ir_candidates", lambda *args, **kwargs: [])

    result = discover_navigation_seeds(
        Company(symbol="EX", name="Example"),
        http=FakeHttp({}),  # type: ignore[arg-type]
        model="test-model",
    )

    assert result.seeds == []
    assert result.failure_type == "navigation_fetch_failed"
    assert result.failure_urls == ["https://www.example.com"]


def test_verified_homepage_name_guides_navigation_host_matching(monkeypatch) -> None:
    class FakeHomepagePredictionAgent:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def predict(self, **kwargs):
            return HomepagePrediction(
                homepage_urls=["https://www.amazon.com"],
                confidence=0.9,
                reason="Amazon ticker",
            )

    class FakeHomepageValidationAgent:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def validate(self, **kwargs):
            return HomepageValidationDecision(
                is_official=True,
                confidence=0.9,
                official_company_name="Amazon",
                linked_ir_urls=["https://www.amazon.com/ir"],
                reason="official Amazon homepage evidence",
            )

    monkeypatch.setattr("ir_transcripts.navigation.HomepagePredictionAgent", FakeHomepagePredictionAgent)
    monkeypatch.setattr("ir_transcripts.navigation.HomepageValidationAgent", FakeHomepageValidationAgent)
    monkeypatch.setattr("ir_transcripts.navigation.IRNavigationAgent", FakeNavigationAgent)
    monkeypatch.setattr("ir_transcripts.navigation.discover_ir_candidates", lambda *args, **kwargs: [])

    result = discover_navigation_seeds(
        Company(symbol="AMZN", name=None),
        http=FakeHttp(
            {
                "https://www.amazon.com": """
                <html><head><title>Amazon.com</title></head>
                <body><a href="/ir">Investor Relations</a></body></html>
                """,
                "https://www.amazon.com/ir": """
                <html><head><title>Amazon Investor Relations</title></head>
                <body>
                  <a href="https://ir.aboutamazon.com/quarterly-results/default.aspx">Quarterly results</a>
                </body></html>
                """,
            }
        ),  # type: ignore[arg-type]
        model="test-model",
        max_steps=2,
        disable_official_homepage_overrides=True,
    )

    assert result.trace.company.name == "Amazon"
    assert "https://ir.aboutamazon.com/quarterly-results/default.aspx" in result.seeds


def test_predictive_homepage_uses_ticker_without_name_hint_and_discards_ir_predictions(monkeypatch) -> None:
    captured = {}

    class FakeHomepagePredictionAgent:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def predict(self, **kwargs):
            captured.update(kwargs)
            return HomepagePrediction(
                homepage_urls=[
                    "https://www.amazon.com",
                    "https://ir.aboutamazon.com",
                ],
                confidence=0.9,
                reason="Amazon ticker",
            )

    class FakeHomepageValidationAgent:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def validate(self, **kwargs):
            assert kwargs["name_hint"] is None
            return HomepageValidationDecision(
                is_official=True,
                confidence=0.9,
                official_company_name="Amazon",
                linked_ir_urls=["https://ir.aboutamazon.com"],
                reason="official Amazon homepage evidence",
            )

    monkeypatch.setattr("ir_transcripts.navigation.HomepagePredictionAgent", FakeHomepagePredictionAgent)
    monkeypatch.setattr("ir_transcripts.navigation.HomepageValidationAgent", FakeHomepageValidationAgent)

    result = predict_and_validate_homepage_starts(
        Company(symbol="AMZN", name=None),
        http=FakeHttp(
            {
                "https://www.amazon.com": """
                <html><head><title>Amazon.com</title></head>
                <body><a href="https://ir.aboutamazon.com">Investor Relations</a></body></html>
                """,
            }
        ),  # type: ignore[arg-type]
        model="test-model",
    )

    assert captured == {"ticker": "AMZN", "name_hint": None}
    assert "https://www.amazon.com" in result.starts
    assert "https://ir.aboutamazon.com" in result.starts
    assert result.verified_homepage_urls == ["https://www.amazon.com"]
    assert result.verified_company_name == "Amazon"


def test_predictive_homepage_passes_memory_name_hint(monkeypatch) -> None:
    captured = {}

    class FakeHomepagePredictionAgent:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def predict(self, **kwargs):
            captured.update(kwargs)
            return HomepagePrediction(homepage_urls=[], confidence=0.1, reason="uncertain")

    monkeypatch.setattr("ir_transcripts.navigation.HomepagePredictionAgent", FakeHomepagePredictionAgent)

    result = predict_and_validate_homepage_starts(
        Company(symbol="AMZN", name=None),
        http=FakeHttp({}),  # type: ignore[arg-type]
        model="test-model",
        identity_name_hint="Amazon.com",
    )

    assert captured == {"ticker": "AMZN", "name_hint": "Amazon.com"}
    assert result.failure_type == "identity_low_confidence"


def test_predictive_homepage_rejects_ir_only_prediction(monkeypatch) -> None:
    class FakeHomepagePredictionAgent:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def predict(self, **kwargs):
            return HomepagePrediction(
                homepage_urls=["https://ir.aboutamazon.com"],
                confidence=0.9,
                reason="bad IR-only prediction",
            )

    monkeypatch.setattr("ir_transcripts.navigation.HomepagePredictionAgent", FakeHomepagePredictionAgent)

    result = predict_and_validate_homepage_starts(
        Company(symbol="AMZN", name=None),
        http=FakeHttp({}),  # type: ignore[arg-type]
        model="test-model",
    )

    assert result.failure_type == "homepage_unverified"


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


def test_earnings_event_detail_outranks_sec_filings() -> None:
    company = Company(symbol="GOOG", name="Alphabet Google")
    current_url = "https://abc.xyz/investor/"
    event = extract_links(
        '<a href="/investor/events/event-details/2026/2026-Q1-Earnings-Call/default.aspx">2026 Q1 Earnings Call</a>',
        current_url,
    )[0]
    sec = extract_links(
        '<a href="/investor/sec-filings/default.aspx">SEC Filings</a>',
        current_url,
    )[0]

    assert navigation_link_score(event, current_url=current_url, allowed_hosts={"abc.xyz"}, company=company) > navigation_link_score(
        sec,
        current_url=current_url,
        allowed_hosts={"abc.xyz"},
        company=company,
    )


def test_order_chosen_urls_preserves_agent_preference() -> None:
    links = [
        CandidateLink(
            url="https://abc.xyz/investor/sec-filings/default.aspx",
            label="SEC Filings",
            source_url="https://abc.xyz/investor/",
        ),
        CandidateLink(
            url="https://abc.xyz/investor/events/event-details/2026/2026-Q1-Earnings-Call/default.aspx",
            label="2026 Q1 Earnings Call",
            source_url="https://abc.xyz/investor/",
        ),
    ]

    ordered = order_chosen_urls(
        [
            "https://abc.xyz/investor/events/event-details/2026/2026-Q1-Earnings-Call/default.aspx",
            "https://abc.xyz/investor/sec-filings/default.aspx",
        ],
        links,
    )

    assert ordered == [
        "https://abc.xyz/investor/events/event-details/2026/2026-Q1-Earnings-Call/default.aspx",
        "https://abc.xyz/investor/sec-filings/default.aspx",
    ]


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


def test_navigation_browser_ua_fallback_for_sparse_official_domain_homepage(monkeypatch) -> None:
    sparse_homepage = """
    <html><head><title>Amazon.com</title></head>
    <body>Amazon.com Continue shopping Conditions of Use Privacy Policy</body></html>
    """
    full_homepage = """
    <html><head><title>Amazon.com. Spend less. Smile more.</title></head>
    <body>
      <footer>
        <a href="/ir">Investor Relations</a>
        <a href="https://www.aboutamazon.com/">About Amazon</a>
      </footer>
    </body></html>
    """

    class FakeRenderer:
        def __init__(self, http) -> None:
            self.http = http

        def render_html(self, url: str) -> str:
            assert url == "https://www.amazon.com"
            return sparse_homepage

        def render_html_with_user_agent(self, url: str, user_agent: str) -> str:
            assert url == "https://www.amazon.com"
            assert "Chrome" in user_agent
            return full_homepage

    monkeypatch.setattr("ir_transcripts.navigation.IRNavigationAgent", FakeNavigationAgent)
    monkeypatch.setattr("ir_transcripts.navigation.PlaywrightRenderer", FakeRenderer)
    monkeypatch.setattr("ir_transcripts.navigation.discover_ir_candidates", lambda *args, **kwargs: [])

    result = discover_navigation_seeds(
        Company(symbol="AMZN", name="Amazon"),
        http=FakeHttp({"https://www.amazon.com": sparse_homepage}),  # type: ignore[arg-type]
        model="test-model",
        max_steps=1,
        playwright_mode="always",
        disable_official_homepage_overrides=True,
    )

    assert result.trace.steps[0].render_strategy == "browser_ua_fallback"
    assert result.trace.steps[0].chosen_urls == ["https://www.amazon.com/ir"]
    assert "https://www.amazon.com/ir" in result.seeds
