from io import BytesIO
from zipfile import ZipFile

from ir_transcripts.crawler import (
    TranscriptCrawler,
    artifact_stem,
    content_hash,
    crawl_identity_url,
    fiscal_period_key,
    is_docx_response,
    is_non_english_variant,
    keep_latest_transcripts,
    link_score,
    low_value_after_transcript_document,
)
from ir_transcripts.http import RobotsDisallowedError, RobotsUnavailableError
from ir_transcripts.models import CandidateLink, Company, NavigationTrace, PageDecision, TranscriptRecord
from ir_transcripts.navigation import NavigationDiscoveryResult


def test_link_score_prefers_transcripts() -> None:
    transcript = CandidateLink(
        url="https://investors.example.com/q1-earnings-call-transcript",
        label="Q1 earnings call transcript",
        source_url="https://investors.example.com",
    )
    careers = CandidateLink(
        url="https://example.com/careers",
        label="Careers",
        source_url="https://investors.example.com",
    )

    assert link_score(transcript) > link_score(careers)


def test_heuristic_links_prioritize_transcript_document_after_page_chrome(tmp_path) -> None:
    chrome_links = [
        CandidateLink(
            url=f"https://investor.example.com/financial-info/archive-{index}",
            label=f"Financial archive {index}",
            source_url="https://investor.example.com/financial-reports",
        )
        for index in range(40)
    ]
    transcript = CandidateLink(
        url="https://cdn.example.com/files/doc_financials/2027/q1/EX-Q1-2027-Earnings-Call.pdf",
        label="Q1 Transcript 2027 (opens in new window)",
        source_url="https://investor.example.com/financial-reports",
    )
    crawler = TranscriptCrawler(model="test-model", out_dir=tmp_path, seed_urls=["https://investor.example.com"])

    prioritized = crawler._heuristic_links([*chrome_links, transcript])

    assert prioritized[0] == transcript
    assert transcript in prioritized


def test_low_value_after_transcript_document_keeps_transcript_links() -> None:
    sec = CandidateLink(
        url="https://investor.example.com/financial-info/sec-filings/default.aspx",
        label="SEC Filings",
        source_url="https://investor.example.com",
    )
    presentation = CandidateLink(
        url="https://investor.example.com/events-and-presentations/presentations/default.aspx",
        label="Presentations",
        source_url="https://investor.example.com",
    )
    transcript = CandidateLink(
        url="https://cdn.example.com/files/Example-Q1-Earnings-Call-Transcript.pdf",
        label="Q1 transcript",
        source_url="https://investor.example.com",
    )

    assert low_value_after_transcript_document(sec)
    assert low_value_after_transcript_document(presentation)
    assert not low_value_after_transcript_document(transcript)


def test_artifact_stem_and_content_hash_are_stable() -> None:
    assert artifact_stem("Q1 Transcript", "https://example.com/a") == artifact_stem(
        "Q1 Transcript", "https://example.com/a"
    )
    assert content_hash("hello   world") == content_hash("hello world")


def test_crawl_identity_url_ignores_http_https_scheme() -> None:
    assert crawl_identity_url("http://example.com/investors/") == crawl_identity_url("https://example.com/investors")


def test_non_english_variant_is_skipped_from_english_page() -> None:
    assert is_non_english_variant(
        "https://investor.example.com/japanese/quarterly-results/2026/q1",
        "https://investor.example.com/english/quarterly-results/2026/q1",
    )
    assert not is_non_english_variant(
        "https://investor.example.com/english/quarterly-results/2026/q2",
        "https://investor.example.com/english/quarterly-results/2026/q1",
    )


def test_fiscal_period_key_handles_common_quarter_shapes() -> None:
    assert fiscal_period_key("2026 Q1 Earnings Call") == (2026, 1)
    assert fiscal_period_key("Q4 FY2025 transcript") == (2025, 4)
    assert fiscal_period_key("first quarter 2026 earnings") == (2026, 1)


def test_keep_latest_transcripts_prefers_newest_fiscal_period() -> None:
    old = TranscriptRecord(
        company=Company(symbol="EX", name="Example"),
        source_url="https://example.com/2025",
        fiscal_period="Q1 2025",
        title="Example Q1 2025 Earnings Call",
        text="OPERATOR: Welcome.",
    )
    latest = TranscriptRecord(
        company=Company(symbol="EX", name="Example"),
        source_url="https://example.com/2026",
        fiscal_period="Q1 2026",
        title="Example Q1 2026 Earnings Call",
        text="OPERATOR: Welcome.",
    )

    assert keep_latest_transcripts([old, latest]) == [latest]


def test_is_docx_response_detects_word_content_type_without_extension() -> None:
    assert is_docx_response(
        "https://cdn.example.com/is/content/example/Transcript",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        b"",
    )


def test_crawler_records_robots_unavailable(tmp_path) -> None:
    class FakeHttp:
        def get(self, url: str):
            raise RobotsUnavailableError(f"Could not verify robots.txt for {url}")

    crawler = TranscriptCrawler(
        model="test-model",
        out_dir=tmp_path,
        seed_urls=["https://example.com/investors"],
        http=FakeHttp(),  # type: ignore[arg-type]
    )

    result = crawler.crawl_company(Company(symbol="EX", name="Example"))

    assert result.failures[0].failure_type == "robots_unavailable"


def test_crawler_records_robots_disallowed(tmp_path) -> None:
    class FakeHttp:
        def get(self, url: str):
            raise RobotsDisallowedError(f"Disallowed by robots.txt: {url}")

    crawler = TranscriptCrawler(
        model="test-model",
        out_dir=tmp_path,
        seed_urls=["https://example.com/private"],
        http=FakeHttp(),  # type: ignore[arg-type]
    )

    result = crawler.crawl_company(Company(symbol="EX", name="Example"))

    assert result.failures[0].failure_type == "robots_disallowed"


def test_crawler_does_not_save_press_release_like_html(tmp_path) -> None:
    class FakeResponse:
        headers = {"content-type": "text/html"}
        content = b""
        text = """
        <html><head><title>FY26 Q3 - Press Releases - Investor Relations - Microsoft</title></head>
        <body>
        <h1>Press Release & Webcast</h1>
        <p>Business Highlights include cloud revenue and Azure growth.</p>
        <p>Webcast Details: executives will host a conference call for analysts.</p>
        <p>Forward-looking statements and non-GAAP financial measures follow.</p>
        </body></html>
        """

    class FakeHttp:
        def get(self, url: str):
            return FakeResponse()

    class FakeAgent:
        def decide(self, **kwargs):
            return PageDecision(page_type="transcript", confidence=1.0, reason="LLM thought transcript")

    crawler = TranscriptCrawler(
        model="test-model",
        out_dir=tmp_path,
        seed_urls=["https://example.com/press-release-webcast"],
        http=FakeHttp(),  # type: ignore[arg-type]
    )
    crawler.agent = FakeAgent()  # type: ignore[assignment]

    result = crawler.crawl_company(Company(symbol="EX", name="Example"))

    assert result.transcripts == []
    assert result.candidates[0].reason.endswith("transcript_rejected_press_release_like")
    assert not [path for path in (tmp_path / "EX").glob("*.json") if not path.name.startswith("_")]


def test_crawler_always_playwright_mode_renders_plain_html(monkeypatch, tmp_path) -> None:
    class FakeResponse:
        headers = {"content-type": "text/html"}
        content = b""
        text = """
        <html><head><title>Example Events</title></head>
        <body><p>This static page has enough text that auto mode would not need rendering.</p></body></html>
        """

    class FakeHttp:
        def get(self, url: str):
            return FakeResponse()

    class FakeRenderer:
        def __init__(self, http) -> None:
            self.http = http

        def render_html(self, url: str) -> str:
            return """
            <html><head><title>Example Q1 Earnings Call Transcript</title></head>
            <body>
            <p>OPERATOR: Welcome to the call.</p>
            <p>JANE DOE: Thank you.</p>
            <p>JOHN SMITH: Prepared remarks.</p>
            <p>ANALYST: My question is about margins.</p>
            <p>QUESTION-AND-ANSWER SESSION</p>
            <p>END</p>
            </body></html>
            """

    class FakeAgent:
        def decide(self, **kwargs):
            return PageDecision(page_type="transcript", confidence=1.0, reason="rendered transcript")

    monkeypatch.setattr("ir_transcripts.crawler.PlaywrightRenderer", FakeRenderer)
    crawler = TranscriptCrawler(
        model="test-model",
        out_dir=tmp_path,
        seed_urls=["https://investor.example.com/events/q1"],
        http=FakeHttp(),  # type: ignore[arg-type]
        playwright_mode="always",
    )
    crawler.agent = FakeAgent()  # type: ignore[assignment]

    result = crawler.crawl_company(Company(symbol="EX", name="Example"))

    assert len(result.transcripts) == 1
    assert result.transcripts[0].metadata["rendered_with_playwright"] == "true"


def test_crawler_saves_docx_transcript(tmp_path) -> None:
    class FakeResponse:
        headers = {"content-type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"}

        def __init__(self, content: bytes) -> None:
            self.content = content
            self.text = ""

    class FakeHttp:
        def __init__(self, content: bytes) -> None:
            self.content = content

        def get(self, url: str):
            return FakeResponse(self.content)

    content = docx_fixture(
        [
            "Example Co Q1 Earnings Call Transcript",
            "OPERATOR: Welcome to the call.",
            "JANE DOE: Thank you.",
            "JOHN SMITH: I will discuss results.",
            "ANALYST: My question is about margins.",
            "QUESTION-AND-ANSWER SESSION",
            "END",
        ]
    )
    crawler = TranscriptCrawler(
        model="test-model",
        out_dir=tmp_path,
        seed_urls=["https://cdn.example.com/ExampleTranscript"],
        http=FakeHttp(content),  # type: ignore[arg-type]
    )

    result = crawler.crawl_company(Company(symbol="EX", name="Example"))

    assert len(result.transcripts) == 1
    assert result.transcripts[0].metadata["format"] == "docx"
    assert list((tmp_path / "EX").glob("*.docx"))


def test_official_linked_docx_can_fetch_when_document_robots_unavailable_with_flag(tmp_path) -> None:
    docx_content = docx_fixture(
        [
            "Example Co Q1 Earnings Call Transcript",
            "OPERATOR: Welcome to the call.",
            "JANE DOE: Thank you.",
            "JOHN SMITH: I will discuss results.",
            "ANALYST: My question is about margins.",
            "QUESTION-AND-ANSWER SESSION",
            "END",
        ]
    )

    class FakeResponse:
        def __init__(self, text: str = "", content: bytes = b"", content_type: str = "text/html") -> None:
            self.text = text
            self.content = content
            self.headers = {"content-type": content_type}

    class FakeHttp:
        def get(self, url: str):
            if url == "https://investor.example.com/events/q1":
                return FakeResponse(
                    """
                    <html><head><title>Example Q1 Earnings</title></head>
                    <body>
                      <a href="https://cdn.example.com/Example-Q1-Earnings-Call-Transcript.docx">
                        Q1 Earnings Call Transcript
                      </a>
                    </body></html>
                    """
                )
            raise RobotsUnavailableError(f"Could not verify robots.txt for {url}")

        def robots_unavailable(self, url: str) -> bool:
            return url == "https://cdn.example.com/Example-Q1-Earnings-Call-Transcript.docx"

        def get_without_robots_check(self, url: str):
            return FakeResponse(
                content=docx_content,
                content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )

    class FakeAgent:
        def decide(self, **kwargs):
            return PageDecision(page_type="earnings_event", confidence=0.7, reason="earnings page", useful_links=[])

    crawler = TranscriptCrawler(
        model="test-model",
        out_dir=tmp_path,
        seed_urls=["https://investor.example.com/events/q1"],
        http=FakeHttp(),  # type: ignore[arg-type]
        allow_official_linked_documents_on_robots_unavailable=True,
    )
    crawler.agent = FakeAgent()  # type: ignore[assignment]

    result = crawler.crawl_company(Company(symbol="EX", name="Example"))

    assert len(result.transcripts) == 1
    assert result.transcripts[0].source_url == "https://cdn.example.com/Example-Q1-Earnings-Call-Transcript.docx"
    assert any(candidate.reason == "robots_unavailable_allowed_official_linked_document" for candidate in result.candidates)


def test_official_linked_docx_still_blocks_when_delegated_policy_disabled(tmp_path) -> None:
    class FakeResponse:
        headers = {"content-type": "text/html"}
        content = b""
        text = """
        <html><head><title>Example Q1 Earnings</title></head>
        <body>
          <a href="https://cdn.example.com/Example-Q1-Earnings-Call-Transcript.docx">
            Q1 Earnings Call Transcript
          </a>
        </body></html>
        """

    class FakeHttp:
        def get(self, url: str):
            if url == "https://investor.example.com/events/q1":
                return FakeResponse()
            raise RobotsUnavailableError(f"Could not verify robots.txt for {url}")

    class FakeAgent:
        def decide(self, **kwargs):
            return PageDecision(page_type="earnings_event", confidence=0.7, reason="earnings page", useful_links=[])

    crawler = TranscriptCrawler(
        model="test-model",
        out_dir=tmp_path,
        seed_urls=["https://investor.example.com/events/q1"],
        http=FakeHttp(),  # type: ignore[arg-type]
    )
    crawler.agent = FakeAgent()  # type: ignore[assignment]

    result = crawler.crawl_company(Company(symbol="EX", name="Example"))

    assert result.transcripts == []
    assert [failure.failure_type for failure in result.failures] == ["robots_unavailable"]


def test_latest_only_prunes_low_value_queue_after_transcript_document(monkeypatch, tmp_path) -> None:
    class FakeResponse:
        def __init__(self, text: str = "", content: bytes = b"", content_type: str = "text/html") -> None:
            self.text = text
            self.content = content
            self.headers = {"content-type": content_type}

    pdf_content = b"%PDF transcript fixture"
    fetched: list[str] = []

    class FakeHttp:
        def get(self, url: str):
            fetched.append(url)
            if url == "https://investor.example.com/quarterly-results":
                return FakeResponse(
                    """
                    <html><head><title>Quarterly Results</title></head>
                    <body>
                      <a href="https://cdn.example.com/Example-Q1-Earnings-Call-Transcript.pdf">Q1 Transcript</a>
                      <a href="https://investor.example.com/financial-info/sec-filings/default.aspx">SEC Filings</a>
                      <a href="https://investor.example.com/financial-info/annual-meeting/default.aspx">Annual Meeting</a>
                    </body></html>
                    """
                )
            if url == "https://cdn.example.com/Example-Q1-Earnings-Call-Transcript.pdf":
                return FakeResponse(content=pdf_content, content_type="application/pdf")
            return FakeResponse("<html><head><title>Low Value</title></head><body></body></html>")

    class FakeAgent:
        def decide(self, **kwargs):
            return PageDecision(page_type="earnings_event", confidence=0.7, reason="earnings page", useful_links=[])

    monkeypatch.setattr(
        "ir_transcripts.crawler.pdf_text",
        lambda content: (
            "Example Co Q1 2027 Earnings Call Transcript\n"
            "OPERATOR: Welcome to the call.\n"
            "JANE DOE: Thank you.\n"
            "JOHN SMITH: Prepared remarks.\n"
            "ANALYST: My question is about margins.\n"
            "QUESTION-AND-ANSWER SESSION\n"
            "END"
        ),
    )
    crawler = TranscriptCrawler(
        model="test-model",
        out_dir=tmp_path,
        seed_urls=["https://investor.example.com/quarterly-results"],
        http=FakeHttp(),  # type: ignore[arg-type]
        latest_only=True,
        max_pages_per_company=10,
        max_depth=2,
    )
    crawler.agent = FakeAgent()  # type: ignore[assignment]

    result = crawler.crawl_company(Company(symbol="EX", name="Example"))

    assert len(result.transcripts) == 1
    assert "https://investor.example.com/financial-info/sec-filings/default.aspx" not in fetched
    assert "https://investor.example.com/financial-info/annual-meeting/default.aspx" not in fetched


def test_latest_only_stops_after_first_saved_transcript_document(monkeypatch, tmp_path) -> None:
    class FakeResponse:
        def __init__(self, text: str = "", content: bytes = b"", content_type: str = "text/html") -> None:
            self.text = text
            self.content = content
            self.headers = {"content-type": content_type}

    fetched: list[str] = []

    class FakeHttp:
        def get(self, url: str):
            fetched.append(url)
            if url == "https://investor.example.com/quarterly-results":
                return FakeResponse(
                    """
                    <html><head><title>Quarterly Results</title></head>
                    <body>
                      <a href="https://investor.example.com/q1-transcript.pdf">Q1 Transcript</a>
                      <a href="https://investor.example.com/quarterly-results/2025/q4">Older quarter</a>
                    </body></html>
                    """
                )
            if url == "https://investor.example.com/q1-transcript.pdf":
                return FakeResponse(content=b"%PDF transcript fixture", content_type="application/pdf")
            return FakeResponse("<html><head><title>Older quarter</title></head><body></body></html>")

    class FakeAgent:
        def decide(self, **kwargs):
            return PageDecision(page_type="earnings_event", confidence=0.7, reason="earnings page", useful_links=[])

    monkeypatch.setattr(
        "ir_transcripts.crawler.pdf_text",
        lambda content: (
            "LSEG STREETEVENTS EDITED TRANSCRIPT\n"
            "Example Co Q1 2027 Earnings Call\n"
            "CORPORATE PARTICIPANTS\n"
            "JANE DOE Example Co - CFO\n"
            "CONFERENCE CALL PARTICIPANTS\n"
            "Analyst One\n"
            "PRESENTATION\n"
            "Operator Welcome.\n"
            "QUESTION AND ANSWER\n"
            "END"
        ),
    )
    crawler = TranscriptCrawler(
        model="test-model",
        out_dir=tmp_path,
        seed_urls=["https://investor.example.com/quarterly-results"],
        http=FakeHttp(),  # type: ignore[arg-type]
        latest_only=True,
        max_pages_per_company=10,
        max_depth=2,
    )
    crawler.agent = FakeAgent()  # type: ignore[assignment]

    result = crawler.crawl_company(Company(symbol="EX", name="Example"))

    assert len(result.transcripts) == 1
    assert "https://investor.example.com/quarterly-results/2025/q4" not in fetched


def test_crawler_latest_only_keeps_newest_transcript_artifacts(tmp_path) -> None:
    class FakeResponse:
        headers = {"content-type": "text/html"}
        content = b""

        def __init__(self, text: str) -> None:
            self.text = text

    class FakeHttp:
        def get(self, url: str):
            year = "2026" if "2026" in url else "2025"
            return FakeResponse(
                f"""
                <html><head><title>Example Q1 {year} Earnings Call Transcript</title></head>
                <body>
                  <p>OPERATOR: Welcome everyone.</p>
                  <p>JANE DOE: Thank you.</p>
                  <p>JOHN SMITH: Prepared remarks.</p>
                  <p>ANALYST: My question is about margins.</p>
                  <p>QUESTION-AND-ANSWER SESSION</p>
                  <p>END</p>
                </body></html>
                """
            )

    class FakeAgent:
        def decide(self, **kwargs):
            return PageDecision(page_type="transcript", confidence=1.0, reason="transcript", useful_links=[])

    crawler = TranscriptCrawler(
        model="test-model",
        out_dir=tmp_path,
        seed_urls=["https://example.com/q1-2025", "https://example.com/q1-2026"],
        http=FakeHttp(),  # type: ignore[arg-type]
        latest_only=True,
    )
    crawler.agent = FakeAgent()  # type: ignore[assignment]

    result = crawler.crawl_company(Company(symbol="EX", name="Example"))

    assert len(result.transcripts) == 1
    assert result.transcripts[0].fiscal_period == "Q1 2026"
    remaining = [path.name for path in (tmp_path / "EX").glob("Example_Q1_*") if not path.name.startswith("_")]
    assert any("2026" in name for name in remaining)
    assert not any("2025" in name for name in remaining)


def test_nav_first_uses_navigation_before_search(monkeypatch, tmp_path) -> None:
    company = Company(symbol="EX", name="Example")
    monkeypatch.setattr(
        "ir_transcripts.crawler.discover_navigation_seeds",
        lambda *args, **kwargs: NavigationDiscoveryResult(
            seeds=["https://example.com/investors"],
            trace=NavigationTrace(company=company),
        ),
    )
    monkeypatch.setattr("ir_transcripts.crawler.find_ir_candidates", lambda *args, **kwargs: ["https://search.example.com"])
    crawler = TranscriptCrawler(model="test-model", out_dir=tmp_path, discovery_mode="nav-first")

    assert crawler._discover_seeds(company) == ["https://example.com/investors"]


def test_crawler_delegates_unavailable_robots_for_navigation_verified_ir_subdomain(monkeypatch, tmp_path) -> None:
    company = Company(symbol="EX", name="Example")
    monkeypatch.setattr(
        "ir_transcripts.crawler.discover_navigation_seeds",
        lambda *args, **kwargs: NavigationDiscoveryResult(
            seeds=["https://ir.example.com/earnings"],
            trace=NavigationTrace(company=company),
            robots_verified_official_urls=["https://www.example.com"],
        ),
    )

    class FakeResponse:
        headers = {"content-type": "text/html"}
        text = """
        <html><head><title>Example Q1 2026 Earnings Call Transcript</title></head>
        <body>
          <p>OPERATOR: Welcome to the Example Q1 2026 earnings call.</p>
          <p>JANE DOE: Thank you.</p>
          <p>QUESTION-AND-ANSWER SESSION</p>
          <p>END</p>
        </body></html>
        """
        content = text.encode("utf-8")

    class FakeHttp:
        def __init__(self) -> None:
            self.delegated_fetches: list[str] = []

        def get(self, url: str):
            raise RobotsUnavailableError(f"Could not verify robots.txt for {url}")

        def get_without_robots_check(self, url: str):
            self.delegated_fetches.append(url)
            return FakeResponse()

    class FakeAgent:
        def decide(self, **kwargs):
            return PageDecision(page_type="transcript", confidence=1.0, reason="transcript", useful_links=[])

    http = FakeHttp()
    crawler = TranscriptCrawler(
        model="test-model",
        out_dir=tmp_path,
        discovery_mode="nav-first",
        http=http,  # type: ignore[arg-type]
    )
    crawler.agent = FakeAgent()  # type: ignore[assignment]

    result = crawler.crawl_company(company)

    assert http.delegated_fetches == ["https://ir.example.com/earnings"]
    assert len(result.transcripts) == 1


def test_nav_first_does_not_fallback_to_search_when_navigation_empty(monkeypatch, tmp_path) -> None:
    company = Company(symbol="EX", name="Example")
    monkeypatch.setattr(
        "ir_transcripts.crawler.discover_navigation_seeds",
        lambda *args, **kwargs: NavigationDiscoveryResult(
            seeds=[],
            trace=NavigationTrace(company=company),
        ),
    )

    def fail_search(*args, **kwargs):
        raise AssertionError("search fallback should not run")

    monkeypatch.setattr("ir_transcripts.crawler.find_ir_candidates", fail_search)
    crawler = TranscriptCrawler(model="test-model", out_dir=tmp_path, discovery_mode="nav-first")

    assert crawler._discover_seeds(company) == []


def test_search_first_preserves_search_discovery(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("ir_transcripts.crawler.find_ir_candidates", lambda *args, **kwargs: ["https://search.example.com"])
    crawler = TranscriptCrawler(model="test-model", out_dir=tmp_path, discovery_mode="search-first")

    assert crawler._discover_seeds(Company(symbol="EX", name="Example")) == ["https://search.example.com"]


def test_seed_url_bypasses_discovery(monkeypatch, tmp_path) -> None:
    def fail_discovery(*args, **kwargs):
        raise AssertionError("discovery should not run")

    monkeypatch.setattr("ir_transcripts.crawler.discover_navigation_seeds", fail_discovery)
    monkeypatch.setattr("ir_transcripts.crawler.find_ir_candidates", fail_discovery)
    crawler = TranscriptCrawler(
        model="test-model",
        out_dir=tmp_path,
        seed_urls=["https://seed.example.com"],
        discovery_mode="nav-first",
    )

    assert crawler._discover_seeds(Company(symbol="EX", name="Example")) == ["https://seed.example.com"]


def docx_fixture(paragraphs: list[str]) -> bytes:
    buffer = BytesIO()
    body = "".join(f"<w:p><w:r><w:t>{paragraph}</w:t></w:r></w:p>" for paragraph in paragraphs)
    document_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body>{body}</w:body>"
        "</w:document>"
    )
    with ZipFile(buffer, "w") as archive:
        archive.writestr("word/document.xml", document_xml)
    return buffer.getvalue()
