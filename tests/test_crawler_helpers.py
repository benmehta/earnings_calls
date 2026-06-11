from io import BytesIO
from zipfile import ZipFile

from ir_transcripts.crawler import (
    TranscriptCrawler,
    artifact_stem,
    content_hash,
    fiscal_period_key,
    is_docx_response,
    keep_latest_transcripts,
    link_score,
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


def test_artifact_stem_and_content_hash_are_stable() -> None:
    assert artifact_stem("Q1 Transcript", "https://example.com/a") == artifact_stem(
        "Q1 Transcript", "https://example.com/a"
    )
    assert content_hash("hello   world") == content_hash("hello world")


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
