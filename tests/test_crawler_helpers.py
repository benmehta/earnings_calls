from io import BytesIO
from zipfile import ZipFile

from ir_transcripts.crawler import TranscriptCrawler, artifact_stem, content_hash, is_docx_response, link_score
from ir_transcripts.http import RobotsDisallowedError, RobotsUnavailableError
from ir_transcripts.models import CandidateLink, Company, PageDecision


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


def test_artifact_stem_and_content_hash_are_stable() -> None:
    assert artifact_stem("Q1 Transcript", "https://example.com/a") == artifact_stem(
        "Q1 Transcript", "https://example.com/a"
    )
    assert content_hash("hello   world") == content_hash("hello world")


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
