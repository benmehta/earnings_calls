from pathlib import Path
from zipfile import ZipFile
from io import BytesIO

from ir_transcripts.parsing import classify_transcript, docx_text, extract_links, looks_like_js_shell, looks_like_transcript, visible_text


FIXTURES = Path(__file__).parent / "fixtures"


def test_visible_text_and_transcript_detection() -> None:
    html = (FIXTURES / "transcript.html").read_text(encoding="utf-8")
    text = visible_text(html)

    assert "Operator" in text
    assert looks_like_transcript(text)


def test_press_release_webcast_text_is_not_transcript() -> None:
    text = """
    Microsoft Cloud and AI Strength Fuels Third Quarter Results.
    Business Highlights include Microsoft 365 and Azure growth.
    Webcast Details: Satya Nadella, Amy Hood, and investor relations will host
    a conference call and webcast to discuss performance. Forward-looking
    statements and non-GAAP financial measures are included below. Analysts and
    investors may access the webcast from the investor relations site.
    """

    detection = classify_transcript(
        text,
        title="FY26 Q3 - Press Releases - Investor Relations - Microsoft",
        url="https://www.microsoft.com/en-us/investor/earnings/fy-2026-q3/press-release-webcast",
    )

    assert not detection.is_transcript
    assert detection.reason == "transcript_rejected_press_release_like"


def test_ir_index_with_transcript_links_is_not_transcript() -> None:
    html = (FIXTURES / "ir_index.html").read_text(encoding="utf-8")

    detection = classify_transcript(visible_text(html), title="Example Investor Relations")

    assert not detection.is_transcript
    assert detection.reason == "transcript_rejected_missing_speaker_structure"


def test_extract_links_normalizes_relative_urls() -> None:
    html = (FIXTURES / "ir_index.html").read_text(encoding="utf-8")
    links = extract_links(html, "https://investors.example.com")

    assert any(link.url.startswith("https://investors.example.com/events/") for link in links)
    assert any(link.url == "https://ir-vendor.example/events/q2-2025-earnings" for link in links)


def test_extract_links_reads_custom_link_attributes() -> None:
    html = (
        '<li link="https://aka.ms/transcriptfy26q3" '
        'arialabel="View Earnings Call Transcript of FY26Q3">Earnings Call Transcript</li>'
    )
    links = extract_links(html, "https://www.microsoft.com/en-us/investor")

    assert links[0].url == "https://cdn-dynmedia-1.microsoft.com/is/content/microsoftcorp/TranscriptQandAFY26Q3"
    assert links[0].label == "Earnings Call Transcript"


def test_extract_links_preserves_context_for_generic_transcript_links() -> None:
    html = """
    <html><body>
      <p>For a PDF version of the transcript, please <a href="/investor/static-files/q1-transcript.pdf">click here</a>.</p>
    </body></html>
    """
    links = extract_links(html, "https://abc.xyz/investor/events/event-details/q1")

    assert links[0].url == "https://abc.xyz/investor/static-files/q1-transcript.pdf"
    assert "PDF version of the transcript" in links[0].label


def test_docx_text_extracts_paragraphs() -> None:
    buffer = BytesIO()
    document_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
    <w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
      <w:body>
        <w:p><w:r><w:t>Operator</w:t></w:r></w:p>
        <w:p><w:r><w:t>Prepared remarks</w:t></w:r></w:p>
      </w:body>
    </w:document>
    """
    with ZipFile(buffer, "w") as archive:
        archive.writestr("word/document.xml", document_xml)

    assert docx_text(buffer.getvalue()) == "Operator\n\nPrepared remarks"


def test_docx_transcript_text_is_detected() -> None:
    text = "\n\n".join(
        [
            "Microsoft Fiscal Year 2026 Third Quarter Earnings Call Transcript",
            "OPERATOR: Welcome to the Microsoft earnings conference call.",
            "SATYA NADELLA: Thank you, and welcome everyone.",
            "AMY HOOD: I will cover our financial results.",
            "JONATHAN NEILSON: Operator, next question, please.",
            "QUESTION-AND-ANSWER SESSION",
            "END",
        ]
    )

    assert looks_like_transcript(text, title="TranscriptQandAFY26Q3")


def test_js_shell_detection() -> None:
    html = (FIXTURES / "js_shell.html").read_text(encoding="utf-8")
    assert looks_like_js_shell(html, visible_text(html))


def test_q4_financial_widget_triggers_browser_render() -> None:
    html = """
    <html><body>
      <p>Investor relations page with enough static text that it is not an app shell.</p>
      <script>$(".module-financial-table").financialTable({ categories: [] });</script>
    </body></html>
    """
    text = " ".join(["static investor relations text"] * 200)

    assert looks_like_js_shell(html, text)


def test_q4_evergreen_event_widget_triggers_browser_render() -> None:
    html = """
    <html><body>
      <h1>Events & Presentations</h1>
      <div class="evergreen evergreen-event">
        <script id="tplEvergreenEventList" type="text/template"></script>
      </div>
    </body></html>
    """
    text = " ".join(["static investor relations events text"] * 200)

    assert looks_like_js_shell(html, text)
