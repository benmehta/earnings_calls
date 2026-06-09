from pathlib import Path
from zipfile import ZipFile
from io import BytesIO

from ir_transcripts.parsing import docx_text, extract_links, looks_like_js_shell, looks_like_transcript, visible_text


FIXTURES = Path(__file__).parent / "fixtures"


def test_visible_text_and_transcript_detection() -> None:
    html = (FIXTURES / "transcript.html").read_text(encoding="utf-8")
    text = visible_text(html)

    assert "Operator" in text
    assert looks_like_transcript(text)


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


def test_js_shell_detection() -> None:
    html = (FIXTURES / "js_shell.html").read_text(encoding="utf-8")
    assert looks_like_js_shell(html, visible_text(html))
