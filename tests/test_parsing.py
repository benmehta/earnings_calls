from pathlib import Path

from ir_transcripts.parsing import extract_links, looks_like_js_shell, looks_like_transcript, visible_text


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

    assert links[0].url == "https://aka.ms/transcriptfy26q3"
    assert links[0].label == "Earnings Call Transcript"


def test_js_shell_detection() -> None:
    html = (FIXTURES / "js_shell.html").read_text(encoding="utf-8")
    assert looks_like_js_shell(html, visible_text(html))
