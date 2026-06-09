from pathlib import Path

from ir_transcripts.state import CrawlState
from ir_transcripts.urls import normalize_url


def test_normalize_url_removes_tracking_and_trailing_slash() -> None:
    assert (
        normalize_url("HTTPS://Investors.Example.com/events/?utm_source=x&b=2&a=1")
        == "https://investors.example.com/events?a=1&b=2"
    )


def test_crawl_state_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    state = CrawlState.load(path)
    state.mark_visited("https://example.com/a/")
    state.mark_transcript_url("https://example.com/t?utm_campaign=x")
    state.mark_content_hash("abc")
    state.save()

    loaded = CrawlState.load(path)
    assert loaded.has_visited("https://example.com/a")
    assert loaded.has_transcript_url("https://example.com/t")
    assert loaded.has_content_hash("abc")

