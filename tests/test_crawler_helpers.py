from ir_transcripts.crawler import TranscriptCrawler, artifact_stem, content_hash, link_score
from ir_transcripts.http import RobotsDisallowedError, RobotsUnavailableError
from ir_transcripts.models import CandidateLink, Company


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
