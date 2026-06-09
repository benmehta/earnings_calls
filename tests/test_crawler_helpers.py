from ir_transcripts.crawler import artifact_stem, content_hash, link_score
from ir_transcripts.models import CandidateLink


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

