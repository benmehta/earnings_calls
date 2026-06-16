from __future__ import annotations

from dataclasses import dataclass

from ir_transcripts.models import Company, IRDiscoveryCandidate, TranscriptResearchJudgment, TranscriptResearchProposal
from ir_transcripts.research import discover_transcript_research_seeds


@dataclass
class FakeResponse:
    text: str
    headers: dict[str, str]


class FakeHttp:
    def __init__(self) -> None:
        self.urls: list[str] = []

    def get(self, url: str) -> FakeResponse:
        self.urls.append(url)
        return FakeResponse(
            text=(
                "<html><title>Example Investor Relations</title>"
                "<body>Example Corp investor relations quarterly earnings transcript"
                "<a href='https://investor.example.com/events'>Quarterly results</a></body></html>"
            ),
            headers={"content-type": "text/html"},
        )


def test_research_agent_and_judge_can_seed_official_urls(monkeypatch) -> None:
    candidate = IRDiscoveryCandidate(
        url="https://investor.example.com/events",
        title="Example Corp Quarterly Results",
        snippet="Official investor relations quarterly results and transcripts.",
        source="search",
        score=90,
    )
    monkeypatch.setattr("ir_transcripts.research.search_ir_candidates", lambda *args, **kwargs: [candidate])

    class FakeResearchAgent:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def research(self, **kwargs) -> TranscriptResearchProposal:
            return TranscriptResearchProposal(
                issuer_name="Example Corp",
                official_ir_urls=["https://investor.example.com/events"],
                confidence=0.9,
                reason="official IR result",
            )

    class FakeJudgeAgent:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def judge(self, **kwargs) -> TranscriptResearchJudgment:
            return TranscriptResearchJudgment(
                accepted=True,
                accepted_ir_urls=["https://investor.example.com/events"],
                official_company_name="Example Corp",
                confidence=0.9,
                reason="official IR evidence verified",
            )

    monkeypatch.setattr("ir_transcripts.research.OfficialTranscriptResearchAgent", FakeResearchAgent)
    monkeypatch.setattr("ir_transcripts.research.TranscriptResearchJudgeAgent", FakeJudgeAgent)

    result = discover_transcript_research_seeds(
        Company(symbol="EXM", name=None),
        http=FakeHttp(),  # type: ignore[arg-type]
        model="test-model",
    )

    assert result.failure_type is None
    assert result.seeds == ["https://investor.example.com/events"]
    assert result.seed_roles == {"https://investor.example.com/events": "ir"}
    assert result.verified_company_name == "Example Corp"


def test_research_seed_roles_preserve_homepage_context(monkeypatch) -> None:
    candidate = IRDiscoveryCandidate(
        url="https://www.example.com/",
        title="Example Corp",
        snippet="Official company homepage with investor relations footer.",
        source="search",
        score=90,
    )
    monkeypatch.setattr("ir_transcripts.research.search_ir_candidates", lambda *args, **kwargs: [candidate])

    class FakeResearchAgent:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def research(self, **kwargs) -> TranscriptResearchProposal:
            return TranscriptResearchProposal(
                issuer_name="Example Corp",
                official_homepage_urls=["https://www.example.com/"],
                confidence=0.9,
                reason="official homepage",
            )

    class FakeJudgeAgent:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def judge(self, **kwargs) -> TranscriptResearchJudgment:
            return TranscriptResearchJudgment(
                accepted=True,
                accepted_homepage_urls=["https://www.example.com/"],
                official_company_name="Example Corp",
                confidence=0.9,
                reason="official homepage evidence verified",
            )

    monkeypatch.setattr("ir_transcripts.research.OfficialTranscriptResearchAgent", FakeResearchAgent)
    monkeypatch.setattr("ir_transcripts.research.TranscriptResearchJudgeAgent", FakeJudgeAgent)

    result = discover_transcript_research_seeds(
        Company(symbol="EXM", name=None),
        http=FakeHttp(),  # type: ignore[arg-type]
        model="test-model",
    )

    assert result.seeds == ["https://www.example.com/"]
    assert result.seed_roles == {"https://www.example.com/": "homepage"}


def test_research_judge_rejection_stops_before_crawl(monkeypatch) -> None:
    candidate = IRDiscoveryCandidate(
        url="https://finance.example.com/transcripts/exm",
        title="Example transcript archive",
        snippet="Third-party transcript archive.",
        source="search",
        score=50,
    )
    monkeypatch.setattr("ir_transcripts.research.search_ir_candidates", lambda *args, **kwargs: [candidate])

    class FakeResearchAgent:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def research(self, **kwargs) -> TranscriptResearchProposal:
            return TranscriptResearchProposal(
                transcript_candidate_urls=["https://finance.example.com/transcripts/exm"],
                confidence=0.8,
                reason="candidate contains transcript",
            )

    class FakeJudgeAgent:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def judge(self, **kwargs) -> TranscriptResearchJudgment:
            return TranscriptResearchJudgment(
                accepted=False,
                confidence=0.9,
                reason="third-party transcript archive",
                retry_guidance="Find official company or IR domain evidence first.",
            )

    monkeypatch.setattr("ir_transcripts.research.OfficialTranscriptResearchAgent", FakeResearchAgent)
    monkeypatch.setattr("ir_transcripts.research.TranscriptResearchJudgeAgent", FakeJudgeAgent)

    result = discover_transcript_research_seeds(
        Company(symbol="EXM", name=None),
        http=FakeHttp(),  # type: ignore[arg-type]
        model="test-model",
    )

    assert result.seeds == []
    assert result.failure_type == "official_research_unverified"
    assert "official company" in result.failure_message


def test_research_runtime_rejects_third_party_seed_even_if_judge_accepts(monkeypatch) -> None:
    candidates = [
        IRDiscoveryCandidate(
            url="https://abc.xyz/investor/",
            title="Alphabet Investor Relations",
            snippet="Official Alphabet investor relations.",
            source="search",
            score=90,
        ),
        IRDiscoveryCandidate(
            url="https://quartr.com/companies/alphabet-inc_4122",
            title="Alphabet (GOOG) Investor Relations Material - 100% Free Access",
            snippet="Third-party investor material and transcripts.",
            source="search",
            score=80,
        ),
    ]
    monkeypatch.setattr("ir_transcripts.research.search_ir_candidates", lambda *args, **kwargs: candidates)

    class FakeResearchAgent:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def research(self, **kwargs) -> TranscriptResearchProposal:
            return TranscriptResearchProposal(
                issuer_name="Alphabet Inc.",
                official_ir_urls=["https://abc.xyz/investor/"],
                transcript_candidate_urls=["https://quartr.com/companies/alphabet-inc_4122"],
                confidence=0.9,
                reason="official IR plus transcript candidate",
            )

    class FakeJudgeAgent:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def judge(self, **kwargs) -> TranscriptResearchJudgment:
            return TranscriptResearchJudgment(
                accepted=True,
                accepted_transcript_urls=["https://quartr.com/companies/alphabet-inc_4122"],
                official_company_name="Alphabet Inc.",
                confidence=0.9,
                reason="mistakenly accepted third-party result",
            )

    monkeypatch.setattr("ir_transcripts.research.OfficialTranscriptResearchAgent", FakeResearchAgent)
    monkeypatch.setattr("ir_transcripts.research.TranscriptResearchJudgeAgent", FakeJudgeAgent)

    result = discover_transcript_research_seeds(
        Company(symbol="GOOG", name=None),
        http=FakeHttp(),  # type: ignore[arg-type]
        model="test-model",
    )

    assert result.seeds == []
    assert result.failure_type == "official_research_unverified"
    assert "third-party" in result.failure_message
