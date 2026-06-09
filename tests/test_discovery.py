from ir_transcripts.models import Company, IRDiscoveryCandidate
from ir_transcripts.search import discover_ir_candidates


def test_discovery_omits_deterministic_guesses_by_default(monkeypatch) -> None:
    company = Company(symbol="ZZZ", name="Zzz Example")
    monkeypatch.setattr("ir_transcripts.search.search_ir_candidates", lambda company, max_results=10: [])

    candidates = discover_ir_candidates(company)

    assert candidates == []


def test_discovery_can_include_guesses_when_search_finds_nothing(monkeypatch) -> None:
    company = Company(symbol="ZZZ", name="Zzz Example")
    monkeypatch.setattr("ir_transcripts.search.search_ir_candidates", lambda company, max_results=10: [])

    candidates = discover_ir_candidates(company, include_guesses=True)

    assert candidates
    assert all(candidate.source == "deterministic" for candidate in candidates)


def test_discovery_prefers_search_over_guesses(monkeypatch) -> None:
    company = Company(symbol="ZZZ", name="Zzz Example")
    search_candidate = IRDiscoveryCandidate(
        url="https://www.zzzexample.com/investor",
        title="Zzz Example Investor Relations",
        source="search",
        score=80,
    )
    monkeypatch.setattr("ir_transcripts.search.search_ir_candidates", lambda company, max_results=10: [search_candidate])

    candidates = discover_ir_candidates(company, include_guesses=True)

    assert candidates == [search_candidate]

