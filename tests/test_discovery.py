from ir_transcripts.models import Company, CompanyNavigationMemory, IRDiscoveryCandidate, PromptGuidance
from ir_transcripts.search import apply_discovery_memory, company_domain_slug, discover_ir_candidates


def test_discovery_omits_deterministic_guesses_by_default(monkeypatch) -> None:
    company = Company(symbol="ZZZ", name="Zzz Example")
    monkeypatch.setattr("ir_transcripts.search.search_ir_candidates", lambda company, max_results=10, **kwargs: [])

    candidates = discover_ir_candidates(company)

    assert candidates == []


def test_discovery_can_include_guesses_when_search_finds_nothing(monkeypatch) -> None:
    company = Company(symbol="ZZZ", name="Zzz Example")
    monkeypatch.setattr("ir_transcripts.search.search_ir_candidates", lambda company, max_results=10, **kwargs: [])

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
    monkeypatch.setattr("ir_transcripts.search.search_ir_candidates", lambda company, max_results=10, **kwargs: [search_candidate])

    candidates = discover_ir_candidates(company, include_guesses=True)

    assert candidates == [search_candidate]


def test_discovery_passes_search_timeout(monkeypatch) -> None:
    captured = {}

    def fake_search(company, max_results=10, **kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr("ir_transcripts.search.search_ir_candidates", fake_search)

    discover_ir_candidates(Company(symbol="ZZZ", name="Zzz Example"), search_timeout_seconds=7)

    assert captured["timeout_seconds"] == 7


def test_discovery_memory_prefers_successful_host_over_raw_search_rank() -> None:
    candidates = [
        IRDiscoveryCandidate(
            url="https://www.taiwansemi.com/en/investor-relations/",
            title="Taiwan Semi Investor Relations",
            score=95,
            reasons=["search ranked first"],
        ),
        IRDiscoveryCandidate(
            url="https://investor.tsmc.com/english",
            title="TSMC Investor Relations",
            score=50,
            reasons=["search ranked lower"],
        ),
    ]
    memory = CompanyNavigationMemory(
        preferred_hosts=["investor.tsmc.com"],
        successful_hosts=["investor.tsmc.com"],
        low_value_hosts=["www.taiwansemi.com"],
    )

    ranked = apply_discovery_memory(candidates, navigation_memory=memory)

    assert ranked[0].url == "https://investor.tsmc.com/english"
    assert "memory preferred host" in ranked[0].reasons
    assert "memory low-value host" in ranked[1].reasons


def test_discovery_memory_scoring_is_idempotent() -> None:
    candidates = [
        IRDiscoveryCandidate(
            url="https://investor.example.com",
            title="Example Investor Relations",
            score=50,
        )
    ]
    memory = CompanyNavigationMemory(preferred_hosts=["investor.example.com"])

    once = apply_discovery_memory(candidates, navigation_memory=memory)
    twice = apply_discovery_memory(once, navigation_memory=memory)

    assert twice[0].score == once[0].score
    assert twice[0].reasons.count("memory preferred host") == 1


def test_discovery_passes_guidance_and_memory_to_reranker(monkeypatch) -> None:
    captured = {}
    candidate = IRDiscoveryCandidate(
        url="https://investor.example.com",
        title="Example Investor Relations",
        source="search",
        score=90,
    )
    memory = CompanyNavigationMemory(preferred_hosts=["investor.example.com"])
    guidance = PromptGuidance(priority_terms=["quarterly results detail pages"])

    class FakeDiscoveryAgent:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def rerank(self, **kwargs):
            captured.update(kwargs)

            class Decision:
                selections = []

            return Decision()

    monkeypatch.setattr("ir_transcripts.search.search_ir_candidates", lambda *args, **kwargs: [candidate])
    monkeypatch.setattr("ir_transcripts.search.IRDiscoveryAgent", FakeDiscoveryAgent)

    discover_ir_candidates(
        Company(symbol="EX", name="Example"),
        rerank_model="test-model",
        prompt_guidance=guidance,
        navigation_memory=memory,
    )

    assert captured["guidance"] == guidance
    assert captured["navigation_memory"] == memory


def test_company_domain_slug_handles_dot_com_legal_names() -> None:
    assert company_domain_slug("Amazon.com, Inc.") == "amazon"
    assert company_domain_slug("Salesforce.com, Inc.") == "salesforce"
