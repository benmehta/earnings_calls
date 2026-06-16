import json

from ir_transcripts.agent import (
    IRNavigationAgent,
    IRPageAgent,
    IRSectionAgent,
    NavigationValidationAgent,
    compact_candidate_links,
    extract_json_object,
    fallback_links_for_kind,
    guidance_from_memory,
    hard_validate_navigation_decision,
    navigation_agent_kind,
    prompt_planner_memory_summary,
    sanitize_crawl_reflection,
    sanitize_prompt_guidance,
)
from ir_transcripts.agent_prompts import load_agent_prompt, parse_prompt_yaml
from ir_transcripts.models import CandidateLink, Company, CompanyMemory, CrawlReflection, IRDiscoverySelection, NavigationDecision, NavigationValidationResult, PromptGuidance


def test_page_agent_converts_useful_urls_to_links(monkeypatch) -> None:
    agent = IRPageAgent.__new__(IRPageAgent)
    agent.max_links = 12
    agent.text_chars = 900

    class FakeChain:
        def invoke(self, values):
            class Message:
                content = (
                    '{"page_type":"ir_index","confidence":0.8,'
                    '"useful_urls":["https://example.com/events"],'
                    '"reason":"events page"}'
                )

            return Message()

    agent.chain = FakeChain()
    links = [
        CandidateLink(
            url="https://example.com/events",
            label="Events",
            source_url="https://example.com/investors",
        ),
        CandidateLink(
            url="https://example.com/careers",
            label="Careers",
            source_url="https://example.com/investors",
        ),
    ]

    decision = agent.decide(
        company_name="Example",
        ticker="EX",
        url="https://example.com/investors",
        title="Investors",
        text="Investor relations",
        links=links,
    )

    assert decision.page_type == "ir_index"
    assert [link.url for link in decision.useful_links] == ["https://example.com/events"]


def test_page_agent_accepts_object_useful_urls(monkeypatch) -> None:
    agent = IRPageAgent.__new__(IRPageAgent)
    agent.max_links = 12
    agent.text_chars = 900

    class FakeChain:
        def invoke(self, values):
            class Message:
                content = (
                    '{"page_type":"ir_index","confidence":0.8,'
                    '"useful_urls":[{"url":"https://example.com/events"}],'
                    '"reason":"events page"}'
                )

            return Message()

    agent.chain = FakeChain()
    links = [
        CandidateLink(
            url="https://example.com/events",
            label="Events",
            source_url="https://example.com/investors",
        )
    ]

    decision = agent.decide(
        company_name="Example",
        ticker="EX",
        url="https://example.com/investors",
        title="Investors",
        text="Investor relations",
        links=links,
    )

    assert [link.url for link in decision.useful_links] == ["https://example.com/events"]


def test_extract_json_object_handles_fenced_output() -> None:
    assert extract_json_object('```json\n{"page_type":"ir_index"}\n```') == {"page_type": "ir_index"}


def test_navigation_prompts_prefer_quarterly_results_over_stock_pages() -> None:
    ir_prompt = IRSectionAgent.SYSTEM_PROMPT.lower()
    validation_prompt = NavigationValidationAgent.SYSTEM_PROMPT.lower()

    for prompt in (ir_prompt, validation_prompt):
        assert "prefer" in prompt
        assert "quarterly-results" in prompt
        assert "transcript" in prompt
        assert "stock-info" in prompt
        assert "only if no preferred" in prompt
        assert "avoid stock-info paths only when no preferred" not in prompt


def test_agent_prompt_yaml_loader_supports_multiline_values() -> None:
    parsed = parse_prompt_yaml(
        """
name: ExampleAgent
description: Example prompt
system_prompt: |
  Prefer quarterly-results.
  Choose stock-info only if no preferred links exist.
"""
    )

    assert parsed["name"] == "ExampleAgent"
    assert parsed["system_prompt"] == (
        "Prefer quarterly-results.\n"
        "Choose stock-info only if no preferred links exist."
    )


def test_agent_prompts_load_from_yaml() -> None:
    prompt = load_agent_prompt("ir_section")

    assert prompt.name == "IRSectionAgent"
    assert "quarterly-results" in prompt.system_prompt
    assert "only if no preferred" in prompt.system_prompt

    navigator_prompt = load_agent_prompt("crawl_navigator")
    assert navigator_prompt.name == "CrawlNavigatorAgent"
    assert "main navigator" in navigator_prompt.system_prompt
    assert "latest official quarterly earnings-call transcript" in navigator_prompt.system_prompt
    assert "never output transcript_link" in navigator_prompt.system_prompt

    document_prompt = load_agent_prompt("document_link_triage")
    assert document_prompt.name == "DocumentLinkTriageAgent"
    assert "earnings-call transcript" in document_prompt.system_prompt

    transcript_prompt = load_agent_prompt("transcript_evidence")
    assert transcript_prompt.name == "TranscriptEvidenceAgent"
    assert "true written public-company earnings-call transcript" in transcript_prompt.system_prompt

    link_prompt = load_agent_prompt("link_batch_triage")
    assert link_prompt.name == "LinkBatchTriageAgent"
    assert "official written earnings-call transcript artifacts" in link_prompt.system_prompt
    assert "rank the newest/latest fiscal period first" in link_prompt.system_prompt

    artifact_prompt = load_agent_prompt("earnings_artifact_extraction")
    assert artifact_prompt.name == "EarningsArtifactExtractionAgent"
    assert "Transcript, Q&A Transcript" in artifact_prompt.system_prompt

    latest_prompt = load_agent_prompt("latest_transcript_selection")
    assert latest_prompt.name == "LatestTranscriptSelectionAgent"
    assert "newest/latest official written earnings-call transcript" in latest_prompt.system_prompt

    ranking_prompt = load_agent_prompt("transcript_document_ranking")
    assert ranking_prompt.name == "TranscriptDocumentRankingAgent"
    assert "Q1 2027" in ranking_prompt.system_prompt

    nav_rank_prompt = load_agent_prompt("navigation_candidate_ranking")
    assert nav_rank_prompt.name == "NavigationCandidateRankingAgent"
    assert "Rank candidate links" in nav_rank_prompt.system_prompt

    discovery_prompt = load_agent_prompt("ir_discovery")
    assert discovery_prompt.name == "IRDiscoveryAgent"
    assert "not responsible for predicting or validating company homepages" in discovery_prompt.system_prompt
    assert "official IR/navigation seeds" in discovery_prompt.system_prompt

    search_rank_prompt = load_agent_prompt("search_candidate_ranking")
    assert search_rank_prompt.name == "SearchCandidateRankingAgent"
    assert "ranks IR/navigation seeds, not company homepages" in search_rank_prompt.system_prompt

    planner_prompt = load_agent_prompt("prompt_planner")
    assert planner_prompt.name == "PromptPlannerAgent"
    assert "Compile a safe run strategy" in planner_prompt.system_prompt
    assert "agent_guidance" in planner_prompt.human_prompt


def test_guidance_for_agent_includes_run_strategy_and_agent_specific_guidance() -> None:
    from ir_transcripts.agent import guidance_for_agent

    guidance = PromptGuidance(
        run_objective="Find the latest official quarterly earnings-call transcript.",
        strategy="Start from official homepage, then official IR, then quarterly results.",
        navigation_guidance="Prefer earnings result pages over generic stock pages.",
        agent_guidance={"HomepageNavAgent": "Use footer and corporate navigation to find Investors."},
    )

    text = guidance_for_agent(guidance, "navigation", agent_name="HomepageNavAgent")

    assert "Objective: Find the latest official quarterly earnings-call transcript." in text
    assert "Run strategy: Start from official homepage" in text
    assert "HomepageNavAgent: Use footer" in text


def test_search_query_sanitizer_strips_invented_site_operator() -> None:
    from ir_transcripts.agent import sanitize_search_queries

    queries = sanitize_search_queries(
        ["site:investor.goo.gl 'latest quarterly earnings report'"],
        limit=4,
        ticker="GOOG",
    )

    assert queries == ["GOOG 'latest quarterly earnings report'"]


def test_search_query_plan_accepts_object_query_items() -> None:
    from ir_transcripts.models import SearchQueryPlan

    plan = SearchQueryPlan.model_validate(
        {
            "queries": [
                {"query": "NVDA quarterly earnings call transcript"},
                "NVDA investor relations quarterly results",
            ],
            "reason": "planner output",
        }
    )

    assert plan.queries == [
        "NVDA quarterly earnings call transcript",
        "NVDA investor relations quarterly results",
    ]


def test_guidance_from_playbook_compiles_company_specific_strategy() -> None:
    from ir_transcripts.agent import guidance_from_playbook
    from ir_transcripts.models import CompanyPlaybook

    guidance = guidance_from_playbook(
        CompanyPlaybook(
            issuer_name="Alphabet",
            brand_names=["Google"],
            official_homepage_candidates=["https://abc.xyz/"],
            planner_prompt="Treat Alphabet as issuer and Google as brand.",
            homepage_strategy="Prefer abc.xyz over google.com when finding investor relations.",
            ir_strategy="Use Alphabet investor event pages.",
            transcript_strategy="Prefer official earnings-call transcript artifacts.",
            avoid_strategy="Avoid fake composite hosts.",
        ),
        ticker="GOOG",
    )

    assert "Alphabet" in guidance.run_objective
    assert "issuer: Alphabet" in guidance.priority_terms
    assert guidance.homepage_guidance.startswith("Prefer abc.xyz")
    assert guidance.agent_guidance["HomepagePredictionAgent"].startswith("Prefer abc.xyz")


def test_discovery_selection_accepts_percentage_confidence() -> None:
    selection = IRDiscoverySelection(url="https://investor.example.com", confidence=72)

    assert selection.confidence == 0.72


def test_navigation_agent_only_returns_provided_urls() -> None:
    agent = IRNavigationAgent.__new__(IRNavigationAgent)

    class FakeSpecialist:
        def decide(self, **kwargs):
            return NavigationDecision(
                chosen_urls=["https://example.com/investors"],
                confidence=0.9,
                reason="footer Investors link",
            )

    agent.agents = {"homepage": FakeSpecialist()}
    decision = agent.decide(
        company_name="Example",
        ticker="EX",
        url="https://example.com",
        title="Example",
        text="Home",
        page_context="homepage",
        links=[
            CandidateLink(
                url="https://example.com/investors",
                label="Investors",
                source_url="https://example.com",
                reason="footer",
            )
        ],
    )

    assert decision.chosen_urls == ["https://example.com/investors"]
    assert decision.reason == "homepage: footer Investors link"


def test_navigation_agent_sends_compact_prompt() -> None:
    from ir_transcripts.agent import HomepageNavAgent

    agent = HomepageNavAgent.__new__(HomepageNavAgent)
    captured = {}

    class FakeChain:
        def invoke(self, values):
            captured.update(values)

            class Message:
                content = (
                    '{"chosen_urls":["https://example.com/investors"],'
                    '"confidence":0.9,"reason":"investors","stop_reason":null}'
                )

            return Message()

    agent.chain = FakeChain()
    agent.max_links = 2
    agent.text_chars = 25
    links = [
        CandidateLink(
            url=f"https://example.com/{index}",
            label="Very long label " * 20,
            source_url="https://example.com",
            reason="footer " * 20,
        )
        for index in range(3)
    ]
    links[0].url = "https://example.com/investors"

    agent.decide(
        company_name="Example",
        ticker="EX",
        url="https://example.com",
        title="Example",
        text="word " * 100,
        links=links,
    )

    assert len(captured["text"]) <= 25
    assert captured["links_json"].count("https://example.com/") == 2
    assert "source_url" not in captured["links_json"]
    assert captured["guidance"] == "None."


def test_compact_candidate_links_respects_character_budget() -> None:
    links = [
        CandidateLink(
            url="https://www.amazon.com/ir",
            label="Investor Relations",
            source_url="https://www.amazon.com",
            reason="footer",
        ),
        CandidateLink(
            url="https://www.amazon.com/events/devicedeals/?" + "tracking=1&" * 30,
            label="Device Deals",
            source_url="https://www.amazon.com",
            reason="body",
        ),
        CandidateLink(
            url="https://www.amazon.com/s/?" + "tracking=2&" * 60,
            label="Retail Search",
            source_url="https://www.amazon.com",
            reason="body",
        ),
    ]

    compact_links = compact_candidate_links(links, max_links=10, char_budget=500)

    assert [link["url"] for link in compact_links] == ["https://www.amazon.com/ir"]


def test_navigation_agent_includes_prompt_guidance() -> None:
    from ir_transcripts.agent import HomepageNavAgent

    agent = HomepageNavAgent.__new__(HomepageNavAgent)
    captured = {}

    class FakeChain:
        def invoke(self, values):
            captured.update(values)

            class Message:
                content = (
                    '{"chosen_urls":["https://example.com/investors"],'
                    '"confidence":0.9,"reason":"investors","stop_reason":null}'
                )

            return Message()

    agent.chain = FakeChain()
    agent.max_links = 2
    agent.text_chars = 25
    agent.guidance = PromptGuidance(
        priority_terms=["investor events"],
        avoid_terms=["blog", "youtube"],
        navigation_guidance="Prefer official IR event pages.",
    )

    agent.decide(
        company_name="Example",
        ticker="EX",
        url="https://example.com",
        title="Example",
        text="Home",
        links=[
            CandidateLink(
                url="https://example.com/investors",
                label="Investors",
                source_url="https://example.com",
            )
        ],
    )

    assert "Prefer: investor events" in captured["guidance"]
    assert "Avoid: blog, youtube" in captured["guidance"]
    assert "Prefer official IR event pages" in captured["guidance"]


def test_navigation_agent_routes_to_specialists() -> None:
    links = [
        CandidateLink(
            url="https://example.com/investor/events/event-details/2026/q1",
            label="2026 Q1 Earnings Call",
            source_url="https://example.com/investor/events",
        )
    ]

    assert (
        navigation_agent_kind(
            url="https://example.com",
            title="Example",
            text="",
            links=links,
            page_context="homepage",
        )
        == "homepage"
    )
    assert (
        navigation_agent_kind(
            url="https://example.com/investor/events",
            title="Events",
            text="",
            links=links,
            page_context="ir",
        )
        == "event_listing"
    )
    assert (
        navigation_agent_kind(
            url="https://example.com/investor/events/event-details/2026/q1-earnings-call",
            title="2026 Q1 Earnings Call",
            text="",
            links=[],
            page_context="ir",
        )
        == "transcript_link"
    )


def test_hard_validation_rejects_avoid_terms_when_better_link_exists() -> None:
    links = [
        CandidateLink(
            url="https://blog.google/alphabet/earnings-q1-2026",
            label="Alphabet earnings Q1 2026",
            source_url="https://abc.xyz/investor/events",
        ),
        CandidateLink(
            url="https://abc.xyz/investor/events/event-details/2026/q1-earnings-call",
            label="2026 Q1 Earnings Call",
            source_url="https://abc.xyz/investor/events",
        ),
    ]
    result = hard_validate_navigation_decision(
        kind="event_listing",
        decision=NavigationDecision(
            chosen_urls=["https://blog.google/alphabet/earnings-q1-2026"],
            confidence=0.9,
        ),
        links=links,
        guidance=PromptGuidance(avoid_terms=["blog.google"]),
    )

    assert not result.is_valid
    assert result.reason == "agent_rejected_wrong_task"


def test_hard_validation_partially_accepts_mixed_decision() -> None:
    links = [
        CandidateLink(
            url="https://abc.xyz/investor/events/event-details/2026/q1-earnings-call",
            label="2026 Q1 Earnings Call",
            source_url="https://abc.xyz/investor/events",
        ),
        CandidateLink(
            url="https://www.youtube.com/watch?v=abc",
            label="Webcast",
            source_url="https://abc.xyz/investor/events",
        ),
    ]
    result = hard_validate_navigation_decision(
        kind="event_listing",
        decision=NavigationDecision(
            chosen_urls=[
                "https://abc.xyz/investor/events/event-details/2026/q1-earnings-call",
                "https://www.youtube.com/watch?v=abc",
            ],
            confidence=0.8,
        ),
        links=links,
        guidance=PromptGuidance(avoid_terms=["youtube"]),
    )

    assert result.is_valid
    assert result.accepted_urls == ["https://abc.xyz/investor/events/event-details/2026/q1-earnings-call"]
    assert result.rejected_urls == ["https://www.youtube.com/watch?v=abc"]


def test_navigation_inner_graph_repairs_invalid_decision() -> None:
    class FakeSpecialist:
        def __init__(self) -> None:
            self.calls = 0

        def decide(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return NavigationDecision(
                    chosen_urls=["https://blog.google/alphabet/earnings"],
                    confidence=0.9,
                    reason="blog looked relevant",
                )
            assert "Previous decision rejected" in kwargs["repair_guidance"]
            return NavigationDecision(
                chosen_urls=["https://abc.xyz/investor/events/event-details/2026/q1-earnings-call"],
                confidence=0.9,
                reason="event detail",
            )

    class FakeValidator:
        def validate(self, **kwargs):
            return NavigationValidationResult(
                is_valid=True,
                accepted_urls=kwargs["decision"].chosen_urls,
                reason="semantic ok",
            )

    agent = IRNavigationAgent.__new__(IRNavigationAgent)
    specialist = FakeSpecialist()
    agent.guidance = PromptGuidance(avoid_terms=["blog.google"])
    agent.confidence_floor = 0.55
    agent.agents = {"event_listing": specialist}
    agent.validator = FakeValidator()
    agent.graph = agent._build_graph()

    links = [
        CandidateLink(
            url="https://blog.google/alphabet/earnings",
            label="Alphabet earnings",
            source_url="https://abc.xyz/investor/events",
        ),
        CandidateLink(
            url="https://abc.xyz/investor/events/event-details/2026/q1-earnings-call",
            label="2026 Q1 Earnings Call",
            source_url="https://abc.xyz/investor/events",
        ),
    ]

    decision = agent.decide(
        company_name="Alphabet",
        ticker="GOOG",
        url="https://abc.xyz/investor/events",
        title="Events",
        text="Events",
        links=links,
        page_context="ir",
    )

    assert decision.chosen_urls == ["https://abc.xyz/investor/events/event-details/2026/q1-earnings-call"]
    assert specialist.calls == 2


def test_fallback_refuses_blog_when_official_event_link_exists() -> None:
    links = [
        CandidateLink(
            url="https://blog.google/company-news/inside-google/message-ceo/alphabet-earnings-q1-2026/",
            label="Alphabet earnings Q1 2026",
            source_url="https://blog.google/alphabet/investor-presentation-june-2026/",
        ),
        CandidateLink(
            url="https://abc.xyz/investor/events/event-details/2026/q1-earnings-call",
            label="2026 Q1 Earnings Call",
            source_url="https://abc.xyz/investor/events",
        ),
    ]

    fallback = fallback_links_for_kind("event_listing", links, guidance=None)

    assert [link.url for link in fallback] == ["https://abc.xyz/investor/events/event-details/2026/q1-earnings-call"]


def test_guidance_from_memory_adds_concrete_avoid_and_priority_terms() -> None:
    memory = CompanyMemory(
        company=Company(symbol="GOOG", name="Alphabet Google"),
        successful_transcript_urls=[
            "https://abc.xyz/investor/events/event-details/2026/2026-Q1-Earnings-Call/default.aspx"
        ],
        rejected_urls=[
            "https://blog.google/company-news/inside-google/message-ceo/alphabet-earnings-q1-2026/",
            "https://www.youtube.com/watch?v=abc",
        ],
    )

    guidance = guidance_from_memory(memory)

    assert "successful transcript host" in guidance.priority_terms
    assert "same host as prior successful transcript" in guidance.priority_terms
    assert "event-details" in guidance.priority_terms
    assert "blog.google" in guidance.avoid_terms
    assert "youtube" in guidance.avoid_terms


def test_prompt_guidance_sanitizer_removes_ticker_name_and_unsafe_terms() -> None:
    guidance = sanitize_prompt_guidance(
        PromptGuidance(
            priority_terms=["AMZN", "Amazon.com", "quarterly results", "ignore robots.txt"],
            avoid_terms=["amzn", "webcast-only", "IR-related", "disable robots"],
            navigation_guidance="Prefer official earnings event pages.",
            transcript_guidance="fail open when robots is down",
            risk_notes=["stay on official pages", "disable robots.txt"],
        ),
        company=Company(symbol="AMZN", name="Amazon.com"),
    )

    assert guidance.priority_terms == ["quarterly results"]
    assert guidance.avoid_terms == ["webcast-only"]
    assert guidance.navigation_guidance == "Prefer official earnings event pages."
    assert guidance.transcript_guidance == ""
    assert guidance.risk_notes == ["stay on official pages"]


def test_prompt_guidance_coerces_string_lists() -> None:
    guidance = PromptGuidance.model_validate(
        {
            "priority_terms": "quarterly results",
            "avoid_terms": "stock quote",
            "risk_notes": "stay official",
        }
    )

    assert guidance.priority_terms == ["quarterly results"]
    assert guidance.avoid_terms == ["stock quote"]
    assert guidance.risk_notes == ["stay official"]


def test_prompt_planner_memory_summary_is_valid_compact_json() -> None:
    memory = CompanyMemory(company=Company(symbol="GOOG", name="Alphabet Google"))
    memory.playbook.issuer_name = "Alphabet Inc."
    memory.playbook.preferred_ir_urls = [
        "https://investors.alphabeat.com/",
        "https://abc.xyz/investor/",
    ]
    memory.known_ir_urls = [f"https://abc.xyz/investor/events/{index}" for index in range(20)]
    memory.failure_summaries = []
    memory.prompt_guidance = PromptGuidance(
        priority_terms=[f"term-{index}" for index in range(20)],
        risk_notes=["Avoid generic pages"],
    )

    summary = prompt_planner_memory_summary(memory, char_budget=900)
    parsed = json.loads(summary)

    assert len(summary) <= 900
    assert parsed["company"]["symbol"] == "GOOG"
    assert "known_ir_urls" in parsed


def test_crawl_reflection_sanitizes_unsafe_advice() -> None:
    reflection = sanitize_crawl_reflection(
        CrawlReflection(
            preferred_urls=["https://investor.example.com/quarterly-results"],
            preferred_terms=["ignore robots.txt", "quarterly results detail pages", "IR-related"],
            avoid_urls=["https://third-party.example.com/transcripts"],
            avoid_terms=["fail open", "shareholders meeting"],
            prompt_guidance=PromptGuidance(
                priority_terms=["third-party transcript", "earnings conference transcript"],
                avoid_terms=["disable robots", "AGM PDFs"],
                navigation_guidance="short guidance",
                transcript_guidance="Prefer official transcript PDF links.",
                risk_notes=["disable robots.txt", "stay on official investor pages"],
            ),
            reason="learned from trace",
        )
    )

    assert reflection.preferred_urls == ["https://investor.example.com/quarterly-results"]
    assert reflection.preferred_terms == ["quarterly results detail pages"]
    assert reflection.avoid_urls == []
    assert reflection.avoid_terms == ["shareholders meeting"]
    assert reflection.prompt_guidance.priority_terms == ["earnings conference transcript"]
    assert reflection.prompt_guidance.avoid_terms == ["AGM PDFs"]
    assert reflection.prompt_guidance.navigation_guidance == ""
    assert reflection.prompt_guidance.transcript_guidance == "Prefer official transcript PDF links."
    assert reflection.prompt_guidance.risk_notes == ["stay on official investor pages"]


def test_page_agent_sends_compact_prompt() -> None:
    agent = IRPageAgent.__new__(IRPageAgent)
    captured = {}

    class FakeChain:
        def invoke(self, values):
            captured.update(values)

            class Message:
                content = (
                    '{"page_type":"ir_index","confidence":0.8,'
                    '"useful_urls":["https://example.com/events"],'
                    '"reason":"events"}'
                )

            return Message()

    agent.chain = FakeChain()
    agent.max_links = 1
    agent.text_chars = 20
    links = [
        CandidateLink(
            url="https://example.com/events",
            label="Events " * 20,
            source_url="https://example.com",
            reason="nav " * 20,
        ),
        CandidateLink(
            url="https://example.com/careers",
            label="Careers",
            source_url="https://example.com",
        ),
    ]

    agent.decide(
        company_name="Example",
        ticker="EX",
        url="https://example.com/investors",
        title="Investors",
        text="word " * 100,
        links=links,
    )

    assert len(captured["text"]) <= 20
    assert captured["links_json"].count("https://example.com/") == 1
    assert "source_url" not in captured["links_json"]
