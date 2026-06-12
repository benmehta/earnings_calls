from ir_transcripts.agent import (
    IRNavigationAgent,
    IRPageAgent,
    extract_json_object,
    fallback_links_for_kind,
    guidance_from_memory,
    hard_validate_navigation_decision,
    navigation_agent_kind,
)
from ir_transcripts.models import CandidateLink, Company, CompanyMemory, NavigationDecision, NavigationValidationResult, PromptGuidance


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


def test_extract_json_object_handles_fenced_output() -> None:
    assert extract_json_object('```json\n{"page_type":"ir_index"}\n```') == {"page_type": "ir_index"}


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
