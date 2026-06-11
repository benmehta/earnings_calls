from ir_transcripts.agent import IRNavigationAgent, IRPageAgent, extract_json_object, navigation_agent_kind
from ir_transcripts.models import CandidateLink, NavigationDecision


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
