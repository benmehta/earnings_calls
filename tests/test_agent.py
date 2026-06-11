from ir_transcripts.agent import IRNavigationAgent, IRPageAgent, extract_json_object
from ir_transcripts.models import CandidateLink


def test_page_agent_converts_useful_urls_to_links(monkeypatch) -> None:
    agent = IRPageAgent.__new__(IRPageAgent)

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

    class FakeChain:
        def invoke(self, values):
            class Message:
                content = (
                    '{"chosen_urls":["https://example.com/investors",'
                    '"https://invented.example.com"],'
                    '"confidence":0.9,"reason":"footer Investors link",'
                    '"stop_reason":null}'
                )

            return Message()

    agent.homepage_chain = FakeChain()
    agent.ir_chain = FakeChain()
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
