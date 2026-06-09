from urllib.robotparser import RobotFileParser

from ir_transcripts.http import HttpClient


def parser_from(text: str) -> RobotFileParser:
    parser = RobotFileParser()
    parser.parse(text.splitlines())
    return parser


def test_robots_disallow_blocks_url() -> None:
    client = HttpClient(user_agent="test-bot/0.1")
    client._robots["https://example.com"] = parser_from(
        "User-agent: *\nDisallow: /private\n"
    )

    assert not client.allowed("https://example.com/private/transcript")
    assert client.allowed("https://example.com/public/transcript")


def test_crawl_delay_uses_robot_policy() -> None:
    client = HttpClient(user_agent="test-bot/0.1")
    client._robots["https://example.com"] = parser_from(
        "User-agent: *\nCrawl-delay: 7\n"
    )

    assert client.crawl_delay("https://example.com/events") == 7


def test_robots_fetch_error_fails_closed_by_default() -> None:
    client = HttpClient(user_agent="test-bot/0.1")
    client._robots_for = lambda origin: None  # type: ignore[method-assign]

    assert not client.allowed("https://example.com/events")

