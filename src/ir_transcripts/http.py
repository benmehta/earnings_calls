from __future__ import annotations

import time
from dataclasses import dataclass
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential


DEFAULT_USER_AGENT = "local-ir-transcript-research/0.1 (+mailto:you@example.com)"


class RobotsError(PermissionError):
    """Base class for robots.txt permission failures."""


class RobotsDisallowedError(RobotsError):
    """The site's robots.txt explicitly disallows the URL."""


class RobotsUnavailableError(RobotsError):
    """robots.txt could not be fetched or parsed and fail-closed is enabled."""


@dataclass
class HttpClient:
    user_agent: str = DEFAULT_USER_AGENT
    delay_seconds: float = 1.5
    timeout_seconds: float = 25.0
    respect_robots: bool = True
    fail_closed_on_robots_error: bool = True

    def __post_init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.user_agent})
        self._robots: dict[str, RobotFileParser] = {}
        self._last_request_at: dict[str, float] = {}

    def allowed(self, url: str) -> bool:
        try:
            self.check_robots(url)
        except RobotsError:
            return False
        return True

    def check_robots(self, url: str) -> None:
        if not self.respect_robots:
            return

        origin = self._origin(url)
        parser = self._robots_for(origin)
        if parser is None:
            if self.fail_closed_on_robots_error:
                raise RobotsUnavailableError(f"Could not verify robots.txt for {url}")
            return

        if not parser.can_fetch(self.user_agent, url):
            raise RobotsDisallowedError(f"Disallowed by robots.txt: {url}")

    def crawl_delay(self, url: str) -> float | None:
        if not self.respect_robots:
            return None

        parser = self._robots_for(self._origin(url))
        if parser is None:
            return None

        return parser.crawl_delay(self.user_agent)

    @retry(
        retry=retry_if_exception_type(requests.RequestException),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
    )
    def get(self, url: str) -> requests.Response:
        self.check_robots(url)

        self.wait_for_host(url)
        response = self.session.get(url, timeout=self.timeout_seconds)
        response.raise_for_status()
        return response

    def _robots_for(self, origin: str) -> RobotFileParser | None:
        if origin in self._robots:
            return self._robots[origin]

        parser = RobotFileParser()
        parser.set_url(f"{origin}/robots.txt")
        try:
            response = self.session.get(f"{origin}/robots.txt", timeout=self.timeout_seconds)
            if response.status_code == 404:
                parser.parse([])
                self._robots[origin] = parser
                return parser
            if response.status_code >= 400:
                return None
            parser.parse(response.text.splitlines())
        except Exception:
            return None

        self._robots[origin] = parser
        return parser

    def wait_for_host(self, url: str) -> None:
        origin = self._origin(url)
        delay = max(self.delay_seconds, self.crawl_delay(url) or 0.0)
        elapsed = time.monotonic() - self._last_request_at.get(origin, 0.0)
        if elapsed < delay:
            time.sleep(delay - elapsed)
        self._last_request_at[origin] = time.monotonic()

    def _origin(self, url: str) -> str:
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}"


def random_browser_user_agent() -> str:
    try:
        from fake_useragent import UserAgent

        return UserAgent().random
    except Exception:
        return DEFAULT_USER_AGENT
