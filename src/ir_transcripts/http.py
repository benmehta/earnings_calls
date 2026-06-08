from __future__ import annotations

import time
from dataclasses import dataclass
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import requests
from tenacity import retry, stop_after_attempt, wait_exponential


DEFAULT_USER_AGENT = "local-ir-transcript-research/0.1 (+contact: you@example.com)"


@dataclass
class HttpClient:
    user_agent: str = DEFAULT_USER_AGENT
    delay_seconds: float = 1.5
    timeout_seconds: float = 25.0
    respect_robots: bool = True

    def __post_init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.user_agent})
        self._robots: dict[str, RobotFileParser] = {}

    def allowed(self, url: str) -> bool:
        if not self.respect_robots:
            return True

        parsed = urlparse(url)
        base = f"{parsed.scheme}://{parsed.netloc}"
        if base not in self._robots:
            parser = RobotFileParser()
            parser.set_url(f"{base}/robots.txt")
            try:
                parser.read()
            except Exception:
                return True
            self._robots[base] = parser

        return self._robots[base].can_fetch(self.user_agent, url)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    def get(self, url: str) -> requests.Response:
        if not self.allowed(url):
            raise PermissionError(f"Blocked by robots.txt: {url}")

        time.sleep(self.delay_seconds)
        response = self.session.get(url, timeout=self.timeout_seconds)
        response.raise_for_status()
        return response

