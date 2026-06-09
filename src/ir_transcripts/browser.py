from __future__ import annotations

from dataclasses import dataclass

from .http import HttpClient


@dataclass
class PlaywrightRenderer:
    """Render JavaScript-heavy IR pages after robots.txt has allowed the URL."""

    http: HttpClient
    wait_until: str = "networkidle"
    timeout_ms: int = 30_000

    def render_html(self, url: str) -> str:
        if not self.http.allowed(url):
            raise PermissionError(f"Blocked by robots.txt: {url}")

        self.http.wait_for_host(url)

        from playwright.sync_api import sync_playwright

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            try:
                page = browser.new_page(user_agent=self.http.user_agent)
                page.goto(url, wait_until=self.wait_until, timeout=self.timeout_ms)
                return page.content()
            finally:
                browser.close()

