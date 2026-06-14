from __future__ import annotations

from dataclasses import dataclass

from .http import HttpClient


BROWSER_COMPATIBLE_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/125.0.0.0 Safari/537.36"
)


@dataclass
class PlaywrightRenderer:
    """Render JavaScript-heavy IR pages after robots.txt has allowed the URL."""

    http: HttpClient
    wait_until: str = "domcontentloaded"
    timeout_ms: int = 15_000

    def render_html(self, url: str) -> str:
        return self.render_html_with_user_agent(url, self.http.user_agent)

    def render_html_with_user_agent(self, url: str, user_agent: str) -> str:
        self.http.check_robots(url)

        self.http.wait_for_host(url)

        from playwright.sync_api import sync_playwright

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            try:
                page = browser.new_page(
                    user_agent=user_agent,
                    locale="en-US",
                    extra_http_headers={"Accept-Language": "en-US,en;q=0.9"},
                )
                page.goto(url, wait_until=self.wait_until, timeout=self.timeout_ms)
                return page.content()
            finally:
                browser.close()
