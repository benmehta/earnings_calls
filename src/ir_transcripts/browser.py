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

    def recover_html_with_actions(
        self,
        url: str,
        actions: list[str],
        user_agent: str | None = None,
        targets: list[dict[str, str]] | None = None,
    ) -> str:
        self.http.check_robots(url)

        self.http.wait_for_host(url)

        from playwright.sync_api import sync_playwright

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            try:
                page = browser.new_page(
                    user_agent=user_agent or self.http.user_agent,
                    locale="en-US",
                    extra_http_headers={"Accept-Language": "en-US,en;q=0.9"},
                )
                page.goto(url, wait_until=self.wait_until, timeout=self.timeout_ms)
                if "wait_for_dynamic_content" in actions:
                    try:
                        page.wait_for_load_state("networkidle", timeout=min(self.timeout_ms, 10_000))
                    except Exception:
                        page.wait_for_timeout(1_500)
                self._apply_recovery_targets(page, targets or [])
                if "select_latest_option" in actions:
                    self._select_first_real_options(page)
                if "expand_disclosure_controls" in actions:
                    self._expand_disclosure_controls(page)
                if actions:
                    try:
                        page.wait_for_load_state("networkidle", timeout=min(self.timeout_ms, 10_000))
                    except Exception:
                        page.wait_for_timeout(1_500)
                return page.content()
            finally:
                browser.close()

    def _apply_recovery_targets(self, page, targets: list[dict[str, str]]) -> None:
        for target in targets[:12]:
            selector = target.get("selector", "")
            action = target.get("action", "")
            if not selector:
                continue
            try:
                locator = page.locator(selector).first
                if not locator.is_visible(timeout=800):
                    continue
                if action == "select_option":
                    value = target.get("value", "")
                    if value:
                        locator.select_option(value=value, timeout=1_500)
                        page.wait_for_timeout(500)
                elif action == "click":
                    locator.click(timeout=1_500)
                    page.wait_for_timeout(500)
            except Exception:
                continue

    def _select_first_real_options(self, page) -> None:
        selects = page.locator("select")
        for index in range(min(selects.count(), 8)):
            select = selects.nth(index)
            try:
                if not select.is_visible(timeout=500):
                    continue
                options = select.locator("option")
                for option_index in range(min(options.count(), 12)):
                    option = options.nth(option_index)
                    value = (option.get_attribute("value") or "").strip()
                    label = option.inner_text(timeout=500).strip()
                    if not value or label.lower() in {"select", "select year", "all"}:
                        continue
                    select.select_option(value=value, timeout=1_500)
                    break
            except Exception:
                continue

    def _expand_disclosure_controls(self, page) -> None:
        selectors = [
            "button[aria-expanded='false']",
            "[role='button'][aria-expanded='false']",
            "summary",
            "button:has-text('Show')",
            "button:has-text('View')",
            "button:has-text('Expand')",
            "button:has-text('Results')",
            "button:has-text('Quarter')",
        ]
        clicked = 0
        for selector in selectors:
            controls = page.locator(selector)
            for index in range(min(controls.count(), 12)):
                if clicked >= 20:
                    return
                control = controls.nth(index)
                try:
                    if not control.is_visible(timeout=500):
                        continue
                    control.click(timeout=1_500)
                    clicked += 1
                    page.wait_for_timeout(250)
                except Exception:
                    continue
