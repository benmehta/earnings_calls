from collections import deque
from io import BytesIO
from zipfile import ZipFile

from ir_transcripts.crawler import (
    SeedDiscoveryResult,
    TranscriptCrawler,
    artifact_stem,
    content_hash,
    crawl_identity_url,
    fiscal_period_key,
    is_document_like_link,
    is_docx_response,
    is_non_english_variant,
    keep_latest_transcripts,
    link_score,
    low_value_after_transcript_document,
    merge_candidate_links,
    rendered_page_dynamic_hints,
    rendered_page_recovery_targets,
    structurally_prioritized_links,
)
from ir_transcripts.http import RobotsDisallowedError, RobotsUnavailableError
from ir_transcripts.models import CandidateLink, Company, CrawlNavigatorDecision, CrawlResult, DocumentLinkTriageDecision, EarningsArtifactExtractionDecision, EarningsArtifactSelection, LatestTranscriptSelectionDecision, LinkBatchTriageDecision, LinkTriageSelection, NavigationTrace, PageDecision, PageDecisionDraft, RenderedPageRecoveryDecision, TranscriptDocumentRankingDecision, TranscriptEvidenceDecision, TranscriptRecord
from ir_transcripts.navigation import NavigationDiscoveryResult


class FakeDocumentTriageAgent:
    def triage(self, *, link: CandidateLink, **kwargs):
        haystack = f"{link.url} {link.label}".lower()
        is_transcript = "transcript" in haystack or "earnings-call" in haystack or "earnings call" in haystack
        return DocumentLinkTriageDecision(
            is_priority_transcript_document=is_transcript,
            confidence=0.9 if is_transcript else 0.2,
            document_type="earnings_call_transcript" if is_transcript else "other",
            reason="fake triage",
        )


class FakeTranscriptEvidenceAgent:
    def classify(self, *, text: str, title: str = "", url: str = "", **kwargs):
        haystack = f"{title} {url} {text}".lower()
        negative = "press release" in haystack and "operator:" not in haystack
        is_transcript = not negative and (
            "lseg streetevents edited transcript" in haystack
            or "operator:" in haystack
            or "question-and-answer" in haystack
            or "question and answer" in haystack
        )
        return TranscriptEvidenceDecision(
            is_transcript=is_transcript,
            confidence=0.9 if is_transcript else 0.1,
            evidence=["fake transcript evidence"] if is_transcript else [],
            rejection_reason="transcript_rejected_press_release_like" if negative else "transcript_rejected_fake_evidence",
        )


class CapturingTranscriptEvidenceAgent(FakeTranscriptEvidenceAgent):
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def classify(self, **kwargs):
        self.calls.append(kwargs)
        return super().classify(**kwargs)


class FakeLinkBatchTriageAgent:
    def triage(self, *, links: list[CandidateLink], **kwargs):
        selections = []
        for link in links:
            haystack = f"{link.url} {link.label}".lower()
            is_useful = any(token in haystack for token in ("transcript", "earnings-call", "quarterly", "events", "financial"))
            priority = 95 if "transcript" in haystack or "earnings-call" in haystack else 60
            selections.append(
                LinkTriageSelection(
                    url=link.url,
                    priority=priority,
                    should_follow=is_useful,
                    reason="fake link triage",
                )
            )
        return LinkBatchTriageDecision(selections=selections)


class FakeEarningsArtifactExtractionAgent:
    def extract(self, *, links: list[CandidateLink], **kwargs):
        selections = []
        for link in links:
            haystack = f"{link.url} {link.label} {link.reason}".lower()
            if "transcript" in haystack:
                selections.append(
                    EarningsArtifactSelection(
                        url=link.url,
                        role="transcript",
                        priority=95,
                        confidence=0.9,
                        reason="fake transcript artifact",
                    )
                )
            elif any(token in haystack for token in ("income statement", "financial statements", "performance", "metrics")):
                selections.append(
                    EarningsArtifactSelection(
                        url=link.url,
                        role="financial_statement",
                        priority=30,
                        confidence=0.8,
                        reason="fake financial table",
                    )
                )
        return EarningsArtifactExtractionDecision(selections=selections, reason="fake artifacts")


class FakeLatestTranscriptSelectionAgent:
    def select(self, *, links: list[CandidateLink], **kwargs):
        selected = next((link for link in links if "2027" in f"{link.url} {link.label}"), None)
        return LatestTranscriptSelectionDecision(
            selected_urls=[selected.url] if selected else [],
            confidence=0.9 if selected else 0.0,
            reason="fake latest selection",
        )


class NoopTranscriptDocumentRankingAgent:
    def rank(self, *, links: list[CandidateLink], **kwargs):
        return TranscriptDocumentRankingDecision(
            ordered_urls=[],
            confidence=0.0,
            reason="fake ranking disabled",
        )


def install_fake_transcript_evidence(crawler: TranscriptCrawler) -> TranscriptCrawler:
    crawler.transcript_evidence_agent = FakeTranscriptEvidenceAgent()  # type: ignore[assignment]
    return crawler


def test_link_score_prefers_transcripts() -> None:
    transcript = CandidateLink(
        url="https://investors.example.com/q1-earnings-call-transcript",
        label="Q1 earnings call transcript",
        source_url="https://investors.example.com",
    )
    careers = CandidateLink(
        url="https://example.com/careers",
        label="Careers",
        source_url="https://investors.example.com",
    )

    assert link_score(transcript) > link_score(careers)


def test_heuristic_links_prioritize_transcript_document_after_page_chrome(tmp_path) -> None:
    chrome_links = [
        CandidateLink(
            url=f"https://investor.example.com/financial-info/archive-{index}",
            label=f"Financial archive {index}",
            source_url="https://investor.example.com/financial-reports",
        )
        for index in range(40)
    ]
    transcript = CandidateLink(
        url="https://cdn.example.com/files/doc_financials/2027/q1/EX-Q1-2027-Earnings-Call.pdf",
        label="Q1 Transcript 2027 (opens in new window)",
        source_url="https://investor.example.com/financial-reports",
    )
    crawler = TranscriptCrawler(model="test-model", out_dir=tmp_path, seed_urls=["https://investor.example.com"])
    crawler.link_triage_agent = FakeLinkBatchTriageAgent()  # type: ignore[assignment]

    prioritized = crawler._heuristic_links(
        Company(symbol="EX", name="Example"),
        "https://investor.example.com/financial-reports",
        "Financial Reports",
        [*chrome_links, transcript],
    )

    assert prioritized[0] == transcript
    assert transcript in prioritized


def test_homepage_prioritize_fallback_keeps_navigation_links(tmp_path) -> None:
    homepage_link = CandidateLink(
        url="https://www.example.com/investors",
        label="Investor Relations",
        source_url="https://www.example.com",
    )
    corporate_pdf = CandidateLink(
        url="https://www.example.com/content/dam/corporate-overview.pdf",
        label="Corporate overview PDF",
        source_url="https://www.example.com",
    )
    crawler = TranscriptCrawler(model="test-model", out_dir=tmp_path, seed_urls=["https://www.example.com"])
    crawler._triaged_links = lambda company, url, title, links, **kwargs: []  # type: ignore[method-assign]

    prioritized = crawler._prioritize_links(
        Company(symbol="EX", name="Example"),
        "https://www.example.com",
        "Example",
        [homepage_link, corporate_pdf],
        page_context="homepage",
    )

    assert homepage_link in prioritized
    assert corporate_pdf in prioritized


def test_structural_link_fallback_prefers_body_links_before_nav() -> None:
    nav_link = CandidateLink(
        url="https://investor.example.com/stock-info",
        label="Stock Info",
        source_url="https://investor.example.com/events",
        reason="nav",
    )
    body_link = CandidateLink(
        url="https://investor.example.com/events/event-details/2027/q1",
        label="Q1 Financial Results",
        source_url="https://investor.example.com/events",
        reason="body",
    )

    assert structurally_prioritized_links([nav_link, body_link])[0] == body_link


def test_link_batch_triage_receives_structurally_prioritized_links(tmp_path) -> None:
    nav_links = [
        CandidateLink(
            url=f"https://investor.example.com/nav/{index}",
            label=f"Navigation {index}",
            source_url="https://investor.example.com/quarterly-results",
            reason="nav",
        )
        for index in range(20)
    ]
    transcript = CandidateLink(
        url="https://cdn.example.com/files/EX-Q1-2027-Earnings-Call.pdf",
        label="Transcript",
        source_url="https://investor.example.com/quarterly-results",
        reason="body",
    )

    class RecordingLinkBatchTriageAgent:
        def __init__(self) -> None:
            self.received_links: list[CandidateLink] = []

        def triage(self, *, links: list[CandidateLink], **kwargs):
            self.received_links = links
            return LinkBatchTriageDecision(selections=[])

    agent = RecordingLinkBatchTriageAgent()
    crawler = TranscriptCrawler(model="test-model", out_dir=tmp_path, seed_urls=["https://investor.example.com"])
    crawler.link_triage_agent = agent  # type: ignore[assignment]

    crawler._prioritize_links(
        Company(symbol="EX", name="Example"),
        "https://investor.example.com/quarterly-results",
        "Quarterly Results",
        [*nav_links, transcript],
    )

    assert agent.received_links[0] == transcript


def test_earnings_artifact_extraction_prioritizes_msft_style_transcript_link(tmp_path) -> None:
    transcript = CandidateLink(
        url="https://cdn.example.com/is/content/examplecorp/TranscriptQandAFY26Q3",
        label="Transcript",
        source_url="https://www.example.com/en-us/investor/events/fy-2026/earnings-fy-2026-q3",
        reason="body",
    )
    webcast = CandidateLink(
        url="https://www.example.com/en-us/Investor/earnings/FY-2026-Q3/press-release-webcast",
        label="Press Release & Webcast",
        source_url="https://www.example.com/en-us/investor/events/fy-2026/earnings-fy-2026-q3",
        reason="nav",
    )
    financials = CandidateLink(
        url="https://www.example.com/en-us/Investor/earnings/FY-2026-Q3/income-statements",
        label="Financial Statements",
        source_url="https://www.example.com/en-us/investor/events/fy-2026/earnings-fy-2026-q3",
        reason="nav",
    )
    performance = CandidateLink(
        url="https://www.example.com/en-us/Investor/earnings/FY-2026-Q3/performance",
        label="Performance",
        source_url="https://www.example.com/en-us/investor/events/fy-2026/earnings-fy-2026-q3",
        reason="nav",
    )
    metrics = CandidateLink(
        url="https://www.example.com/en-us/Investor/earnings/FY-2026-Q3/metrics",
        label="Metrics",
        source_url="https://www.example.com/en-us/investor/events/fy-2026/earnings-fy-2026-q3",
        reason="nav",
    )

    class FinancialsFirstLinkBatchTriageAgent:
        def triage(self, *, links: list[CandidateLink], **kwargs):
            return LinkBatchTriageDecision(
                selections=[
                    LinkTriageSelection(url=financials.url, priority=95, should_follow=True, reason="financials first"),
                    LinkTriageSelection(url=performance.url, priority=90, should_follow=True, reason="performance second"),
                    LinkTriageSelection(url=metrics.url, priority=85, should_follow=True, reason="metrics third"),
                    LinkTriageSelection(url=webcast.url, priority=80, should_follow=True, reason="webcast fourth"),
                    LinkTriageSelection(url=transcript.url, priority=60, should_follow=True, reason="transcript too low"),
                ]
            )

    crawler = TranscriptCrawler(model="test-model", out_dir=tmp_path, seed_urls=["https://www.example.com/investor"])
    crawler.earnings_artifact_agent = FakeEarningsArtifactExtractionAgent()  # type: ignore[assignment]
    crawler.link_triage_agent = FinancialsFirstLinkBatchTriageAgent()  # type: ignore[assignment]
    diagnostics = CrawlResult(company=Company(symbol="EX", name="Example"), ir_url="https://www.example.com/investor")

    ranked = crawler._prioritize_links(
        Company(symbol="EX", name="Example"),
        "https://www.example.com/en-us/investor/events/fy-2026/earnings-fy-2026-q3",
        "Example Fiscal Year 2026 Third Quarter Earnings Conference Call",
        [financials, performance, webcast, metrics, transcript],
        text="Wednesday, April 29, 2026. Transcript. Press Release & Webcast.",
        page_context="earnings_event",
        result=diagnostics,
    )

    ranked_urls = [link.url for link in ranked]
    assert ranked[0].url == transcript.url
    assert "earnings_artifact role=transcript" in ranked[0].reason
    assert ranked_urls.index(transcript.url) < ranked_urls.index(financials.url)
    assert any(
        candidate.url == transcript.url and "earnings_artifact role=transcript" in candidate.reason
        for candidate in diagnostics.candidates
    )


def test_latest_transcript_selection_prepends_newest_document(tmp_path) -> None:
    older = CandidateLink(
        url="https://cdn.example.com/files/EX-Q1-2026-Earnings-Call.pdf",
        label="Q1 Transcript 2026",
        source_url="https://investor.example.com/financial-reports",
        reason="body",
    )
    newer = CandidateLink(
        url="https://cdn.example.com/files/EX-Q1-2027-Earnings-Call.pdf",
        label="Q1 Transcript 2027",
        source_url="https://investor.example.com/financial-reports",
        reason="body",
    )

    class OlderFirstLinkBatchTriageAgent:
        def triage(self, *, links: list[CandidateLink], **kwargs):
            return LinkBatchTriageDecision(
                selections=[
                    LinkTriageSelection(url=older.url, priority=95, should_follow=True, reason="older transcript"),
                    LinkTriageSelection(url=newer.url, priority=80, should_follow=True, reason="newer transcript"),
                ]
            )

    crawler = TranscriptCrawler(model="test-model", out_dir=tmp_path, seed_urls=["https://investor.example.com"])
    crawler.link_triage_agent = OlderFirstLinkBatchTriageAgent()  # type: ignore[assignment]
    crawler.latest_transcript_selection_agent = FakeLatestTranscriptSelectionAgent()  # type: ignore[assignment]

    ranked = crawler._triaged_links(
        Company(symbol="EX", name="Example"),
        "https://investor.example.com/financial-reports",
        "Financial Reports",
        [older, newer],
    )

    assert ranked[0] == newer
    assert ranked[1] == older


def test_document_triage_rescues_latest_selection_uncertainty(tmp_path) -> None:
    release = CandidateLink(
        url="https://cdn.example.com/files/EX-Q1-2027-Earnings-Release.pdf",
        label="Q1 2027 Earnings Release",
        source_url="https://investor.example.com/quarterly-results",
        reason="document link",
    )
    transcript = CandidateLink(
        url="https://cdn.example.com/files/EX-Q1-2027-Earnings-Call.pdf",
        label="Q1 2027 Earnings Call Transcript",
        source_url="https://investor.example.com/quarterly-results",
        reason="document link",
    )

    class ReleaseFirstLinkBatchTriageAgent:
        def triage(self, *, links: list[CandidateLink], **kwargs):
            return LinkBatchTriageDecision(
                selections=[
                    LinkTriageSelection(url=release.url, priority=95, should_follow=True, reason="release first"),
                    LinkTriageSelection(url=transcript.url, priority=60, should_follow=True, reason="transcript lower"),
                ]
            )

    class UncertainLatestTranscriptSelectionAgent:
        def select(self, *, links: list[CandidateLink], **kwargs):
            return LatestTranscriptSelectionDecision(
                selected_urls=[],
                confidence=0.0,
                reason="no likely written earnings-call transcript found",
            )

    crawler = TranscriptCrawler(model="test-model", out_dir=tmp_path, seed_urls=["https://investor.example.com"])
    crawler.link_triage_agent = ReleaseFirstLinkBatchTriageAgent()  # type: ignore[assignment]
    crawler.latest_transcript_selection_agent = UncertainLatestTranscriptSelectionAgent()  # type: ignore[assignment]
    crawler.transcript_document_ranking_agent = NoopTranscriptDocumentRankingAgent()  # type: ignore[assignment]
    crawler.document_triage_agent = FakeDocumentTriageAgent()  # type: ignore[assignment]

    ranked = crawler._triaged_links(
        Company(symbol="EX", name="Example"),
        "https://investor.example.com/quarterly-results",
        "Quarterly Results",
        [release, transcript],
    )

    assert ranked[0].url == transcript.url
    assert "[agent-selected]" in ranked[0].reason
    assert "document_triage type=earnings_call_transcript" in ranked[0].reason


def test_latest_selection_receives_document_triage_evidence(tmp_path) -> None:
    older = CandidateLink(
        url="https://cdn.example.com/files/EX-Q1-2026-Earnings-Call.pdf",
        label="Q1 2026 Earnings Call",
        source_url="https://investor.example.com/quarterly-results",
        reason="older document",
    )
    newer = CandidateLink(
        url="https://cdn.example.com/files/EX-Q1-2027-Earnings-Call.pdf",
        label="Q1 2027 Earnings Call",
        source_url="https://investor.example.com/quarterly-results",
        reason="newer document",
    )

    class EmptyLinkBatchTriageAgent:
        def triage(self, *, links: list[CandidateLink], **kwargs):
            return LinkBatchTriageDecision(selections=[])

    class TwoStepLatestTranscriptSelectionAgent:
        def __init__(self) -> None:
            self.calls: list[list[CandidateLink]] = []

        def select(self, *, links: list[CandidateLink], **kwargs):
            self.calls.append(links)
            if len(self.calls) == 1:
                return LatestTranscriptSelectionDecision(
                    selected_urls=[],
                    confidence=0.0,
                    reason="uncertain without triage evidence",
                )
            assert all("document_triage type=earnings_call_transcript" in link.reason for link in links)
            return LatestTranscriptSelectionDecision(
                selected_urls=[newer.url],
                confidence=0.9,
                reason="newest triaged transcript",
            )

    latest_agent = TwoStepLatestTranscriptSelectionAgent()
    crawler = TranscriptCrawler(model="test-model", out_dir=tmp_path, seed_urls=["https://investor.example.com"])
    crawler.link_triage_agent = EmptyLinkBatchTriageAgent()  # type: ignore[assignment]
    crawler.latest_transcript_selection_agent = latest_agent  # type: ignore[assignment]
    crawler.transcript_document_ranking_agent = NoopTranscriptDocumentRankingAgent()  # type: ignore[assignment]
    crawler.document_triage_agent = FakeDocumentTriageAgent()  # type: ignore[assignment]

    ranked = crawler._triaged_links(
        Company(symbol="EX", name="Example"),
        "https://investor.example.com/quarterly-results",
        "Quarterly Results",
        [older, newer],
    )

    assert len(latest_agent.calls) == 2
    assert ranked[0].url == newer.url
    assert "newest triaged transcript" in ranked[0].reason


def test_transcript_document_ranking_orders_newest_fiscal_document_first(tmp_path) -> None:
    q4_2026 = CandidateLink(
        url="https://s201.q4cdn.com/141608511/files/doc_financials/2026/q4/NVDA-Q4-2026-Earnings-Call-25-February-2026-5_00-PM-ET.pdf",
        label="Q4 2026 Earnings Call",
        source_url="https://investor.nvidia.com/financial-info/quarterly-results/default.aspx",
        reason="document link",
    )
    q1_2027 = CandidateLink(
        url="https://s201.q4cdn.com/141608511/files/doc_financials/2027/q1/NVDA-Q1-2027-Earnings-Call-20-May-2026-5_00-PM-ET.pdf",
        label="Q1 2027 Earnings Call",
        source_url="https://investor.nvidia.com/financial-info/quarterly-results/default.aspx",
        reason="document link",
    )

    class OlderFirstLinkBatchTriageAgent:
        def triage(self, *, links: list[CandidateLink], **kwargs):
            return LinkBatchTriageDecision(
                selections=[
                    LinkTriageSelection(url=q4_2026.url, priority=95, should_follow=True, reason="older first"),
                    LinkTriageSelection(url=q1_2027.url, priority=90, should_follow=True, reason="newer second"),
                ]
            )

    class UncertainLatestTranscriptSelectionAgent:
        def select(self, *, links: list[CandidateLink], **kwargs):
            return LatestTranscriptSelectionDecision(
                selected_urls=[],
                confidence=0.0,
                reason="uncertain",
            )

    class NewestFirstTranscriptDocumentRankingAgent:
        def rank(self, *, links: list[CandidateLink], **kwargs):
            assert all("document_triage type=earnings_call_transcript" in link.reason for link in links)
            return TranscriptDocumentRankingDecision(
                ordered_urls=[q1_2027.url, q4_2026.url],
                confidence=0.9,
                reason="Q1 2027 has later May 2026 call date than Q4 2026",
            )

    crawler = TranscriptCrawler(model="test-model", out_dir=tmp_path, seed_urls=["https://investor.nvidia.com"])
    crawler.link_triage_agent = OlderFirstLinkBatchTriageAgent()  # type: ignore[assignment]
    crawler.latest_transcript_selection_agent = UncertainLatestTranscriptSelectionAgent()  # type: ignore[assignment]
    crawler.transcript_document_ranking_agent = NewestFirstTranscriptDocumentRankingAgent()  # type: ignore[assignment]
    crawler.document_triage_agent = FakeDocumentTriageAgent()  # type: ignore[assignment]

    ranked = crawler._triaged_links(
        Company(symbol="NVDA", name="NVIDIA"),
        "https://investor.nvidia.com/financial-info/quarterly-results/default.aspx",
        "Quarterly Results",
        [q4_2026, q1_2027],
    )

    assert ranked[0].url == q1_2027.url
    assert ranked[1].url == q4_2026.url
    assert "transcript document ranking after document triage" in ranked[0].reason


def test_crawl_navigator_decision_accepts_object_chosen_urls() -> None:
    decision = CrawlNavigatorDecision.model_validate(
        {
            "page_type": "ir_index",
            "confidence": 0.8,
            "chosen_urls": [{"url": "https://investor.example.com/events"}],
            "reason": "object-shaped model output",
        }
    )

    assert decision.chosen_urls == ["https://investor.example.com/events"]


def test_crawl_navigator_decision_normalizes_page_type_alias() -> None:
    decision = CrawlNavigatorDecision.model_validate(
        {
            "page_type": "transcript_link",
            "confidence": 0.8,
            "chosen_urls": ["https://investor.example.com/results"],
            "reason": "page links to transcript documents",
        }
    )

    assert decision.page_type == "ir_index"


def test_page_decision_draft_normalizes_page_type_alias() -> None:
    decision = PageDecisionDraft.model_validate(
        {
            "page_type": "quarterly_results",
            "confidence": 0.8,
            "useful_urls": ["https://investor.example.com/results"],
            "reason": "quarterly results page with document links",
        }
    )

    assert decision.page_type == "ir_index"


def test_low_value_after_transcript_document_keeps_transcript_links() -> None:
    sec = CandidateLink(
        url="https://investor.example.com/financial-info/sec-filings/default.aspx",
        label="SEC Filings",
        source_url="https://investor.example.com",
    )
    presentation = CandidateLink(
        url="https://investor.example.com/events-and-presentations/presentations/default.aspx",
        label="Presentations",
        source_url="https://investor.example.com",
    )
    transcript = CandidateLink(
        url="https://cdn.example.com/files/Example-Q1-Earnings-Call-Transcript.pdf",
        label="Q1 transcript",
        source_url="https://investor.example.com",
    )
    financial_statement = CandidateLink(
        url="https://investor.example.com/earnings/fy-2026-q3/income-statements",
        label="Income Statements",
        source_url="https://investor.example.com/events/fy-2026/q3",
        reason="[agent-selected] earnings_artifact role=financial_statement confidence=0.90",
    )

    assert low_value_after_transcript_document(sec)
    assert low_value_after_transcript_document(presentation)
    assert low_value_after_transcript_document(financial_statement)
    assert not low_value_after_transcript_document(transcript)


def test_latest_only_queue_pruning_does_not_call_link_triage(tmp_path) -> None:
    sec = CandidateLink(
        url="https://investor.example.com/financial-info/sec-filings/default.aspx",
        label="SEC Filings",
        source_url="https://investor.example.com",
    )
    transcript = CandidateLink(
        url="https://investor.example.com/q1-earnings-call-transcript.pdf",
        label="Q1 Earnings Call Transcript",
        source_url="https://investor.example.com",
    )

    class FailingLinkBatchTriageAgent:
        def triage(self, **kwargs):
            raise AssertionError("queue pruning should not ask the LLM")

    crawler = TranscriptCrawler(
        model="test-model",
        out_dir=tmp_path,
        seed_urls=["https://investor.example.com"],
        latest_only=True,
    )
    crawler.link_triage_agent = FailingLinkBatchTriageAgent()  # type: ignore[assignment]

    queue = crawler._trim_queue_after_transcript_document(
        deque(
            [
                (sec.url, 1, sec, None),
                (transcript.url, 1, transcript, None),
            ]
        )
    )

    assert [item[0] for item in queue] == [transcript.url]


def test_document_like_link_is_mechanical_not_semantic() -> None:
    assert is_document_like_link("https://cdn.example.com/files/release.pdf")
    assert is_document_like_link("https://cdn.example.com/is/content/example/TranscriptQandA")
    assert not is_document_like_link("https://investor.example.com/quarterly-results")


def test_merge_candidate_links_preserves_rendered_and_recovered_documents() -> None:
    latest = CandidateLink(
        url="https://cdn.example.com/EX-Q1-2027-Earnings-Call.pdf",
        label="Q1 2027 Transcript",
        source_url="https://investor.example.com/quarterly-results",
        reason="rendered",
    )
    older = CandidateLink(
        url="https://cdn.example.com/EX-Q4-2026-Earnings-Call.pdf",
        label="Q4 2026 Transcript",
        source_url="https://investor.example.com/quarterly-results",
        reason="recovered",
    )
    duplicate_latest = CandidateLink(
        url="https://cdn.example.com/EX-Q1-2027-Earnings-Call.pdf#maincontent",
        label="Duplicate latest",
        source_url="https://investor.example.com/quarterly-results",
        reason="recovered",
    )

    merged = merge_candidate_links([latest], [older, duplicate_latest])

    assert [link.url for link in merged] == [latest.url, older.url]


def test_artifact_stem_and_content_hash_are_stable() -> None:
    assert artifact_stem("Q1 Transcript", "https://example.com/a") == artifact_stem(
        "Q1 Transcript", "https://example.com/a"
    )
    assert content_hash("hello   world") == content_hash("hello world")


def test_crawl_identity_url_ignores_http_https_scheme() -> None:
    assert crawl_identity_url("http://example.com/investors/") == crawl_identity_url("https://example.com/investors")


def test_crawl_identity_url_dedupes_page_casing_and_fragments() -> None:
    upper = "https://www.example.com/en-us/Investor/earnings/FY-2026-Q3/press-release-webcast#mainContent"
    lower = "https://www.example.com/en-us/investor/earnings/fy-2026-q3/press-release-webcast"

    assert crawl_identity_url(upper) == crawl_identity_url(lower)


def test_crawl_identity_url_preserves_document_path_casing() -> None:
    upper = "https://cdn.example.com/is/content/examplecorp/TranscriptQandAFY26Q3"
    lower = "https://cdn.example.com/is/content/examplecorp/transcriptqandafy26q3"

    assert crawl_identity_url(upper) != crawl_identity_url(lower)


def test_non_english_variant_is_skipped_from_english_page() -> None:
    assert is_non_english_variant(
        "https://investor.example.com/japanese/quarterly-results/2026/q1",
        "https://investor.example.com/english/quarterly-results/2026/q1",
    )
    assert not is_non_english_variant(
        "https://investor.example.com/english/quarterly-results/2026/q2",
        "https://investor.example.com/english/quarterly-results/2026/q1",
    )


def test_fiscal_period_key_handles_common_quarter_shapes() -> None:
    assert fiscal_period_key("2026 Q1 Earnings Call") == (2026, 1)
    assert fiscal_period_key("Q4 FY2025 transcript") == (2025, 4)
    assert fiscal_period_key("first quarter 2026 earnings") == (2026, 1)


def test_keep_latest_transcripts_prefers_newest_fiscal_period() -> None:
    old = TranscriptRecord(
        company=Company(symbol="EX", name="Example"),
        source_url="https://example.com/2025",
        fiscal_period="Q1 2025",
        title="Example Q1 2025 Earnings Call",
        text="OPERATOR: Welcome.",
    )
    latest = TranscriptRecord(
        company=Company(symbol="EX", name="Example"),
        source_url="https://example.com/2026",
        fiscal_period="Q1 2026",
        title="Example Q1 2026 Earnings Call",
        text="OPERATOR: Welcome.",
    )

    assert keep_latest_transcripts([old, latest]) == [latest]


def test_is_docx_response_detects_word_content_type_without_extension() -> None:
    assert is_docx_response(
        "https://cdn.example.com/is/content/example/Transcript",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        b"",
    )


def test_crawler_records_robots_unavailable(tmp_path) -> None:
    class FakeHttp:
        def get(self, url: str):
            raise RobotsUnavailableError(f"Could not verify robots.txt for {url}")

    crawler = TranscriptCrawler(
        model="test-model",
        out_dir=tmp_path,
        seed_urls=["https://example.com/investors"],
        http=FakeHttp(),  # type: ignore[arg-type]
    )

    result = crawler.crawl_company(Company(symbol="EX", name="Example"))

    assert result.failures[0].failure_type == "robots_unavailable"


def test_crawler_records_robots_disallowed(tmp_path) -> None:
    class FakeHttp:
        def get(self, url: str):
            raise RobotsDisallowedError(f"Disallowed by robots.txt: {url}")

    crawler = TranscriptCrawler(
        model="test-model",
        out_dir=tmp_path,
        seed_urls=["https://example.com/private"],
        http=FakeHttp(),  # type: ignore[arg-type]
    )

    result = crawler.crawl_company(Company(symbol="EX", name="Example"))

    assert result.failures[0].failure_type == "robots_disallowed"


def test_crawler_does_not_save_press_release_like_html(tmp_path) -> None:
    class FakeResponse:
        headers = {"content-type": "text/html"}
        content = b""
        text = """
        <html><head><title>FY26 Q3 - Press Releases - Investor Relations - Microsoft</title></head>
        <body>
        <h1>Press Release & Webcast</h1>
        <p>Business Highlights include cloud revenue and Azure growth.</p>
        <p>Webcast Details: executives will host a conference call for analysts.</p>
        <p>Forward-looking statements and non-GAAP financial measures follow.</p>
        </body></html>
        """

    class FakeHttp:
        def get(self, url: str):
            return FakeResponse()

    class FakeAgent:
        def decide(self, **kwargs):
            return PageDecision(page_type="transcript", confidence=1.0, reason="LLM thought transcript")

    crawler = TranscriptCrawler(
        model="test-model",
        out_dir=tmp_path,
        seed_urls=["https://example.com/press-release-webcast"],
        http=FakeHttp(),  # type: ignore[arg-type]
    )
    crawler.agent = FakeAgent()  # type: ignore[assignment]
    crawler.document_triage_agent = FakeDocumentTriageAgent()  # type: ignore[assignment]
    install_fake_transcript_evidence(crawler)

    result = crawler.crawl_company(Company(symbol="EX", name="Example"))

    assert result.transcripts == []
    assert result.candidates[0].reason.endswith("transcript_rejected_press_release_like")
    assert not [path for path in (tmp_path / "EX").glob("*.json") if not path.name.startswith("_")]


def test_crawler_does_not_save_ir_homepage_on_vague_transcript_evidence(tmp_path) -> None:
    class FakeResponse:
        headers = {"content-type": "text/html"}
        content = b""
        text = """
        <html><head><title>Example Investor Relations - Home</title></head>
        <body>
        <nav>Home Investors Results & Financials Earnings SEC Filings News Events Governance</nav>
        <h1>Investor Relations</h1>
        <p>Find earnings, financial results, SEC filings, and investor events.</p>
        </body></html>
        """

    class FakeHttp:
        def get(self, url: str):
            return FakeResponse()

    class FalsePositiveTranscriptEvidenceAgent:
        def classify(self, **kwargs):
            return TranscriptEvidenceDecision(
                is_transcript=True,
                confidence=0.9,
                evidence=["earnings and investor relations page"],
                rejection_reason="",
            )

    class FakeAgent:
        def decide(self, **kwargs):
            return PageDecision(page_type="ir_index", confidence=0.9, reason="IR homepage")

    crawler = TranscriptCrawler(
        model="test-model",
        out_dir=tmp_path,
        seed_urls=["https://investor.example.com/default.aspx"],
        http=FakeHttp(),  # type: ignore[arg-type]
    )
    crawler.agent = FakeAgent()  # type: ignore[assignment]
    crawler.transcript_evidence_agent = FalsePositiveTranscriptEvidenceAgent()  # type: ignore[assignment]
    crawler.document_triage_agent = FakeDocumentTriageAgent()  # type: ignore[assignment]

    result = crawler.crawl_company(Company(symbol="EX", name="Example"))

    assert result.transcripts == []
    assert result.candidates[0].reason.endswith("transcript_rejected_agent_evidence")
    assert not [path for path in (tmp_path / "EX").glob("*.json") if not path.name.startswith("_")]


def test_research_homepage_seed_passes_homepage_context_to_navigator(tmp_path) -> None:
    class FakeResponse:
        headers = {"content-type": "text/html"}
        content = b""
        text = """
        <html><head><title>Example Corp</title></head>
        <body>
          <a href="/content/dam/corporate-overview.pdf">Corporate overview PDF</a>
          <a href="https://investor.example.com/">Investor Relations</a>
        </body></html>
        """

    class FakeHttp:
        def get(self, url: str):
            return FakeResponse()

    seen_contexts: list[str] = []

    class FakeAgent:
        def decide_page(self, **kwargs):
            seen_contexts.append(kwargs.get("page_context"))
            links = {link.label: link for link in kwargs["links"]}
            return PageDecision(
                page_type="ir_index",
                confidence=0.9,
                useful_links=[links["Investor Relations"]],
                reason="homepage-to-IR",
            )

    crawler = TranscriptCrawler(
        model="test-model",
        out_dir=tmp_path,
        max_pages_per_company=1,
        seed_urls=[],
        http=FakeHttp(),  # type: ignore[arg-type]
    )
    crawler.agent = FakeAgent()  # type: ignore[assignment]
    crawler.document_triage_agent = FakeDocumentTriageAgent()  # type: ignore[assignment]
    install_fake_transcript_evidence(crawler)
    crawler._triaged_links = lambda company, url, title, links, **kwargs: links  # type: ignore[method-assign]
    crawler._discover_seed_result = lambda company: SeedDiscoveryResult(  # type: ignore[method-assign]
        seeds=["https://www.example.com/"],
        seed_roles={"https://www.example.com/": "homepage"},
        failures=[],
    )

    result = crawler.crawl_company(Company(symbol="EX", name="Example Corp"))

    assert seen_contexts == ["homepage"]
    assert result.transcripts == []
    assert result.candidates[0].reason.startswith("homepage-to-IR")


def test_homepage_context_skips_non_transcript_documents(tmp_path) -> None:
    class FakeResponse:
        headers = {"content-type": "text/html"}
        content = b""

        def __init__(self, text: str) -> None:
            self.text = text

    class FakeHttp:
        def __init__(self) -> None:
            self.urls: list[str] = []

        def get(self, url: str):
            self.urls.append(url)
            if url == "https://www.example.com/":
                return FakeResponse(
                    """
                    <html><head><title>Example Corp</title></head>
                    <body>
                      <a href="/content/dam/corporate-overview.pdf">Corporate overview PDF</a>
                      <a href="/investors">Investor Relations</a>
                    </body></html>
                    """
                )
            return FakeResponse("<html><head><title>Investors</title></head><body>Investor relations</body></html>")

    class FakeAgent:
        def decide_page(self, **kwargs):
            links = {link.label: link for link in kwargs["links"]}
            useful = []
            if "Corporate overview PDF" in links:
                useful.append(links["Corporate overview PDF"])
            if "Investor Relations" in links:
                useful.append(links["Investor Relations"])
            return PageDecision(page_type="ir_index", confidence=0.9, useful_links=useful, reason="homepage")

    http = FakeHttp()
    crawler = TranscriptCrawler(
        model="test-model",
        out_dir=tmp_path,
        max_pages_per_company=3,
        seed_urls=[],
        http=http,  # type: ignore[arg-type]
    )
    crawler.agent = FakeAgent()  # type: ignore[assignment]
    crawler.document_triage_agent = FakeDocumentTriageAgent()  # type: ignore[assignment]
    crawler.link_triage_agent = FakeLinkBatchTriageAgent()  # type: ignore[assignment]
    install_fake_transcript_evidence(crawler)
    crawler._discover_seed_result = lambda company: SeedDiscoveryResult(  # type: ignore[method-assign]
        seeds=["https://www.example.com/"],
        seed_roles={"https://www.example.com/": "homepage"},
        failures=[],
    )

    crawler.crawl_company(Company(symbol="EX", name="Example Corp"))

    assert "https://www.example.com/content/dam/corporate-overview.pdf" not in http.urls
    assert "https://www.example.com/investors" in http.urls


def test_rendered_page_recovery_agent_exposes_dynamic_transcript_link(monkeypatch, tmp_path) -> None:
    initial_html = """
    <html><head><title>Example Quarterly Results</title></head>
    <body>
      <label for="year">Select Year</label>
      <select id="year"><option value="">Select Year</option><option value="2027">2027</option></select>
      <button aria-expanded="false">First Quarter 2027</button>
      <div id="results">Loading...</div>
      <script type="text/template">{{docUrl}}</script>
    </body></html>
    """
    recovered_html = """
    <html><head><title>Example Quarterly Results</title></head>
    <body>
      <a href="https://cdn.example.com/EX-Q1-2027-Earnings-Call-Transcript.pdf">Q1 Transcript</a>
    </body></html>
    """

    class FakeResponse:
        content = b""

        def __init__(self, text: str = "", content_type: str = "text/html", content: bytes | None = None) -> None:
            self.text = text
            self.content = content if content is not None else text.encode("utf-8")
            self.headers = {"content-type": content_type}

    class FakeHttp:
        def __init__(self) -> None:
            self.urls: list[str] = []

        def get(self, url: str):
            self.urls.append(url)
            if url.endswith(".pdf"):
                return FakeResponse(content_type="application/pdf", content=b"%PDF transcript")
            return FakeResponse("<html><body>shell</body></html>")

    class FakeRenderer:
        def __init__(self) -> None:
            self.actions: list[str] = []

        def render_html(self, url: str) -> str:
            return initial_html

        def recover_html_with_actions(self, url: str, actions: list[str], targets=None) -> str:
            self.actions = actions
            return recovered_html

    class FakeRecoveryAgent:
        def decide(self, **kwargs):
            assert "Select Year" in kwargs["dynamic_hints"]
            return RenderedPageRecoveryDecision(
                should_recover=True,
                confidence=0.9,
                actions=["wait_for_dynamic_content", "select_latest_option", "expand_disclosure_controls"],
                reason="dynamic controls likely hide document links",
            )

    class FakeAgent:
        def decide_page(self, **kwargs):
            return PageDecision(page_type="ir_index", confidence=0.9, useful_links=kwargs["links"], reason="follow recovered links")

    monkeypatch.setattr(
        "ir_transcripts.crawler.pdf_text",
        lambda content: "Operator: Welcome.\nJane Doe: Thanks.\nQuestion-and-answer session\nEND",
    )

    http = FakeHttp()
    renderer = FakeRenderer()
    crawler = TranscriptCrawler(
        model="test-model",
        out_dir=tmp_path,
        max_pages_per_company=3,
        seed_urls=["https://investor.example.com/quarterly-results"],
        playwright_mode="always",
        http=http,  # type: ignore[arg-type]
    )
    crawler.renderer = renderer  # type: ignore[assignment]
    crawler.rendered_page_recovery_agent = FakeRecoveryAgent()  # type: ignore[assignment]
    crawler.agent = FakeAgent()  # type: ignore[assignment]
    crawler.document_triage_agent = FakeDocumentTriageAgent()  # type: ignore[assignment]
    install_fake_transcript_evidence(crawler)

    result = crawler.crawl_company(Company(symbol="EX", name="Example"))

    assert renderer.actions == ["wait_for_dynamic_content", "select_latest_option", "expand_disclosure_controls"]
    assert result.transcripts
    assert result.transcripts[0].source_url == "https://cdn.example.com/EX-Q1-2027-Earnings-Call-Transcript.pdf"


def test_rendered_page_dynamic_hints_describe_generic_controls() -> None:
    hints = rendered_page_dynamic_hints(
        """
        <html><body>
          <label for="year">Select Year</label>
          <select id="year"><option value="">Select Year</option><option value="2027">2027</option></select>
          <button aria-expanded="false">First Quarter</button>
          <script type="text/template">{{docUrl}}</script>
        </body></html>
        """
    )

    assert "Select controls" in hints
    assert "Collapsed controls" in hints
    assert "Dynamic markers" in hints


def test_rendered_page_recovery_targets_include_specific_safe_controls() -> None:
    targets = rendered_page_recovery_targets(
        """
        <html><body>
          <label for="year">Select Year</label>
          <select id="year"><option value="">Select Year</option><option value="2027">2027</option></select>
          <button id="q1" aria-expanded="false">First Quarter 2027</button>
          <a href="/events/event-details/2027/q1">Q1 Financial Results</a>
          <a href="/privacy">Privacy</a>
        </body></html>
        """
    )

    labels = [target["label"] for target in targets]

    assert "Select Year -> 2027" in labels
    assert "First Quarter 2027" in labels
    assert "Q1 Financial Results" in labels
    assert "Privacy" not in labels
    assert all(target["id"].startswith("t") for target in targets)


def test_crawler_always_playwright_mode_renders_plain_html(monkeypatch, tmp_path) -> None:
    class FakeResponse:
        headers = {"content-type": "text/html"}
        content = b""
        text = """
        <html><head><title>Example Events</title></head>
        <body><p>This static page has enough text that auto mode would not need rendering.</p></body></html>
        """

    class FakeHttp:
        def get(self, url: str):
            return FakeResponse()

    class FakeRenderer:
        def __init__(self, http) -> None:
            self.http = http

        def render_html(self, url: str) -> str:
            return """
            <html><head><title>Example Q1 Earnings Call Transcript</title></head>
            <body>
            <p>OPERATOR: Welcome to the call.</p>
            <p>JANE DOE: Thank you.</p>
            <p>JOHN SMITH: Prepared remarks.</p>
            <p>ANALYST: My question is about margins.</p>
            <p>QUESTION-AND-ANSWER SESSION</p>
            <p>END</p>
            </body></html>
            """

    class FakeAgent:
        def decide(self, **kwargs):
            return PageDecision(page_type="transcript", confidence=1.0, reason="rendered transcript")

    monkeypatch.setattr("ir_transcripts.crawler.PlaywrightRenderer", FakeRenderer)
    crawler = TranscriptCrawler(
        model="test-model",
        out_dir=tmp_path,
        seed_urls=["https://investor.example.com/events/q1"],
        http=FakeHttp(),  # type: ignore[arg-type]
        playwright_mode="always",
    )
    crawler.agent = FakeAgent()  # type: ignore[assignment]
    install_fake_transcript_evidence(crawler)

    result = crawler.crawl_company(Company(symbol="EX", name="Example"))

    assert len(result.transcripts) == 1
    assert result.transcripts[0].metadata["rendered_with_playwright"] == "true"


def test_crawler_saves_docx_transcript(tmp_path) -> None:
    class FakeResponse:
        headers = {"content-type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"}

        def __init__(self, content: bytes) -> None:
            self.content = content
            self.text = ""

    class FakeHttp:
        def __init__(self, content: bytes) -> None:
            self.content = content

        def get(self, url: str):
            return FakeResponse(self.content)

    content = docx_fixture(
        [
            "Example Co Q1 Earnings Call Transcript",
            "OPERATOR: Welcome to the call.",
            "JANE DOE: Thank you.",
            "JOHN SMITH: I will discuss results.",
            "ANALYST: My question is about margins.",
            "QUESTION-AND-ANSWER SESSION",
            "END",
        ]
    )
    crawler = TranscriptCrawler(
        model="test-model",
        out_dir=tmp_path,
        seed_urls=["https://cdn.example.com/ExampleTranscript"],
        http=FakeHttp(content),  # type: ignore[arg-type]
    )
    install_fake_transcript_evidence(crawler)

    result = crawler.crawl_company(Company(symbol="EX", name="Example"))

    assert len(result.transcripts) == 1
    assert result.transcripts[0].metadata["format"] == "docx"
    assert list((tmp_path / "EX").glob("*.docx"))


def test_official_linked_docx_can_fetch_when_document_robots_unavailable_with_flag(tmp_path) -> None:
    docx_content = docx_fixture(
        [
            "Example Co Q1 Earnings Call Transcript",
            "OPERATOR: Welcome to the call.",
            "JANE DOE: Thank you.",
            "JOHN SMITH: I will discuss results.",
            "ANALYST: My question is about margins.",
            "QUESTION-AND-ANSWER SESSION",
            "END",
        ]
    )

    class FakeResponse:
        def __init__(self, text: str = "", content: bytes = b"", content_type: str = "text/html") -> None:
            self.text = text
            self.content = content
            self.headers = {"content-type": content_type}

    class FakeHttp:
        def get(self, url: str):
            if url == "https://investor.example.com/events/q1":
                return FakeResponse(
                    """
                    <html><head><title>Example Q1 Earnings</title></head>
                    <body>
                      <a href="https://cdn.example.com/Example-Q1-Earnings-Call-Transcript.docx">
                        Q1 Earnings Call Transcript
                      </a>
                    </body></html>
                    """
                )
            raise RobotsUnavailableError(f"Could not verify robots.txt for {url}")

        def robots_unavailable(self, url: str) -> bool:
            return url == "https://cdn.example.com/Example-Q1-Earnings-Call-Transcript.docx"

        def get_without_robots_check(self, url: str):
            return FakeResponse(
                content=docx_content,
                content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )

    class FakeAgent:
        def decide(self, **kwargs):
            return PageDecision(page_type="earnings_event", confidence=0.7, reason="earnings page", useful_links=[])

    crawler = TranscriptCrawler(
        model="test-model",
        out_dir=tmp_path,
        seed_urls=["https://investor.example.com/events/q1"],
        http=FakeHttp(),  # type: ignore[arg-type]
        allow_official_linked_documents_on_robots_unavailable=True,
    )
    crawler.agent = FakeAgent()  # type: ignore[assignment]
    crawler.document_triage_agent = FakeDocumentTriageAgent()  # type: ignore[assignment]
    install_fake_transcript_evidence(crawler)

    result = crawler.crawl_company(Company(symbol="EX", name="Example"))

    assert len(result.transcripts) == 1
    assert result.transcripts[0].source_url == "https://cdn.example.com/Example-Q1-Earnings-Call-Transcript.docx"
    assert any(candidate.reason == "robots_unavailable_allowed_official_linked_document" for candidate in result.candidates)


def test_official_linked_docx_still_blocks_when_delegated_policy_disabled(tmp_path) -> None:
    class FakeResponse:
        headers = {"content-type": "text/html"}
        content = b""
        text = """
        <html><head><title>Example Q1 Earnings</title></head>
        <body>
          <a href="https://cdn.example.com/Example-Q1-Earnings-Call-Transcript.docx">
            Q1 Earnings Call Transcript
          </a>
        </body></html>
        """

    class FakeHttp:
        def get(self, url: str):
            if url == "https://investor.example.com/events/q1":
                return FakeResponse()
            raise RobotsUnavailableError(f"Could not verify robots.txt for {url}")

    class FakeAgent:
        def decide(self, **kwargs):
            return PageDecision(page_type="earnings_event", confidence=0.7, reason="earnings page", useful_links=[])

    crawler = TranscriptCrawler(
        model="test-model",
        out_dir=tmp_path,
        seed_urls=["https://investor.example.com/events/q1"],
        http=FakeHttp(),  # type: ignore[arg-type]
    )
    crawler.agent = FakeAgent()  # type: ignore[assignment]
    crawler.document_triage_agent = FakeDocumentTriageAgent()  # type: ignore[assignment]
    install_fake_transcript_evidence(crawler)

    result = crawler.crawl_company(Company(symbol="EX", name="Example"))

    assert result.transcripts == []
    assert [failure.failure_type for failure in result.failures] == ["robots_unavailable"]


def test_latest_only_prunes_low_value_queue_after_transcript_document(monkeypatch, tmp_path) -> None:
    class FakeResponse:
        def __init__(self, text: str = "", content: bytes = b"", content_type: str = "text/html") -> None:
            self.text = text
            self.content = content
            self.headers = {"content-type": content_type}

    pdf_content = b"%PDF transcript fixture"
    fetched: list[str] = []

    class FakeHttp:
        def get(self, url: str):
            fetched.append(url)
            if url == "https://investor.example.com/quarterly-results":
                return FakeResponse(
                    """
                    <html><head><title>Quarterly Results</title></head>
                    <body>
                      <a href="https://cdn.example.com/Example-Q1-Earnings-Call-Transcript.pdf">Q1 Transcript</a>
                      <a href="https://investor.example.com/financial-info/sec-filings/default.aspx">SEC Filings</a>
                      <a href="https://investor.example.com/financial-info/annual-meeting/default.aspx">Annual Meeting</a>
                    </body></html>
                    """
                )
            if url == "https://cdn.example.com/Example-Q1-Earnings-Call-Transcript.pdf":
                return FakeResponse(content=pdf_content, content_type="application/pdf")
            return FakeResponse("<html><head><title>Low Value</title></head><body></body></html>")

    class FakeAgent:
        def decide(self, **kwargs):
            return PageDecision(page_type="earnings_event", confidence=0.7, reason="earnings page", useful_links=[])

    monkeypatch.setattr(
        "ir_transcripts.crawler.pdf_text",
        lambda content: (
            "Example Co Q1 2027 Earnings Call Transcript\n"
            "OPERATOR: Welcome to the call.\n"
            "JANE DOE: Thank you.\n"
            "JOHN SMITH: Prepared remarks.\n"
            "ANALYST: My question is about margins.\n"
            "QUESTION-AND-ANSWER SESSION\n"
            "END"
        ),
    )
    crawler = TranscriptCrawler(
        model="test-model",
        out_dir=tmp_path,
        seed_urls=["https://investor.example.com/quarterly-results"],
        http=FakeHttp(),  # type: ignore[arg-type]
        latest_only=True,
        max_pages_per_company=10,
        max_depth=2,
    )
    crawler.agent = FakeAgent()  # type: ignore[assignment]
    crawler.document_triage_agent = FakeDocumentTriageAgent()  # type: ignore[assignment]
    install_fake_transcript_evidence(crawler)

    result = crawler.crawl_company(Company(symbol="EX", name="Example"))

    assert len(result.transcripts) == 1
    assert "https://investor.example.com/financial-info/sec-filings/default.aspx" not in fetched
    assert "https://investor.example.com/financial-info/annual-meeting/default.aspx" not in fetched


def test_priority_document_triage_fetches_transcript_before_remaining_seed_pages(monkeypatch, tmp_path) -> None:
    class FakeResponse:
        def __init__(self, text: str = "", content: bytes = b"", content_type: str = "text/html") -> None:
            self.text = text
            self.content = content
            self.headers = {"content-type": content_type}

    fetched: list[str] = []

    class FakeHttp:
        def get(self, url: str):
            fetched.append(url)
            if url == "https://investor.example.com/financial-info/quarterly-results/default.aspx":
                return FakeResponse(
                    """
                    <html><head><title>Quarterly Results</title></head>
                    <body>
                      <a href="https://cdn.example.com/EX-Q1-2027-Earnings-Call.pdf"
                         aria-label="Transcript of First Quarter 2027, PDF file">Transcript</a>
                      <a href="https://cdn.example.com/EX-Q1-2027-Earnings-Release.pdf">Earnings Release</a>
                    </body></html>
                    """
                )
            if url == "https://cdn.example.com/EX-Q1-2027-Earnings-Call.pdf":
                return FakeResponse(content=b"%PDF transcript fixture", content_type="application/pdf")
            return FakeResponse("<html><head><title>Low Value</title></head><body></body></html>")

    class FakeAgent:
        def decide(self, **kwargs):
            return PageDecision(page_type="ir_index", confidence=0.7, reason="quarterly page", useful_links=[])

    monkeypatch.setattr(
        "ir_transcripts.crawler.pdf_text",
        lambda content: (
            "LSEG STREETEVENTS EDITED TRANSCRIPT\n"
            "Example Co Q1 2027 Earnings Call\n"
            "CORPORATE PARTICIPANTS\n"
            "JANE DOE Example Co - CFO\n"
            "CONFERENCE CALL PARTICIPANTS\n"
            "Analyst One\n"
            "PRESENTATION\n"
            "Operator Welcome.\n"
            "QUESTION AND ANSWER\n"
            "END"
        ),
    )
    crawler = TranscriptCrawler(
        model="test-model",
        out_dir=tmp_path,
        seed_urls=[
            "https://investor.example.com/financial-info/quarterly-results/default.aspx",
            "https://investor.example.com/stock-info/stock-quote-and-chart/default.aspx",
            "https://investor.example.com/financial-info/sec-filings/default.aspx",
        ],
        http=FakeHttp(),  # type: ignore[arg-type]
        max_pages_per_company=2,
        max_depth=2,
    )
    crawler.agent = FakeAgent()  # type: ignore[assignment]
    crawler.document_triage_agent = FakeDocumentTriageAgent()  # type: ignore[assignment]
    transcript_agent = CapturingTranscriptEvidenceAgent()
    crawler.transcript_evidence_agent = transcript_agent  # type: ignore[assignment]
    crawler.document_triage_agent = FakeDocumentTriageAgent()  # type: ignore[assignment]

    result = crawler.crawl_company(Company(symbol="EX", name="Example"))

    assert len(result.transcripts) == 1
    assert transcript_agent.calls
    assert "Source page: https://investor.example.com/financial-info/quarterly-results/default.aspx" in transcript_agent.calls[-1]["source_context"]
    assert "Link label: Transcript" in transcript_agent.calls[-1]["source_context"]
    assert fetched[:2] == [
        "https://investor.example.com/financial-info/quarterly-results/default.aspx",
        "https://cdn.example.com/EX-Q1-2027-Earnings-Call.pdf",
    ]


def test_latest_only_stops_after_first_saved_transcript_document(monkeypatch, tmp_path) -> None:
    class FakeResponse:
        def __init__(self, text: str = "", content: bytes = b"", content_type: str = "text/html") -> None:
            self.text = text
            self.content = content
            self.headers = {"content-type": content_type}

    fetched: list[str] = []

    class FakeHttp:
        def get(self, url: str):
            fetched.append(url)
            if url == "https://investor.example.com/quarterly-results":
                return FakeResponse(
                    """
                    <html><head><title>Quarterly Results</title></head>
                    <body>
                      <a href="https://investor.example.com/q1-transcript.pdf">Q1 Transcript</a>
                      <a href="https://investor.example.com/quarterly-results/2025/q4">Older quarter</a>
                    </body></html>
                    """
                )
            if url == "https://investor.example.com/q1-transcript.pdf":
                return FakeResponse(content=b"%PDF transcript fixture", content_type="application/pdf")
            return FakeResponse("<html><head><title>Older quarter</title></head><body></body></html>")

    class FakeAgent:
        def decide(self, **kwargs):
            return PageDecision(page_type="earnings_event", confidence=0.7, reason="earnings page", useful_links=[])

    monkeypatch.setattr(
        "ir_transcripts.crawler.pdf_text",
        lambda content: (
            "LSEG STREETEVENTS EDITED TRANSCRIPT\n"
            "Example Co Q1 2027 Earnings Call\n"
            "CORPORATE PARTICIPANTS\n"
            "JANE DOE Example Co - CFO\n"
            "CONFERENCE CALL PARTICIPANTS\n"
            "Analyst One\n"
            "PRESENTATION\n"
            "Operator Welcome.\n"
            "QUESTION AND ANSWER\n"
            "END"
        ),
    )
    crawler = TranscriptCrawler(
        model="test-model",
        out_dir=tmp_path,
        seed_urls=["https://investor.example.com/quarterly-results"],
        http=FakeHttp(),  # type: ignore[arg-type]
        latest_only=True,
        max_pages_per_company=10,
        max_depth=2,
    )
    crawler.agent = FakeAgent()  # type: ignore[assignment]
    install_fake_transcript_evidence(crawler)

    result = crawler.crawl_company(Company(symbol="EX", name="Example"))

    assert len(result.transcripts) == 1
    assert "https://investor.example.com/quarterly-results/2025/q4" not in fetched


def test_crawler_latest_only_keeps_newest_transcript_artifacts(tmp_path) -> None:
    class FakeResponse:
        headers = {"content-type": "text/html"}
        content = b""

        def __init__(self, text: str) -> None:
            self.text = text

    class FakeHttp:
        def get(self, url: str):
            year = "2026" if "2026" in url else "2025"
            return FakeResponse(
                f"""
                <html><head><title>Example Q1 {year} Earnings Call Transcript</title></head>
                <body>
                  <p>OPERATOR: Welcome everyone.</p>
                  <p>JANE DOE: Thank you.</p>
                  <p>JOHN SMITH: Prepared remarks.</p>
                  <p>ANALYST: My question is about margins.</p>
                  <p>QUESTION-AND-ANSWER SESSION</p>
                  <p>END</p>
                </body></html>
                """
            )

    class FakeAgent:
        def decide(self, **kwargs):
            return PageDecision(page_type="transcript", confidence=1.0, reason="transcript", useful_links=[])

    crawler = TranscriptCrawler(
        model="test-model",
        out_dir=tmp_path,
        seed_urls=["https://example.com/q1-2025", "https://example.com/q1-2026"],
        http=FakeHttp(),  # type: ignore[arg-type]
        latest_only=True,
    )
    crawler.agent = FakeAgent()  # type: ignore[assignment]
    install_fake_transcript_evidence(crawler)

    result = crawler.crawl_company(Company(symbol="EX", name="Example"))

    assert len(result.transcripts) == 1
    assert result.transcripts[0].fiscal_period == "Q1 2026"
    remaining = [path.name for path in (tmp_path / "EX").glob("Example_Q1_*") if not path.name.startswith("_")]
    assert any("2026" in name for name in remaining)
    assert not any("2025" in name for name in remaining)


def test_nav_first_uses_navigation_before_search(monkeypatch, tmp_path) -> None:
    company = Company(symbol="EX", name="Example")
    monkeypatch.setattr(
        "ir_transcripts.crawler.discover_navigation_seeds",
        lambda *args, **kwargs: NavigationDiscoveryResult(
            seeds=["https://example.com/investors"],
            trace=NavigationTrace(company=company),
        ),
    )
    monkeypatch.setattr("ir_transcripts.crawler.find_ir_candidates", lambda *args, **kwargs: ["https://search.example.com"])
    crawler = TranscriptCrawler(model="test-model", out_dir=tmp_path, discovery_mode="nav-first")

    assert crawler._discover_seeds(company) == ["https://example.com/investors"]


def test_crawler_delegates_unavailable_robots_for_navigation_verified_ir_subdomain(monkeypatch, tmp_path) -> None:
    company = Company(symbol="EX", name="Example")
    monkeypatch.setattr(
        "ir_transcripts.crawler.discover_navigation_seeds",
        lambda *args, **kwargs: NavigationDiscoveryResult(
            seeds=["https://ir.example.com/earnings"],
            trace=NavigationTrace(company=company),
            robots_verified_official_urls=["https://www.example.com"],
        ),
    )

    class FakeResponse:
        headers = {"content-type": "text/html"}
        text = """
        <html><head><title>Example Q1 2026 Earnings Call Transcript</title></head>
        <body>
          <p>OPERATOR: Welcome to the Example Q1 2026 earnings call.</p>
          <p>JANE DOE: Thank you.</p>
          <p>QUESTION-AND-ANSWER SESSION</p>
          <p>END</p>
        </body></html>
        """
        content = text.encode("utf-8")

    class FakeHttp:
        def __init__(self) -> None:
            self.delegated_fetches: list[str] = []

        def get(self, url: str):
            raise RobotsUnavailableError(f"Could not verify robots.txt for {url}")

        def get_without_robots_check(self, url: str):
            self.delegated_fetches.append(url)
            return FakeResponse()

    class FakeAgent:
        def decide(self, **kwargs):
            return PageDecision(page_type="transcript", confidence=1.0, reason="transcript", useful_links=[])

    http = FakeHttp()
    crawler = TranscriptCrawler(
        model="test-model",
        out_dir=tmp_path,
        discovery_mode="nav-first",
        http=http,  # type: ignore[arg-type]
    )
    crawler.agent = FakeAgent()  # type: ignore[assignment]
    install_fake_transcript_evidence(crawler)

    result = crawler.crawl_company(company)

    assert http.delegated_fetches == ["https://ir.example.com/earnings"]
    assert len(result.transcripts) == 1


def test_nav_first_does_not_fallback_to_search_when_navigation_empty(monkeypatch, tmp_path) -> None:
    company = Company(symbol="EX", name="Example")
    monkeypatch.setattr(
        "ir_transcripts.crawler.discover_navigation_seeds",
        lambda *args, **kwargs: NavigationDiscoveryResult(
            seeds=[],
            trace=NavigationTrace(company=company),
        ),
    )

    def fail_search(*args, **kwargs):
        raise AssertionError("search fallback should not run")

    monkeypatch.setattr("ir_transcripts.crawler.find_ir_candidates", fail_search)
    crawler = TranscriptCrawler(model="test-model", out_dir=tmp_path, discovery_mode="nav-first")

    assert crawler._discover_seeds(company) == []


def test_nav_first_empty_result_preserves_verified_homepage_identity(monkeypatch, tmp_path) -> None:
    company = Company(symbol="AMZN", name=None)
    monkeypatch.setattr(
        "ir_transcripts.crawler.discover_navigation_seeds",
        lambda *args, **kwargs: NavigationDiscoveryResult(
            seeds=[],
            trace=NavigationTrace(company=company),
            verified_homepage_urls=["https://www.amazon.com"],
            verified_company_name="Amazon",
        ),
    )

    crawler = TranscriptCrawler(
        model="test-model",
        out_dir=tmp_path,
        discovery_mode="nav-first",
    )

    result = crawler.crawl_company(company)

    assert result.verified_homepage_urls == ["https://www.amazon.com"]
    assert result.verified_company_name == "Amazon"


def test_search_first_preserves_search_discovery(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("ir_transcripts.crawler.find_ir_candidates", lambda *args, **kwargs: ["https://search.example.com"])
    crawler = TranscriptCrawler(model="test-model", out_dir=tmp_path, discovery_mode="search-first")

    assert crawler._discover_seeds(Company(symbol="EX", name="Example")) == ["https://search.example.com"]


def test_seed_url_bypasses_discovery(monkeypatch, tmp_path) -> None:
    def fail_discovery(*args, **kwargs):
        raise AssertionError("discovery should not run")

    monkeypatch.setattr("ir_transcripts.crawler.discover_navigation_seeds", fail_discovery)
    monkeypatch.setattr("ir_transcripts.crawler.find_ir_candidates", fail_discovery)
    crawler = TranscriptCrawler(
        model="test-model",
        out_dir=tmp_path,
        seed_urls=["https://seed.example.com"],
        discovery_mode="nav-first",
    )

    assert crawler._discover_seeds(Company(symbol="EX", name="Example")) == ["https://seed.example.com"]


def docx_fixture(paragraphs: list[str]) -> bytes:
    buffer = BytesIO()
    body = "".join(f"<w:p><w:r><w:t>{paragraph}</w:t></w:r></w:p>" for paragraph in paragraphs)
    document_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body>{body}</w:body>"
        "</w:document>"
    )
    with ZipFile(buffer, "w") as archive:
        archive.writestr("word/document.xml", document_xml)
    return buffer.getvalue()
