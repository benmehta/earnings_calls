from pathlib import Path

from ir_transcripts.memory import load_company_memory
from ir_transcripts.models import (
    CandidatePage,
    Company,
    CrawlAttemptConfig,
    CrawlFailure,
    CrawlResult,
    TranscriptRecord,
)
from ir_transcripts.orchestration import (
    SupervisedCrawler,
    analyze_crawl_result,
    plan_next_action,
    resolve_company_identity,
)


def test_identity_resolver_corrects_goog_parent_company() -> None:
    resolved = resolve_company_identity(Company(symbol="GOOG", name="GOOG"))

    assert resolved.name == "Alphabet Google"


def test_playwright_retry_for_js_heavy_ir_page() -> None:
    result = CrawlResult(
        company=Company(symbol="EX", name="Example"),
        candidates=[
            CandidatePage(
                company=Company(symbol="EX", name="Example"),
                url="https://investor.example.com/financial-reports",
                title="Example Financial Reports",
                depth=0,
                reason="transcript_rejected_missing_speaker_structure",
            )
        ],
        visited_count=1,
    )
    config = CrawlAttemptConfig(use_playwright=False)

    analysis = analyze_crawl_result(result, config)
    action = plan_next_action(
        analysis,
        company=result.company,
        config=config,
        attempt_count=1,
        max_attempts=2,
    )

    assert analysis.category == "render_needed"
    assert action.action_type == "playwright_retry"
    assert action.next_config
    assert action.next_config.use_playwright


def test_deeper_crawl_retry_for_shallow_promising_page() -> None:
    result = CrawlResult(
        company=Company(symbol="EX", name="Example"),
        candidates=[
            CandidatePage(
                company=Company(symbol="EX", name="Example"),
                url="https://investor.example.com/events",
                title="Example Events",
                depth=1,
                llm_page_type="ir_index",
            )
        ],
        visited_count=5,
    )
    config = CrawlAttemptConfig(max_depth=1, use_playwright=True)

    analysis = analyze_crawl_result(result, config)
    action = plan_next_action(
        analysis,
        company=result.company,
        config=config,
        attempt_count=1,
        max_attempts=2,
    )

    assert analysis.category == "needs_deeper_crawl"
    assert action.action_type == "deeper_crawl"
    assert action.next_config
    assert action.next_config.max_depth == 2


def test_robots_unavailable_transcript_doc_requires_manual_review() -> None:
    result = CrawlResult(
        company=Company(symbol="NVDA", name="NVIDIA"),
        failures=[
            CrawlFailure(
                company=Company(symbol="NVDA", name="NVIDIA"),
                url="https://s201.q4cdn.com/files/NVDA-Q1-2027-Earnings-Call.pdf",
                failure_type="robots_unavailable",
            )
        ],
    )
    config = CrawlAttemptConfig()

    analysis = analyze_crawl_result(result, config)
    action = plan_next_action(
        analysis,
        company=result.company,
        config=config,
        attempt_count=1,
        max_attempts=2,
    )

    assert analysis.category == "robots_unavailable"
    assert action.action_type == "manual_review"
    assert action.next_config is None
    assert "--robots-fail-open" in action.manual_recommendations[0]


def test_successful_transcript_stops_graph_and_updates_memory(tmp_path: Path) -> None:
    company = Company(symbol="EX", name="Example")

    def runner(run_company: Company, config: CrawlAttemptConfig) -> CrawlResult:
        return CrawlResult(
            company=run_company,
            ir_url="https://investor.example.com",
            transcripts=[
                TranscriptRecord(
                    company=run_company,
                    source_url="https://investor.example.com/q1-transcript.pdf",
                    title="Q1 Transcript",
                    text="OPERATOR: Welcome.\nJANE DOE: Thanks.\nQUESTION-AND-ANSWER SESSION\nEND",
                )
            ],
            visited_count=1,
        )

    supervisor = SupervisedCrawler(
        model="test-model",
        out_dir=tmp_path,
        http=None,  # type: ignore[arg-type]
        attempt_runner=runner,
    )

    result = supervisor.run_company(company)
    memory = load_company_memory(tmp_path, company)

    assert result.status == "success"
    assert len(result.attempts) == 1
    assert memory.successful_transcript_urls == ["https://investor.example.com/q1-transcript.pdf"]


def test_supervised_retry_uses_playwright_after_nvidia_like_first_pass(tmp_path: Path) -> None:
    company = Company(symbol="NVDA", name="NVIDIA")
    seen_configs: list[CrawlAttemptConfig] = []

    def runner(run_company: Company, config: CrawlAttemptConfig) -> CrawlResult:
        seen_configs.append(config)
        if not config.use_playwright:
            return CrawlResult(
                company=run_company,
                ir_url="https://investor.nvidia.com/financial-info/financial-reports/default.aspx",
                candidates=[
                    CandidatePage(
                        company=run_company,
                        url="https://investor.nvidia.com/financial-info/financial-reports/default.aspx",
                        title="NVIDIA Financial Reports",
                        reason="transcript_rejected_missing_speaker_structure",
                    )
                ],
                visited_count=1,
            )
        return CrawlResult(
            company=run_company,
            ir_url="https://investor.nvidia.com/financial-info/financial-reports/default.aspx",
            transcripts=[
                TranscriptRecord(
                    company=run_company,
                    source_url="https://s201.q4cdn.com/NVDA-Q1-2027-Earnings-Call.pdf",
                    title="NVDA Q1 2027 Earnings Call",
                    text="Operator: Welcome.\nJensen Huang: Thank you.\nQuestion-and-answer session\nEND",
                )
            ],
            visited_count=2,
        )

    supervisor = SupervisedCrawler(
        model="test-model",
        out_dir=tmp_path,
        http=None,  # type: ignore[arg-type]
        max_attempts=2,
        attempt_runner=runner,
    )

    result = supervisor.run_company(company)

    assert result.status == "success"
    assert len(seen_configs) == 2
    assert seen_configs[1].use_playwright


def test_alphabet_supervisor_corrects_identity_without_third_party_retry(tmp_path: Path) -> None:
    company = Company(symbol="GOOG", name="GOOG")

    def runner(run_company: Company, config: CrawlAttemptConfig) -> CrawlResult:
        return CrawlResult(
            company=run_company,
            ir_url="https://abc.xyz/investor/",
            candidates=[
                CandidatePage(
                    company=run_company,
                    url="https://abc.xyz/investor/earnings/",
                    title="Alphabet Investor Relations - Earnings",
                    depth=1,
                    llm_page_type="ir_index",
                )
            ],
            visited_count=2,
        )

    supervisor = SupervisedCrawler(
        model="test-model",
        out_dir=tmp_path,
        http=None,  # type: ignore[arg-type]
        max_attempts=1,
        attempt_runner=runner,
    )

    result = supervisor.run_company(company)

    assert result.final_company.name == "Alphabet Google"
    assert all("quartr" not in url for analysis in result.analyses for url in analysis.evidence_urls)


def test_microsoft_like_robots_failure_is_summarized_without_retry(tmp_path: Path) -> None:
    company = Company(symbol="MSFT", name="Microsoft")

    def runner(run_company: Company, config: CrawlAttemptConfig) -> CrawlResult:
        return CrawlResult(
            company=run_company,
            failures=[
                CrawlFailure(
                    company=run_company,
                    url="https://cdn-dynmedia-1.microsoft.com/is/content/microsoftcorp/TranscriptQandAFY26Q3",
                    failure_type="robots_unavailable",
                    message="Could not verify robots.txt",
                )
            ],
        )

    supervisor = SupervisedCrawler(
        model="test-model",
        out_dir=tmp_path,
        http=None,  # type: ignore[arg-type]
        max_attempts=2,
        attempt_runner=runner,
    )

    result = supervisor.run_company(company)

    assert result.status == "manual_review"
    assert len(result.attempts) == 1
    assert result.actions[-1].action_type == "manual_review"
