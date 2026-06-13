from pathlib import Path

from ir_transcripts.memory import (
    apply_crawl_reflection,
    load_company_memory,
    memory_path,
    remember_crawl_result,
    save_company_memory,
)
from ir_transcripts.models import (
    CandidatePage,
    Company,
    CompanyMemory,
    CrawlAttemptConfig,
    CrawlFailure,
    CrawlReflection,
    CrawlResult,
    FailureAnalysis,
    PromptGuidance,
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
    assert memory.navigation_memory.successful_hosts == ["investor.example.com"]
    assert memory.navigation_memory.preferred_hosts == ["investor.example.com"]
    assert memory.navigation_memory.known_transcript_urls == ["https://investor.example.com/q1-transcript.pdf"]


def test_memory_does_not_mark_blog_or_youtube_as_official_hosts() -> None:
    company = Company(symbol="GOOG", name="Alphabet Google")
    memory = remember_crawl_result(
        load_company_memory(Path("/tmp/nonexistent-memory-root"), company),
        CrawlResult(
            company=company,
            ir_url="https://abc.xyz/investor/",
            candidates=[
                CandidatePage(company=company, url="https://blog.google/alphabet/earnings"),
                CandidatePage(company=company, url="https://www.youtube.com/watch?v=abc"),
            ],
        ),
        FailureAnalysis(category="no_transcript_found", summary="No transcript"),
    )

    assert "abc.xyz" in memory.official_hosts
    assert "blog.google" not in memory.official_hosts
    assert "www.youtube.com" not in memory.official_hosts
    assert "blog.google" in memory.navigation_memory.low_value_hosts
    assert "www.youtube.com" in memory.navigation_memory.low_value_hosts


def test_reflection_memory_learns_preferred_and_avoid_paths() -> None:
    company = Company(symbol="TSM", name="Taiwan Semiconductor Manufacturing Company")
    memory = apply_crawl_reflection(
        load_company_memory(Path("/tmp/nonexistent-memory-root"), company),
        CrawlReflection(
            preferred_urls=["https://investor.example.com/english/quarterly-results/2026/q1"],
            preferred_terms=["quarterly results detail pages"],
            avoid_urls=[
                "https://investor.example.com/english/shareholders-meeting",
                "https://investor.example.com/japanese/shareholders-meeting/2026",
            ],
            avoid_terms=["shareholders-meeting", "japanese/"],
            prompt_guidance=PromptGuidance(
                priority_terms=["earnings conference transcript"],
                avoid_terms=["AGM PDFs"],
                navigation_guidance="Prefer quarterly result detail pages before shareholder meeting branches.",
                transcript_guidance="Prefer official earnings conference transcript PDF links.",
            ),
        ),
    )

    assert "https://investor.example.com/english/quarterly-results/2026/q1" in memory.navigation_memory.known_event_listing_urls
    assert "https://investor.example.com/english/shareholders-meeting" in memory.rejected_urls
    assert "shareholders-meeting" in memory.navigation_memory.low_value_path_terms
    assert "japanese/" in memory.navigation_memory.low_value_path_terms
    assert memory.prompt_guidance
    assert "quarterly results detail pages" in memory.prompt_guidance.priority_terms
    assert "AGM PDFs" in memory.prompt_guidance.avoid_terms


def test_reflection_memory_marks_wrong_start_host_low_value_without_poisoning_good_host() -> None:
    company = Company(symbol="TSM", name="Taiwan Semiconductor Manufacturing Company")
    memory = apply_crawl_reflection(
        load_company_memory(Path("/tmp/nonexistent-memory-root"), company),
        CrawlReflection(
            preferred_urls=["https://investor.tsmc.com/english/quarterly-results"],
            avoid_urls=[
                "https://www.taiwansemi.com/en/investor-relations/",
                "https://investor.tsmc.com/english/shareholders-meeting",
            ],
            avoid_terms=["shareholders-meeting"],
        ),
    )

    assert "www.taiwansemi.com" in memory.navigation_memory.low_value_hosts
    assert "investor.tsmc.com" not in memory.navigation_memory.low_value_hosts


def test_goog_parent_company_identity_is_not_penalized_by_reflection_memory() -> None:
    company = Company(symbol="GOOG", name="Alphabet Google")
    memory = apply_crawl_reflection(
        load_company_memory(Path("/tmp/nonexistent-memory-root"), company),
        CrawlReflection(
            preferred_urls=["https://abc.xyz/investor/events/event-details/2026/q1"],
            preferred_terms=["official investor event detail pages"],
            avoid_terms=["blog.google"],
        ),
    )

    assert "https://abc.xyz/investor/events/event-details/2026/q1" in memory.known_ir_urls
    assert "abc.xyz" not in memory.navigation_memory.low_value_hosts
    assert "blog.google" in memory.navigation_memory.low_value_path_terms


def test_supervised_retry_uses_playwright_after_nvidia_like_first_pass(tmp_path: Path) -> None:
    company = Company(symbol="NVDA", name="NVIDIA")
    seen_configs: list[CrawlAttemptConfig] = []

    def runner(run_company: Company, config: CrawlAttemptConfig) -> CrawlResult:
        seen_configs.append(config.model_copy(deep=True))
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


def test_homepage_unverified_triggers_supervised_retry(tmp_path: Path) -> None:
    company = Company(symbol="AMZN", name=None)
    seen_configs: list[CrawlAttemptConfig] = []

    def runner(run_company: Company, config: CrawlAttemptConfig) -> CrawlResult:
        seen_configs.append(config)
        if len(seen_configs) == 1:
            return CrawlResult(
                company=run_company,
                failures=[
                    CrawlFailure(
                        company=run_company,
                        url="https://www.example.com",
                        failure_type="homepage_unverified",
                        message="no predicted homepage could be verified",
                    )
                ],
            )
        return CrawlResult(
            company=run_company,
            transcripts=[
                TranscriptRecord(
                    company=run_company,
                    source_url="https://ir.example.com/q1-transcript.pdf",
                    title="Q1 Transcript",
                    text="Operator: Welcome.\nJane Doe: Thanks.\nQuestion-and-answer session\nEND",
                )
            ],
            visited_count=1,
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
    assert result.analyses[0].category == "homepage_unverified"
    assert result.actions[0].action_type == "identity_retry"
    assert len(seen_configs) == 2


def test_crawl_step_failure_triggers_supervised_retry(tmp_path: Path) -> None:
    company = Company(symbol="EX", name="Example")
    seen_configs: list[CrawlAttemptConfig] = []

    def runner(run_company: Company, config: CrawlAttemptConfig) -> CrawlResult:
        seen_configs.append(config)
        if len(seen_configs) == 1:
            return CrawlResult(
                company=run_company,
                failures=[
                    CrawlFailure(
                        company=run_company,
                        url="https://www.example.com/investors",
                        failure_type="page_classification_failed",
                        message="LLM failed",
                    )
                ],
                visited_count=1,
            )
        return CrawlResult(
            company=run_company,
            transcripts=[
                TranscriptRecord(
                    company=run_company,
                    source_url="https://www.example.com/q1-transcript",
                    title="Q1 Transcript",
                    text="Operator: Welcome.\nJane Doe: Thanks.\nQuestion-and-answer session\nEND",
                )
            ],
            visited_count=1,
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
    assert result.analyses[0].category == "crawl_step_failed"
    assert result.actions[0].action_type == "step_retry"
    assert len(seen_configs) == 2


def test_retryable_no_useful_links_triggers_supervised_retry(tmp_path: Path) -> None:
    company = Company(symbol="AMZN", name=None)
    seen_configs: list[CrawlAttemptConfig] = []

    def runner(run_company: Company, config: CrawlAttemptConfig) -> CrawlResult:
        seen_configs.append(config)
        if len(seen_configs) == 1:
            return CrawlResult(
                company=run_company,
                skipped_reason="No investor-relations candidates found",
            )
        return CrawlResult(
            company=run_company,
            transcripts=[
                TranscriptRecord(
                    company=run_company,
                    source_url="https://investors.example.com/q1-transcript",
                    title="Q1 Transcript",
                    text="Operator: Welcome.\nJane Doe: Thanks.\nQuestion-and-answer session\nEND",
                )
            ],
            visited_count=1,
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
    assert result.analyses[0].category == "no_useful_links"
    assert result.actions[0].action_type == "retry"
    assert seen_configs[1].reason == "no_useful_links_retry"


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


def test_prompt_planner_guidance_updates_retry_attempt(monkeypatch, tmp_path: Path) -> None:
    company = Company(symbol="GOOG", name="Alphabet Google")
    seen_guidance: list[PromptGuidance | None] = []

    class FakePromptPlannerAgent:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def plan(self, *, company: Company, memory):
            return PromptGuidance(
                priority_terms=["successful transcript host", "official investor event-detail pages"],
                avoid_terms=["blog.google"] if memory.rejected_urls else [],
                navigation_guidance="Prefer official investor event pages.",
                transcript_guidance="Prefer speaker-turn transcripts.",
            )

    class FakeReflectionAgent:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def reflect(self, *, company: Company, evidence):
            return CrawlReflection(avoid_terms=["blog.google"])

    monkeypatch.setattr("ir_transcripts.orchestration.PromptPlannerAgent", FakePromptPlannerAgent)
    monkeypatch.setattr("ir_transcripts.orchestration.CrawlReflectionAgent", FakeReflectionAgent)

    def runner(run_company: Company, config: CrawlAttemptConfig) -> CrawlResult:
        seen_guidance.append(config.prompt_guidance)
        if config.attempt == 1:
            return CrawlResult(
                company=run_company,
                candidates=[
                    CandidatePage(
                        company=run_company,
                        url="https://abc.xyz/investor/events",
                        title="Alphabet Investor Events",
                        reason="transcript_rejected_missing_speaker_structure",
                    )
                ],
                failures=[
                    CrawlFailure(
                        company=run_company,
                        url="https://blog.google/alphabet/earnings/",
                        failure_type="http_error",
                    )
                ],
                visited_count=10,
            )
        return CrawlResult(
            company=run_company,
            transcripts=[
                TranscriptRecord(
                    company=run_company,
                    source_url="https://abc.xyz/investor/events/event-details/2026/q1",
                    title="Alphabet Q1 2026 Earnings Call",
                    text="OPERATOR: Welcome.\nQUESTION-AND-ANSWER SESSION\nEND",
                )
            ],
            visited_count=1,
        )

    supervisor = SupervisedCrawler(
        model="test-model",
        out_dir=tmp_path,
        http=None,  # type: ignore[arg-type]
        max_attempts=2,
        base_config=CrawlAttemptConfig(use_prompt_planner=True, max_pages_per_company=10),
        attempt_runner=runner,
    )

    result = supervisor.run_company(company)
    memory = load_company_memory(tmp_path, company)

    assert result.status == "success"
    assert seen_guidance[0]
    assert seen_guidance[0].avoid_terms == []
    assert seen_guidance[1]
    assert seen_guidance[1].avoid_terms == ["blog.google"]
    seen_configs = [attempt for attempt in result.attempts]
    assert seen_configs
    assert seen_configs[1].navigation_memory
    assert memory.prompt_guidance
    assert memory.prompt_guidance.avoid_terms == ["blog.google"]


def test_disable_memory_ignores_persisted_memory_but_keeps_run_memory_for_retries(tmp_path: Path) -> None:
    company = Company(symbol="AMZN", name=None)
    persisted = CompanyMemory(company=Company(symbol="AMZN", name="Amazon.com"))
    persisted.navigation_memory.preferred_hosts = ["investors.amazon.com"]
    persisted.navigation_memory.known_event_listing_urls = ["https://investors.amazon.com/earnings.aspx"]
    persisted.prompt_guidance = PromptGuidance(priority_terms=["persisted-only"])
    saved_path = save_company_memory(tmp_path, persisted)
    original_memory_json = saved_path.read_text(encoding="utf-8")
    seen_configs: list[CrawlAttemptConfig] = []

    def runner(run_company: Company, config: CrawlAttemptConfig) -> CrawlResult:
        seen_configs.append(config.model_copy(deep=True))
        if config.attempt == 1:
            return CrawlResult(
                company=run_company,
                failures=[
                    CrawlFailure(
                        company=run_company,
                        url="https://blog.example.com/earnings",
                        failure_type="homepage_unverified",
                    )
                ],
                visited_count=0,
            )
        return CrawlResult(
            company=run_company,
            transcripts=[
                TranscriptRecord(
                    company=run_company,
                    source_url="https://investors.amazon.com/q1-transcript.pdf",
                    title="Amazon Q1 2026 Earnings Call Transcript",
                    text="OPERATOR: Welcome.\nQUESTION-AND-ANSWER SESSION\nEND",
                )
            ],
            visited_count=1,
        )

    supervisor = SupervisedCrawler(
        model="test-model",
        out_dir=tmp_path,
        http=None,  # type: ignore[arg-type]
        max_attempts=2,
        base_config=CrawlAttemptConfig(disable_memory=True),
        attempt_runner=runner,
    )

    result = supervisor.run_company(company)

    assert result.status == "success"
    assert result.memory_path is None
    assert len(seen_configs) == 2
    assert seen_configs[0].identity_name_hint is None
    assert seen_configs[0].prompt_guidance is None
    assert seen_configs[0].navigation_memory
    assert seen_configs[0].navigation_memory.preferred_hosts == []
    assert seen_configs[0].navigation_memory.known_event_listing_urls == []
    assert seen_configs[1].navigation_memory
    assert seen_configs[1].navigation_memory.low_value_hosts == ["blog.example.com"]
    assert "investors.amazon.com" not in seen_configs[1].navigation_memory.preferred_hosts
    assert result.attempts[0].navigation_memory
    assert result.attempts[0].navigation_memory.low_value_hosts == []
    assert result.attempts[1].navigation_memory
    assert result.attempts[1].navigation_memory.low_value_hosts == ["blog.example.com"]
    assert result.actions[0].next_config
    assert result.actions[0].next_config.navigation_memory
    assert result.actions[0].next_config.navigation_memory.low_value_hosts == []
    assert memory_path(tmp_path, company).read_text(encoding="utf-8") == original_memory_json


def test_reflection_filters_urls_that_are_absent_from_crawl_evidence(monkeypatch, tmp_path: Path) -> None:
    company = Company(symbol="AMZN", name=None)
    seen_configs: list[CrawlAttemptConfig] = []

    class FakePromptPlannerAgent:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def plan(self, *, company: Company, memory):
            return PromptGuidance(priority_terms=["AMZN"])

    class FakeReflectionAgent:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def reflect(self, *, company: Company, evidence):
            assert evidence["navigation_steps"] == []
            return CrawlReflection(
                preferred_urls=["https://investors.amazon.com/earnings.aspx"],
                avoid_urls=["https://aboutamazon.com/news/company-news"],
                preferred_terms=["quarterly earnings reports"],
                avoid_terms=["news pages"],
            )

    monkeypatch.setattr("ir_transcripts.orchestration.PromptPlannerAgent", FakePromptPlannerAgent)
    monkeypatch.setattr("ir_transcripts.orchestration.CrawlReflectionAgent", FakeReflectionAgent)

    def runner(run_company: Company, config: CrawlAttemptConfig) -> CrawlResult:
        seen_configs.append(config.model_copy(deep=True))
        if config.attempt == 1:
            return CrawlResult(company=run_company, skipped_reason="No investor-relations candidates found")
        return CrawlResult(
            company=run_company,
            transcripts=[
                TranscriptRecord(
                    company=run_company,
                    source_url="https://example.com/q1-transcript.pdf",
                    title="Amazon Q1 2026 Earnings Call Transcript",
                    text="OPERATOR: Welcome.\nQUESTION-AND-ANSWER SESSION\nEND",
                )
            ],
            visited_count=1,
        )

    supervisor = SupervisedCrawler(
        model="test-model",
        out_dir=tmp_path,
        http=None,  # type: ignore[arg-type]
        max_attempts=2,
        base_config=CrawlAttemptConfig(use_prompt_planner=True, disable_memory=True),
        attempt_runner=runner,
    )

    result = supervisor.run_company(company)

    assert result.status == "success"
    assert len(seen_configs) == 2
    assert seen_configs[1].navigation_memory
    assert seen_configs[1].navigation_memory.known_event_listing_urls == []
    assert seen_configs[1].navigation_memory.low_value_hosts == []
    assert seen_configs[1].prompt_guidance
    assert "quarterly earnings reports" in seen_configs[1].prompt_guidance.priority_terms
    assert "news pages" in seen_configs[1].prompt_guidance.avoid_terms


def test_reflection_guidance_updates_retry_attempt(monkeypatch, tmp_path: Path) -> None:
    company = Company(symbol="TSM", name="Taiwan Semiconductor Manufacturing Company")
    seen_guidance: list[PromptGuidance | None] = []

    class FakePromptPlannerAgent:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def plan(self, *, company: Company, memory):
            return PromptGuidance(
                priority_terms=["known event listing URL"],
                navigation_guidance="Prefer known event listing URLs.",
            )

    class FakeReflectionAgent:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def reflect(self, *, company: Company, evidence):
            assert evidence["candidates"][0]["url"] == "https://investor.example.com/english/shareholders-meeting"
            return CrawlReflection(
                preferred_urls=["https://investor.example.com/english/quarterly-results/2026/q1"],
                preferred_terms=["quarterly results detail pages"],
                avoid_urls=["https://investor.example.com/english/shareholders-meeting"],
                avoid_terms=["shareholders-meeting"],
                prompt_guidance=PromptGuidance(
                    priority_terms=["earnings conference transcript"],
                    avoid_terms=["AGM PDFs"],
                    navigation_guidance="Prefer quarterly result detail pages before shareholder branches.",
                    transcript_guidance="Prefer official earnings conference transcript PDFs.",
                ),
            )

    monkeypatch.setattr("ir_transcripts.orchestration.PromptPlannerAgent", FakePromptPlannerAgent)
    monkeypatch.setattr("ir_transcripts.orchestration.CrawlReflectionAgent", FakeReflectionAgent)

    def runner(run_company: Company, config: CrawlAttemptConfig) -> CrawlResult:
        seen_guidance.append(config.prompt_guidance)
        if config.attempt == 1:
            return CrawlResult(
                company=run_company,
                candidates=[
                    CandidatePage(
                        company=run_company,
                        url="https://investor.example.com/english/shareholders-meeting",
                        title="Shareholders Meeting",
                        reason="transcript_rejected_missing_speaker_structure",
                    )
                ],
                failures=[
                    CrawlFailure(
                        company=run_company,
                        url="https://investor.example.com/sites/ir/shareholders-meeting/2026/AGM.pdf",
                        failure_type="not_transcript",
                    )
                ],
                visited_count=10,
            )
        return CrawlResult(
            company=run_company,
            transcripts=[
                TranscriptRecord(
                    company=run_company,
                    source_url="https://investor.example.com/english/reports/TSM-Transcript.pdf",
                    title="TSM Earnings Conference Transcript",
                    text="OPERATOR: Welcome.\nQUESTION-AND-ANSWER SESSION\nEND",
                )
            ],
            visited_count=2,
        )

    supervisor = SupervisedCrawler(
        model="test-model",
        out_dir=tmp_path,
        http=None,  # type: ignore[arg-type]
        max_attempts=2,
        base_config=CrawlAttemptConfig(use_prompt_planner=True, max_pages_per_company=10),
        attempt_runner=runner,
    )

    result = supervisor.run_company(company)
    memory = load_company_memory(tmp_path, company)

    assert result.status == "success"
    assert seen_guidance[1]
    assert "quarterly results detail pages" in seen_guidance[1].priority_terms
    assert "shareholders-meeting" in seen_guidance[1].avoid_terms
    assert memory.prompt_guidance
    assert "earnings conference transcript" in memory.prompt_guidance.priority_terms
