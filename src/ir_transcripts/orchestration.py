from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from .agent import (
    CompanyPlaybookAgent,
    CrawlReflectionAgent,
    MemoryGuidanceAgent,
    OrchestrationAnalysisAgent,
    PromptPlannerAgent,
    RetryPlanningAgent,
    guidance_from_playbook,
    merge_prompt_guidance as merge_agent_prompt_guidance,
)
from .crawler import TranscriptCrawler
from .http import HttpClient
from .identity import company_display_name, resolve_company_identity_with_overrides
from .memory import apply_crawl_reflection, load_company_memory, memory_path, remember_crawl_result, save_company_memory
from .models import (
    CandidatePage,
    Company,
    CompanyMemory,
    CrawlAttemptConfig,
    CrawlFailure,
    CrawlReflection,
    CrawlResult,
    FailureAnalysis,
    NavigationTrace,
    OrchestrationAnalysisDecision,
    RetryPlanningDecision,
    SupervisorAction,
    SupervisorRunResult,
)
from .runtime import ProgressReporter, timeout_after
from .urls import normalize_url


AttemptRunner = Callable[[Company, CrawlAttemptConfig], CrawlResult]


def resolve_company_identity(company: Company) -> Company:
    return resolve_company_identity_with_overrides(company)


def should_save_memory_incrementally(config: CrawlAttemptConfig) -> bool:
    return not config.disable_memory and not config.no_memory_write


class SupervisorState(TypedDict, total=False):
    original_company: Company
    current_company: Company
    current_config: CrawlAttemptConfig
    memory: CompanyMemory
    result: CrawlResult
    attempts: list[CrawlAttemptConfig]
    analyses: list[FailureAnalysis]
    actions: list[SupervisorAction]
    max_attempts: int
    out_dir: Path
    should_retry: bool


class SupervisedCrawler:
    """AutoData-inspired supervisor around the existing transcript crawler."""

    def __init__(
        self,
        *,
        model: str,
        out_dir: Path,
        http: HttpClient,
        ollama_base_url: str | None = None,
        base_config: CrawlAttemptConfig | None = None,
        max_attempts: int = 2,
        attempt_runner: AttemptRunner | None = None,
        progress: ProgressReporter | None = None,
    ) -> None:
        self.model = model
        self.out_dir = out_dir
        self.http = http
        self.ollama_base_url = ollama_base_url
        self.base_config = base_config or CrawlAttemptConfig()
        self.max_attempts = max_attempts
        self.attempt_runner = attempt_runner or self._run_crawler_attempt
        self.progress = progress or ProgressReporter(enabled=False)
        self.graph = self._build_graph()

    def run_company(self, company: Company) -> SupervisorRunResult:
        self.progress.log(f"{company.symbol}: supervised run starting")
        load_persisted_memory = not self.base_config.disable_memory
        write_memory = not self.base_config.no_memory_write
        initial_state: SupervisorState = {
            "original_company": company,
            "current_company": company,
            "current_config": self.base_config.model_copy(update={"attempt": 1, "reason": "initial"}),
            "memory": load_company_memory(self.out_dir, company) if load_persisted_memory else CompanyMemory(company=company),
            "attempts": [],
            "analyses": [],
            "actions": [],
            "max_attempts": self.max_attempts,
            "out_dir": self.out_dir,
            "should_retry": False,
        }
        final_state = self.graph.invoke(initial_state)
        result = final_state.get("result")
        analyses = final_state.get("analyses", [])
        actions = final_state.get("actions", [])
        memory = final_state.get("memory") or CompanyMemory(company=company)
        path = save_company_memory(self.out_dir, memory) if write_memory else None
        self.progress.log(f"{company.symbol}: supervised run finished with status={supervisor_status(result, actions)}")
        return SupervisorRunResult(
            company=company,
            final_company=final_state.get("current_company", company),
            attempts=final_state.get("attempts", []),
            analyses=analyses,
            actions=actions,
            result=result,
            memory_path=path,
            status=supervisor_status(result, actions),
        )

    def _build_graph(self):
        graph = StateGraph(SupervisorState)
        graph.add_node("identity", self._identity_node)
        graph.add_node("prompt", self._prompt_node)
        graph.add_node("crawl", self._crawl_node)
        graph.add_node("analyze", self._analyze_node)
        graph.add_node("reflect", self._reflect_node)
        graph.add_node("plan", self._plan_node)
        graph.add_node("retry", self._retry_node)
        graph.add_node("validate", self._validate_node)
        graph.add_edge(START, "identity")
        graph.add_edge("identity", "prompt")
        graph.add_edge("prompt", "crawl")
        graph.add_edge("crawl", "analyze")
        graph.add_edge("analyze", "reflect")
        graph.add_edge("reflect", "plan")
        graph.add_conditional_edges(
            "plan",
            lambda state: "retry" if state.get("should_retry") else "validate",
            {"retry": "retry", "validate": "validate"},
        )
        graph.add_edge("retry", "prompt")
        graph.add_edge("validate", END)
        return graph.compile()

    def _identity_node(self, state: SupervisorState) -> SupervisorState:
        company = state["current_company"]
        config = state["current_config"]
        resolved = resolve_company_identity_with_overrides(
            company,
            allow_homepage_overrides=not config.disable_official_homepage_overrides,
        )
        if resolved != company:
            self.progress.log(f"{company.symbol}: identity resolved to {company_display_name(resolved)}")
            action = SupervisorAction(
                action_type="identity_correction",
                reason=f"Resolved {company.symbol} from {company_display_name(company)} to {company_display_name(resolved)}",
                next_company=resolved,
            )
            state["current_company"] = resolved
            state["actions"] = [*state.get("actions", []), action]
            state["memory"] = state["memory"].model_copy(update={"company": resolved})
        return state

    def _prompt_node(self, state: SupervisorState) -> SupervisorState:
        config = state["current_config"]
        memory = state["memory"]
        memory_name_hint = memory.company.name if memory.company.name and memory.company.name.upper() != memory.company.symbol.upper() else None
        navigation_memory = memory.navigation_memory.model_copy(deep=True)
        homepage_candidates = list(memory.playbook.official_homepage_candidates)
        if not config.use_prompt_planner:
            updates = {
                "navigation_memory": navigation_memory,
                "identity_name_hint": memory_name_hint,
                "homepage_candidates": homepage_candidates,
            }
            if memory.prompt_guidance:
                updates["prompt_guidance"] = memory.prompt_guidance.model_copy(deep=True)
            state["current_config"] = config.model_copy(deep=True, update=updates)
            return state

        company = state["current_company"]
        try:
            self.progress.log(f"{company.symbol}: company playbook agent starting")
            with timeout_after(config.llm_timeout_seconds, f"building company playbook for {company.symbol}"):
                playbook = CompanyPlaybookAgent(
                    self.model,
                    base_url=self.ollama_base_url,
                    text_chars=config.llm_text_chars * 3,
                ).build(company=company, memory=memory)
            if playbook.issuer_name or playbook.official_homepage_candidates or playbook.preferred_ir_urls:
                self.progress.log(f"{company.symbol}: company playbook active")
            memory.playbook = playbook
            if playbook.issuer_name and (not memory.company.name or memory.company.name.upper() == memory.company.symbol.upper()):
                memory.company = memory.company.model_copy(update={"name": playbook.issuer_name})
                state["current_company"] = state["current_company"].model_copy(update={"name": playbook.issuer_name})
                company = state["current_company"]
        except Exception as exc:
            self.progress.log(f"{company.symbol}: company playbook skipped ({type(exc).__name__}: {exc})")

        playbook_guidance = guidance_from_playbook(memory.playbook, ticker=company.symbol)
        memory_guidance = None
        try:
            self.progress.log(f"{company.symbol}: memory guidance agent starting")
            with timeout_after(config.llm_timeout_seconds, f"planning memory guidance for {company.symbol}"):
                memory_guidance = MemoryGuidanceAgent(
                    self.model,
                    base_url=self.ollama_base_url,
                    text_chars=config.llm_text_chars * 2,
                ).guide(company=company, memory=memory)
        except Exception as exc:
            self.progress.log(f"{company.symbol}: memory guidance skipped ({type(exc).__name__}: {exc})")
            memory_guidance = None

        try:
            self.progress.log(f"{company.symbol}: prompt planner starting")
            with timeout_after(config.llm_timeout_seconds, f"planning prompts for {company.symbol}"):
                guidance = PromptPlannerAgent(
                    self.model,
                    base_url=self.ollama_base_url,
                    text_chars=config.llm_text_chars * 2,
                ).plan(company=company, memory=memory)
        except Exception as exc:
            self.progress.log(f"{company.symbol}: prompt planner skipped ({type(exc).__name__}: {exc})")
            guidance = memory.prompt_guidance

        if guidance and memory_guidance:
            guidance = merge_agent_prompt_guidance(guidance, memory_guidance)
        elif memory_guidance:
            guidance = memory_guidance

        if guidance and playbook_guidance:
            guidance = merge_agent_prompt_guidance(guidance, playbook_guidance)
        elif playbook_guidance:
            guidance = playbook_guidance

        if guidance and memory.prompt_guidance:
            guidance = merge_agent_prompt_guidance(guidance, memory.prompt_guidance)

        if guidance:
            self.progress.log(f"{company.symbol}: prompt planner guidance active")
            memory.prompt_guidance = guidance
            state["memory"] = memory
            state["current_config"] = config.model_copy(
                deep=True,
                update={
                    "prompt_guidance": guidance.model_copy(deep=True),
                    "navigation_memory": memory.navigation_memory.model_copy(deep=True),
                    "identity_name_hint": memory_name_hint,
                    "homepage_candidates": homepage_candidates,
                }
            )
            if should_save_memory_incrementally(config):
                save_company_memory(self.out_dir, memory)
        else:
            state["current_config"] = config.model_copy(
                deep=True,
                update={
                    "navigation_memory": memory.navigation_memory.model_copy(deep=True),
                    "identity_name_hint": memory_name_hint,
                    "homepage_candidates": homepage_candidates,
                }
            )
        return state

    def _crawl_node(self, state: SupervisorState) -> SupervisorState:
        company = state["current_company"]
        config = state["current_config"]
        self.progress.log(
            f"{company.symbol}: attempt {config.attempt} starting "
            f"(mode={config.discovery_mode}, depth={config.max_depth}, pages={config.max_pages_per_company})"
        )
        result = self.attempt_runner(company, config)
        self.progress.log(
            f"{company.symbol}: attempt {config.attempt} finished "
            f"({len(result.transcripts)} transcript(s), {result.visited_count} page(s))"
        )
        state["result"] = result
        config_snapshot = config.model_copy(deep=True)
        state["attempts"] = [*state.get("attempts", []), config_snapshot]
        memory = state["memory"]
        memory.attempted_configs.append(config_snapshot.model_copy(deep=True))
        state["memory"] = memory
        return state

    def _analyze_node(self, state: SupervisorState) -> SupervisorState:
        result = state["result"]
        config = state["current_config"]
        analysis_agent = None
        if config.use_prompt_planner:
            analysis_agent = OrchestrationAnalysisAgent(
                self.model,
                base_url=self.ollama_base_url,
                text_chars=config.llm_text_chars * 3,
            )
        analysis = analyze_crawl_result(
            result,
            config,
            analysis_agent=analysis_agent,
            timeout_seconds=config.llm_timeout_seconds,
            progress=self.progress,
        )
        self.progress.log(f"{result.company.symbol}: analysis={analysis.category}: {analysis.summary}")
        memory = remember_crawl_result(state["memory"], result, analysis)
        state["memory"] = memory
        state["analyses"] = [*state.get("analyses", []), analysis]
        if should_save_memory_incrementally(config):
            save_company_memory(self.out_dir, memory)
        return state

    def _reflect_node(self, state: SupervisorState) -> SupervisorState:
        result = state["result"]
        if result.transcripts:
            return state
        analysis = state["analyses"][-1]
        if analysis.manual_recommendations:
            return state

        company = state["current_company"]
        config = state["current_config"]
        if not config.use_prompt_planner:
            return state
        try:
            self.progress.log(f"{company.symbol}: reflection agent starting")
            with timeout_after(config.llm_timeout_seconds, f"reflecting on {company.symbol} crawl"):
                evidence = reflection_evidence(
                    result,
                    analysis,
                    load_navigation_trace(self.out_dir, company),
                )
                reflection = CrawlReflectionAgent(
                    self.model,
                    base_url=self.ollama_base_url,
                    text_chars=config.llm_text_chars * 3,
                ).reflect(
                    company=company,
                    evidence=evidence,
                )
        except Exception as exc:
            self.progress.log(f"{company.symbol}: reflection skipped ({type(exc).__name__}: {exc})")
            return state

        reflection = filter_reflection_urls_to_evidence(reflection, evidence)
        if reflection.reason or reflection.preferred_urls or reflection.avoid_urls or reflection.avoid_terms:
            self.progress.log(f"{company.symbol}: reflection guidance active")
        memory = apply_crawl_reflection(state["memory"], reflection)
        state["memory"] = memory
        if should_save_memory_incrementally(config):
            save_company_memory(self.out_dir, memory)
        return state

    def _plan_node(self, state: SupervisorState) -> SupervisorState:
        analysis = state["analyses"][-1]
        attempts = state.get("attempts", [])
        planner = None
        if state["current_config"].use_prompt_planner:
            planner = RetryPlanningAgent(
                self.model,
                base_url=self.ollama_base_url,
                text_chars=state["current_config"].llm_text_chars * 2,
            )
        action = plan_next_action(
            analysis,
            company=state["current_company"],
            config=state["current_config"],
            attempt_count=len(attempts),
            max_attempts=state.get("max_attempts", self.max_attempts),
            planner=planner,
            timeout_seconds=state["current_config"].llm_timeout_seconds,
            progress=self.progress,
        )
        self.progress.log(f"{state['current_company'].symbol}: supervisor action={action.action_type}: {action.reason}")
        state["actions"] = [*state.get("actions", []), action.model_copy(deep=True)]
        state["should_retry"] = action.next_config is not None and action.action_type not in {"finish", "manual_review"}
        return state

    def _retry_node(self, state: SupervisorState) -> SupervisorState:
        action = state["actions"][-1]
        if action.next_company:
            state["current_company"] = action.next_company
            state["memory"] = state["memory"].model_copy(update={"company": action.next_company})
        if action.next_config:
            state["current_config"] = action.next_config
        state["should_retry"] = False
        return state

    def _validate_node(self, state: SupervisorState) -> SupervisorState:
        if should_save_memory_incrementally(state["current_config"]):
            save_company_memory(self.out_dir, state["memory"])
        return state

    def _run_crawler_attempt(self, company: Company, config: CrawlAttemptConfig) -> CrawlResult:
        crawler = TranscriptCrawler(
            model=self.model,
            out_dir=self.out_dir,
            max_pages_per_company=config.max_pages_per_company,
            max_depth=config.max_depth,
            use_playwright=config.use_playwright,
            playwright_mode=config.playwright_mode,
            ollama_base_url=self.ollama_base_url,
            resume=config.resume,
            review_only=config.review_only,
            seed_urls=config.seed_urls,
            discovery_mode=config.discovery_mode,
            include_discovery_guesses=config.include_discovery_guesses,
            disable_official_homepage_overrides=config.disable_official_homepage_overrides,
            disable_predictive_identity=config.disable_predictive_identity,
            rerank_discovery=config.rerank_discovery,
            extract_metadata_with_llm=config.extract_metadata_with_llm,
            latest_only=config.latest_only,
            allow_official_linked_documents_on_robots_unavailable=config.allow_official_linked_documents_on_robots_unavailable,
            prompt_guidance=config.prompt_guidance,
            navigation_memory=config.navigation_memory,
            identity_name_hint=config.identity_name_hint,
            homepage_candidates=config.homepage_candidates,
            search_timeout_seconds=config.search_timeout_seconds,
            llm_timeout_seconds=config.llm_timeout_seconds,
            navigation_llm_max_links=config.navigation_llm_max_links,
            page_llm_max_links=config.page_llm_max_links,
            llm_text_chars=config.llm_text_chars,
            progress=self.progress,
            http=self.http,
        )
        return crawler.crawl_company(company)


def analyze_crawl_result(
    result: CrawlResult,
    config: CrawlAttemptConfig,
    *,
    analysis_agent: OrchestrationAnalysisAgent | None = None,
    timeout_seconds: float = 45.0,
    progress: ProgressReporter | None = None,
) -> FailureAnalysis:
    if result.transcripts:
        return FailureAnalysis(
            category="success",
            summary=f"Saved {len(result.transcripts)} transcript(s).",
            retryable=False,
            evidence_urls=[str(record.source_url) for record in result.transcripts],
        )

    identity_low_confidence = [failure for failure in result.failures if failure.failure_type == "identity_low_confidence"]
    if identity_low_confidence:
        return FailureAnalysis(
            category="identity_low_confidence",
            summary="Homepage prediction was low confidence before crawling.",
            retryable=True,
            evidence_urls=[failure.url for failure in identity_low_confidence if failure.url],
        )

    homepage_unverified = [failure for failure in result.failures if failure.failure_type == "homepage_unverified"]
    if homepage_unverified:
        return FailureAnalysis(
            category="homepage_unverified",
            summary="No predicted homepage could be verified before crawling.",
            retryable=True,
            evidence_urls=[failure.url for failure in homepage_unverified if failure.url],
        )

    navigation_step_failures = [
        failure
        for failure in result.failures
        if failure.failure_type in {
            "navigation_fetch_failed",
            "navigation_render_failed",
            "navigation_llm_failed",
        }
    ]
    if navigation_step_failures:
        return FailureAnalysis(
            category="navigation_step_failed",
            summary="Navigation discovery failed before crawl pages were visited.",
            retryable=True,
            evidence_urls=[failure.url for failure in navigation_step_failures if failure.url],
        )

    crawl_step_failures = [
        failure
        for failure in result.failures
        if failure.failure_type in {
            "timeout",
            "http_error",
            "robots_blocked",
            "page_render_failed",
            "page_classification_failed",
            "document_parse_failed",
        }
    ]
    if crawl_step_failures:
        return FailureAnalysis(
            category="crawl_step_failed",
            summary="A crawl step failed and the current attempt stopped early.",
            retryable=True,
            evidence_urls=[failure.url for failure in crawl_step_failures if failure.url],
        )

    robots_unavailable = [failure for failure in result.failures if failure.failure_type == "robots_unavailable"]
    if robots_unavailable:
        manual = [
            "At least one URL could not verify robots.txt. Review before rerunning with --robots-fail-open."
        ]
        return FailureAnalysis(
            category="robots_unavailable",
            summary="robots.txt could not be verified for one or more URLs.",
            retryable=False,
            evidence_urls=[failure.url for failure in robots_unavailable],
            manual_recommendations=manual,
        )

    agent_analysis = analyze_with_orchestration_agent(
        result,
        config,
        analysis_agent=analysis_agent,
        timeout_seconds=timeout_seconds,
        progress=progress,
    )
    if agent_analysis and agent_analysis.category in {"render_needed", "needs_deeper_crawl"}:
        return agent_analysis
    if not agent_analysis:
        if needs_playwright_retry(result, config):
            return FailureAnalysis(
                category="render_needed",
                summary="IR or earnings pages appear likely to need browser rendering.",
                retryable=True,
                evidence_urls=[candidate.url for candidate in result.candidates],
            )

        if needs_deeper_crawl(result, config):
            return FailureAnalysis(
                category="needs_deeper_crawl",
                summary="Crawl reached promising IR pages but may need more depth or page budget.",
                retryable=True,
                evidence_urls=[candidate.url for candidate in result.candidates],
            )

    if not result.candidates and not result.failures:
        if agent_analysis:
            return agent_analysis
        return FailureAnalysis(
            category="no_useful_links",
            summary=result.skipped_reason or "No useful IR candidates were found.",
            retryable=True,
        )

    not_transcript = [failure for failure in result.failures if failure.failure_type == "not_transcript"]
    if not_transcript and len(not_transcript) == len(result.failures):
        if agent_analysis:
            return agent_analysis
        return FailureAnalysis(
            category="not_transcript",
            summary="Fetched document candidates were rejected by strict transcript detection.",
            retryable=False,
            evidence_urls=[failure.url for failure in not_transcript],
        )

    if agent_analysis:
        return agent_analysis

    return FailureAnalysis(
        category="no_transcript_found",
        summary="No transcript artifact was saved.",
        retryable=config.discovery_mode == "nav-first",
        evidence_urls=[candidate.url for candidate in result.candidates],
    )


def plan_next_action(
    analysis: FailureAnalysis,
    *,
    company: Company,
    config: CrawlAttemptConfig,
    attempt_count: int,
    max_attempts: int,
    planner: RetryPlanningAgent | None = None,
    timeout_seconds: float = 45.0,
    progress: ProgressReporter | None = None,
) -> SupervisorAction:
    if analysis.category == "success":
        return SupervisorAction(action_type="finish", reason=analysis.summary)

    if analysis.manual_recommendations:
        return SupervisorAction(
            action_type="manual_review",
            reason=analysis.summary,
            manual_recommendations=analysis.manual_recommendations,
        )

    if attempt_count >= max_attempts:
        return SupervisorAction(action_type="finish", reason=f"Reached supervisor max attempts ({max_attempts}).")

    next_attempt = attempt_count + 1
    planned = plan_with_retry_agent(
        analysis,
        company=company,
        config=config,
        attempt_count=attempt_count,
        max_attempts=max_attempts,
        planner=planner,
        timeout_seconds=timeout_seconds,
        progress=progress,
    )
    if planned:
        return planned

    if analysis.category == "render_needed" and config.playwright_mode == "off" and not config.use_playwright:
        return SupervisorAction(
            action_type="playwright_retry",
            reason=analysis.summary,
            next_config=config.model_copy(
                deep=True,
                update={
                    "attempt": next_attempt,
                    "use_playwright": True,
                    "playwright_mode": "auto",
                    "reason": "playwright_retry",
                }
            ),
        )

    if analysis.category == "needs_deeper_crawl":
        return SupervisorAction(
            action_type="deeper_crawl",
            reason=analysis.summary,
            next_config=config.model_copy(
                deep=True,
                update={
                    "attempt": next_attempt,
                    "max_depth": min(config.max_depth + 1, 6),
                    "max_pages_per_company": min(max(config.max_pages_per_company + 5, 10), 80),
                    "reason": "deeper_crawl_retry",
                }
            ),
        )

    if analysis.category in {"identity_low_confidence", "homepage_unverified"}:
        return SupervisorAction(
            action_type="identity_retry",
            reason=analysis.summary,
            next_config=config.model_copy(
                deep=True,
                update={
                    "attempt": next_attempt,
                    "reason": f"{analysis.category}_retry",
                }
            ),
        )

    if analysis.category in {"navigation_step_failed", "crawl_step_failed"}:
        return SupervisorAction(
            action_type="step_retry",
            reason=analysis.summary,
            next_config=config.model_copy(
                deep=True,
                update={
                    "attempt": next_attempt,
                    "reason": f"{analysis.category}_retry",
                }
            ),
        )

    if analysis.retryable:
        return SupervisorAction(
            action_type="retry",
            reason=analysis.summary,
            next_config=config.model_copy(
                deep=True,
                update={
                    "attempt": next_attempt,
                    "reason": f"{analysis.category}_retry",
                }
            ),
        )

    resolved = resolve_company_identity_with_overrides(
        company,
        allow_homepage_overrides=not config.disable_official_homepage_overrides,
    )
    if resolved != company:
        return SupervisorAction(
            action_type="identity_correction",
            reason=f"Resolved company identity to {company_display_name(resolved)}.",
            next_company=resolved,
            next_config=config.model_copy(deep=True, update={"attempt": next_attempt, "reason": "identity_correction_retry"}),
        )

    return SupervisorAction(action_type="finish", reason=analysis.summary)


def analyze_with_orchestration_agent(
    result: CrawlResult,
    config: CrawlAttemptConfig,
    *,
    analysis_agent: OrchestrationAnalysisAgent | None,
    timeout_seconds: float,
    progress: ProgressReporter | None = None,
) -> FailureAnalysis | None:
    if not analysis_agent:
        return None
    progress = progress or ProgressReporter(enabled=False)
    evidence = orchestration_analysis_evidence(result, config)
    try:
        progress.log(f"{result.company.symbol}: orchestration analysis agent starting")
        with timeout_after(timeout_seconds, f"analyzing {result.company.symbol} crawl outcome"):
            decision = analysis_agent.analyze(company=result.company, evidence=evidence)
    except Exception as exc:
        progress.log(f"{result.company.symbol}: orchestration analysis skipped ({type(exc).__name__}: {exc})")
        return None
    return failure_analysis_from_decision(decision, result)


def failure_analysis_from_decision(
    decision: OrchestrationAnalysisDecision,
    result: CrawlResult,
) -> FailureAnalysis:
    evidence_urls = [
        url
        for url in decision.evidence_urls[:20]
        if normalize_url(url) in {normalize_url(candidate.url) for candidate in result.candidates}
        or normalize_url(url) in {normalize_url(failure.url) for failure in result.failures if failure.url}
        or normalize_url(url) in {normalize_url(str(record.source_url)) for record in result.transcripts}
    ]
    return FailureAnalysis(
        category=decision.category,
        summary=decision.summary[:240] or "Crawl outcome analyzed by orchestration agent.",
        retryable=decision.retryable,
        evidence_urls=evidence_urls,
        manual_recommendations=decision.manual_recommendations[:5],
    )


def plan_with_retry_agent(
    analysis: FailureAnalysis,
    *,
    company: Company,
    config: CrawlAttemptConfig,
    attempt_count: int,
    max_attempts: int,
    planner: RetryPlanningAgent | None,
    timeout_seconds: float,
    progress: ProgressReporter | None = None,
) -> SupervisorAction | None:
    if not planner:
        return None
    progress = progress or ProgressReporter(enabled=False)
    evidence = retry_planning_evidence(
        analysis,
        config=config,
        attempt_count=attempt_count,
        max_attempts=max_attempts,
    )
    try:
        progress.log(f"{company.symbol}: retry planning agent starting")
        with timeout_after(timeout_seconds, f"planning {company.symbol} retry"):
            decision = planner.plan(company=company, evidence=evidence)
    except Exception as exc:
        progress.log(f"{company.symbol}: retry planning skipped ({type(exc).__name__}: {exc})")
        return None
    return supervisor_action_from_retry_decision(
        decision,
        analysis=analysis,
        config=config,
        attempt_count=attempt_count,
        max_attempts=max_attempts,
    )


VALID_RETRY_ACTIONS = {
    "finish",
    "identity_correction",
    "identity_retry",
    "step_retry",
    "retry",
    "search_first",
    "playwright_retry",
    "deeper_crawl",
    "manual_review",
}

CONFIG_UPDATE_FIELDS = set(CrawlAttemptConfig.model_fields)


def supervisor_action_from_retry_decision(
    decision: RetryPlanningDecision,
    *,
    analysis: FailureAnalysis,
    config: CrawlAttemptConfig,
    attempt_count: int,
    max_attempts: int,
) -> SupervisorAction | None:
    if decision.action_type not in VALID_RETRY_ACTIONS:
        return None
    if attempt_count >= max_attempts and decision.action_type != "finish":
        return None
    if decision.action_type == "manual_review":
        return SupervisorAction(
            action_type="manual_review",
            reason=decision.reason or analysis.summary,
            manual_recommendations=decision.manual_recommendations or analysis.manual_recommendations,
        )
    if decision.action_type == "finish":
        return SupervisorAction(action_type="finish", reason=decision.reason or analysis.summary)

    updates = {
        key: value
        for key, value in decision.config_updates.items()
        if key in CONFIG_UPDATE_FIELDS
    }
    updates["attempt"] = attempt_count + 1
    updates.setdefault("reason", f"{analysis.category}_retry")

    if decision.action_type == "playwright_retry":
        updates.setdefault("use_playwright", True)
        updates.setdefault("playwright_mode", "auto")
    elif decision.action_type == "deeper_crawl":
        updates.setdefault("max_depth", min(config.max_depth + 1, 6))
        updates.setdefault("max_pages_per_company", min(max(config.max_pages_per_company + 5, 10), 80))

    return SupervisorAction(
        action_type=decision.action_type,  # type: ignore[arg-type]
        reason=decision.reason or analysis.summary,
        next_config=config.model_copy(deep=True, update=updates),
        manual_recommendations=decision.manual_recommendations,
    )


def needs_playwright_retry(result: CrawlResult, config: CrawlAttemptConfig) -> bool:
    if config.use_playwright or config.playwright_mode != "off":
        return False
    haystack = " ".join(page_haystack(candidate) for candidate in result.candidates).lower()
    js_markers = ("q4web", "q4cdn", "q4app", "financial reports", "earnings", "events")
    return bool(result.candidates) and any(marker in haystack for marker in js_markers)


def needs_deeper_crawl(result: CrawlResult, config: CrawlAttemptConfig) -> bool:
    if not result.candidates:
        return False
    if result.visited_count >= config.max_pages_per_company:
        return True
    return any(candidate.depth >= config.max_depth and promising_candidate(candidate) for candidate in result.candidates)


def promising_candidate(candidate: CandidatePage) -> bool:
    haystack = page_haystack(candidate).lower()
    return any(token in haystack for token in ("investor", "earnings", "financial", "events", "webcast"))


def page_haystack(candidate: CandidatePage) -> str:
    return f"{candidate.url} {candidate.title} {candidate.llm_page_type or ''} {candidate.reason}"


def orchestration_analysis_evidence(result: CrawlResult, config: CrawlAttemptConfig) -> dict[str, Any]:
    return {
        "config": {
            "attempt": config.attempt,
            "max_depth": config.max_depth,
            "max_pages_per_company": config.max_pages_per_company,
            "use_playwright": config.use_playwright,
            "playwright_mode": config.playwright_mode,
            "discovery_mode": config.discovery_mode,
            "latest_only": config.latest_only,
            "reason": config.reason,
        },
        "visited_count": result.visited_count,
        "ir_url": result.ir_url,
        "skipped_reason": result.skipped_reason,
        "transcripts": [
            {"url": str(record.source_url), "title": record.title}
            for record in result.transcripts[:10]
        ],
        "candidates": [
            {
                "url": candidate.url,
                "title": candidate.title,
                "depth": candidate.depth,
                "page_type": candidate.llm_page_type,
                "confidence": candidate.llm_confidence,
                "reason": candidate.reason,
            }
            for candidate in result.candidates[:30]
        ],
        "failures": [
            {
                "url": failure.url,
                "failure_type": failure.failure_type,
                "message": failure.message[:180],
            }
            for failure in result.failures[:30]
        ],
    }


def retry_planning_evidence(
    analysis: FailureAnalysis,
    *,
    config: CrawlAttemptConfig,
    attempt_count: int,
    max_attempts: int,
) -> dict[str, Any]:
    return {
        "analysis": analysis.model_dump(mode="json"),
        "attempt_count": attempt_count,
        "max_attempts": max_attempts,
        "current_config": config.model_dump(mode="json"),
    }


def load_navigation_trace(out_dir: Path, company: Company) -> NavigationTrace | None:
    path = out_dir / company.symbol / "_navigation_trace.json"
    if not path.exists():
        return None
    try:
        return NavigationTrace.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def reflection_evidence(
    result: CrawlResult,
    analysis: FailureAnalysis,
    trace: NavigationTrace | None,
) -> dict[str, Any]:
    return {
        "analysis": analysis.model_dump(mode="json"),
        "visited_count": result.visited_count,
        "ir_url": result.ir_url,
        "candidates": [
            {
                "url": candidate.url,
                "title": candidate.title,
                "depth": candidate.depth,
                "page_type": candidate.llm_page_type,
                "reason": candidate.reason,
            }
            for candidate in result.candidates[:30]
        ],
        "failures": [
            {
                "url": failure.url,
                "failure_type": failure.failure_type,
                "message": failure.message[:160],
            }
            for failure in result.failures[:30]
        ],
        "navigation_steps": [
            {
                "current_url": step.current_url,
                "title": step.title,
                "chosen_urls": step.chosen_urls,
                "rejected_urls": step.rejected_urls,
                "reason": step.reason,
                "stop_reason": step.stop_reason,
            }
            for step in (trace.steps if trace else [])[:20]
        ],
    }


URL_EVIDENCE_KEYS = {"url", "urls", "ir_url", "source_url", "current_url", "chosen_urls", "rejected_urls", "evidence_urls"}


def filter_reflection_urls_to_evidence(reflection: CrawlReflection, evidence: dict[str, Any]) -> CrawlReflection:
    allowed_urls = evidence_url_set(evidence)
    return reflection.model_copy(
        update={
            "preferred_urls": [url for url in reflection.preferred_urls if normalize_url(url) in allowed_urls],
            "avoid_urls": [url for url in reflection.avoid_urls if normalize_url(url) in allowed_urls],
        }
    )


def evidence_url_set(evidence: dict[str, Any]) -> set[str]:
    urls: set[str] = set()

    def collect(value: Any, *, parent_key: str | None = None) -> None:
        if isinstance(value, dict):
            for key, nested in value.items():
                collect(nested, parent_key=key if key in URL_EVIDENCE_KEYS else None)
            return
        if isinstance(value, list):
            for item in value:
                collect(item, parent_key=parent_key)
            return
        if isinstance(value, str) and parent_key in URL_EVIDENCE_KEYS and value.startswith(("http://", "https://")):
            urls.add(normalize_url(value))

    collect(evidence)
    return urls


def supervisor_status(result: CrawlResult | None, actions: list[SupervisorAction]) -> str:
    if result and result.transcripts:
        return "success"
    if actions and actions[-1].action_type == "manual_review":
        return "manual_review"
    if result and (result.candidates or result.failures):
        return "partial"
    return "failed"


def run_supervised_company(
    company: Company,
    *,
    model: str,
    out_dir: Path,
    http: HttpClient,
    ollama_base_url: str | None = None,
    base_config: CrawlAttemptConfig | None = None,
    max_attempts: int = 2,
    progress: ProgressReporter | None = None,
) -> SupervisorRunResult:
    return SupervisedCrawler(
        model=model,
        out_dir=out_dir,
        http=http,
        ollama_base_url=ollama_base_url,
        base_config=base_config,
        max_attempts=max_attempts,
        progress=progress,
    ).run_company(company)
