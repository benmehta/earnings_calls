from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from .crawler import TranscriptCrawler
from .http import DEFAULT_USER_AGENT, HttpClient, random_browser_user_agent
from .identity import COMMON_COMPANY_NAMES, resolve_company_identity
from .indexing import index_transcripts
from .models import Company, CrawlAttemptConfig
from .orchestration import run_supervised_company
from .runtime import ProgressReporter
from .sp500 import load_sp500


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scrape S&P 500 IR earnings transcripts with Ollama + LangChain.")
    parser.add_argument("--model", default=os.getenv("OLLAMA_MODEL", "llama3.1:8b"), help="Ollama model name")
    parser.add_argument("--ollama-base-url", default=os.getenv("OLLAMA_BASE_URL"), help="Optional Ollama base URL")
    parser.add_argument("--out", type=Path, default=Path("data/transcripts"), help="Output directory")
    parser.add_argument("--limit", type=int, default=None, help="Limit company count for testing")
    parser.add_argument("--symbols", nargs="*", help="Optional ticker symbols to crawl")
    parser.add_argument("--seed-url", action="append", default=[], help="Seed URL to crawl instead of discovery. Can be repeated")
    parser.add_argument("--max-pages", type=int, default=40, help="Max pages per company")
    parser.add_argument("--max-depth", type=int, default=3, help="Max crawl depth from IR candidates")
    parser.add_argument("--user-agent", default=os.getenv("IR_USER_AGENT", DEFAULT_USER_AGENT), help="User agent sent to company websites")
    parser.add_argument(
        "--fake-user-agent",
        action="store_true",
        help="Use fake-useragent to generate one browser-like user agent for this run",
    )
    parser.add_argument("--delay", type=float, default=float(os.getenv("IR_DELAY_SECONDS", "1.5")), help="Minimum delay between requests to the same host")
    parser.add_argument(
        "--search-timeout",
        type=float,
        default=float(os.getenv("IR_SEARCH_TIMEOUT_SECONDS", "30")),
        help="Seconds to allow each discovery search query before skipping it",
    )
    parser.add_argument(
        "--llm-timeout",
        type=float,
        default=float(os.getenv("IR_LLM_TIMEOUT_SECONDS", "45")),
        help="Seconds to allow each local Ollama page/navigation decision before using fallback logic",
    )
    parser.add_argument(
        "--navigation-llm-max-links",
        type=int,
        default=int(os.getenv("IR_NAVIGATION_LLM_MAX_LINKS", "12")),
        help="Maximum candidate links sent to the navigation agent",
    )
    parser.add_argument(
        "--page-llm-max-links",
        type=int,
        default=int(os.getenv("IR_PAGE_LLM_MAX_LINKS", "12")),
        help="Maximum candidate links sent to the page-classification agent",
    )
    parser.add_argument(
        "--llm-text-chars",
        type=int,
        default=int(os.getenv("IR_LLM_TEXT_CHARS", "900")),
        help="Maximum visible-text characters sent to each local Ollama agent call",
    )
    parser.add_argument("--quiet", action="store_true", help="Hide progress logs and only print final per-company summaries")
    parser.add_argument("--playwright", action="store_true", help="Render likely JavaScript app-shell pages")
    parser.add_argument(
        "--playwright-mode",
        choices=("off", "auto", "always"),
        default=os.getenv("IR_PLAYWRIGHT_MODE", "auto"),
        help="Browser rendering policy for HTML pages. --playwright is kept as an alias for auto.",
    )
    parser.add_argument("--review-only", action="store_true", help="Write candidate/failure review files without saving transcript artifacts")
    parser.add_argument("--no-resume", action="store_true", help="Ignore existing per-company crawl state")
    parser.add_argument(
        "--discovery-mode",
        choices=("nav-first", "search-first"),
        default=os.getenv("IR_DISCOVERY_MODE", "nav-first"),
        help="Discovery flow. nav-first predicts/verifies the official homepage, navigates homepage to IR, then crawls IR pages. search-first is explicit legacy search discovery.",
    )
    parser.add_argument("--include-discovery-guesses", action="store_true", help="Use deterministic IR URL guesses only if search/curated discovery finds nothing")
    parser.add_argument(
        "--disable-official-homepage-overrides",
        action="store_true",
        help="Disable curated official homepage starts such as GOOG -> abc.xyz for blind discovery tests",
    )
    parser.add_argument(
        "--disable-predictive-identity",
        action="store_true",
        help="Disable local Ollama homepage prediction for ticker-only company identities",
    )
    parser.add_argument("--rerank-discovery", action="store_true", help="Use local Ollama to rerank discovery search results")
    parser.add_argument(
        "--research-agent",
        action="store_true",
        help="Experimental opt-in: use search plus a high-level official transcript research judge to produce crawl seeds instead of homepage-first discovery",
    )
    parser.add_argument("--metadata-llm", action="store_true", help="Use a second local Ollama pass for confirmed transcript metadata")
    parser.add_argument("--prompt-planner", action="store_true", help="Use a local Ollama planner to create advisory prompts from company memory in supervised mode")
    parser.add_argument(
        "--disable-memory",
        action="store_true",
        help="Start supervised runs from blank company memory, then merge learned memory into the memory file at the end.",
    )
    parser.add_argument(
        "--no-memory-write",
        action="store_true",
        help="Do not write company memory after supervised runs. Retries may still use memory learned during this run.",
    )
    parser.add_argument("--latest-only", action="store_true", help="Keep only the latest detected transcript per company")
    parser.add_argument(
        "--allow-official-linked-documents-on-robots-unavailable",
        action="store_true",
        help=(
            "When an allowed official IR page links directly to a transcript document, "
            "allow fetching that document if the document host's robots.txt is unavailable."
        ),
    )
    parser.add_argument("--index-chroma", action="store_true", help="Index collected transcripts into local Chroma")
    parser.add_argument("--chroma-dir", type=Path, default=Path("data/chroma"), help="Chroma persistence directory")
    parser.add_argument("--embedding-model", default=os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"), help="Ollama embedding model for Chroma")
    parser.add_argument(
        "--orchestration-mode",
        choices=("single-pass", "supervised"),
        default=os.getenv("IR_ORCHESTRATION_MODE", "single-pass"),
        help="Run the crawler once, or use the fault-tolerant supervisor for conservative retries",
    )
    parser.add_argument(
        "--supervisor-max-attempts",
        type=int,
        default=int(os.getenv("IR_SUPERVISOR_MAX_ATTEMPTS", "2")),
        help="Maximum crawl attempts per company in supervised mode",
    )
    parser.add_argument("--ignore-robots", action="store_true", help="Disable robots.txt checks")
    parser.add_argument(
        "--robots-fail-open",
        action="store_true",
        help="Allow crawling when robots.txt cannot be fetched due to an error",
    )
    return parser


def load_requested_companies(symbols: list[str] | None) -> list[Company]:
    wanted_symbols = [symbol.upper() for symbol in symbols] if symbols else []
    if wanted_symbols:
        wanted = set(wanted_symbols)
        try:
            companies = load_sp500()
        except Exception:
            companies = [
                Company(symbol=symbol, name=COMMON_COMPANY_NAMES.get(symbol))
                for symbol in wanted_symbols
            ]
        companies = [
            resolve_company_identity(company)
            for company in companies
            if company.symbol.upper() in wanted
        ]
        found = {company.symbol.upper() for company in companies}
        companies.extend(
            resolve_company_identity(Company(symbol=symbol, name=COMMON_COMPANY_NAMES.get(symbol)))
            for symbol in wanted_symbols
            if symbol not in found
        )
        return companies
    return load_sp500()


def main() -> None:
    load_dotenv()
    args = build_parser().parse_args()

    companies = load_requested_companies(args.symbols)

    if args.limit is not None:
        companies = companies[: args.limit]

    user_agent = random_browser_user_agent() if args.fake_user_agent else args.user_agent
    playwright_mode = "auto" if args.playwright else args.playwright_mode
    progress = ProgressReporter(enabled=not args.quiet)

    http = HttpClient(
        user_agent=user_agent,
        delay_seconds=args.delay,
        respect_robots=not args.ignore_robots,
        fail_closed_on_robots_error=not args.robots_fail_open,
    )

    crawler = TranscriptCrawler(
        model=args.model,
        out_dir=args.out,
        max_pages_per_company=args.max_pages,
        max_depth=args.max_depth,
        use_playwright=args.playwright,
        playwright_mode=playwright_mode,
        ollama_base_url=args.ollama_base_url,
        resume=not args.no_resume,
        review_only=args.review_only,
        seed_urls=args.seed_url,
        discovery_mode=args.discovery_mode,
        include_discovery_guesses=args.include_discovery_guesses,
        disable_official_homepage_overrides=args.disable_official_homepage_overrides,
        disable_predictive_identity=args.disable_predictive_identity,
        use_research_agent=args.research_agent,
        rerank_discovery=args.rerank_discovery,
        extract_metadata_with_llm=args.metadata_llm,
        latest_only=args.latest_only,
        allow_official_linked_documents_on_robots_unavailable=args.allow_official_linked_documents_on_robots_unavailable,
        search_timeout_seconds=args.search_timeout,
        llm_timeout_seconds=args.llm_timeout,
        navigation_llm_max_links=args.navigation_llm_max_links,
        page_llm_max_links=args.page_llm_max_links,
        llm_text_chars=args.llm_text_chars,
        progress=progress,
        http=http,
    )
    supervised_config = CrawlAttemptConfig(
        max_pages_per_company=args.max_pages,
        max_depth=args.max_depth,
        use_playwright=args.playwright,
        playwright_mode=playwright_mode,
        resume=not args.no_resume,
        review_only=args.review_only,
        seed_urls=args.seed_url,
        discovery_mode=args.discovery_mode,
        include_discovery_guesses=args.include_discovery_guesses,
        disable_official_homepage_overrides=args.disable_official_homepage_overrides,
        disable_predictive_identity=args.disable_predictive_identity,
        use_research_agent=args.research_agent,
        rerank_discovery=args.rerank_discovery,
        extract_metadata_with_llm=args.metadata_llm,
        latest_only=args.latest_only,
        allow_official_linked_documents_on_robots_unavailable=args.allow_official_linked_documents_on_robots_unavailable,
        use_prompt_planner=args.prompt_planner,
        disable_memory=args.disable_memory,
        no_memory_write=args.no_memory_write,
        search_timeout_seconds=args.search_timeout,
        llm_timeout_seconds=args.llm_timeout,
        navigation_llm_max_links=args.navigation_llm_max_links,
        page_llm_max_links=args.page_llm_max_links,
        llm_text_chars=args.llm_text_chars,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    summary_path = args.out / "_summary.jsonl"

    with summary_path.open("a", encoding="utf-8") as summary:
        collected = []
        for company in companies:
            if args.orchestration_mode == "supervised":
                supervised = run_supervised_company(
                    company,
                    model=args.model,
                    out_dir=args.out,
                    http=http,
                    ollama_base_url=args.ollama_base_url,
                    base_config=supervised_config,
                    max_attempts=args.supervisor_max_attempts,
                    progress=progress,
                )
                result = supervised.result
                summary.write(json.dumps(supervised.model_dump(mode="json"), ensure_ascii=True) + "\n")
                summary.flush()
                transcript_count = len(result.transcripts) if result else 0
                visited_count = result.visited_count if result else 0
                print(
                    f"{company.symbol}: {transcript_count} transcript(s), "
                    f"{visited_count} page(s) visited, supervised status={supervised.status}"
                )
                if result:
                    collected.extend(result.transcripts)
                continue

            result = crawler.crawl_company(company)
            collected.extend(result.transcripts)
            summary.write(json.dumps(result.model_dump(mode="json"), ensure_ascii=True) + "\n")
            summary.flush()
            print(
                f"{company.symbol}: {len(result.transcripts)} transcript(s), "
                f"{result.visited_count} page(s) visited"
            )

    if args.index_chroma:
        chunks = index_transcripts(
            collected,
            persist_dir=args.chroma_dir,
            embedding_model=args.embedding_model,
            ollama_base_url=args.ollama_base_url,
        )
        print(f"Indexed {chunks} transcript chunk(s) into {args.chroma_dir}")


if __name__ == "__main__":
    main()
