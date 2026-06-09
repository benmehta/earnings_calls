from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from .crawler import TranscriptCrawler
from .http import DEFAULT_USER_AGENT, HttpClient, random_browser_user_agent
from .indexing import index_transcripts
from .models import Company
from .sp500 import load_sp500


COMMON_COMPANY_NAMES = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "NVDA": "NVIDIA",
}


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
    parser.add_argument("--playwright", action="store_true", help="Render likely JavaScript app-shell pages")
    parser.add_argument("--review-only", action="store_true", help="Write candidate/failure review files without saving transcript artifacts")
    parser.add_argument("--no-resume", action="store_true", help="Ignore existing per-company crawl state")
    parser.add_argument("--include-discovery-guesses", action="store_true", help="Use deterministic IR URL guesses only if search/curated discovery finds nothing")
    parser.add_argument("--rerank-discovery", action="store_true", help="Use local Ollama to rerank discovery search results")
    parser.add_argument("--metadata-llm", action="store_true", help="Use a second local Ollama pass for confirmed transcript metadata")
    parser.add_argument("--index-chroma", action="store_true", help="Index collected transcripts into local Chroma")
    parser.add_argument("--chroma-dir", type=Path, default=Path("data/chroma"), help="Chroma persistence directory")
    parser.add_argument("--embedding-model", default=os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"), help="Ollama embedding model for Chroma")
    parser.add_argument("--ignore-robots", action="store_true", help="Disable robots.txt checks")
    parser.add_argument(
        "--robots-fail-open",
        action="store_true",
        help="Allow crawling when robots.txt cannot be fetched due to an error",
    )
    return parser


def main() -> None:
    load_dotenv()
    args = build_parser().parse_args()

    if args.symbols:
        wanted = {symbol.upper() for symbol in args.symbols}
        try:
            companies = load_sp500()
        except Exception:
            companies = [
                Company(symbol=symbol, name=COMMON_COMPANY_NAMES.get(symbol, symbol))
                for symbol in sorted(wanted)
            ]
    else:
        companies = load_sp500()

    if args.symbols:
        wanted = {symbol.upper() for symbol in args.symbols}
        companies = [company for company in companies if company.symbol.upper() in wanted]

    if args.limit is not None:
        companies = companies[: args.limit]

    user_agent = random_browser_user_agent() if args.fake_user_agent else args.user_agent

    crawler = TranscriptCrawler(
        model=args.model,
        out_dir=args.out,
        max_pages_per_company=args.max_pages,
        max_depth=args.max_depth,
        use_playwright=args.playwright,
        ollama_base_url=args.ollama_base_url,
        resume=not args.no_resume,
        review_only=args.review_only,
        seed_urls=args.seed_url,
        include_discovery_guesses=args.include_discovery_guesses,
        rerank_discovery=args.rerank_discovery,
        extract_metadata_with_llm=args.metadata_llm,
        http=HttpClient(
            user_agent=user_agent,
            delay_seconds=args.delay,
            respect_robots=not args.ignore_robots,
            fail_closed_on_robots_error=not args.robots_fail_open,
        ),
    )

    args.out.mkdir(parents=True, exist_ok=True)
    summary_path = args.out / "_summary.jsonl"

    with summary_path.open("a", encoding="utf-8") as summary:
        collected = []
        for company in companies:
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
