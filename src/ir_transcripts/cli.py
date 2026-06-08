from __future__ import annotations

import argparse
import json
from pathlib import Path

from .crawler import TranscriptCrawler
from .sp500 import load_sp500


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scrape S&P 500 IR earnings transcripts with Ollama + LangChain.")
    parser.add_argument("--model", default="llama3.1:8b", help="Ollama model name")
    parser.add_argument("--out", type=Path, default=Path("data/transcripts"), help="Output directory")
    parser.add_argument("--limit", type=int, default=None, help="Limit company count for testing")
    parser.add_argument("--symbols", nargs="*", help="Optional ticker symbols to crawl")
    parser.add_argument("--max-pages", type=int, default=40, help="Max pages per company")
    parser.add_argument("--max-depth", type=int, default=3, help="Max crawl depth from IR candidates")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    companies = load_sp500()

    if args.symbols:
        wanted = {symbol.upper() for symbol in args.symbols}
        companies = [company for company in companies if company.symbol.upper() in wanted]

    if args.limit is not None:
        companies = companies[: args.limit]

    crawler = TranscriptCrawler(
        model=args.model,
        out_dir=args.out,
        max_pages_per_company=args.max_pages,
        max_depth=args.max_depth,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    summary_path = args.out / "_summary.jsonl"

    with summary_path.open("a", encoding="utf-8") as summary:
        for company in companies:
            result = crawler.crawl_company(company)
            summary.write(json.dumps(result.model_dump(mode="json"), ensure_ascii=True) + "\n")
            summary.flush()
            print(
                f"{company.symbol}: {len(result.transcripts)} transcript(s), "
                f"{result.visited_count} page(s) visited"
            )


if __name__ == "__main__":
    main()

