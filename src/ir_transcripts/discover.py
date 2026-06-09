from __future__ import annotations

import argparse
import json

from dotenv import load_dotenv

from .cli import COMMON_COMPANY_NAMES
from .models import Company
from .search import discover_ir_candidates


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Discover likely company investor-relations URLs.")
    parser.add_argument("symbol", help="Ticker symbol to discover")
    parser.add_argument("--name", help="Company name. Defaults to a small known-name map or the ticker.")
    parser.add_argument("--max-results", type=int, default=10, help="Search results to inspect")
    parser.add_argument("--include-guesses", action="store_true", help="Include deterministic guesses only if search/curated discovery finds nothing")
    parser.add_argument("--rerank-ollama", action="store_true", help="Use local Ollama to rerank discovery results")
    parser.add_argument("--model", default="llama3.1:8b", help="Ollama model for --rerank-ollama")
    parser.add_argument("--ollama-base-url", help="Optional Ollama base URL")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of a table")
    return parser


def main() -> None:
    load_dotenv()
    args = build_parser().parse_args()
    symbol = args.symbol.upper()
    company = Company(symbol=symbol, name=args.name or COMMON_COMPANY_NAMES.get(symbol, symbol))
    candidates = discover_ir_candidates(
        company,
        max_results=args.max_results,
        include_guesses=args.include_guesses,
        rerank_model=args.model if args.rerank_ollama else None,
        ollama_base_url=args.ollama_base_url,
    )

    if args.json:
        print(json.dumps([candidate.model_dump(mode="json") for candidate in candidates], indent=2))
        return

    for index, candidate in enumerate(candidates, start=1):
        reasons = ", ".join(candidate.reasons[:4])
        print(f"{index:>2}. [{candidate.score:>3}] {candidate.source:<13} {candidate.url}")
        if candidate.title:
            print(f"    title: {candidate.title}")
        if reasons:
            print(f"    why: {reasons}")


if __name__ == "__main__":
    main()
