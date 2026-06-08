# IR Transcript Scraper Architecture

## Package Layout

Use modules with narrow responsibilities:

- `sp500.py`: load the company universe at runtime or from a pinned fixture.
- `search.py`: find likely investor-relations entry points.
- `http.py`: own sessions, user agent, robots.txt, retry, and delay behavior.
- `parsing.py`: extract visible HTML text, links, PDF text, and cheap transcript heuristics.
- `agent.py`: contain LangChain/Ollama prompts, chains, and structured decisions.
- `crawler.py`: coordinate breadth/depth limits, visited state, page decisions, and persistence.
- `cli.py`: expose flags for model, symbols, limits, output directory, and crawl depth.

## Crawl Strategy

Start from a small set of IR candidates discovered through search or known IR URL patterns. Allow host expansion only when the new host is clearly part of the company's IR flow, such as a linked vendor-hosted IR page.

Good crawl hints:

- URL or label contains `transcript`, `earnings`, `quarter`, `events`, `financial-results`, `webcast`, `presentation`.
- Page text contains multiple transcript markers such as `operator`, `prepared remarks`, `question-and-answer session`, `analyst`, `conference call`.

Bad crawl hints:

- SEC filings pages unless the user is collecting filings too.
- Press release indexes with no call material.
- Stock quote pages, governance pages, careers, product marketing, support, privacy, terms.

## Agent Responsibilities

Use the local model for judgments that are brittle with heuristics:

- classify a page as transcript, earnings event, IR index, press release, filings, or irrelevant
- rank links that are likely to lead toward transcripts
- extract fiscal period or call date only from strong evidence

Do not use the model to fabricate missing text. If the transcript is behind JavaScript, gated, missing, or only available through audio, record the limitation.

## Persistence

Store raw HTML/PDF files and normalized JSON together under an output directory grouped by ticker. Keep a crawl summary JSONL at the root for full-run auditing.

Use safe filenames derived from titles or URLs, and include enough metadata to trace each normalized record back to its raw artifact.

## Practical Limits

Default to conservative limits:

- `max_pages_per_company`: 20-50 during development
- `max_depth`: 2-3
- delay: at least 1 second per request

Expose flags so the user can intentionally broaden the crawl later.

