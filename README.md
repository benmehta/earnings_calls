# Local IR Transcript Scraper

Prototype for scraping earnings call transcripts from investor-relations pages for
S&P 500 companies using local Ollama models through LangChain.

This code is intentionally written but not run. Install dependencies later, make
sure Ollama is running, then start with a small limit before attempting the full
S&P 500.

## Setup

Using your existing conda env:

```bash
conda activate ollama
pip install -e .
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

Or with a new virtualenv:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

Optional `.env`:

```bash
cp .env.example .env
```

To recreate the known-good direct package set from this repo:

```bash
conda env create -f environment.yml
```

## Example

```bash
python -m ir_transcripts \
  --model llama3.1:8b \
  --out data/transcripts \
  --user-agent "local-ir-transcript-research/0.1 (+mailto:you@example.com)" \
  --limit 5
```

The crawler defaults to auto Playwright rendering for pages that look like
JavaScript-driven IR widgets, such as Q4/Evergreen events and financial tables.
Install the browser binary once:

```bash
playwright install chromium
```

You can force a rendering policy when debugging:

```bash
python -m ir_transcripts --symbols AAPL --max-pages 5 --playwright-mode off
python -m ir_transcripts --symbols AAPL --max-pages 5 --playwright-mode always
```

The older `--playwright` flag is still accepted and maps to
`--playwright-mode auto`.

To build a local Chroma index from collected transcripts:

```bash
python -m ir_transcripts \
  --symbols AAPL MSFT \
  --index-chroma \
  --embedding-model nomic-embed-text
```

To review likely transcript pages before saving transcript artifacts:

```bash
python -m ir_transcripts --symbols AAPL MSFT --review-only --max-pages 10
```

To preview discovery without crawling company IR pages:

```bash
python -m ir_transcripts.discover MSFT --max-results 8
```

Discovery uses official-site navigation first by default. The local Ollama
navigation agent chooses from extracted homepage/IR links while the crawler keeps
control of robots.txt, rate limits, and host expansion. Search remains the
fallback when navigation does not find usable seeds.

To compare with the older search-first behavior:

```bash
python -m ir_transcripts --symbols NVDA --discovery-mode search-first --review-only
```

Deterministic URL guesses are disabled unless search/curated discovery finds
nothing and you explicitly opt in:

```bash
python -m ir_transcripts.discover MSFT --include-guesses
python -m ir_transcripts.discover MSFT --rerank-ollama
```

To skip discovery entirely:

```bash
python -m ir_transcripts \
  --symbols MSFT \
  --seed-url https://www.microsoft.com/en-us/Investor \
  --review-only
```

Each company directory gets `_candidates.jsonl`, `_failures.jsonl`,
`_crawl_state.json`, and `_page_cache/` files so broad crawls can be audited,
resumed, and reclassified without repeatedly fetching the same pages.

To use a second local Ollama pass for confirmed transcript metadata:

```bash
python -m ir_transcripts --symbols AAPL --metadata-llm
```

Use `--limit` while developing. Full-index crawling across 500 public companies
will take time and should respect each site's robots.txt, terms, and rate limits.

## What It Does

1. Loads the current S&P 500 constituent table from Wikipedia.
2. Locates a likely investor-relations site for each company.
3. Navigates official company and IR pages toward earnings materials.
4. Crawls only that official IR path, politely and with robots.txt checks.
5. Uses local Ollama-backed LangChain agents to choose navigation links and classify pages.
6. Downloads likely transcript pages, PDFs, or DOCX files.
7. Optionally renders JavaScript-heavy pages with Playwright.
8. Saves normalized transcript JSON plus raw artifacts.
9. Optionally indexes transcripts into local Chroma using Ollama embeddings.

## No-Network Verification

The test fixtures under `tests/fixtures/` exercise parsing, URL normalization,
robots behavior, state persistence, and scoring without calling company
websites, Ollama, Chroma, or package indexes.

```bash
python -m compileall src
python -m pytest
```

## Notes

- Some companies publish transcripts on third-party IR platforms such as
  `investors.company.com`, `ir.company.com`, `investor.company.com`, or hosted
  vendors. This still counts as company IR when linked from the company's site.
- Many companies do not publish complete call transcripts themselves. The code
  records misses instead of silently inventing data.
- This is a research crawler scaffold, not legal advice. Review each site's
  terms before large-scale collection.

## Robots.txt And Rate Limits

The crawler checks `robots.txt` before every page fetch by default. It fetches
each site's `robots.txt` with the same user agent used for crawling, caches the
rules per origin, honors `Crawl-delay` when provided, and otherwise applies the
CLI `--delay` value between requests to the same host.

Use an honest, contactable user agent rather than a fake browser user agent:

```text
local-ir-transcript-research/0.1 (+mailto:you@example.com)
```

If you intentionally need a browser-like user agent for a small diagnostic run,
use an explicit opt-in:

```bash
python -m ir_transcripts --symbols AAPL --max-pages 5 --review-only --fake-user-agent
```

Defaults are intentionally conservative:

- `robots.txt` checks are enabled.
- If `robots.txt` cannot be fetched because of a server/network error, the
  crawler fails closed and skips that host.
- A missing `robots.txt` file (`404`) is treated as no published robots policy.
- Officially linked transcript documents on CDN/vendor hosts can be fetched
  when that document host's `robots.txt` is unavailable only with the explicit
  `--allow-official-linked-documents-on-robots-unavailable` opt-in. The source
  page must already be an allowed official IR page, and the target must look
  transcript-document-like.

Useful options:

```bash
python -m ir_transcripts --symbols AAPL --user-agent "your-project/0.1 (+mailto:you@example.com)"
python -m ir_transcripts --symbols AAPL --delay 5
python -m ir_transcripts --symbols AAPL --robots-fail-open
python -m ir_transcripts --symbols AAPL --allow-official-linked-documents-on-robots-unavailable
python -m ir_transcripts --symbols AAPL --ignore-robots
```

Use `--ignore-robots` only for controlled testing or when you have another
explicit permission model.
