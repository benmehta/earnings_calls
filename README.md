# Local IR Transcript Scraper

Prototype for scraping earnings call transcripts from investor-relations pages for
S&P 500 companies using local Ollama models through LangChain.

This code is intentionally written but not run. Install dependencies later, make
sure Ollama is running, then start with a small limit before attempting the full
S&P 500.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
ollama pull llama3.1:8b
```

## Example

```bash
python -m ir_transcripts \
  --model llama3.1:8b \
  --out data/transcripts \
  --limit 5
```

Use `--limit` while developing. Full-index crawling across 500 public companies
will take time and should respect each site's robots.txt, terms, and rate limits.

## What It Does

1. Loads the current S&P 500 constituent table from Wikipedia.
2. Locates a likely investor-relations site for each company.
3. Crawls only that IR domain/path, politely and with robots.txt checks.
4. Uses a local Ollama-backed LangChain agent to classify pages and links.
5. Downloads likely transcript pages or PDFs.
6. Saves normalized transcript JSON plus raw artifacts.

## Notes

- Some companies publish transcripts on third-party IR platforms such as
  `investors.company.com`, `ir.company.com`, `investor.company.com`, or hosted
  vendors. This still counts as company IR when linked from the company's site.
- Many companies do not publish complete call transcripts themselves. The code
  records misses instead of silently inventing data.
- This is a research crawler scaffold, not legal advice. Review each site's
  terms before large-scale collection.

