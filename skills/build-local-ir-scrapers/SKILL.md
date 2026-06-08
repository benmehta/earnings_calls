---
name: build-local-ir-scrapers
description: Build, modify, or review Python scrapers that use local Ollama models through LangChain for investor-relations earnings call transcript collection, especially S&P 500 company IR sites, transcript discovery, polite crawling, page classification agents, PDF/HTML extraction, and durable local datasets.
---

# Build Local IR Scrapers

## Workflow

Use this skill when working on local-first transcript collection code built around Ollama, LangChain, and company investor-relations websites.

1. Inspect the repo before editing. Prefer existing package layout, CLI style, and dependency conventions.
2. Keep the crawler local-first. Do not require hosted LLM APIs for page classification or extraction unless the user asks.
3. Separate concerns: company universe loading, IR discovery, HTTP/robots/rate limiting, parsing, agent classification, transcript normalization, and output persistence.
4. Make full-index S&P 500 crawling opt-in. Include `--limit` or `--symbols` paths so development starts with one or a few companies.
5. Treat missing transcripts as a valid result. Many companies publish webcasts, releases, or prepared remarks but not full transcripts.
6. Avoid running network crawls, installs, or Ollama inference unless the user explicitly asks.

## LangChain And Ollama

Prefer current package boundaries:

```python
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
```

Use structured outputs with Pydantic models for page decisions, extracted metadata, and transcript records. Keep prompts high precision: agents should identify likely transcript pages and useful links, not invent transcript metadata.

Recommended default model examples:

- `llama3.1:8b` for a common local baseline
- `qwen2.5:7b` or a user-provided model when already installed

Do not assume Ollama models are installed. Document `ollama pull ...` commands, but do not run them unless requested.

## Scraping Guardrails

Read [references/ir-transcript-architecture.md](references/ir-transcript-architecture.md) when implementing or changing crawler behavior.

Core rules:

- Respect robots.txt by default.
- Use a clear user agent and conservative per-host delay.
- Keep crawl scope to company IR hosts or hosts directly discovered from company IR pages.
- Save raw artifacts beside normalized JSON so extraction bugs can be repaired later.
- Record skipped companies and failures in machine-readable output.
- Prefer precise transcript evidence over broad keyword scraping.

## Output Shape

For each transcript, aim to persist:

- `company`: ticker, name, sector, CIK when available
- `source_url`
- `title`
- `call_date` and `fiscal_period` when confidently extracted
- `text`
- `raw_path`
- `metadata`

Write per-company crawl summaries as JSON so interrupted large runs can be audited and resumed.

## Verification

When the user allows running code, validate in escalating steps:

1. Static checks or imports.
2. One ticker with a tiny page limit.
3. A small symbol list across different IR platforms.
4. Larger S&P 500 crawl only after rate limits, output, and failure logging look sane.

