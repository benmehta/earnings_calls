# Temporary Design Note: Memory/Reflection Reranker Gap

## Problem

Memory/reflection may be working mechanically, but it is not yet influencing
every decision point strongly enough. In particular, the discovery reranker can
still choose a weak or wrong domain first even when later navigation agents have
better memory guidance.

Example issue:

```text
Reranker chooses taiwansemi.com first.
Official investor.tsmc.com/english is still present, but lower.
```

## Why This Happens

1. Memory is per-symbol and may be empty on the first run.

   If this is the first TSM/TSMC crawl, there is little prior evidence to guide
   the planner.

2. Prompt guidance currently helps navigation/page agents more than discovery.

   The prompt planner guidance is injected into specialist navigation agents and
   page classification, but the search reranker may still operate mostly from
   raw search candidates and its static prompt.

3. Bad domains may not be recorded as bad soon enough.

   If a domain appears as a candidate but is never marked rejected or low-value,
   memory cannot learn to avoid it.

4. Guidance may only deprioritize, not ban.

   A wrong domain can remain in the start set unless validation/policy removes
   it or the agent explicitly excludes it.

## Architectural Gap

Current rough shape:

```text
memory/reflection
   |
   |-- prompt planner
          |
          |-- navigation agents
          |-- page classifier

search reranker is less memory-aware
```

Better target shape:

```text
memory/reflection
   |
   |-- discovery reranker
   |-- navigation specialists
   |-- page classifier
   |-- validation/repair loop
```

## Recommended Changes

1. Feed `PromptGuidance` into `IRDiscoveryAgent`.

   The reranker should see priority terms, avoid terms, known successful hosts,
   and rejected hosts.

2. Add memory-derived discovery guidance.

   For example:

   ```text
   Prefer known successful official hosts: investor.tsmc.com
   Avoid previously rejected/low-value hosts: taiwansemi.com
   Prefer investor-relations pages over generic corporate/news/brand pages.
   ```

3. Record low-value discovery starts.

   If a start URL produces no useful IR links, no candidates, or repeated
   non-transcript pages, record its host as low-value in company memory.

4. Add a rerank validator.

   After the discovery reranker returns selections:

   - reject invented URLs
   - reject or demote avoid-term hosts
   - prefer known successful official hosts
   - require official-domain evidence in strict mode

5. Add optional strict agent mode.

   In `--agent-strict` mode:

   - no heuristic rerank fallback
   - if reranker fails validation, repair prompt once
   - if still invalid, stop or mark manual review instead of silently using
     heuristic order

## Main Takeaway

Memory/reflection is not useless; it is just not attached to the earliest and
most important decision point yet. The discovery reranker needs to become
memory-aware, and its output should be validated before start URLs are accepted.

