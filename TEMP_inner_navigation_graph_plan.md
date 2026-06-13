# Temporary Design Note: Inner Navigation Decision Graph

## Goal

Add a bounded validation-and-repair loop around each navigation subagent decision.
The outer supervised LangGraph still controls whole crawl attempts. This inner
flow would control one page-level navigation decision.

## Why

Specialist agents can still choose weak links, such as blog, YouTube, SEC-only,
or webcast-only paths. Instead of immediately falling back to heuristics, the
system should validate the agent output, feed the validation failure back into
prompt guidance, and retry the same specialist once.

## Proposed Inner Flow

```text
START
  |
select_specialist
  |
call_specialist
  |
validate_decision
  |
  |-- valid --> END
  |
  |-- invalid and repair unused --> repair_prompt --> call_specialist
  |
  |-- invalid and repair already used --> fallback_or_stop --> END
```

## Validation Checks

- Chosen URLs must be from the provided candidate links.
- Chosen URLs must satisfy allowed-host policy.
- Chosen URLs must fit the specialist task:
  - `HomepageNavAgent`: investor/IR/shareholder-like links.
  - `IRSectionAgent`: earnings/events/financial reports/quarterly results.
  - `EventListingAgent`: latest earnings-call event or event-detail links.
  - `TranscriptLinkAgent`: transcript/PDF/DOCX/Q&A/prepared-remarks links.
- Chosen URLs should not match planner `avoid_terms` unless no better links exist.
- Confidence should meet a floor, for example `0.55`.
- Number of followed links should stay bounded.

## Repair Step

If validation fails, create compact repair guidance such as:

```text
Previous decision rejected: EventListingAgent chose blog.google, but avoid_terms
include blog. Choose official investor event-detail or earnings-call links only.
```

Then retry the same specialist once with updated guidance.

## Fallback

If the repaired decision also fails:

- Use deterministic top-scored links, or
- Stop navigation on that page and record a trace reason.

Suggested trace reasons:

- `agent_validated`
- `agent_rejected_low_confidence`
- `agent_rejected_avoid_term`
- `agent_rejected_wrong_task`
- `agent_repair_attempted`
- `agent_repair_failed`
- `heuristic_repair_after_agent`

## Safety Boundary

The inner graph must remain advisory. It must not override:

- robots.txt enforcement
- allowed-host policy
- configured page/depth limits
- third-party-source restrictions
- transcript save heuristics

The LLM proposes link choices and prompt repairs. Deterministic code validates
and enforces policy.

## Implementation Sketch

- Add `NavigationValidationResult`.
- Add `validate_navigation_decision(kind, decision, links, guidance, company)`.
- Add optional `NavigationDecisionGraph` or equivalent LangGraph subgraph.
- Start with one repair attempt maximum.
- Keep heuristic fallback as the final emergency path.
- Add fixture tests for:
  - blog link rejected when avoid terms include blog
  - event listing repair chooses event-detail link
  - invented URL rejected
  - low confidence triggers repair
  - repaired failure falls back deterministically

