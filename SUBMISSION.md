# MaxShield AI - AGI House Submission Packet

## Team

Unique team name: **MaxShield AI**

Team member:
- Sylesh Kona - syleshkona@gmail.com

## 2-3 Sentence Summary

MaxShield AI is a multi-agent medical claim scrubbing system that catches denial risks before a claim is submitted. It runs clinical documentation review and payer-policy compliance in parallel, uses deterministic triage to decide whether a deeper audit is needed, then synthesizes a corrected claim with NCCI guardrails and a financial impact estimate. The live demo shows a same-day office visit plus knee injection where the system catches the missing Modifier 25 and returns an optimized `99214-25` claim.

## What It Does and Why It Matters

Medical billing teams lose time and money when preventable claims are rejected for missing modifiers, NCCI bundling edits, payer-specific documentation rules, or diagnosis-code mismatches. MaxShield AI acts as a pre-submission denial prevention engine:

- Accepts a clinical note, payer, provider ID, and proposed CPT/ICD-10 codes.
- Runs multiple specialized agents over the same claim.
- Applies deterministic NCCI edits so the LLM does not invent billing rules.
- Streams each agent's result to a dashboard in real time.
- Produces an optimized claim payload, prioritized revisions, denial probability, and estimated rework cost avoided.

## How It Is Built

Core orchestration:
- **LangGraph StateGraph** with parallel fan-out, fan-in, conditional routing, and final synthesis.
- **Clinical Validator Agent** checks documentation sufficiency against billed code complexity.
- **Payer Compliance Agent** reviews the claim against injected carrier-specific policy text.
- **Triage Router** is deterministic and routes high-risk claims to deeper review when max risk exceeds 75%.
- The demo's missing Modifier 25 also contributes a deterministic 0.92 NCCI triage risk, so the conditional Deep Audit path is guaranteed to be visible in the live run.
- **Deep Audit Agent** performs line-by-line review only on high-risk claims.
- **Denial Predictor / Orchestrator** synthesizes the final result and runs deterministic NCCI validation.

Interfaces:
- **FastAPI** backend with `/api/v1/scrub-claim`, `/api/v1/scrub-claim/stream`, `/api/v1/mock-demo`, and `/api/v1/health`.
- **Server-Sent Events** stream node completions into the live UI.
- **Pydantic v2 structured outputs** enforce typed agent-to-agent communication.
- **Alpine.js + Tailwind** dashboard shows claim input, agent timeline, risk gauge, before/after payload, and assessment details.
- **Model split for speed:** Clinical Validator, Payer Compliance, and Deep Audit use `claude-haiku-4-5`; only the final Orchestrator uses `claude-sonnet-4-6`.
- **Stage-safe demo mode:** the built-in critical NCCI demo uses a deterministic fast path by default, so judges see every agent event immediately instead of waiting on live model latency. Set `MAXSHIELD_FAST_DEMO=0` to force all live LLM calls.

Guardrails:
- `verify_against_ncci_edits()` detects raw code-pair edits.
- `verify_claim_against_ncci_edits()` checks the actual submitted modifiers and separates unresolved violations from resolved edits.
- Carrier rules are injected into the payer agent prompt so it reasons from policy text instead of generic memory.

## Sponsor Tools Used

W&B Weave:
- `weave.init("maxshield-ai-scrubber")` initializes tracing.
- `@weave.op()` wraps all agent nodes, routing, evaluation scorers, and pipeline eval runs.
- `weave.attributes()` logs structured metadata including agent name, payer, CPT codes, risk level, route decision, and NCCI findings.
- `weave.Evaluation` runs a 5-case golden dataset with 5 scorers for model and prompt comparison.

Anthropic Claude:
- Claude is used through `langchain-anthropic`.
- All LLM agents use `.with_structured_output()` against Pydantic models.
- The design keeps legally significant code-pair validation deterministic and uses Claude for reasoning, prioritization, and claim-correction synthesis.

## Demo Flow

1. Start the app with `python start.py`.
2. Open `http://127.0.0.1:8000`.
3. Click **Load Mock Demo - Knee Injection**.
4. Explain the intentional flaw: CPT `99214` and CPT `20610` are billed together, but the E&M code is missing Modifier 25.
5. Click **Stream Live**.
6. Narrate the orchestration:
   - Clinical Validator and Payer Compliance run in parallel on Haiku for speed.
   - Triage Router deterministically routes based on max agent risk plus NCCI risk.
   - Deep Audit fires only if risk is high.
   - Denial Predictor uses Sonnet, applies deterministic NCCI validation, and returns the optimized payload.
7. Show the final result:
   - Missing Modifier 25 found.
   - `99214` corrected to `99214-25`.
   - Unresolved NCCI violation converted into an actionable fix.
   - W&B Weave trace captures the full run.

## Judging Criteria Mapping

Agent Orchestration:
- Multiple specialized agents coordinate through a LangGraph state machine.
- Parallel fan-out and deterministic fan-in are visible in code and UI.
- Conditional Deep Audit proves the pipeline changes behavior based on earlier agent results.

Utility:
- Prevents real healthcare claim denials before submission.
- Produces output a billing team can use: prioritized fixes and corrected claim payload.

Technical Execution:
- Deterministic tests cover schemas, NCCI rules, payer rules, graph structure, API endpoints, and modifier-aware validation.
- SSE streaming keeps the demo transparent.
- Typed schemas avoid freeform JSON handoffs.

Creativity:
- Hybrid architecture combines agents with deterministic compliance checks.
- The system is not just detecting risk; it rewrites the claim into a cleaner payload.

Sponsor Usage:
- W&B Weave is used for tracing and evaluation, not just a badge.
- Claude structured outputs power the agent reasoning while deterministic rules handle compliance-critical checks.

## Verification Commands

```bash
python test_maxshield.py
python eval.py --dry-run
python start.py
```

Expected deterministic test result:

```text
Results: 40/40 passed - ALL PASS
```

## Honest Boundaries

This is a hackathon prototype, not production medical billing software. The NCCI and payer-policy tables are representative subsets built for the event demo; production use would require licensed/up-to-date rule feeds, payer contract integration, HIPAA-grade deployment controls, and compliance review by certified billing specialists.
