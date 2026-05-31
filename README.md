# MaxShield AI

**Autonomous medical claim scrubbing and real-time denial prevention engine**

Built at the AGI House × W&B × SundAI Multi-Agent Orchestration Build Day — May 31, 2026, Cambridge MA.

---

## The Problem

The US healthcare system loses **$265 billion per year** to claim denials and rework cycles. The average cost to rework a single denied claim is **$118 (MGMA benchmark)**. Most denials are preventable — they stem from missing modifiers, NCCI bundling violations, and carrier-specific policy mismatches that a billing team catches only *after* the payer rejects the claim.

MaxShield AI scrubs claims **before** submission, catching those errors autonomously.

---

## Demo — The Knee Injection Scenario

The built-in mock demo loads a real-world orthopedic encounter: a 68-year-old patient with right knee osteoarthritis who receives both an office visit (CPT 99214) and a corticosteroid injection (CPT 20610) on the same day. The codes are submitted *without* Modifier 25 on the E&M — a critical NCCI bundling violation that virtually every commercial payer will auto-deny.

MaxShield AI autonomously:

1. Detects the missing Modifier 25 via the Payer Compliance Agent and Aetna-specific injected policy
2. Confirms the NCCI bundling violation via the deterministic rules engine (no LLM guessing)
3. Returns an optimized claim with `99214-25` corrected and the financial impact quantified

```
Denial probability:  87%  →  <25% after applying fixes
Financial impact:    $408.74 in admin rework cost prevented
NCCI violations:     1 CRITICAL (99214 + 20610, Modifier 25 missing)
Agents fired:        4  (Deep Audit Agent triggered by high risk)
```

---

## Architecture

### Multi-Agent Graph (LangGraph 1.x)

```
                        ┌─────────────────────────────┐
                        │           START              │
                        └──────────────┬──────────────┘
                                       │
                         ┌─────────────┴──────────────┐
                         │  PARALLEL FAN-OUT           │
                         ▼                             ▼
          ┌──────────────────────┐    ┌──────────────────────────┐
          │  Clinical Validator  │    │    Payer Compliance       │
          │  (Claude Sonnet 4.6) │    │  (Claude + Dynamic Rules) │
          │  Doc sufficiency vs  │    │  Carrier-specific policy  │
          │  code complexity     │    │  & modifier requirements  │
          └──────────┬───────────┘    └────────────┬─────────────┘
                     │                             │
                     └──────────────┬──────────────┘
                                    │  FAN-IN (both must complete)
                                    ▼
                         ┌──────────────────────┐
                         │    Triage Router      │
                         │  Deterministic — no   │
                         │  LLM. Computes max    │
                         │  risk score & routes  │
                         └──────────┬────────────┘
                                    │
                      max_risk > 0.75?
                    YES ─────────────────────────── NO
                     │                               │
                     ▼                               │
          ┌──────────────────────┐                  │
          │   Deep Audit Agent   │                  │
          │  Line-by-line code   │                  │
          │  analysis. Fires     │                  │
          │  only on high risk.  │                  │
          └──────────┬───────────┘                  │
                     │                              │
                     └──────────────┬───────────────┘
                                    ▼
                         ┌──────────────────────┐
                         │   Denial Predictor   │
                         │  NCCI deterministic  │
                         │  check + synthesis   │
                         │  + financial metrics │
                         └──────────┬───────────┘
                                    │
                                   END
```

**Key design decisions:**

| Decision | Rationale |
|---|---|
| Parallel fan-out (Clinical + Payer) | Neither agent depends on the other's output. Running simultaneously halves wall-clock time and demonstrates real multi-agent coordination. |
| Deterministic fan-in triage | The routing decision (risk > 0.75 → Deep Audit) combines agent risk with deterministic NCCI risk. The demo's missing Modifier 25 forces the 4th-agent path even if an LLM under-scores the issue. |
| Conditional Deep Audit Agent | A 4th agent pass is expensive. It only fires when the combined risk from the parallel agents justifies it — keeps low-risk claims fast. |
| NCCI guardrail at orchestrator | Billing code combinations are legally defined rules. LLMs must not hallucinate these. The NCCI engine runs deterministically *before* the LLM synthesises the final report. |
| `Annotated[list, operator.add]` state | Parallel nodes each return one assessment. LangGraph merges them via the reducer — no race conditions, no overwriting. |

### Hallucination Prevention

The single biggest risk in an agentic billing system is an LLM inventing or misremembering billing rules. MaxShield AI prevents this at two layers:

1. **`verify_against_ncci_edits(cpt_codes)` + `verify_claim_against_ncci_edits(code_lines)`** — A pure Python rules engine with 9 hardcoded NCCI edit pairs. The claim-aware validator also checks submitted modifiers, separating unresolved violations from edits already resolved by Modifier 25/50. The LLM never decides whether two codes are bundled; it only decides how to explain and fix deterministic findings.

2. **Dynamic payer rule injection** — `fetch_payer_rules(payer_name)` injects the exact text of carrier-specific policy into the Payer Compliance Agent's system prompt. The LLM reasons against real policy language, not generic billing guidelines.

---

## W&B Weave Integration

Every agent call in the pipeline is fully traced in W&B Weave under the project `maxshield-ai-scrubber`.

**Tracing depth:**
- `@weave.op()` on all 4 agent nodes — captures inputs, outputs, token counts, latency
- `weave.attributes({...})` inside every agent with structured metadata:
  ```python
  weave.attributes({
      "agent": "Payer_Compliance_Agent",
      "payer": "Aetna",
      "cpt_codes": ["99214", "20610"],
      "risk_level": "high",
  })
  ```
- The Triage Router logs its routing decision as a Weave op for full pipeline auditability

**Evaluation harness (`eval.py`):**

```bash
python eval.py                           # run full evaluation, log to Weave
python eval.py --dry-run                 # validate dataset schema, no LLM calls
python eval.py --model claude-opus-4-8   # compare models side-by-side in Weave UI
```

5 golden labelled cases × 5 scorers logged as a `weave.Evaluation`:

| Scorer | What it measures |
|---|---|
| `score_denial_flag_accuracy` | Did we correctly call the claim high-risk vs low-risk? |
| `score_ncci_detection` | Did the deterministic engine catch (or correctly pass) bundling? |
| `score_denial_probability_range` | Is the predicted score within the expected range? |
| `score_modifier_25_correction` | Does the optimized payload have Modifier 25 applied where required? |
| `score_actionable_revisions_non_empty` | Do high-risk claims produce at least one revision? |

Composite score = mean of all binary pass/fail metrics, logged per model per run.

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Claude Haiku 4.5 for fast assessment agents; Claude Sonnet 4.6 for final orchestration via `langchain-anthropic` |
| Agent orchestration | LangGraph 1.2.2 — `StateGraph` with `Annotated` reducers |
| LLM tracing & evaluation | W&B Weave 0.52.x — `@weave.op()`, `weave.Evaluation`, `weave.Dataset` |
| Structured outputs | Pydantic v2 `.with_structured_output()` — no freeform text between nodes |
| API layer | FastAPI 0.136+ with `asyncio.to_thread` (non-blocking LangGraph calls) |
| SSE streaming | `asyncio.Queue` + thread pool — real-time agent progress events |
| Frontend | Alpine.js 3 + Tailwind CSS — single-file SPA, no build step |
| Validation | 40 deterministic tests across schemas, NCCI engine, payer rules, graph wiring, HTTP endpoints |

---

## Project Structure

```
maxshield-ai/
├── schemas.py          # Pydantic v2 domain + LangGraph state models (204 lines)
├── tools.py            # NCCI rules engine + 5-carrier payer rule injector (270 lines)
├── agents.py           # 4 Claude agent nodes, @weave.op() + weave.attributes() (389 lines)
├── graph.py            # LangGraph StateGraph — parallel, fan-in, conditional (235 lines)
├── main.py             # FastAPI app — REST + SSE streaming endpoints (429 lines)
├── eval.py             # W&B Weave evaluation harness, 5 cases × 5 scorers (409 lines)
├── start.py            # Single-command launcher with env-var validation (67 lines)
├── test_maxshield.py   # 40 deterministic tests, no LLM required
├── frontend/
│   └── index.html      # Alpine.js SPA — agent timeline, gauge, before/after (1029 lines)
├── requirements.txt
├── .env.example
└── README.md
```

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/Sylesh29/maxshield-ai.git
cd maxshield-ai
pip install -r requirements.txt
```

### 2. Set your API keys

```bash
cp .env.example .env
```

Edit `.env`:
```
ANTHROPIC_API_KEY=sk-ant-...    # console.anthropic.com
WANDB_API_KEY=...               # wandb.ai/authorize
```

### 3. Run the server

```bash
python start.py
```

```
MaxShield AI starting on http://0.0.0.0:8000
Dashboard   :  http://127.0.0.1:8000/
Swagger UI  :  http://127.0.0.1:8000/docs
```

### 4. Open the dashboard

Navigate to **http://127.0.0.1:8000** — click **Load Mock Demo** to pre-populate the orthopedic knee injection encounter, then click **Stream Live** to watch all 5 agents fire in real-time.

### 5. Run the test suite (no API key required)

```bash
python test_maxshield.py
# Results: 40/40 passed - ALL PASS
```

### 6. Run the Weave evaluation (requires API keys)

```bash
python eval.py
# Logs 5-case × 5-scorer evaluation to W&B Weave
# Add --model claude-opus-4-8 to compare models
```

---

## API Reference

### `POST /api/v1/scrub-claim`
Submit a claim and receive a complete `FinalDenialPreventionReport`.

**Request body:** `ClaimPayload`
```json
{
  "clinical_note": {
    "raw_text": "68-year-old male, right knee OA...",
    "provider_id": "NPI-1234567890",
    "payer_name": "Aetna"
  },
  "proposed_codes": [
    { "code": "99214", "code_type": "CPT", "description": "E&M moderate complexity", "modifiers": [] },
    { "code": "20610", "code_type": "CPT", "description": "Arthrocentesis major joint", "modifiers": [] },
    { "code": "M17.11", "code_type": "ICD-10-CM", "description": "Primary OA right knee", "modifiers": [] }
  ]
}
```

**Response:** `FinalDenialPreventionReport`
```json
{
  "transaction_id": "uuid",
  "denial_probability_score": 87,
  "financial_impact_saved_usd": 408.74,
  "deep_audit_triggered": true,
  "pipeline_agents_run": ["Clinical_Validator_Agent", "Payer_Compliance_Agent", "Deep_Audit_Agent"],
  "actionable_revisions": [
    "CRITICAL: Append Modifier 25 to CPT 99214 to unbundle from same-day CPT 20610..."
  ],
  "ncci_edit_details": { "passed": false, "violations": [...] },
  "optimized_claim_payload": { "proposed_codes": [ { "code": "99214", "modifiers": ["25"] }, ... ] },
  "agent_assessments": [...]
}
```

### `POST /api/v1/scrub-claim/stream`
Same pipeline, streamed as Server-Sent Events. Each agent completion yields one event.

```bash
curl -N -X POST http://localhost:8000/api/v1/scrub-claim/stream \
     -H "Content-Type: application/json" \
     -d @claim.json
```

```
data: {"event":"node_complete","node":"clinical_validator","risk_score":0.42,"approval_status":"FLAGGED"}
data: {"event":"node_complete","node":"payer_compliance","risk_score":0.88,"approval_status":"REJECTED"}
data: {"event":"node_complete","node":"triage_router","routing_decision":"deep_audit","triage_risk_max":0.92}
data: {"event":"node_complete","node":"deep_audit","risk_score":0.71,"flaws_found":2}
data: {"event":"node_complete","node":"denial_predictor"}
data: {"event":"done","final_report":{...}}
```

### `GET /api/v1/mock-demo`
Returns the pre-built knee injection encounter with the intentional Modifier 25 flaw. Ready to POST directly to `/scrub-claim`.

### `GET /api/v1/health`
Liveness probe — returns `{"status": "healthy"}`.

---

## The NCCI Rules Engine

The deterministic guardrail layer covers 9 code-pair rules and reports whether each edit is unresolved or resolved by a submitted modifier. The raw pair detector catches bundling relationships; the claim-aware validator inspects the actual modifier list on each CPT line.

Included rules:

| Pair | Rule | Modifier Required |
|---|---|---|
| 99213 + 20610 | E&M bundled with major joint injection | **25** on E&M |
| 99214 + 20610 | E&M bundled with major joint injection | **25** on E&M |
| 99215 + 20610 | E&M bundled with major joint injection | **25** on E&M |
| 99213 + 20600 | E&M bundled with small joint injection | **25** on E&M |
| 99214 + 20600 | E&M bundled with small joint injection | **25** on E&M |
| 99213 + 12001 | E&M bundled with laceration repair | **25** on E&M |
| 99214 + 12001 | E&M bundled with laceration repair | **25** on E&M |
| 27447 + 27447 | Bilateral TKA billed twice | **50** (bilateral) |
| 45378 + 45380 | Diagnostic colonoscopy bundled into colonoscopy w/biopsy | Remove 45378 |

---

## Supported Payers

Distinct carrier-specific policy rules are injected for:

- **Aetna** — Modifier 25 documentation standard, prior auth triggers, diagnosis specificity requirements
- **UnitedHealthcare** — ClaimLogic™ bundling edits, separate HPI/ROS documentation standard
- **BlueCross BlueShield** — FEP 4-element Modifier 25 threshold, frequency limitations, referral requirements
- **Cigna** — MDM complexity documentation, multiple procedure reduction (Modifier 51), J-code NDC requirements
- **Medicare** — CMS NCCI binding edits, LCD medical necessity, ABN requirements, incident-to rules

---

## Judging Criteria Mapping

| Criterion | Implementation |
|---|---|
| **Agent Orchestration** | 5-node LangGraph graph: parallel fan-out, deterministic fan-in, conditional routing to a 4th agent on risk > 75%. Agents communicate exclusively via Pydantic structured outputs — no freeform text. |
| **Utility** | Targets a real $265B/year problem. The demo shows a live claim correction with exact dollar impact. |
| **Technical Execution** | Non-blocking `asyncio.to_thread` for all LLM calls, deterministic fast path for the built-in NCCI demo, fast Haiku assessment agents, real-time SSE via `asyncio.Queue` + thread pool, 40 deterministic tests. |
| **Creativity** | Combines deterministic guardrails (NCCI engine) with LLM reasoning — a hybrid architecture that prevents the failure mode (hallucination) that makes most medical AI non-deployable. |
| **Sponsor Usage — Claude** | `claude-sonnet-4-6` via `langchain-anthropic` with `.with_structured_output()` on all 4 agents. Structured outputs enforce Pydantic schemas — no JSON parsing, no prompt engineering for format. |
| **Sponsor Usage — W&B Weave** | `weave.init()` + `@weave.op()` on all agents + `weave.attributes()` per call + `weave.Evaluation` with 5 scorers across a 5-case golden dataset with `asyncio.to_thread` for concurrent evaluation. |

---

## Built at

**AGI House × W&B × TNT × SundAI Club × E14 — Multi-Agent Orchestration Build Day**
May 31, 2026 · The Engine, 750 Main St, Cambridge MA

**Team:** Sylesh Kona (syleshkona@gmail.com)
