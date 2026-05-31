"""
MaxShield AI -- FastAPI application entry point.

Endpoints:
  POST /api/v1/scrub-claim         -- Full pipeline, returns FinalDenialPreventionReport
  POST /api/v1/scrub-claim/stream  -- SSE stream: watch agents fire node-by-node in real-time
  GET  /api/v1/mock-demo           -- Pre-built messy orthopedic encounter for demos
  GET  /api/v1/health              -- Liveness probe
  GET  /                           -- Frontend SPA
"""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager

import weave
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

# Load .env before anything else so all env vars are available at import time.
# override=True ensures .env values win even when the parent process has stale
# empty-string env vars from a previous failed load (e.g., BOM encoding issue).
load_dotenv(override=True)

from graph import execute_scrubbing_pipeline, stream_scrubbing_pipeline
from schemas import (
    ClaimPayload,
    ClinicalNote,
    CodeType,
    FinalDenialPreventionReport,
    MedicalCode,
)


# ---------------------------------------------------------------------------
# Lifespan -- initialise Weave tracing at startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialise W&B Weave tracing at startup if WANDB_API_KEY is present.
    When no key is configured the app starts normally with tracing disabled.
    """
    wandb_key = os.environ.get("WANDB_API_KEY", "").strip()
    wandb_mode = os.environ.get("WANDB_MODE", "").strip().lower()

    if wandb_key and wandb_mode != "disabled":
        try:
            weave.init("maxshield-ai-scrubber")
            print("W&B Weave initialised -- traces at https://wandb.ai/ project 'maxshield-ai-scrubber'")
        except Exception as exc:
            print(f"[WARNING] Weave init failed (tracing disabled): {exc}")
    else:
        os.environ.setdefault("WANDB_MODE", "disabled")
        print("W&B Weave tracing disabled -- set WANDB_API_KEY in .env to enable")

    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    model = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6")
    if anthropic_key:
        print(f"Claude model: {model}")
    else:
        print("WARNING: ANTHROPIC_API_KEY not set -- POST /api/v1/scrub-claim will return 503")

    yield


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="MaxShield AI",
    description=(
        "Autonomous medical claim scrubbing and real-time denial prevention engine. "
        "Powered by LangGraph multi-agent orchestration with W&B Weave observability."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the frontend SPA
_FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "frontend")
if os.path.isdir(_FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=_FRONTEND_DIR), name="static")


@app.get("/", include_in_schema=False)
async def serve_frontend():
    return FileResponse(os.path.join(_FRONTEND_DIR, "index.html"))


# ---------------------------------------------------------------------------
# Health probe
# ---------------------------------------------------------------------------

@app.get("/api/v1/health", tags=["System"])
async def health_check() -> dict:
    return {"status": "healthy", "service": "MaxShield AI", "version": "1.0.0"}


# ---------------------------------------------------------------------------
# Primary scrubbing endpoint
# ---------------------------------------------------------------------------

@app.post(
    "/api/v1/scrub-claim",
    response_model=FinalDenialPreventionReport,
    tags=["Claims Scrubbing"],
    summary="Submit a claim for multi-agent AI scrubbing and denial prevention analysis.",
)
async def scrub_claim(payload: ClaimPayload) -> FinalDenialPreventionReport:
    """
    Routes the claim through the full MaxShield AI LangGraph pipeline:
    1. Clinical Validator + Payer Compliance (parallel)
    2. Triage Router (fan-in, deterministic routing)
    3. Deep Audit Agent (conditional, risk > 0.75 only)
    4. Denial Predictor (NCCI check + final synthesis)

    All agent steps are traced in W&B Weave under 'maxshield-ai-scrubber'.
    The synchronous LangGraph pipeline runs in a thread pool so it never
    blocks the asyncio event loop.
    """
    if not os.environ.get("ANTHROPIC_API_KEY", "").strip():
        raise HTTPException(
            status_code=503,
            detail=(
                "ANTHROPIC_API_KEY is not configured. "
                "Set it in your .env file (copy .env.example) and restart. "
                "Get your key at https://console.anthropic.com/"
            ),
        )

    try:
        # Run synchronous LangGraph pipeline in a thread to avoid blocking the event loop
        report = await asyncio.to_thread(execute_scrubbing_pipeline, payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline execution failed: {str(exc)}",
        ) from exc

    return report


# ---------------------------------------------------------------------------
# SSE streaming endpoint -- watch agents fire in real-time
# ---------------------------------------------------------------------------

@app.post(
    "/api/v1/scrub-claim/stream",
    tags=["Claims Scrubbing"],
    summary="Submit a claim and receive real-time SSE events as each agent completes.",
)
async def scrub_claim_stream(payload: ClaimPayload, request: Request):
    """
    Streams node-by-node completion events via Server-Sent Events (SSE).
    The synchronous LangGraph generator runs in a thread pool and posts
    chunks to an asyncio.Queue so the event loop is never blocked.

    Event format:
        data: {"event": "node_complete", "node": "<name>", ...}

    Terminal event:
        data: {"event": "done", "final_report": {...}}

    Connect with:
        curl -N -X POST http://localhost:8000/api/v1/scrub-claim/stream
             -H "Content-Type: application/json"
             -d @claim_payload.json
    """
    if not os.environ.get("ANTHROPIC_API_KEY", "").strip():
        raise HTTPException(
            status_code=503,
            detail="ANTHROPIC_API_KEY is not configured. Set it in your .env file.",
        )

    async def event_generator():
        final_report_data = None
        deep_audit_triggered = False
        assessments_seen = []

        try:
            # Run the synchronous LangGraph generator in a thread pool so it
            # never blocks the asyncio event loop during LLM inference.
            # Chunks are posted back via a Queue for true real-time SSE.
            loop = asyncio.get_running_loop()
            queue: asyncio.Queue = asyncio.Queue()

            def _run_pipeline():
                try:
                    for _chunk in stream_scrubbing_pipeline(payload):
                        loop.call_soon_threadsafe(queue.put_nowait, _chunk)
                except Exception as _exc:
                    loop.call_soon_threadsafe(queue.put_nowait, _exc)
                finally:
                    loop.call_soon_threadsafe(queue.put_nowait, None)  # sentinel

            loop.run_in_executor(None, _run_pipeline)

            while True:
                item = await queue.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    raise item

                chunk = item
                node_name = chunk["node"]
                update = chunk["update"]

                new_assessments = update.get("assessments", [])
                assessments_seen.extend(new_assessments)

                if update.get("deep_audit_triggered"):
                    deep_audit_triggered = True

                if update.get("final_report"):
                    final_report_data = update["final_report"]

                triage_risk = update.get("triage_risk_max", 0.0)
                event_data: dict = {
                    "event": "node_complete",
                    "node": node_name,
                    "agents_run_so_far": len(assessments_seen),
                    "deep_audit_triggered": deep_audit_triggered,
                    "ncci_violations": len(
                        update.get("ncci_edit_details", {}).get("violations", [])
                    ),
                }

                # Expose routing decision so the frontend doesn't have to guess
                if node_name == "triage_router" and triage_risk > 0:
                    event_data["triage_risk_max"] = triage_risk
                    event_data["routing_decision"] = (
                        "deep_audit" if triage_risk > 0.75 else "denial_predictor"
                    )

                if node_name in ("clinical_validator", "payer_compliance", "deep_audit"):
                    if new_assessments:
                        a = (
                            new_assessments[0]
                            if isinstance(new_assessments[0], dict)
                            else new_assessments[0].model_dump()
                        )
                        event_data["agent_name"] = a.get("agent_name", node_name)
                        event_data["risk_score"] = a.get("risk_score", 0)
                        event_data["approval_status"] = a.get("approval_status", "")
                        event_data["flaws_found"] = len(a.get("identified_flaws", []))

                yield f"data: {json.dumps(event_data)}\n\n"

            # Terminal event -- include optimized payload for before/after UI but
            # strip verbose agent_assessments to keep the SSE message size reasonable
            if final_report_data:
                compact = {
                    k: v for k, v in final_report_data.items()
                    if k != "agent_assessments"
                }
                yield f"data: {json.dumps({'event': 'done', 'final_report': compact})}\n\n"
            else:
                yield "data: {\"event\": \"done\", \"error\": \"pipeline returned no report\"}\n\n"

        except Exception as exc:
            yield f"data: {json.dumps({'event': 'error', 'detail': str(exc)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ---------------------------------------------------------------------------
# Mock demo endpoint
# ---------------------------------------------------------------------------

@app.get(
    "/api/v1/mock-demo",
    response_model=dict,
    tags=["Demo"],
    summary="Returns a pre-built messy orthopedic claim payload for live demonstration.",
)
async def mock_demo() -> dict:
    """
    Returns a deliberately flawed orthopedic knee injection encounter.
    CPT 99214 + CPT 20610 billed WITHOUT Modifier 25 -- the critical NCCI bundling flaw.
    Submit the returned claim_payload to POST /api/v1/scrub-claim to see the full analysis.
    """
    messy_claim = ClaimPayload(
        clinical_note=ClinicalNote(
            raw_text=(
                "DATE OF SERVICE: 05/31/2026\n"
                "PROVIDER: Dr. Marcus Webb, MD -- Orthopaedic Surgery\n"
                "NPI: 1234567890\n"
                "PATIENT: James Holloway, DOB 03/14/1958, Male\n"
                "PAYER: Aetna | Policy ID: ATN-9981234-A\n\n"
                "CHIEF COMPLAINT:\n"
                "Mr. Holloway is a 68-year-old male presenting today with a 3-month history "
                "of worsening right knee pain, rated 7/10 at rest and 9/10 with ambulation. "
                "He reports significant difficulty with stair climbing and getting in/out "
                "of his vehicle. Previous treatments include acetaminophen and a 6-week "
                "formal physical therapy course completed in March 2026, with minimal relief.\n\n"
                "HISTORY OF PRESENT ILLNESS:\n"
                "The patient first noticed knee pain approximately 18 months ago, which has "
                "progressively worsened. He denies any recent trauma or fall. He has a known "
                "history of right knee primary osteoarthritis (diagnosed 2022) with prior "
                "X-ray showing moderate joint space narrowing at the medial compartment, "
                "subchondral sclerosis, and marginal osteophyte formation. He is not yet "
                "a surgical candidate due to cardiac comorbidities (prior CABG 2019, on "
                "Eliquis). He was last seen 8 months ago for the same knee.\n\n"
                "PAST MEDICAL HISTORY:\n"
                "- Right knee primary osteoarthritis, medial compartment (M17.11)\n"
                "- Hypertension, well-controlled on Lisinopril\n"
                "- Type 2 Diabetes Mellitus, HbA1c 7.1% (E11.65)\n"
                "- Prior CABG x4 (2019) -- Cardiology clearance obtained 04/2026\n\n"
                "MEDICATIONS: Eliquis 5mg BID (held x5 days pre-procedure per protocol), "
                "Lisinopril 10mg daily, Metformin 1000mg BID, Atorvastatin 40mg QHS.\n\n"
                "ALLERGIES: Penicillin (anaphylaxis), Sulfa drugs (rash).\n\n"
                "REVIEW OF SYSTEMS (10 systems reviewed):\n"
                "Positive: Musculoskeletal pain (right knee), limited ROM, antalgic gait.\n"
                "Negative: Fever, chills, chest pain, dyspnea, nausea/vomiting, urinary "
                "symptoms, neurological deficits, skin changes, ophthalmologic complaints.\n\n"
                "PHYSICAL EXAMINATION:\n"
                "Vital Signs: BP 138/82, HR 72, RR 16, Temp 98.6F, SpO2 98% on RA, "
                "Weight 194 lbs, Height 5'10\", BMI 27.8.\n\n"
                "RIGHT KNEE EXAMINATION:\n"
                "- Inspection: Mild periarticular swelling noted medially. No skin changes.\n"
                "- Palpation: Medial joint line tenderness ++. No effusion on ballottement.\n"
                "- Range of Motion: Flexion 0-105 degrees (limited by pain). Full extension.\n"
                "- Stability: Valgus/varus stress testing negative bilaterally. Lachman negative.\n"
                "- Neurovascular: Distal pulses intact. Sensation intact L3-S1 bilaterally.\n\n"
                "DIAGNOSTIC REVIEW:\n"
                "Standing AP and lateral X-rays of the right knee (obtained today in office): "
                "Moderate-to-severe medial compartment narrowing. Subchondral sclerosis with "
                "osteophyte formation. Grade III Kellgren-Lawrence osteoarthritis.\n\n"
                "MEDICAL DECISION MAKING (Moderate Complexity):\n"
                "Problems addressed: 1 chronic illness with exacerbation (right knee OA). "
                "Data reviewed: independent X-ray interpretation (in-office). "
                "Risk: Prescription drug management with monitoring (corticosteroid injection "
                "in context of Type 2 DM -- blood glucose monitoring counselling provided). "
                "Time spent in this E&M service, separate from procedure: 35 minutes total "
                "face-to-face time. This E&M service represents a significant, separately "
                "identifiable evaluation and management service beyond the pre/post-operative "
                "care associated with the knee injection procedure.\n\n"
                "ASSESSMENT:\n"
                "1. Right knee primary osteoarthritis, medial compartment (M17.11).\n"
                "2. Type 2 Diabetes Mellitus without complications (E11.65).\n\n"
                "PLAN & PROCEDURE NOTE -- INTRA-ARTICULAR KNEE INJECTION (CPT 20610):\n"
                "After discussion of risks, benefits, and alternatives, informed consent "
                "was obtained. Medial parapatellar approach used. 22-gauge 1.5-inch needle "
                "introduced into medial joint space. Injected: 1 mL Kenalog 40 + 3 mL 1% "
                "plain Lidocaine. Good flow without resistance. No complications.\n\n"
                "Electronically signed: Dr. Marcus Webb, MD\n"
                "Date/Time: 05/31/2026 14:47 EST"
            ),
            provider_id="NPI-1234567890",
            payer_name="Aetna",
        ),
        proposed_codes=[
            MedicalCode(
                code="99214",
                code_type=CodeType.CPT,
                description="Office/outpatient visit, established patient, moderate complexity MDM",
                modifiers=[],  # CRITICAL FLAW: Modifier 25 is MISSING
            ),
            MedicalCode(
                code="20610",
                code_type=CodeType.CPT,
                description="Arthrocentesis/aspiration/injection, major joint -- knee",
                modifiers=[],
            ),
            MedicalCode(
                code="M17.11",
                code_type=CodeType.ICD10_CM,
                description="Primary osteoarthritis, right knee",
                modifiers=[],
            ),
            MedicalCode(
                code="E11.65",
                code_type=CodeType.ICD10_CM,
                description="Type 2 diabetes mellitus with hyperglycemia",
                modifiers=[],
            ),
        ],
    )

    return {
        "description": (
            "MaxShield AI Live Demo -- Orthopedic Knee Injection Encounter. "
            "This payload intentionally omits Modifier 25 on CPT 99214, creating a "
            "critical NCCI bundling violation with CPT 20610. Submit this payload to "
            "POST /api/v1/scrub-claim to see the full multi-agent denial prevention "
            "analysis in action."
        ),
        "scenario_summary": {
            "patient_age": 68,
            "encounter_type": "Orthopedic Office Visit + Intra-articular Knee Injection",
            "payer": "Aetna",
            "intentional_flaw": "CPT 99214 billed same-day as CPT 20610 WITHOUT Modifier 25",
            "expected_denial_type": "NCCI Bundling Edit -- E&M bundled with minor procedure",
            "estimated_revenue_at_risk_usd": 165.00,
            "estimated_admin_rework_cost_usd": 118.00,
            "corrected_code": "99214-25 (Modifier 25 appended to E&M to unbundle from injection)",
        },
        "claim_payload": messy_claim.model_dump(),
        "instructions": (
            "Copy the 'claim_payload' object and POST it to /api/v1/scrub-claim "
            "to run the full MaxShield AI scrubbing pipeline."
        ),
    }
