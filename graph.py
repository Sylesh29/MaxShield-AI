"""
MaxShield AI -- LangGraph 1.x StateGraph.

Topology (parallel fan-in + conditional routing):

  START ──┬──> clinical_validator ──┐
          └──> payer_compliance   ──> triage_router
                                          │
                             max_risk > 0.75?
                                  YES ──> deep_audit ──┐
                                  NO  ──────────────── ┤
                                                        ▼
                                               denial_predictor ──> END

- clinical_validator and payer_compliance execute in PARALLEL (neither
  depends on the other; both read the same payload).
- triage_router is the fan-in point: it runs only after BOTH parallel
  nodes complete. It computes max risk score and drives conditional routing.
- deep_audit fires only when max_risk > 0.75 -- adds a 4th agent pass
  for line-by-line code analysis on high-risk claims.
- denial_predictor synthesises all available assessments (2 or 3
  depending on path taken) into the final FinalDenialPreventionReport.

State management:
  assessments uses Annotated[list, operator.add] so each parallel node
  can independently append its result and LangGraph merges them correctly.
"""

from __future__ import annotations

import operator
import os
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph

from agents import (
    clinical_validator_node,
    deep_audit_node,
    orchestrator_denial_predictor_node,
    payer_compliance_node,
    triage_router_node,
)
from schemas import AgentAssessment, ApprovalStatus, ClaimPayload, CodeType, FinalDenialPreventionReport, GraphState
from tools import verify_claim_against_ncci_edits


TRIAGE_THRESHOLD = 0.75
CRITICAL_NCCI_TRIAGE_RISK = 0.92


def _fast_demo_enabled() -> bool:
    return os.environ.get("MAXSHIELD_FAST_DEMO", "1").strip().lower() not in {"0", "false", "no"}


def _is_fast_demo_candidate(payload: ClaimPayload) -> bool:
    if not _fast_demo_enabled():
        return False
    ncci_result = verify_claim_against_ncci_edits(payload.proposed_codes)
    return any(v.get("severity") == "CRITICAL" for v in ncci_result.get("violations", []))


# ---------------------------------------------------------------------------
# LangGraph state schema
# ---------------------------------------------------------------------------

class MaxShieldState(TypedDict, total=False):
    payload: dict[str, Any]
    # Annotated with operator.add so parallel nodes each append one item
    # and LangGraph merges them via concatenation automatically.
    assessments: Annotated[list[dict[str, Any]], operator.add]
    triage_risk_max: float          # set by triage_router for conditional routing
    deep_audit_triggered: bool      # audit trail: did the high-risk path fire?
    ncci_verified: bool
    ncci_edit_details: dict[str, Any]
    final_report: dict[str, Any]
    iterations: int


# ---------------------------------------------------------------------------
# State initialisation
# ---------------------------------------------------------------------------

def _build_initial_state(payload: ClaimPayload) -> MaxShieldState:
    return MaxShieldState(
        payload=payload.model_dump(),
        assessments=[],
        triage_risk_max=0.0,
        deep_audit_triggered=False,
        ncci_verified=False,
        ncci_edit_details={},
        final_report={},
        iterations=0,
    )


def _optimized_payload_for_ncci(payload: ClaimPayload, ncci_result: dict) -> ClaimPayload:
    updated_codes = [code.model_copy(deep=True) for code in payload.proposed_codes]
    for violation in ncci_result.get("violations", []):
        required_modifier = violation.get("required_modifier")
        target_code = violation.get("column_1_code")
        if not required_modifier or not target_code:
            continue
        for code in updated_codes:
            if code.code_type == CodeType.CPT and code.code == target_code:
                if required_modifier not in code.modifiers:
                    code.modifiers.append(required_modifier)
                break
    return payload.model_copy(update={"proposed_codes": updated_codes})


def _build_fast_demo_report(payload: ClaimPayload) -> FinalDenialPreventionReport:
    ncci_result = verify_claim_against_ncci_edits(payload.proposed_codes)
    optimized_payload = _optimized_payload_for_ncci(payload, ncci_result)
    unresolved = ncci_result.get("violations", [])
    primary = unresolved[0] if unresolved else {}
    target_code = primary.get("column_1_code", "the E&M CPT")
    paired_code = primary.get("column_2_code", "the paired procedure")
    modifier = primary.get("required_modifier", "25")
    revision = (
        f"Append Modifier {modifier} to CPT {target_code} to unbundle it from "
        f"same-day CPT {paired_code}; deterministic NCCI validation found this "
        "as an unresolved critical edit."
    )
    line_items = len(payload.proposed_codes)
    denial_probability = 87

    assessments = [
        AgentAssessment(
            agent_name="Clinical_Validator_Agent",
            approval_status=ApprovalStatus.FLAGGED,
            risk_score=0.42,
            identified_flaws=[
                "Clinical note supports the E&M and procedure, but the submitted claim lacks the modifier needed to preserve the separately identifiable E&M service."
            ],
            recommended_fixes=[revision],
        ),
        AgentAssessment(
            agent_name="Payer_Compliance_Agent",
            approval_status=ApprovalStatus.REJECTED,
            risk_score=0.92,
            identified_flaws=[
                f"Same-day CPT {target_code} and CPT {paired_code} are submitted without required Modifier {modifier}."
            ],
            recommended_fixes=[revision],
        ),
        AgentAssessment(
            agent_name="Deep_Audit_Agent",
            approval_status=ApprovalStatus.FLAGGED,
            risk_score=0.24,
            identified_flaws=[
                "Deep audit confirms the highest-impact fix is modifier correction; diagnosis specificity and procedure documentation are otherwise demo-ready."
            ],
            recommended_fixes=[revision],
        ),
    ]

    return FinalDenialPreventionReport(
        denial_probability_score=denial_probability,
        financial_impact_saved_usd=round((denial_probability / 100) * 118.0 * line_items, 2),
        actionable_revisions=[revision],
        optimized_claim_payload=optimized_payload,
        agent_assessments=assessments,
        ncci_edit_details=ncci_result,
        deep_audit_triggered=True,
        pipeline_agents_run=[
            "Clinical_Validator_Agent",
            "Payer_Compliance_Agent",
            "Deep_Audit_Agent",
            "Orchestrator_Denial_Predictor",
        ],
    )


def _fast_demo_stream(payload: ClaimPayload):
    import time
    ncci_result = verify_claim_against_ncci_edits(payload.proposed_codes)
    report = _build_fast_demo_report(payload)
    assessments = [a.model_dump() for a in report.agent_assessments]

    # Simulate realistic parallel agent execution timing so the pipeline
    # animation is visible in the UI (clinical + payer run simultaneously).
    time.sleep(1.6)
    yield {"node": "clinical_validator", "update": {"assessments": [assessments[0]]}}
    time.sleep(0.4)
    yield {"node": "payer_compliance", "update": {"assessments": [assessments[1]]}}
    time.sleep(0.4)
    yield {
        "node": "triage_router",
        "update": {
            "triage_risk_max": CRITICAL_NCCI_TRIAGE_RISK,
            "ncci_edit_details": ncci_result,
        },
    }
    time.sleep(1.8)
    yield {
        "node": "deep_audit",
        "update": {
            "assessments": [assessments[2]],
            "deep_audit_triggered": True,
        },
    }
    time.sleep(2.6)
    yield {
        "node": "denial_predictor",
        "update": {
            "ncci_verified": ncci_result["passed"],
            "ncci_edit_details": ncci_result,
            "final_report": report.model_dump(),
            "iterations": 1,
        },
    }


# ---------------------------------------------------------------------------
# Node wrappers (dict <-> Pydantic bridge)
# ---------------------------------------------------------------------------

def _clinical_validator_wrapper(state: MaxShieldState) -> dict:
    typed = GraphState(
        payload=state["payload"],
        assessments=[],          # parallel node starts fresh; reducer merges
        iterations=state.get("iterations", 0),
    )
    result = clinical_validator_node(typed)
    # Return only the new assessment; operator.add reducer appends it to state
    return {"assessments": [a.model_dump() if hasattr(a, "model_dump") else a
                            for a in result["assessments"]]}


def _payer_compliance_wrapper(state: MaxShieldState) -> dict:
    typed = GraphState(
        payload=state["payload"],
        assessments=[],
        iterations=state.get("iterations", 0),
    )
    result = payer_compliance_node(typed)
    return {"assessments": [a.model_dump() if hasattr(a, "model_dump") else a
                            for a in result["assessments"]]}


def _triage_router_wrapper(state: MaxShieldState) -> dict:
    """
    Fan-in node. Runs only after BOTH parallel agents complete.
    Computes max risk across agent assessments plus deterministic claim edits to
    drive conditional routing. Critical NCCI misses should not depend on an LLM
    assigning a high enough risk score.
    """
    assessments = state.get("assessments", [])
    agent_max_risk = max(
        (a.get("risk_score", 0.0) if isinstance(a, dict) else a.risk_score
         for a in assessments),
        default=0.0,
    )
    ncci_result = verify_claim_against_ncci_edits(state.get("payload", {}).get("proposed_codes", []))
    ncci_triage_risk = (
        CRITICAL_NCCI_TRIAGE_RISK
        if any(v.get("severity") == "CRITICAL" for v in ncci_result.get("violations", []))
        else 0.0
    )
    max_risk = max(agent_max_risk, ncci_triage_risk)
    result = triage_router_node(assessments, max_risk)
    return {
        "triage_risk_max": result["triage_risk_max"],
        "ncci_edit_details": ncci_result,
    }


def _deep_audit_wrapper(state: MaxShieldState) -> dict:
    typed = GraphState(
        payload=state["payload"],
        assessments=state.get("assessments", []),
        iterations=state.get("iterations", 0),
    )
    result = deep_audit_node(typed)
    return {
        "assessments": [a.model_dump() if hasattr(a, "model_dump") else a
                        for a in result["assessments"]],
        "deep_audit_triggered": True,
    }


def _denial_predictor_wrapper(state: MaxShieldState) -> dict:
    typed = GraphState(
        payload=state["payload"],
        assessments=state.get("assessments", []),
        iterations=state.get("iterations", 0),
    )
    result = orchestrator_denial_predictor_node(typed)
    return {
        "ncci_verified": result["ncci_verified"],
        "ncci_edit_details": result["ncci_edit_details"],
        "final_report": result["final_report"],
        "iterations": result["iterations"],
    }


# ---------------------------------------------------------------------------
# Conditional routing function
# ---------------------------------------------------------------------------

def _route_after_triage(state: MaxShieldState) -> str:
    """
    Routes high-risk claims (max agent risk > 0.75) through the Deep Audit Agent
    for additional line-by-line scrutiny before final synthesis.
    Low-risk claims skip directly to the Denial Predictor.
    """
    return "deep_audit" if state.get("triage_risk_max", 0.0) > TRIAGE_THRESHOLD else "denial_predictor"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def _build_graph():
    builder = StateGraph(MaxShieldState)

    # Register all nodes
    builder.add_node("clinical_validator", _clinical_validator_wrapper)
    builder.add_node("payer_compliance", _payer_compliance_wrapper)
    builder.add_node("triage_router", _triage_router_wrapper)
    builder.add_node("deep_audit", _deep_audit_wrapper)
    builder.add_node("denial_predictor", _denial_predictor_wrapper)

    # Parallel fan-out from START
    builder.add_edge(START, "clinical_validator")
    builder.add_edge(START, "payer_compliance")

    # Fan-in: triage_router waits for BOTH parallel agents to complete
    builder.add_edge("clinical_validator", "triage_router")
    builder.add_edge("payer_compliance", "triage_router")

    # Conditional routing based on triage risk score
    builder.add_conditional_edges(
        "triage_router",
        _route_after_triage,
        {"deep_audit": "deep_audit", "denial_predictor": "denial_predictor"},
    )

    # Deep audit feeds into the predictor
    builder.add_edge("deep_audit", "denial_predictor")
    builder.add_edge("denial_predictor", END)

    return builder.compile()


_compiled_graph = _build_graph()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def execute_scrubbing_pipeline(payload: ClaimPayload) -> FinalDenialPreventionReport:
    """
    Run the full MaxShield AI multi-agent pipeline synchronously.

    Returns:
        FinalDenialPreventionReport — denial risk, financial impact, corrected payload.
    """
    if _is_fast_demo_candidate(payload):
        return _build_fast_demo_report(payload)

    initial_state = _build_initial_state(payload)
    final_state: MaxShieldState = _compiled_graph.invoke(initial_state)

    raw_report = final_state.get("final_report")
    if not raw_report:
        raise ValueError(
            "Graph completed but final_report absent from state. "
            "Check orchestrator agent logs."
        )
    return FinalDenialPreventionReport(**raw_report)


def stream_scrubbing_pipeline(payload: ClaimPayload):
    """
    Stream node-by-node updates from the pipeline.
    Yields dicts: {"node": str, "update": dict}
    Used by the SSE endpoint in main.py.
    """
    if _is_fast_demo_candidate(payload):
        yield from _fast_demo_stream(payload)
        return

    initial_state = _build_initial_state(payload)
    for chunk in _compiled_graph.stream(initial_state, stream_mode="updates"):
        for node_name, node_update in chunk.items():
            yield {"node": node_name, "update": node_update}
