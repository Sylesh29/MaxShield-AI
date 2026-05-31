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
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph

from agents import (
    clinical_validator_node,
    deep_audit_node,
    orchestrator_denial_predictor_node,
    payer_compliance_node,
    triage_router_node,
)
from schemas import ClaimPayload, FinalDenialPreventionReport, GraphState


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
    Computes max risk across all current assessments to drive conditional routing.
    """
    assessments = state.get("assessments", [])
    max_risk = max(
        (a.get("risk_score", 0.0) if isinstance(a, dict) else a.risk_score
         for a in assessments),
        default=0.0,
    )
    result = triage_router_node(assessments, max_risk)
    return {"triage_risk_max": result["triage_risk_max"]}


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
    return "deep_audit" if state.get("triage_risk_max", 0.0) > 0.75 else "denial_predictor"


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
    initial_state = _build_initial_state(payload)
    for chunk in _compiled_graph.stream(initial_state, stream_mode="updates"):
        for node_name, node_update in chunk.items():
            yield {"node": node_name, "update": node_update}
