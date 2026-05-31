"""
MaxShield AI -- Agent node definitions.

LLM backend: Anthropic Claude (claude-sonnet-4-6 by default).
Orchestration: LangGraph parallel fan-in + conditional routing (4 agent nodes).
Tracing: every agent is decorated with @weave.op() and enriched with
         weave.attributes() for structured trace metadata in the Weave UI.

Agent roster:
  1. clinical_validator_node    -- documentation sufficiency vs code complexity
  2. payer_compliance_node      -- carrier-specific policy review (parallel with #1)
  3. triage_router_node         -- deterministic fan-in, computes max risk for routing
  4. deep_audit_node            -- line-by-line code audit, fires only on risk > 0.75
  5. orchestrator_denial_predictor_node -- NCCI check + final report synthesis
"""

from __future__ import annotations

import json
import os
from typing import Any

import weave
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from schemas import (
    AgentAssessment,
    ClaimPayload,
    CodeType,
    FinalDenialPreventionReport,
    GraphState,
)
from tools import fetch_payer_rules, verify_claim_against_ncci_edits

if not os.environ.get("WANDB_API_KEY"):
    os.environ.setdefault("WANDB_MODE", "disabled")

_DEFAULT_REASONING_MODEL = "claude-sonnet-4-6"
_DEFAULT_FAST_MODEL = "claude-haiku-4-5"


# ---------------------------------------------------------------------------
# Lazy LLM singletons
# ---------------------------------------------------------------------------

_llm_clients: dict[str, ChatAnthropic] = {}
_llm_assessment_instances: dict[str, Any] = {}
_llm_report_instance = None


def _make_llm(model: str) -> ChatAnthropic:
    return ChatAnthropic(
        model=model,
        anthropic_api_key=os.environ["ANTHROPIC_API_KEY"],
        temperature=0,
        max_tokens=int(os.environ.get("CLAUDE_MAX_TOKENS", "4096")),
        timeout=float(os.environ.get("CLAUDE_TIMEOUT_SECONDS", "90")),
        max_retries=int(os.environ.get("CLAUDE_MAX_RETRIES", "1")),
    )


def _get_llm_client(model: str) -> ChatAnthropic:
    if model not in _llm_clients:
        _llm_clients[model] = _make_llm(model)
    return _llm_clients[model]


def _get_llm_assessment(model_tier: str = "fast"):
    model = (
        os.environ.get("CLAUDE_FAST_MODEL", _DEFAULT_FAST_MODEL)
        if model_tier == "fast"
        else os.environ.get("CLAUDE_REASONING_MODEL")
        or os.environ.get("CLAUDE_MODEL", _DEFAULT_REASONING_MODEL)
    )
    if model not in _llm_assessment_instances:
        _llm_assessment_instances[model] = _get_llm_client(model).with_structured_output(AgentAssessment)
    return _llm_assessment_instances[model]


def _get_llm_report():
    global _llm_report_instance
    if _llm_report_instance is None:
        model = (
            os.environ.get("CLAUDE_REASONING_MODEL")
            or os.environ.get("CLAUDE_MODEL", _DEFAULT_REASONING_MODEL)
        )
        _llm_report_instance = _get_llm_client(model).with_structured_output(FinalDenialPreventionReport)
    return _llm_report_instance


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _deserialise_assessments(raw: list[Any]) -> list[AgentAssessment]:
    result = []
    for item in raw:
        if isinstance(item, AgentAssessment):
            result.append(item)
        elif isinstance(item, dict):
            result.append(AgentAssessment(**item))
    return result


def _serialise_claim(payload: ClaimPayload) -> str:
    codes_block = "\n".join(
        f"  [{c.code_type.value}] {c.code} -- {c.description}"
        + (f" | Modifiers: {', '.join(c.modifiers)}" if c.modifiers else " | No modifiers")
        for c in payload.proposed_codes
    )
    return (
        f"PROVIDER ID: {payload.clinical_note.provider_id}\n"
        f"PAYER: {payload.clinical_note.payer_name}\n\n"
        f"=== CLINICAL NOTE ===\n{payload.clinical_note.raw_text}\n\n"
        f"=== PROPOSED CODES ===\n{codes_block}"
    )


def _cpt_codes_from_payload(payload: ClaimPayload) -> list[str]:
    return [c.code for c in payload.proposed_codes if c.code_type == CodeType.CPT]


# ---------------------------------------------------------------------------
# Agent 1: Clinical Validator Agent (runs in PARALLEL with Agent 2)
# ---------------------------------------------------------------------------

CLINICAL_VALIDATOR_PROMPT = """\
You are the Clinical Validator Agent within MaxShield AI. Evaluate whether the
physician's documentation semantically and clinically SUPPORTS the billed codes.

EVALUATION CRITERIA:
1. E&M LEVEL JUSTIFICATION: For E&M codes (99202-99215):
   - 99214 requires moderate MDM OR 30-39 minutes of total face-to-face time.
   - 99215 requires high MDM OR 40-54 minutes total documented time.
   Documentation must explicitly state MDM complexity level or exact time.
2. PROCEDURE SUPPORT: Every CPT procedure needs documented consent, technique
   description, and post-procedure assessment in the note.
3. DIAGNOSIS SPECIFICITY: Flag unspecified ICD-10 codes (X.X9, unspecified
   variants) when the note contains laterality or causal detail that permits
   a more specific code.
4. DOCUMENTATION COMPLETENESS: Flag missing HPI, unsigned entries, absent
   procedure notes, or undocumented medical decision-making.

Set risk_score to 0.0-1.0 reflecting documentation-only denial probability.
A note missing E&M time or procedure technique always scores above 0.5.
"""


@weave.op()
def clinical_validator_node(state: GraphState) -> dict:
    """Parallel Agent 1 -- clinical documentation sufficiency check."""
    payload = state.payload
    cpt_codes = _cpt_codes_from_payload(payload)

    with weave.attributes({
        "agent": "Clinical_Validator_Agent",
        "provider_id": payload.clinical_note.provider_id,
        "payer": payload.clinical_note.payer_name,
        "cpt_codes": cpt_codes,
        "note_word_count": len(payload.clinical_note.raw_text.split()),
    }):
        messages = [
            SystemMessage(content=CLINICAL_VALIDATOR_PROMPT),
            HumanMessage(content=(
                "Evaluate this claim for clinical documentation sufficiency.\n\n"
                + _serialise_claim(payload)
            )),
        ]
        assessment: AgentAssessment = _get_llm_assessment("fast").invoke(messages)
        assessment = assessment.model_copy(update={"agent_name": "Clinical_Validator_Agent"})

    return {"assessments": [assessment]}


# ---------------------------------------------------------------------------
# Agent 2: Payer Compliance Agent (runs in PARALLEL with Agent 1)
# ---------------------------------------------------------------------------

PAYER_COMPLIANCE_PROMPT_TEMPLATE = """\
You are the Payer Compliance Agent within MaxShield AI. Review the claim
EXCLUSIVELY against the carrier-specific policy rules injected below.
You are a payer policy specialist, not a general coder.

=== {payer_name} POLICY RULES ===
{payer_rules}
=== END OF POLICY ===

EVALUATION CRITERIA:
1. MODIFIER COMPLIANCE: Flag any CPT combinations that this carrier requires
   a modifier for, based ONLY on the injected rules above.
2. PRIOR AUTH: Identify procedures requiring prior authorisation per this carrier.
3. DIAGNOSIS LINKAGE: Verify ICD-10 codes medically justify every procedure per
   this carrier's LCD/NCD coverage policies.
4. CARRIER DOCUMENTATION REQUIREMENTS: Flag unmet requirements specific to
   this carrier (stored images, referral notes, time documentation standards).
5. FREQUENCY LIMITS: Flag potential frequency limit violations.

Set risk_score > 0.80 for a missing Modifier 25 on a same-day E&M + procedure.
"""


@weave.op()
def payer_compliance_node(state: GraphState) -> dict:
    """Parallel Agent 2 -- carrier-specific policy compliance check."""
    payload = state.payload
    payer_name = payload.clinical_note.payer_name
    payer_rules = fetch_payer_rules(payer_name)
    cpt_codes = _cpt_codes_from_payload(payload)

    with weave.attributes({
        "agent": "Payer_Compliance_Agent",
        "payer": payer_name,
        "cpt_codes": cpt_codes,
        "rules_source": payer_name,
    }):
        messages = [
            SystemMessage(content=PAYER_COMPLIANCE_PROMPT_TEMPLATE.format(
                payer_name=payer_name, payer_rules=payer_rules
            )),
            HumanMessage(content=(
                f"Review this claim against {payer_name} policy. Return structured assessment.\n\n"
                + _serialise_claim(payload)
            )),
        ]
        assessment: AgentAssessment = _get_llm_assessment("fast").invoke(messages)
        assessment = assessment.model_copy(update={"agent_name": "Payer_Compliance_Agent"})

    return {"assessments": [assessment]}


# ---------------------------------------------------------------------------
# Triage Router (deterministic fan-in -- NO LLM call)
# ---------------------------------------------------------------------------

@weave.op()
def triage_router_node(assessments: list[dict], max_risk: float) -> dict:
    """
    Deterministic fan-in node. No LLM call.
    Aggregates parallel agent outputs and computes the routing decision.
    """
    with weave.attributes({
        "agent": "Triage_Router",
        "assessments_received": len(assessments),
        "max_risk_score": max_risk,
        "routing_decision": "deep_audit" if max_risk > 0.75 else "denial_predictor",
    }):
        return {
            "triage_risk_max": max_risk,
            "routing_decision": "deep_audit" if max_risk > 0.75 else "denial_predictor",
        }


# ---------------------------------------------------------------------------
# Agent 4: Deep Audit Agent (conditional -- fires only when risk > 0.75)
# ---------------------------------------------------------------------------

DEEP_AUDIT_PROMPT = """\
You are the Deep Audit Agent within MaxShield AI. You are activated ONLY when
prior agents have flagged a HIGH denial risk (> 75%). Your job is surgical,
line-by-line analysis of every single billing code on this claim.

PRIOR AGENT FINDINGS:
{prior_findings}

YOUR DEEP AUDIT RESPONSIBILITIES:
1. PER-CODE ANALYSIS: For each CPT and ICD-10 code, independently assess:
   - Is this code supported by the documentation?
   - Is this code correctly linked to a supporting diagnosis?
   - Are all required modifiers present for THIS specific code?
   - Is there a more precise or correct code that should be used?
2. NCCI CROSS-CHECK: Identify every CPT pair that could trigger a bundling
   edit and specify exactly which modifier prevents the bundle for each pair.
3. UPCODING RISK: Flag any code that appears to be billed at a higher
   complexity/value than the documentation supports.
4. MISSING CODES: Identify any billable services clearly documented in the
   note that are NOT on the proposed code list (lost revenue).
5. PRIORITY RANKING: Rank your identified_flaws from highest to lowest
   financial impact. The first item must be the single change that would
   most dramatically reduce denial probability.

Your risk_score reflects residual denial risk AFTER all your recommended
fixes are applied. If your fixes are applied, it should be below 0.30.
"""


@weave.op()
def deep_audit_node(state: GraphState) -> dict:
    """
    Conditional Agent 4 -- deep line-by-line code audit for high-risk claims.
    Only executes when triage_router routes here (max_risk > 0.75).
    """
    payload = state.payload
    existing = _deserialise_assessments(state.assessments)
    cpt_codes = _cpt_codes_from_payload(payload)

    prior_findings = "\n".join(
        f"[{a.agent_name}] Risk={a.risk_score:.2f} | Flaws: {'; '.join(a.identified_flaws)}"
        for a in existing
    )

    with weave.attributes({
        "agent": "Deep_Audit_Agent",
        "triggered_by_risk": max((a.risk_score for a in existing), default=0),
        "prior_agents": [a.agent_name for a in existing],
        "cpt_codes": cpt_codes,
        "claim_line_items": len(payload.proposed_codes),
    }):
        messages = [
            SystemMessage(content=DEEP_AUDIT_PROMPT.format(prior_findings=prior_findings)),
            HumanMessage(content=(
                "Perform a deep line-by-line audit of this high-risk claim. "
                "Return your structured assessment.\n\n"
                + _serialise_claim(payload)
            )),
        ]
        assessment: AgentAssessment = _get_llm_assessment("fast").invoke(messages)
        assessment = assessment.model_copy(update={"agent_name": "Deep_Audit_Agent"})

    return {"assessments": [assessment]}


# ---------------------------------------------------------------------------
# Agent 5: Orchestrator / Denial Predictor (always the final node)
# ---------------------------------------------------------------------------

ORCHESTRATOR_PROMPT = """\
You are the Orchestrator Denial Predictor within MaxShield AI. Synthesise all
available agent assessments into a final FinalDenialPreventionReport.

SCORING FORMULA:
- Base score = (payer_compliance_risk * 0.60) + (clinical_risk * 0.40) * 100
- If Deep_Audit_Agent ran: weight its risk_score at 50%, redistribute others 30/20
- Each unresolved CRITICAL NCCI violation: +15 points (cap at 98)
- Resolved NCCI edits with the required modifier already present should not
  increase denial probability, but should be mentioned as an audit pass.
- If all recommended fixes are applied: target score should drop below 30

FINANCIAL IMPACT:
  financial_impact_saved_usd = (denial_probability_score / 100) * 118.00 * line_items
  where $118 = MGMA benchmark administrative rework cost per denied claim.

OPTIMIZED PAYLOAD:
Construct a corrected ClaimPayload. Apply ALL recommended fixes:
- Add missing modifiers to the correct codes
- Correct any miscoded ICD-10 specificity
- Remove component codes that are bundled into a more comprehensive code
The optimized payload must be actionable -- a billing team pastes it directly
into their practice management system.

ACTIONABLE REVISIONS: Lead with the single highest-impact fix. Be specific:
  BAD:  "Add modifier"
  GOOD: "Append Modifier 25 to CPT 99214 to unbundle the E&M service from the
         same-day CPT 20610 joint injection per Aetna Clinical Policy Bulletin."
"""


@weave.op()
def orchestrator_denial_predictor_node(state: GraphState) -> dict:
    """
    Final node -- NCCI deterministic check + multi-agent synthesis + report generation.
    """
    payload = state.payload
    assessments = _deserialise_assessments(state.assessments)
    cpt_codes = _cpt_codes_from_payload(payload)

    # Deterministic NCCI check -- always runs, no LLM hallucination risk.
    # This claim-aware variant checks submitted modifiers, so a 99214-25 line
    # is treated as a resolved edit instead of an active denial risk.
    ncci_result = verify_claim_against_ncci_edits(payload.proposed_codes)

    agents_ran = [a.agent_name for a in assessments]
    avg_risk = sum(a.risk_score for a in assessments) / len(assessments) if assessments else 0

    with weave.attributes({
        "agent": "Orchestrator_Denial_Predictor",
        "agents_in_pipeline": agents_ran,
        "deep_audit_ran": "Deep_Audit_Agent" in agents_ran,
        "ncci_violations_found": len(ncci_result.get("violations", [])),
        "average_agent_risk": round(avg_risk, 3),
        "num_line_items": len(payload.proposed_codes),
    }):
        messages = [
            SystemMessage(content=ORCHESTRATOR_PROMPT),
            HumanMessage(content=(
                "Synthesise into a FinalDenialPreventionReport.\n\n"
                f"=== CLAIM CODES (ORIGINAL) ===\n"
                + "\n".join(
                    f"[{c.code_type.value}] {c.code} - {c.description} | modifiers: {c.modifiers or []}"
                    for c in payload.proposed_codes
                ) + "\n"
                f"Provider: {payload.clinical_note.provider_id} | Payer: {payload.clinical_note.payer_name}\n\n"
                f"=== AGENT ASSESSMENTS ({len(assessments)} agents ran) ===\n"
                f"{json.dumps([a.model_dump() for a in assessments], indent=2)}\n\n"
                f"=== NCCI DETERMINISTIC RESULTS ===\n{json.dumps(ncci_result, indent=2)}\n\n"
                f"Line items: {len(payload.proposed_codes)} | MGMA benchmark: $118.00/denial"
            )),
        ]
        report: FinalDenialPreventionReport = _get_llm_report().invoke(messages)
        deep_audit_ran = "Deep_Audit_Agent" in agents_ran
        report = report.model_copy(update={
            "agent_assessments": assessments,
            "ncci_edit_details": ncci_result,
            "deep_audit_triggered": deep_audit_ran,
            "pipeline_agents_run": agents_ran,
        })

    return {
        "ncci_verified": ncci_result["passed"],
        "ncci_edit_details": ncci_result,
        "final_report": report.model_dump(),
        "iterations": state.iterations + 1,
    }
