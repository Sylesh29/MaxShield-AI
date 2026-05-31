"""
MaxShield AI — Pydantic v2 schemas for strict type-safe state management
across the entire LangGraph multi-agent pipeline.
"""

from __future__ import annotations

import uuid
from enum import Enum
from typing import Annotated, Any

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class CodeType(str, Enum):
    ICD10_CM = "ICD-10-CM"
    CPT = "CPT"


# ---------------------------------------------------------------------------
# Core domain objects
# ---------------------------------------------------------------------------

class ClinicalNote(BaseModel):
    """Raw physician documentation plus routing metadata."""

    raw_text: str = Field(
        ...,
        min_length=10,
        description="Full physician encounter / progress note text.",
    )
    provider_id: str = Field(
        ...,
        min_length=3,
        description="NPI or internal provider identifier.",
    )
    payer_name: str = Field(
        ...,
        description="Insurance carrier name used for payer-specific rule injection.",
        examples=["Aetna", "UnitedHealthcare", "BlueCross BlueShield", "Cigna"],
    )


class MedicalCode(BaseModel):
    """A single billing or diagnosis code with its human-readable description."""

    code: str = Field(..., description="Exact billing code string, e.g. '99214' or 'M17.11'.")
    code_type: CodeType = Field(..., description="ICD-10-CM for diagnosis, CPT for procedure.")
    description: str = Field(..., description="Short human-readable code description.")
    modifiers: list[str] = Field(
        default_factory=list,
        description="List of two-digit CMS modifiers appended to this code, e.g. ['25', 'LT'].",
    )

    @field_validator("code")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        return v.strip().upper()


class ClaimPayload(BaseModel):
    """Complete claim submission: one clinical note + one or more proposed codes."""

    clinical_note: ClinicalNote
    proposed_codes: list[MedicalCode] = Field(
        ...,
        min_length=1,
        description="All CPT and ICD-10-CM codes the provider intends to bill.",
    )

    @field_validator("proposed_codes")
    @classmethod
    def at_least_one_cpt(cls, codes: list[MedicalCode]) -> list[MedicalCode]:
        cpt_codes = [c for c in codes if c.code_type == CodeType.CPT]
        if not cpt_codes:
            raise ValueError("ClaimPayload must contain at least one CPT procedure code.")
        return codes


# ---------------------------------------------------------------------------
# Agent communication objects
# ---------------------------------------------------------------------------

class ApprovalStatus(str, Enum):
    APPROVED = "APPROVED"
    FLAGGED = "FLAGGED"
    REJECTED = "REJECTED"


class AgentAssessment(BaseModel):
    """Structured assessment output emitted by every agent node in the graph."""

    agent_name: str = Field(..., description="Canonical name of the agent producing this report.")
    approval_status: ApprovalStatus = Field(
        ..., description="Overall disposition: APPROVED | FLAGGED | REJECTED."
    )
    risk_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Normalised denial-risk probability emitted by this agent (0.0 – 1.0).",
    )
    identified_flaws: list[str] = Field(
        default_factory=list,
        description="Concrete coding defects or documentation gaps identified.",
    )
    recommended_fixes: list[str] = Field(
        default_factory=list,
        description="Actionable remediation steps the billing team must apply.",
    )


# ---------------------------------------------------------------------------
# LangGraph shared state
# ---------------------------------------------------------------------------

class GraphState(BaseModel):
    """
    Pydantic view of LangGraph state — passed into agent node functions.
    The actual LangGraph state is MaxShieldState (TypedDict) in graph.py;
    agent wrappers convert between the two.
    """

    payload: ClaimPayload = Field(
        ...,
        description="Accepts ClaimPayload or dict; Pydantic coerces dict to ClaimPayload.",
    )
    assessments: list[Any] = Field(
        default_factory=list,
        description="Accepts AgentAssessment objects or dicts (after graph serialisation).",
    )
    triage_risk_max: float = Field(
        default=0.0,
        description="Max risk score across parallel agents, set by triage_router.",
    )
    deep_audit_triggered: bool = Field(
        default=False,
        description="True when the deep audit conditional path was taken.",
    )
    ncci_verified: bool = False
    ncci_edit_details: dict[str, Any] = Field(default_factory=dict)
    final_report: dict[str, Any] = Field(default_factory=dict)
    iterations: int = Field(default=0)


# ---------------------------------------------------------------------------
# Final output report
# ---------------------------------------------------------------------------

class FinalDenialPreventionReport(BaseModel):
    """
    The authoritative response returned to the caller from the
    POST /api/v1/scrub-claim endpoint.
    """

    transaction_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique UUID for this scrubbing transaction (audit trail).",
    )
    denial_probability_score: int = Field(
        ...,
        ge=0,
        le=100,
        description="Predicted probability (0-100%) that this claim will be denied as-submitted.",
    )
    financial_impact_saved_usd: float = Field(
        ...,
        ge=0.0,
        description=(
            "Estimated administrative cost avoided by preventing a denial rework cycle, "
            "based on MGMA benchmarks ($118 average rework cost per denial)."
        ),
    )
    actionable_revisions: list[str] = Field(
        default_factory=list,
        description="Prioritised list of specific changes required before resubmission.",
    )
    optimized_claim_payload: ClaimPayload = Field(
        ...,
        description="The corrected ClaimPayload with recommended modifiers and code adjustments applied.",
    )
    agent_assessments: list[AgentAssessment] = Field(
        default_factory=list,
        description="Full audit trail of every agent's individual assessment.",
    )
    ncci_edit_details: dict[str, Any] = Field(
        default_factory=dict,
        description="Verbatim output from the NCCI deterministic bundling check.",
    )
    deep_audit_triggered: bool = Field(
        default=False,
        description=(
            "True when the conditional Deep Audit Agent path fired (claim risk > 75%). "
            "Indicates a 4-agent pipeline ran instead of the standard 3-agent pipeline."
        ),
    )
    pipeline_agents_run: list[str] = Field(
        default_factory=list,
        description="Ordered list of agent names that executed in this scrubbing run.",
    )
