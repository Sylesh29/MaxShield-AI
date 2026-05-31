"""
MaxShield AI — Test suite.

Sections
--------
1. Schema validation tests       — pure Pydantic, no I/O
2. NCCI rules-engine tests        — deterministic, no LLM
3. Payer-rules tests              — deterministic, no LLM
4. Graph structure tests          — compile + node wiring, no LLM
5. FastAPI endpoint tests         — httpx TestClient, no LLM (mocked)
6. Integration test skeleton      — requires ANTHROPIC_API_KEY + WANDB_MODE=disabled

Run all deterministic tests (no API key needed):
    python test_maxshield.py

Run including integration tests (needs ANTHROPIC_API_KEY):
    ANTHROPIC_API_KEY=sk-... python test_maxshield.py --integration
"""

from __future__ import annotations

import json
import os
import sys
import traceback
import unittest
from unittest.mock import MagicMock, patch

# Disable Weave for the entire test run
os.environ.setdefault("WANDB_MODE", "disabled")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"


def run_test(name: str, fn):
    try:
        fn()
        print(f"  {PASS}  {name}")
        return True
    except Exception as exc:
        print(f"  {FAIL}  {name}")
        print(f"         {type(exc).__name__}: {exc}")
        if "--verbose" in sys.argv:
            traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# 1. Schema validation
# ---------------------------------------------------------------------------

def test_schemas():
    from schemas import (
        AgentAssessment,
        ApprovalStatus,
        ClaimPayload,
        ClinicalNote,
        CodeType,
        FinalDenialPreventionReport,
        GraphState,
        MedicalCode,
    )

    results = []

    def t_valid_claim():
        p = ClaimPayload(
            clinical_note=ClinicalNote(
                raw_text="Patient presents with knee pain, Osteoarthritis.",
                provider_id="NPI-001",
                payer_name="Aetna",
            ),
            proposed_codes=[
                MedicalCode(code="99214", code_type=CodeType.CPT, description="E&M moderate"),
                MedicalCode(code="M17.11", code_type=CodeType.ICD10_CM, description="OA right knee"),
            ],
        )
        assert p.clinical_note.payer_name == "Aetna"
        assert len(p.proposed_codes) == 2

    def t_code_stripped_upper():
        m = MedicalCode(code=" 99214 ", code_type=CodeType.CPT, description="test")
        assert m.code == "99214"

    def t_no_cpt_raises():
        try:
            ClaimPayload(
                clinical_note=ClinicalNote(raw_text="note", provider_id="X", payer_name="Aetna"),
                proposed_codes=[
                    MedicalCode(code="M17.11", code_type=CodeType.ICD10_CM, description="OA")
                ],
            )
            assert False, "Should have raised"
        except Exception:
            pass

    def t_risk_score_bounds():
        try:
            AgentAssessment(
                agent_name="test",
                approval_status=ApprovalStatus.APPROVED,
                risk_score=1.5,  # out of bounds
                identified_flaws=[],
                recommended_fixes=[],
            )
            assert False, "Should have raised"
        except Exception:
            pass

    def t_graph_state_coerces_dict_payload():
        payload_dict = {
            "clinical_note": {"raw_text": "Patient with right knee pain.", "provider_id": "NPI-001", "payer_name": "Aetna"},
            "proposed_codes": [{"code": "99214", "code_type": "CPT", "description": "E&M", "modifiers": []}],
        }
        state = GraphState(payload=payload_dict, iterations=0)
        assert state.payload.clinical_note.payer_name == "Aetna"

    def t_denial_report_uuid():
        from schemas import ClaimPayload, ClinicalNote, CodeType, MedicalCode
        payload = ClaimPayload(
            clinical_note=ClinicalNote(raw_text="note text here patient visit", provider_id="NPI-001", payer_name="Cigna"),
            proposed_codes=[MedicalCode(code="99214", code_type=CodeType.CPT, description="E&M")],
        )
        r1 = FinalDenialPreventionReport(
            denial_probability_score=75,
            financial_impact_saved_usd=118.0,
            actionable_revisions=["Add Modifier 25"],
            optimized_claim_payload=payload,
        )
        r2 = FinalDenialPreventionReport(
            denial_probability_score=75,
            financial_impact_saved_usd=118.0,
            actionable_revisions=["Add Modifier 25"],
            optimized_claim_payload=payload,
        )
        assert r1.transaction_id != r2.transaction_id, "Each report must have a unique UUID"

    for fn in [t_valid_claim, t_code_stripped_upper, t_no_cpt_raises,
               t_risk_score_bounds, t_graph_state_coerces_dict_payload, t_denial_report_uuid]:
        results.append(run_test(fn.__name__, fn))
    return results


# ---------------------------------------------------------------------------
# 2. NCCI rules engine
# ---------------------------------------------------------------------------

def test_ncci():
    from tools import verify_against_ncci_edits, verify_claim_against_ncci_edits

    results = []

    def t_99214_20610_flagged():
        r = verify_against_ncci_edits(["99214", "20610"])
        assert r["passed"] is False
        assert len(r["violations"]) == 1
        v = r["violations"][0]
        assert v["required_modifier"] == "25"
        assert v["severity"] == "CRITICAL"

    def t_99213_20610_flagged():
        r = verify_against_ncci_edits(["99213", "20610"])
        assert r["passed"] is False

    def t_reverse_order_detected():
        r = verify_against_ncci_edits(["20610", "99214"])
        assert r["passed"] is False, "Reverse order should still be detected"

    def t_clean_codes_pass():
        r = verify_against_ncci_edits(["99214", "M17.11"])
        assert r["passed"] is True
        assert r["violations"] == []

    def t_single_code_no_pairs():
        r = verify_against_ncci_edits(["99214"])
        assert r["passed"] is True
        assert r["total_pairs_checked"] == 0

    def t_colonoscopy_biopsy_bundled():
        r = verify_against_ncci_edits(["45378", "45380"])
        assert r["passed"] is False
        assert r["violations"][0]["modifier_required"] is False

    def t_checked_pairs_format():
        r = verify_against_ncci_edits(["99214", "20610"])
        assert r["total_pairs_checked"] == 1
        assert "99214<->20610" in r["checked_pairs"]

    def t_modifier_25_present_still_flagged_by_engine():
        # NCCI engine checks base code pairs only — modifier presence is the LLM's job
        r = verify_against_ncci_edits(["99214", "20610"])
        assert r["passed"] is False, (
            "NCCI engine flags the pair; the LLM/agent must verify the modifier "
            "is actually appended to the code in the payload"
        )

    def t_claim_aware_missing_modifier_25_unresolved():
        r = verify_claim_against_ncci_edits([
            {"code": "99214", "code_type": "CPT", "description": "E&M", "modifiers": []},
            {"code": "20610", "code_type": "CPT", "description": "Joint injection", "modifiers": []},
        ])
        assert r["passed"] is False
        assert r["unresolved_count"] == 1
        assert r["violations"][0]["status"] == "unresolved"

    def t_claim_aware_modifier_25_resolves_edit():
        r = verify_claim_against_ncci_edits([
            {"code": "99214", "code_type": "CPT", "description": "E&M", "modifiers": ["25"]},
            {"code": "20610", "code_type": "CPT", "description": "Joint injection", "modifiers": []},
        ])
        assert r["passed"] is True
        assert r["unresolved_count"] == 0
        assert r["resolved_count"] == 1
        assert r["resolved_edits"][0]["status"] == "resolved_by_modifier"

    def t_claim_aware_component_code_stays_unresolved():
        r = verify_claim_against_ncci_edits([
            {"code": "45378", "code_type": "CPT", "description": "Diagnostic colonoscopy", "modifiers": []},
            {"code": "45380", "code_type": "CPT", "description": "Colonoscopy with biopsy", "modifiers": []},
        ])
        assert r["passed"] is False
        assert r["violations"][0]["status"] == "unresolvable_component_code"

    for fn in [t_99214_20610_flagged, t_99213_20610_flagged, t_reverse_order_detected,
               t_clean_codes_pass, t_single_code_no_pairs, t_colonoscopy_biopsy_bundled,
               t_checked_pairs_format, t_modifier_25_present_still_flagged_by_engine,
               t_claim_aware_missing_modifier_25_unresolved,
               t_claim_aware_modifier_25_resolves_edit,
               t_claim_aware_component_code_stays_unresolved]:
        results.append(run_test(fn.__name__, fn))
    return results


# ---------------------------------------------------------------------------
# 3. Payer rules injection
# ---------------------------------------------------------------------------

def test_payer_rules():
    from tools import fetch_payer_rules

    results = []

    def t_aetna_contains_modifier_25():
        rules = fetch_payer_rules("Aetna")
        assert "Modifier 25" in rules or "MODIFIER 25" in rules
        assert "Aetna" in rules or "AETNA" in rules

    def t_case_insensitive_lookup():
        r1 = fetch_payer_rules("aetna")
        r2 = fetch_payer_rules("AETNA")
        r3 = fetch_payer_rules("Aetna")
        assert r1 == r2 == r3

    def t_united_healthcare_distinct():
        aetna = fetch_payer_rules("Aetna")
        uhc = fetch_payer_rules("UnitedHealthcare")
        assert aetna != uhc, "Each payer must return distinct rules"

    def t_all_five_payers_distinct():
        payers = ["Aetna", "UnitedHealthcare", "BlueCross BlueShield", "Cigna", "Medicare"]
        rule_sets = [fetch_payer_rules(p) for p in payers]
        assert len(set(rule_sets)) == 5, "All five payer rule sets must be unique"

    def t_unknown_payer_returns_generic():
        rules = fetch_payer_rules("SomeObscureCarrier LLC")
        assert "GENERIC" in rules or len(rules) > 50

    def t_partial_name_match():
        rules = fetch_payer_rules("UnitedHealth")
        assert "UHC" in rules or "UnitedHealthcare" in rules or "UNITEDHEALTHCARE" in rules

    for fn in [t_aetna_contains_modifier_25, t_case_insensitive_lookup,
               t_united_healthcare_distinct, t_all_five_payers_distinct,
               t_unknown_payer_returns_generic, t_partial_name_match]:
        results.append(run_test(fn.__name__, fn))
    return results


# ---------------------------------------------------------------------------
# 4. Graph structure
# ---------------------------------------------------------------------------

def test_graph_structure():
    results = []

    def t_graph_compiles():
        from graph import _compiled_graph
        assert _compiled_graph is not None

    def t_graph_has_five_nodes():
        from graph import _compiled_graph
        nodes = list(_compiled_graph.nodes)
        for expected in (
            "clinical_validator", "payer_compliance",
            "triage_router", "deep_audit", "denial_predictor",
        ):
            assert expected in nodes, f"Node '{expected}' missing from graph"

    def t_execute_function_exists():
        from graph import execute_scrubbing_pipeline, stream_scrubbing_pipeline
        import inspect
        assert callable(execute_scrubbing_pipeline)
        assert callable(stream_scrubbing_pipeline)
        sig = inspect.signature(execute_scrubbing_pipeline)
        assert "payload" in sig.parameters

    def t_parallel_fan_out_nodes_present():
        from graph import _compiled_graph
        nodes = list(_compiled_graph.nodes)
        for required in ("clinical_validator", "payer_compliance", "triage_router",
                         "deep_audit", "denial_predictor"):
            assert required in nodes, f"Node '{required}' missing from compiled graph"

    def t_state_uses_annotated_assessments():
        from graph import MaxShieldState
        import typing
        hints = typing.get_type_hints(MaxShieldState, include_extras=True)
        assessments_hint = hints.get("assessments")
        assert hasattr(assessments_hint, "__metadata__"), (
            "assessments must use Annotated[list, operator.add] for parallel-safe merging"
        )

    def t_demo_ncci_violation_forces_deep_audit_triage():
        from graph import _route_after_triage, _triage_router_wrapper

        state = {
            "payload": {
                "clinical_note": {
                    "raw_text": "Demo note with separate E&M and same-day joint injection.",
                    "provider_id": "NPI-TEST",
                    "payer_name": "Aetna",
                },
                "proposed_codes": [
                    {"code": "99214", "code_type": "CPT", "description": "E&M", "modifiers": []},
                    {"code": "20610", "code_type": "CPT", "description": "Joint injection", "modifiers": []},
                ],
            },
            "assessments": [
                {"agent_name": "Clinical_Validator_Agent", "approval_status": "FLAGGED", "risk_score": 0.30},
                {"agent_name": "Payer_Compliance_Agent", "approval_status": "FLAGGED", "risk_score": 0.40},
            ],
        }
        update = _triage_router_wrapper(state)
        assert update["triage_risk_max"] > 0.75
        assert _route_after_triage({**state, **update}) == "deep_audit"

    def t_resolved_modifier_25_does_not_force_deep_audit_triage():
        from graph import _route_after_triage, _triage_router_wrapper

        state = {
            "payload": {
                "clinical_note": {
                    "raw_text": "Clean note with Modifier 25 already submitted.",
                    "provider_id": "NPI-TEST",
                    "payer_name": "Aetna",
                },
                "proposed_codes": [
                    {"code": "99214", "code_type": "CPT", "description": "E&M", "modifiers": ["25"]},
                    {"code": "20610", "code_type": "CPT", "description": "Joint injection", "modifiers": []},
                ],
            },
            "assessments": [
                {"agent_name": "Clinical_Validator_Agent", "approval_status": "APPROVED", "risk_score": 0.20},
                {"agent_name": "Payer_Compliance_Agent", "approval_status": "APPROVED", "risk_score": 0.25},
            ],
        }
        update = _triage_router_wrapper(state)
        assert update["triage_risk_max"] == 0.25
        assert update["ncci_edit_details"]["resolved_count"] == 1
        assert _route_after_triage({**state, **update}) == "denial_predictor"

    def t_fast_demo_stream_emits_deep_audit_without_llm():
        from graph import stream_scrubbing_pipeline
        from schemas import ClaimPayload, ClinicalNote, CodeType, MedicalCode

        payload = ClaimPayload(
            clinical_note=ClinicalNote(
                raw_text="Demo note with separate E&M and same-day joint injection.",
                provider_id="NPI-TEST",
                payer_name="Aetna",
            ),
            proposed_codes=[
                MedicalCode(code="99214", code_type=CodeType.CPT, description="E&M", modifiers=[]),
                MedicalCode(code="20610", code_type=CodeType.CPT, description="Joint injection", modifiers=[]),
            ],
        )
        chunks = list(stream_scrubbing_pipeline(payload))
        nodes = [chunk["node"] for chunk in chunks]
        assert nodes == [
            "clinical_validator",
            "payer_compliance",
            "triage_router",
            "deep_audit",
            "denial_predictor",
        ]
        assert chunks[-1]["update"]["final_report"]["deep_audit_triggered"] is True

    for fn in [t_graph_compiles, t_graph_has_five_nodes, t_execute_function_exists,
               t_parallel_fan_out_nodes_present, t_state_uses_annotated_assessments,
               t_demo_ncci_violation_forces_deep_audit_triage,
               t_resolved_modifier_25_does_not_force_deep_audit_triage,
               t_fast_demo_stream_emits_deep_audit_without_llm]:
        results.append(run_test(fn.__name__, fn))
    return results


# ---------------------------------------------------------------------------
# 5. FastAPI endpoint tests (no LLM — mocked pipeline)
# ---------------------------------------------------------------------------

def test_api_endpoints():
    results = []

    try:
        from fastapi.testclient import TestClient
        import importlib.util, sys as _sys

        spec = importlib.util.spec_from_file_location("main_module", "main.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        client = TestClient(mod.app, raise_server_exceptions=True)

        def t_health_200():
            r = client.get("/api/v1/health")
            assert r.status_code == 200
            body = r.json()
            assert body["status"] == "healthy"
            assert body["service"] == "MaxShield AI"

        def t_mock_demo_200():
            r = client.get("/api/v1/mock-demo")
            assert r.status_code == 200
            body = r.json()
            assert "claim_payload" in body
            assert "scenario_summary" in body
            assert body["scenario_summary"]["payer"] == "Aetna"

        def t_mock_demo_has_proposed_codes():
            r = client.get("/api/v1/mock-demo")
            codes = r.json()["claim_payload"]["proposed_codes"]
            cpt_codes = [c["code"] for c in codes if c["code_type"] == "CPT"]
            assert "99214" in cpt_codes
            assert "20610" in cpt_codes

        def t_mock_demo_missing_modifier_25():
            r = client.get("/api/v1/mock-demo")
            codes = r.json()["claim_payload"]["proposed_codes"]
            em_code = next(c for c in codes if c["code"] == "99214")
            assert "25" not in em_code.get("modifiers", []), (
                "Demo payload must intentionally omit Modifier 25 on 99214"
            )

        def t_scrub_claim_503_without_key():
            saved = os.environ.pop("ANTHROPIC_API_KEY", None)
            try:
                payload = {
                    "clinical_note": {
                        "raw_text": "Clean established patient visit, moderate MDM documented.",
                        "provider_id": "NPI-CLEAN",
                        "payer_name": "Aetna",
                    },
                    "proposed_codes": [
                        {"code": "99214", "code_type": "CPT", "description": "E&M", "modifiers": []},
                    ],
                }
                r = client.post("/api/v1/scrub-claim", json=payload)
                assert r.status_code == 503, f"Expected 503, got {r.status_code}"
                assert "ANTHROPIC_API_KEY" in r.json()["detail"]
            finally:
                if saved:
                    os.environ["ANTHROPIC_API_KEY"] = saved

        def t_fast_demo_scrub_200_without_key():
            saved = os.environ.pop("ANTHROPIC_API_KEY", None)
            try:
                payload = client.get("/api/v1/mock-demo").json()["claim_payload"]
                r = client.post("/api/v1/scrub-claim", json=payload)
                assert r.status_code == 200, r.text
                body = r.json()
                assert body["deep_audit_triggered"] is True
                assert "25" in next(
                    c for c in body["optimized_claim_payload"]["proposed_codes"]
                    if c["code"] == "99214"
                )["modifiers"]
            finally:
                if saved:
                    os.environ["ANTHROPIC_API_KEY"] = saved

        def t_scrub_claim_422_bad_payload():
            r = client.post("/api/v1/scrub-claim", json={"bad": "data"})
            assert r.status_code == 422

        def t_openapi_schema_valid():
            r = client.get("/openapi.json")
            assert r.status_code == 200
            schema = r.json()
            paths = list(schema["paths"].keys())
            assert "/api/v1/scrub-claim" in paths
            assert "/api/v1/scrub-claim/stream" in paths, "SSE streaming endpoint must be in schema"
            assert "/api/v1/mock-demo" in paths
            assert "/api/v1/health" in paths

        def t_stream_endpoint_503_without_key():
            saved = os.environ.pop("ANTHROPIC_API_KEY", None)
            try:
                payload = {
                    "clinical_note": {
                        "raw_text": "Clean established patient visit, moderate MDM documented.",
                        "provider_id": "NPI-CLEAN",
                        "payer_name": "Aetna",
                    },
                    "proposed_codes": [
                        {"code": "99214", "code_type": "CPT", "description": "E&M", "modifiers": []},
                    ],
                }
                r = client.post("/api/v1/scrub-claim/stream", json=payload)
                assert r.status_code == 503
            finally:
                if saved:
                    os.environ["ANTHROPIC_API_KEY"] = saved

        for fn in [t_health_200, t_mock_demo_200, t_mock_demo_has_proposed_codes,
                   t_mock_demo_missing_modifier_25, t_scrub_claim_503_without_key,
                   t_fast_demo_scrub_200_without_key, t_scrub_claim_422_bad_payload, t_openapi_schema_valid,
                   t_stream_endpoint_503_without_key]:
            results.append(run_test(fn.__name__, fn))

    except Exception as exc:
        print(f"  {FAIL}  API test setup failed: {exc}")
        results.append(False)

    return results


# ---------------------------------------------------------------------------
# 6. Integration test (requires ANTHROPIC_API_KEY)
# ---------------------------------------------------------------------------

def test_integration():
    if not os.environ.get("ANTHROPIC_API_KEY", "").strip():
        print("  SKIP  integration tests — ANTHROPIC_API_KEY not set")
        return [True]

    results = []

    def t_full_pipeline_runs():
        from graph import execute_scrubbing_pipeline
        from schemas import ClaimPayload, ClinicalNote, CodeType, MedicalCode

        payload = ClaimPayload(
            clinical_note=ClinicalNote(
                raw_text=(
                    "68-year-old male with right knee osteoarthritis. Moderate MDM. "
                    "Performed intra-articular corticosteroid injection CPT 20610. "
                    "E&M conducted separately from procedure. 35 minutes total time."
                ),
                provider_id="NPI-TEST-001",
                payer_name="Aetna",
            ),
            proposed_codes=[
                MedicalCode(code="99214", code_type=CodeType.CPT,
                            description="E&M moderate complexity", modifiers=[]),
                MedicalCode(code="20610", code_type=CodeType.CPT,
                            description="Arthrocentesis major joint", modifiers=[]),
                MedicalCode(code="M17.11", code_type=CodeType.ICD10_CM,
                            description="Primary OA right knee", modifiers=[]),
            ],
        )

        report = execute_scrubbing_pipeline(payload)

        assert 0 <= report.denial_probability_score <= 100
        assert report.financial_impact_saved_usd >= 0
        assert len(report.actionable_revisions) > 0
        assert report.transaction_id
        assert len(report.agent_assessments) >= 2
        assert report.ncci_edit_details.get("passed") is False, (
            "NCCI engine must flag the 99214+20610 bundle"
        )

        print(f"         denial_probability_score = {report.denial_probability_score}%")
        print(f"         financial_impact_saved   = ${report.financial_impact_saved_usd:.2f}")
        print(f"         actionable_revisions     = {len(report.actionable_revisions)}")

    def t_optimized_payload_has_modifier_25():
        from graph import execute_scrubbing_pipeline
        from schemas import ClaimPayload, ClinicalNote, CodeType, MedicalCode

        payload = ClaimPayload(
            clinical_note=ClinicalNote(
                raw_text="Knee injection + E&M visit, moderate complexity MDM, 30 min.",
                provider_id="NPI-TEST-002",
                payer_name="Aetna",
            ),
            proposed_codes=[
                MedicalCode(code="99214", code_type=CodeType.CPT,
                            description="E&M moderate", modifiers=[]),
                MedicalCode(code="20610", code_type=CodeType.CPT,
                            description="Joint injection", modifiers=[]),
                MedicalCode(code="M17.11", code_type=CodeType.ICD10_CM,
                            description="OA right knee", modifiers=[]),
            ],
        )

        report = execute_scrubbing_pipeline(payload)
        cpt_codes = [
            c for c in report.optimized_claim_payload.proposed_codes
            if c.code_type.value == "CPT" and c.code == "99214"
        ]
        assert len(cpt_codes) == 1
        assert "25" in cpt_codes[0].modifiers, (
            f"Expected Modifier 25 on 99214 in optimized payload, got: {cpt_codes[0].modifiers}"
        )

    for fn in [t_full_pipeline_runs, t_optimized_payload_has_modifier_25]:
        results.append(run_test(fn.__name__, fn))
    return results


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main():
    run_integration = "--integration" in sys.argv

    sections = [
        ("Schema Validation", test_schemas),
        ("NCCI Rules Engine", test_ncci),
        ("Payer Rules Injection", test_payer_rules),
        ("Graph Structure", test_graph_structure),
        ("FastAPI Endpoints", test_api_endpoints),
    ]
    if run_integration:
        sections.append(("Integration (LLM)", test_integration))

    all_results = []
    for section_name, fn in sections:
        print(f"\n{'='*55}")
        print(f"  {section_name}")
        print(f"{'='*55}")
        all_results.extend(fn())

    total = len(all_results)
    passed = sum(all_results)
    failed = total - passed

    print(f"\n{'='*55}")
    print(f"  Results: {passed}/{total} passed", end="")
    if failed:
        print(f"  |  {failed} FAILED")
    else:
        print("  — ALL PASS")
    print(f"{'='*55}\n")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
