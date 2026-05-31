"""
MaxShield AI — W&B Weave Evaluation Harness.

This script runs a structured evaluation of the full scrubbing pipeline against
a labelled golden dataset and logs all results to W&B Weave for analysis,
hill-climbing, and model comparison.

Usage:
    python eval.py                      # run full eval, log to Weave
    python eval.py --dry-run            # validate dataset + schema, no LLM calls
    python eval.py --model claude-opus-4-8   # evaluate a specific model

Weave docs:
  https://docs.wandb.ai/weave/guides/evaluation/evaluation_logger
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

import weave
from dotenv import load_dotenv

load_dotenv(override=True)

os.environ.setdefault("WANDB_MODE", "disabled" if not os.environ.get("WANDB_API_KEY") else "online")


# ---------------------------------------------------------------------------
# Golden evaluation dataset
# ---------------------------------------------------------------------------
# Each example contains:
#   input    — the ClaimPayload dict to submit
#   expected — ground-truth labels for scoring
# ---------------------------------------------------------------------------

EVAL_DATASET: list[dict[str, Any]] = [
    # -----------------------------------------------------------------------
    # Case 1: Classic missing Modifier 25 (should be flagged)
    # -----------------------------------------------------------------------
    {
        "id": "eval-001",
        "description": "E&M + joint injection same day, Modifier 25 missing -- SHOULD FLAG",
        "input": {
            "clinical_note": {
                "raw_text": (
                    "68-year-old male, right knee osteoarthritis. Moderate complexity MDM. "
                    "35 minutes total face-to-face time. Separately identifiable E&M service "
                    "documented prior to corticosteroid knee injection. Informed consent obtained. "
                    "22g needle, medial parapatellar approach, 1mL Kenalog 40 + 3mL lidocaine. "
                    "No complications. Follow-up in 6 weeks."
                ),
                "provider_id": "NPI-1234567890",
                "payer_name": "Aetna",
            },
            "proposed_codes": [
                {"code": "99214", "code_type": "CPT", "description": "E&M moderate complexity", "modifiers": []},
                {"code": "20610", "code_type": "CPT", "description": "Arthrocentesis major joint", "modifiers": []},
                {"code": "M17.11", "code_type": "ICD-10-CM", "description": "Primary OA right knee", "modifiers": []},
            ],
        },
        "expected": {
            "should_flag_denial_risk": True,
            "ncci_violation_expected": True,
            "optimized_payload_should_have_modifier_25_on_99214": True,
            "min_denial_probability": 60,
        },
    },
    # -----------------------------------------------------------------------
    # Case 2: Modifier 25 correctly applied (should PASS)
    # -----------------------------------------------------------------------
    {
        "id": "eval-002",
        "description": "E&M + joint injection, Modifier 25 present -- SHOULD PASS",
        "input": {
            "clinical_note": {
                "raw_text": (
                    "55-year-old female, left shoulder osteoarthritis with subacromial impingement. "
                    "Moderate MDM. 32 minutes total time documented. Separate E&M documented with "
                    "distinct HPI, ROS, and MSK exam. Modifier 25 applied to E&M. "
                    "Corticosteroid injection CPT 20610 performed left shoulder under sterile technique."
                ),
                "provider_id": "NPI-9876543210",
                "payer_name": "UnitedHealthcare",
            },
            "proposed_codes": [
                {"code": "99214", "code_type": "CPT", "description": "E&M moderate complexity", "modifiers": ["25"]},
                {"code": "20610", "code_type": "CPT", "description": "Arthrocentesis major joint left shoulder", "modifiers": []},
                {"code": "M19.012", "code_type": "ICD-10-CM", "description": "Primary OA left shoulder", "modifiers": []},
            ],
        },
        "expected": {
            "should_flag_denial_risk": False,
            "ncci_violation_expected": False,
            "optimized_payload_should_have_modifier_25_on_99214": True,
            "max_denial_probability": 35,
        },
    },
    # -----------------------------------------------------------------------
    # Case 3: E&M documentation insufficient for billed level
    # -----------------------------------------------------------------------
    {
        "id": "eval-003",
        "description": "99215 billed but note only supports 99213 -- documentation mismatch",
        "input": {
            "clinical_note": {
                "raw_text": (
                    "Follow-up visit. Patient reports knee pain improving. Takes Tylenol PRN. "
                    "Exam: mild effusion. Assessment: OA stable. Plan: continue current meds."
                ),
                "provider_id": "NPI-1111111111",
                "payer_name": "BlueCross BlueShield",
            },
            "proposed_codes": [
                {"code": "99215", "code_type": "CPT", "description": "E&M high complexity", "modifiers": []},
                {"code": "M17.11", "code_type": "ICD-10-CM", "description": "Primary OA right knee", "modifiers": []},
            ],
        },
        "expected": {
            "should_flag_denial_risk": True,
            "ncci_violation_expected": False,
            "min_denial_probability": 50,
        },
    },
    # -----------------------------------------------------------------------
    # Case 4: Colonoscopy with biopsy -- component code bundling error
    # -----------------------------------------------------------------------
    {
        "id": "eval-004",
        "description": "CPT 45378 billed with 45380 -- component code cannot be separately billed",
        "input": {
            "clinical_note": {
                "raw_text": (
                    "65-year-old male, screening colonoscopy. Scope advanced to cecum. "
                    "2 polyps identified in sigmoid colon. Cold forceps biopsy x2 performed. "
                    "Specimens sent to pathology. Patient tolerated procedure well. "
                    "Bowel prep adequate. No complications."
                ),
                "provider_id": "NPI-2222222222",
                "payer_name": "Medicare",
            },
            "proposed_codes": [
                {"code": "45378", "code_type": "CPT", "description": "Diagnostic colonoscopy", "modifiers": []},
                {"code": "45380", "code_type": "CPT", "description": "Colonoscopy with biopsy", "modifiers": []},
                {"code": "Z12.11", "code_type": "ICD-10-CM", "description": "Encounter for screening colonoscopy", "modifiers": []},
            ],
        },
        "expected": {
            "should_flag_denial_risk": True,
            "ncci_violation_expected": True,
            "min_denial_probability": 70,
        },
    },
    # -----------------------------------------------------------------------
    # Case 5: Clean claim, correct codes, well-documented -- should be low risk
    # -----------------------------------------------------------------------
    {
        "id": "eval-005",
        "description": "Well-documented office visit, single E&M code, no procedures -- CLEAN",
        "input": {
            "clinical_note": {
                "raw_text": (
                    "45-year-old female, established patient presenting for annual diabetes management. "
                    "HbA1c 7.2%, well-controlled. BP 128/78. BMI 26. No hypoglycemic episodes. "
                    "Reviews 10 systems. Detailed exam: cardiovascular, neurological, ophthalmologic, "
                    "podiatric. Moderate MDM: 2 chronic conditions managed, labs reviewed, "
                    "medication adjustment. 30 minutes total documented time. "
                    "Plan: continue Metformin 1000mg BID, recheck HbA1c in 3 months."
                ),
                "provider_id": "NPI-3333333333",
                "payer_name": "Cigna",
            },
            "proposed_codes": [
                {"code": "99214", "code_type": "CPT", "description": "E&M moderate complexity established", "modifiers": []},
                {"code": "E11.65", "code_type": "ICD-10-CM", "description": "T2DM with hyperglycemia", "modifiers": []},
                {"code": "Z79.84", "code_type": "ICD-10-CM", "description": "Long-term insulin use", "modifiers": []},
            ],
        },
        "expected": {
            "should_flag_denial_risk": False,
            "ncci_violation_expected": False,
            "max_denial_probability": 25,
        },
    },
]


# ---------------------------------------------------------------------------
# Scorer functions — each is decorated with @weave.op() for granular tracing
# ---------------------------------------------------------------------------

@weave.op()
def score_denial_flag_accuracy(expected: dict, output: dict) -> dict:
    """
    Checks whether the pipeline correctly identified (or dismissed) denial risk.
    Binary accuracy: did we correctly call it high-risk vs low-risk?
    """
    predicted_high_risk = output.get("denial_probability_score", 0) >= 50
    expected_high_risk = expected.get("should_flag_denial_risk", False)
    correct = predicted_high_risk == expected_high_risk
    return {
        "flag_accuracy": 1.0 if correct else 0.0,
        "predicted_high_risk": predicted_high_risk,
        "expected_high_risk": expected_high_risk,
    }


@weave.op()
def score_ncci_detection(expected: dict, output: dict) -> dict:
    """
    Checks whether the deterministic NCCI engine correctly detected unresolved
    edits. Resolved edits with the required modifier present are audit passes,
    not active denial risks.
    """
    ncci_details = output.get("ncci_edit_details", {})
    ncci_found_violation = ncci_details.get("unresolved_count", 0) > 0
    if "unresolved_count" not in ncci_details:
        ncci_found_violation = not ncci_details.get("passed", True)
    expected_violation = expected.get("ncci_violation_expected", False)
    correct = ncci_found_violation == expected_violation
    return {
        "ncci_detection_accuracy": 1.0 if correct else 0.0,
        "ncci_found_violation": ncci_found_violation,
        "expected_violation": expected_violation,
    }


@weave.op()
def score_denial_probability_range(expected: dict, output: dict) -> dict:
    """
    Checks if denial_probability_score falls within the expected range.
    """
    score = output.get("denial_probability_score", 0)
    min_p = expected.get("min_denial_probability", 0)
    max_p = expected.get("max_denial_probability", 100)
    in_range = min_p <= score <= max_p
    return {
        "probability_in_range": 1.0 if in_range else 0.0,
        "predicted_score": score,
        "expected_min": min_p,
        "expected_max": max_p,
    }


@weave.op()
def score_modifier_25_correction(expected: dict, output: dict) -> dict:
    """
    When Modifier 25 correction is expected, checks the optimized payload has it.
    """
    if not expected.get("optimized_payload_should_have_modifier_25_on_99214"):
        return {"modifier_25_check": "not_applicable", "score": 1.0}

    optimized = output.get("optimized_claim_payload", {})
    codes = optimized.get("proposed_codes", []) if isinstance(optimized, dict) else []
    em_codes = [c for c in codes if c.get("code") == "99214"]

    if not em_codes:
        return {"modifier_25_check": "99214_not_found", "score": 0.0}

    has_25 = "25" in em_codes[0].get("modifiers", [])
    return {
        "modifier_25_check": "present" if has_25 else "missing",
        "score": 1.0 if has_25 else 0.0,
    }


@weave.op()
def score_actionable_revisions_non_empty(expected: dict, output: dict) -> dict:
    """High-risk claims must produce at least one actionable revision."""
    if not expected.get("should_flag_denial_risk"):
        return {"revisions_check": "not_required", "score": 1.0}
    revisions = output.get("actionable_revisions", [])
    has_revisions = len(revisions) > 0
    return {
        "revisions_check": "present" if has_revisions else "empty",
        "revisions_count": len(revisions),
        "score": 1.0 if has_revisions else 0.0,
    }


# ---------------------------------------------------------------------------
# Pipeline runner (called per example by Weave)
# ---------------------------------------------------------------------------

@weave.op()
def run_pipeline_for_eval(claim_payload: dict) -> dict:
    """
    Runs the full MaxShield AI scrubbing pipeline and returns the report as a dict.
    This wrapper is what Weave traces end-to-end per evaluation example.
    """
    from graph import execute_scrubbing_pipeline
    from schemas import ClaimPayload

    payload = ClaimPayload(**claim_payload)
    report = execute_scrubbing_pipeline(payload)
    return report.model_dump()


# ---------------------------------------------------------------------------
# Weave Evaluation
# ---------------------------------------------------------------------------

class MaxShieldEvaluator(weave.Evaluation):
    """
    Weave Evaluation subclass for MaxShield AI.

    Runs each example through the scrubbing pipeline and scores it against
    all five scorer functions. Results are logged to the W&B Weave UI for
    comparison across models and prompt iterations.
    """

    async def evaluate_example(self, example: dict, output: dict) -> dict:
        expected = example.get("expected", {})
        scores = {}
        scores.update(score_denial_flag_accuracy(expected, output))
        scores.update(score_ncci_detection(expected, output))
        scores.update(score_denial_probability_range(expected, output))
        scores.update(score_modifier_25_correction(expected, output))
        scores.update(score_actionable_revisions_non_empty(expected, output))

        # Composite score: mean of all binary pass/fail metrics
        binary_scores = [
            v for k, v in scores.items()
            if isinstance(v, float) and k not in ("predicted_score",)
        ]
        scores["composite_score"] = sum(binary_scores) / len(binary_scores) if binary_scores else 0.0
        return scores


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def run_dry(dataset: list[dict]) -> None:
    """Validate dataset schema without making any LLM calls."""
    from schemas import ClaimPayload
    print(f"Dry-run validating {len(dataset)} examples...")
    for ex in dataset:
        ClaimPayload(**ex["input"])
        assert "expected" in ex
        print(f"  OK  {ex['id']}  {ex['description'][:60]}")
    print("\nAll examples valid -- no LLM calls made.")


def run_evaluation(dataset: list[dict], model: str | None = None) -> None:
    """Run full evaluation with W&B Weave logging."""
    if not os.environ.get("ANTHROPIC_API_KEY", "").strip():
        print("ERROR: ANTHROPIC_API_KEY not set. Cannot run LLM evaluation.")
        sys.exit(1)

    if model:
        os.environ["CLAUDE_MODEL"] = model

    weave.init("maxshield-ai-scrubber")

    weave_dataset = weave.Dataset(
        name="maxshield-claim-eval",
        rows=[{"id": ex["id"], "input": ex["input"], "expected": ex["expected"]} for ex in dataset],
    )

    evaluation = MaxShieldEvaluator(
        name="maxshield-denial-prevention-eval",
        dataset=weave_dataset,
        scorers=[
            score_denial_flag_accuracy,
            score_ncci_detection,
            score_denial_probability_range,
            score_modifier_25_correction,
            score_actionable_revisions_non_empty,
        ],
    )

    import asyncio

    async def _predict(example: dict) -> dict:
        # run_pipeline_for_eval is synchronous and blocks for several seconds.
        # Run it in a thread so Weave can evaluate multiple examples concurrently.
        return await asyncio.to_thread(run_pipeline_for_eval, example["input"])

    print(f"\nRunning MaxShield AI evaluation on {len(dataset)} examples...")
    print(f"Model: {os.environ.get('CLAUDE_MODEL', 'claude-sonnet-4-6')}")
    print(f"Logging to W&B Weave project: maxshield-ai-scrubber\n")

    results = asyncio.run(evaluation.evaluate(_predict))

    print("\n=== EVALUATION RESULTS ===")
    for metric, value in results.items():
        if isinstance(value, (int, float)):
            print(f"  {metric:<40} {value:.3f}")
    print("===========================\n")
    print("Full results in W&B Weave: https://wandb.ai/")


def main():
    parser = argparse.ArgumentParser(description="MaxShield AI — Weave evaluation runner")
    parser.add_argument("--dry-run", action="store_true", help="Validate dataset only, no LLM calls")
    parser.add_argument("--model", type=str, help="Override Claude model (e.g. claude-opus-4-8)")
    args = parser.parse_args()

    if args.dry_run:
        run_dry(EVAL_DATASET)
    else:
        run_evaluation(EVAL_DATASET, model=args.model)


if __name__ == "__main__":
    main()
