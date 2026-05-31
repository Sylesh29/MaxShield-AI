"""
MaxShield AI — Deterministic tool functions used by agent nodes.

These functions act as a guardrail layer that prevents LLM hallucination
on legally/financially significant billing rules. Every code combination
validated here is based on published CMS NCCI Policy Manual logic.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# NCCI Bundling Rules Database (mock — representative subset)
# ---------------------------------------------------------------------------

# Structure: (column_1_code, column_2_code) -> {"modifier_required": bool, "modifier": str, "rationale": str}
_NCCI_EDIT_TABLE: dict[tuple[str, str], dict] = {
    # E&M + minor surgical procedure on the same day requires Modifier 25 on the E&M code
    ("99213", "20610"): {
        "modifier_required": True,
        "modifier": "25",
        "rationale": (
            "CMS NCCI Policy: CPT 99213 (E&M) and CPT 20610 (Arthrocentesis/injection, "
            "major joint) are bundled. A separate, significant E&M service on the same day "
            "requires Modifier 25 appended to the E&M code to unbundle."
        ),
    },
    ("99214", "20610"): {
        "modifier_required": True,
        "modifier": "25",
        "rationale": (
            "CMS NCCI Policy: CPT 99214 (E&M, moderate complexity) and CPT 20610 "
            "(Arthrocentesis/injection, major joint) are bundled. Modifier 25 is required "
            "on the E&M code to establish medical necessity for a separately identifiable "
            "evaluation and management service."
        ),
    },
    ("99215", "20610"): {
        "modifier_required": True,
        "modifier": "25",
        "rationale": (
            "CMS NCCI Policy: CPT 99215 and CPT 20610 are bundled. Modifier 25 on the "
            "E&M is required to unbundle."
        ),
    },
    ("99213", "20600"): {
        "modifier_required": True,
        "modifier": "25",
        "rationale": (
            "CMS NCCI Policy: CPT 99213 and CPT 20600 (Arthrocentesis/injection, small "
            "joint) are bundled. Modifier 25 required on E&M."
        ),
    },
    ("99214", "20600"): {
        "modifier_required": True,
        "modifier": "25",
        "rationale": (
            "CMS NCCI Policy: CPT 99214 and CPT 20600 are bundled. Modifier 25 required "
            "on E&M to unbundle."
        ),
    },
    # Evaluation & management unbundling with laceration repair
    ("99213", "12001"): {
        "modifier_required": True,
        "modifier": "25",
        "rationale": (
            "CMS NCCI Policy: E&M service (99213) bundled with simple laceration repair "
            "(12001). Modifier 25 on E&M required."
        ),
    },
    ("99214", "12001"): {
        "modifier_required": True,
        "modifier": "25",
        "rationale": (
            "CMS NCCI Policy: E&M service (99214) bundled with simple laceration repair "
            "(12001). Modifier 25 on E&M required."
        ),
    },
    # Bilateral procedure without Modifier 50
    ("27447", "27447"): {
        "modifier_required": True,
        "modifier": "50",
        "rationale": (
            "Bilateral total knee arthroplasty (CPT 27447) billed twice requires "
            "Modifier 50 (bilateral) on the second unit or per payer-specific instructions."
        ),
    },
    # Colonoscopy with biopsy — 45380 includes 45378 (diagnostic colonoscopy is component)
    ("45378", "45380"): {
        "modifier_required": False,
        "modifier": "",
        "rationale": (
            "CMS NCCI Policy: CPT 45378 (diagnostic colonoscopy) is a component of "
            "CPT 45380 (colonoscopy with biopsy). Bill only 45380 — 45378 cannot be "
            "separately reported."
        ),
    },
}


def verify_against_ncci_edits(cpt_codes: list[str]) -> dict:
    """
    Deterministic NCCI bundling rules engine.

    Checks every unique ordered pair of CPT codes in ``cpt_codes`` against the
    internal NCCI edit table.  Returns a structured result that agents MUST
    inspect before finalising any claim.

    Args:
        cpt_codes: List of raw CPT code strings (modifiers must be stripped
                   before passing — pass only the 5-character base code).

    Returns:
        dict with keys:
            ``passed``         – True only if NO bundling violations were found.
            ``violations``     – List of violation detail dicts (empty on pass).
            ``checked_pairs``  – All code pairs that were evaluated.
    """
    clean_codes = [c.strip().upper() for c in cpt_codes]
    violations: list[dict] = []
    checked_pairs: list[tuple[str, str]] = []

    for i, code_a in enumerate(clean_codes):
        for code_b in clean_codes[i + 1 :]:
            pair = (code_a, code_b)
            reverse_pair = (code_b, code_a)
            checked_pairs.append(pair)

            rule = _NCCI_EDIT_TABLE.get(pair) or _NCCI_EDIT_TABLE.get(reverse_pair)
            if rule:
                violations.append(
                    {
                        "column_1_code": pair[0],
                        "column_2_code": pair[1],
                        "modifier_required": rule["modifier_required"],
                        "required_modifier": rule["modifier"],
                        "rationale": rule["rationale"],
                        "severity": "CRITICAL" if rule["modifier_required"] else "ERROR",
                    }
                )

    return {
        "passed": len(violations) == 0,
        "violations": violations,
        "checked_pairs": [f"{a}<->{b}" for a, b in checked_pairs],
        "total_pairs_checked": len(checked_pairs),
    }


# ---------------------------------------------------------------------------
# Payer-Specific Policy Rules Database (mock)
# ---------------------------------------------------------------------------

_PAYER_RULES: dict[str, str] = {
    "aetna": (
        "AETNA CLINICAL POLICY BULLETIN — CLAIMS SUBMISSION GUIDELINES (Effective 2026-01):\n"
        "1. MODIFIER 25 MANDATORY: Aetna requires Modifier 25 on ALL same-day E&M codes when "
        "a procedure is performed. Documentation must explicitly state the E&M was a SEPARATE, "
        "IDENTIFIABLE service beyond the pre/post-operative care of the procedure. Without this "
        "documentation language, the E&M will be automatically denied.\n"
        "2. PRIOR AUTH: CPT 20610 (joint injection) requires prior authorization when billed "
        "more than 3 times in a rolling 12-month period for the same joint.\n"
        "3. DIAGNOSIS SPECIFICITY: Aetna mandates the highest level of ICD-10 specificity. "
        "Unspecified codes (e.g., M17.9 — Osteoarthritis of knee, unspecified) will trigger "
        "a medical records request. Use laterality-specific codes (M17.11 right primary, "
        "M17.12 left primary) wherever documented.\n"
        "4. PLACE OF SERVICE: Office (POS 11) claims for joint injections require the "
        "supervising physician's NPI on the claim. Incident-to billing rules apply strictly.\n"
        "5. TIMELY FILING: Claims must be submitted within 90 days of date of service. "
        "Late submissions are denied without appeal rights under standard contracts."
    ),
    "unitedhealthcare": (
        "UNITEDHEALTHCARE CLAIMS EDITING POLICY — PROVIDER HANDBOOK EXCERPT (2026 Q1):\n"
        "1. MODIFIER 25 WITH SUPPORTING DOCUMENTATION: UHC honours Modifier 25 ONLY when "
        "the medical record contains a distinct chief complaint and history for the E&M portion "
        "SEPARATE from the injection consent/procedure note. A single combined SOAP note is "
        "insufficient and will result in denial of the E&M component.\n"
        "2. BUNDLING EDITS: UHC applies ClaimLogic™ edits that are MORE RESTRICTIVE than "
        "standard CMS NCCI. Check the UHC Clinical Editing Tool before submission.\n"
        "3. ICD-10 LINKAGE: Every CPT code must link to a medically appropriate ICD-10 code. "
        "Mismatched diagnosis-to-procedure linkage (e.g., billing an injection for a fracture "
        "code) results in automatic denial.\n"
        "4. FACILITY vs PROFESSIONAL: When the physician performs a procedure in an ASC or "
        "HOPDs, the professional fee component must use POS 24. Incorrect POS results in "
        "payment at the lower facility rate, not the professional rate.\n"
        "5. IMAGING REQUIREMENT: Ultrasound guidance (CPT 76942) billed with a joint injection "
        "requires a stored image in the medical record as proof. Image documentation must note "
        "needle placement confirmation."
    ),
    "bluecross blueshield": (
        "BCBS FEDERAL EMPLOYEE PROGRAM — BILLING AND CODING GUIDELINES (2026):\n"
        "1. MODIFIER 25 THRESHOLD: BCBS-FEP applies the most stringent Modifier 25 standard. "
        "The E&M note must contain: (a) distinct HPI, (b) separate ROS, (c) independent "
        "physical exam elements unrelated to the procedure site, AND (d) an independent "
        "medical decision-making section. Failing any element voids the Modifier 25 unbundling.\n"
        "2. FREQUENCY LIMITATIONS: Viscosupplementation injections (CPT 20610 + J7321-J7325) "
        "are limited to one series per knee per lifetime under the FEP formulary.\n"
        "3. COORDINATION OF BENEFITS: For Medicare Advantage enrollees, Medicare billing rules "
        "take precedence. BCBS FEP is secondary payer — submit Medicare EOB with every claim.\n"
        "4. REFERRAL REQUIRED: Specialist visits (orthopedics, rheumatology) require a valid "
        "PCP referral on file. Date of referral must predate the date of service.\n"
        "5. SIGNATURE REQUIREMENTS: All claims must include a valid treating provider signature "
        "in the medical record dated on or before the service date."
    ),
    "cigna": (
        "CIGNA HEALTHCARE REIMBURSEMENT POLICY — OFFICE PROCEDURES (Updated March 2026):\n"
        "1. MODIFIER 25 DOCUMENTATION STANDARD: Cigna follows AMA CPT guidelines. "
        "Modifier 25 is valid when the E&M represents a significant, separately identifiable "
        "service. Cigna's auditors specifically look for TIME documentation or MDM complexity "
        "justification when the E&M and procedure share the same anatomical site.\n"
        "2. MULTIPLE PROCEDURE REDUCTION: When more than one procedure is billed on the same "
        "date, Cigna applies a 50% reduction to the lesser procedure RVU value. Modifier 51 "
        "(multiple procedures) must be appended to secondary procedures.\n"
        "3. TELEHEALTH RESTRICTIONS: Injections and procedures cannot be billed under a "
        "telehealth POS code. Physical presence of the physician is mandatory and audited.\n"
        "4. DRUG BILLING: J-codes for injectable medications (e.g., corticosteroids, "
        "hyaluronic acid) must be billed with the exact NDC number and units matching "
        "the administered dose per the drug package insert.\n"
        "5. APPEALS WINDOW: Cigna allows 180 days from EOB date to file a reconsideration. "
        "Submit with medical records, operative reports, and a cover letter citing specific "
        "policy language."
    ),
    "medicare": (
        "CMS MEDICARE FEE-FOR-SERVICE — CLAIMS PROCESSING MANUAL CHAPTER 12 (Rev. 2026):\n"
        "1. NCCI EDITS BINDING: All CMS NCCI edits apply without exception. Column 2 codes "
        "cannot be unbundled from Column 1 without an appropriate modifier AND documentation "
        "supporting a distinct service.\n"
        "2. MEDICAL NECESSITY: LCD (Local Coverage Determination) for joint injections "
        "(L34539) requires documented conservative therapy failure (minimum 6 weeks of PT "
        "or NSAID therapy) prior to corticosteroid injection approval.\n"
        "3. ADVANCE BENEFICIARY NOTICE: If Medicare is likely to deny a service as not "
        "medically necessary, an ABN must be signed by the patient BEFORE the service.\n"
        "4. INCIDENT-TO RULES: Mid-level provider (PA/NP) services billed incident-to "
        "require the supervising physician to be present in the office suite (not necessarily "
        "the same room) during the service.\n"
        "5. GLOBAL SURGERY PACKAGE: Post-operative E&M services within the global surgery "
        "period (10 or 90 days) cannot be separately billed without Modifier 24 (unrelated "
        "E&M during postop period)."
    ),
}

_DEFAULT_PAYER_RULES = (
    "GENERIC COMMERCIAL PAYER GUIDELINES:\n"
    "1. Follow AMA CPT Assistant guidance for all modifier usage.\n"
    "2. Apply standard CMS NCCI bundling edits.\n"
    "3. Ensure ICD-10 codes support medical necessity for all billed procedures.\n"
    "4. Obtain prior authorisation for procedures listed on the payer's PA required list.\n"
    "5. Submit claims within 90 days of date of service unless contract specifies otherwise."
)


def fetch_payer_rules(payer_name: str) -> str:
    """
    Returns carrier-specific denial-prevention guidelines for the given payer.

    The returned string is injected verbatim into the Payer Compliance Agent's
    system prompt context so the LLM reasons against real, carrier-specific rules.

    Args:
        payer_name: Human-readable insurance carrier name (case-insensitive).

    Returns:
        Multi-line string of policy constraints for the requested carrier, or
        generic commercial payer guidelines if the carrier is not in the database.
    """
    normalised = payer_name.strip().lower()
    for key, rules in _PAYER_RULES.items():
        if key in normalised or normalised in key:
            return rules
    return _DEFAULT_PAYER_RULES
