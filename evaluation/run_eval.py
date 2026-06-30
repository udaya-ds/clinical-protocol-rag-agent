"""
Scores agent/extract_protocol.py's output against hand-verified ground
truth (eval_set.json).

Run after building data/structured/structured_protocols.json:
    python agent/extract_protocol.py
    python evaluation/run_eval.py

Scoring approach:
  - Scalar fields (protocol_number, phase, indication, study_design,
    treatment_duration_weeks, planned_enrollment, number_of_sites):
    exact match (case-insensitive for strings).
  - treatment_arms: set overlap (did the extractor find the same arms,
    regardless of order).
  - eligibility inclusion/exclusion: count check + substring-presence check
    for a few key phrases (loose match, since LLM wording can vary).
  - endpoints.primary: substring containment (the core claim should appear,
    exact punctuation may differ).

This is intentionally a simple, explainable scorer - the goal is an honest,
inspectable accuracy report for a portfolio project, not a publishable NLP
metric.
"""

from __future__ import annotations
import json
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
EVAL_SET_PATH = Path(__file__).parent / "eval_set.json"
STRUCTURED_PATH = BASE_DIR / "data" / "structured" / "structured_protocols.json"


def _norm(s) -> str:
    return str(s).strip().lower() if s is not None else ""


def score_case(expected: dict, actual: dict) -> dict:
    results = {}

    # Scalar exact-match fields
    for field in ["protocol_number", "phase", "indication", "study_design",
                  "treatment_duration_weeks", "planned_enrollment", "number_of_sites"]:
        exp_val = expected.get(field)
        act_val = actual.get(field)
        if isinstance(exp_val, str):
            match = _norm(exp_val) == _norm(act_val)
        else:
            match = exp_val == act_val
        results[field] = {"expected": exp_val, "actual": act_val, "match": match}

    # Treatment arms: set overlap on description text
    exp_arms = {_norm(a) for a in expected.get("treatment_arms", [])}
    act_arms = {_norm(a.get("description", "")) for a in actual.get("treatment_arms", [])}
    arm_overlap = len(exp_arms & act_arms) / len(exp_arms) if exp_arms else 1.0
    results["treatment_arms"] = {
        "expected": sorted(exp_arms), "actual": sorted(act_arms),
        "overlap_score": round(arm_overlap, 2),
    }

    # Eligibility: count + key-phrase presence
    exp_elig = expected.get("eligibility", {})
    act_elig = actual.get("eligibility", {})
    inclusion_count_match = len(act_elig.get("inclusion", [])) == exp_elig.get("inclusion_count")
    exclusion_count_match = len(act_elig.get("exclusion", [])) == exp_elig.get("exclusion_count")

    inclusion_text = " ".join(act_elig.get("inclusion", [])).lower()
    exclusion_text = " ".join(act_elig.get("exclusion", [])).lower()
    inclusion_phrase_hits = sum(1 for p in exp_elig.get("inclusion_contains", []) if p.lower() in inclusion_text)
    exclusion_phrase_hits = sum(1 for p in exp_elig.get("exclusion_contains", []) if p.lower() in exclusion_text)

    results["eligibility"] = {
        "inclusion_count_match": inclusion_count_match,
        "exclusion_count_match": exclusion_count_match,
        "inclusion_phrase_hits": f"{inclusion_phrase_hits}/{len(exp_elig.get('inclusion_contains', []))}",
        "exclusion_phrase_hits": f"{exclusion_phrase_hits}/{len(exp_elig.get('exclusion_contains', []))}",
    }

    # Primary endpoint: substring containment, loose match
    exp_endpoint = expected.get("endpoints", {}).get("primary", "")
    act_endpoints = " ".join(actual.get("endpoints", {}).get("primary", [])).lower()
    endpoint_match = _norm(exp_endpoint).rstrip(".") in act_endpoints or act_endpoints in _norm(exp_endpoint)
    results["primary_endpoint"] = {
        "expected": exp_endpoint, "actual": actual.get("endpoints", {}).get("primary", []),
        "match": endpoint_match,
    }

    return results


def summarize(all_results: dict) -> dict:
    scalar_fields = ["protocol_number", "phase", "indication", "study_design",
                      "treatment_duration_weeks", "planned_enrollment", "number_of_sites"]
    total, correct = 0, 0
    for case_results in all_results.values():
        for field in scalar_fields:
            total += 1
            if case_results[field]["match"]:
                correct += 1
        total += 1
        if case_results["primary_endpoint"]["match"]:
            correct += 1
    return {"scalar_field_accuracy": f"{correct}/{total} ({100*correct/total:.1f}%)"}


def main():
    if not STRUCTURED_PATH.exists():
        print(f"Missing {STRUCTURED_PATH}. Run agent/extract_protocol.py first.")
        return

    eval_set = json.loads(EVAL_SET_PATH.read_text())
    structured = {r["source_file"]: r for r in json.loads(STRUCTURED_PATH.read_text())}

    all_results = {}
    for case in eval_set["cases"]:
        source_file_txt = case["source_file"]
        if source_file_txt not in structured:
            print(f"WARNING: no extracted record for {source_file_txt}, skipping.")
            continue
        result = score_case(case["expected"], structured[source_file_txt])
        all_results[source_file_txt] = result

        print(f"\n=== {source_file_txt} ===")
        for field, r in result.items():
            if field in ("treatment_arms", "eligibility"):
                print(f"  {field}: {r}")
            else:
                status = "OK" if r.get("match") else "MISMATCH"
                print(f"  [{status}] {field}: expected={r['expected']!r} actual={r['actual']!r}")

    print("\n" + "=" * 50)
    print("SUMMARY")
    print(summarize(all_results))


if __name__ == "__main__":
    main()
