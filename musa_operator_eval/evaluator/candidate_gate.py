"""Apply the B-tier measured-performance admission gate."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

EVALUATOR_DIR = Path(__file__).resolve().parent
if str(EVALUATOR_DIR) not in sys.path:
    sys.path.insert(0, str(EVALUATOR_DIR))

from baseline_readiness import B_REQUIRED_ROLES, assess, declared_readiness_errors  # noqa: E402


def evaluate(baseline: dict) -> dict:
    threshold = baseline.get("scoring", {}).get("minimum_naive_to_expert_ratio", 1.3)
    readiness = assess(baseline)
    readiness_errors = declared_readiness_errors(baseline)
    if not readiness["evidence_complete"] or not readiness["scoreable"] or readiness_errors:
        return {
            "decision": "insufficient_data",
            "threshold": threshold,
            "ratios": {},
            "readiness": readiness,
            "reason": "the mandatory §4.4 baseline roster is incomplete or its readiness record has drifted",
            "problems": readiness_errors or (
                readiness["evidence_problems"] + readiness["scoreability_problems"]
            ),
        }
    by_case: dict[str, dict[str, dict]] = {}
    for result in baseline.get("results", []):
        by_case.setdefault(result["case_id"], {})[result["implementation"]] = result

    # Roster readiness only proves that every mandatory observation is present.
    # The B-tier performance gate is narrower and stricter: its numerator and
    # denominator must both pass, with usable latencies, on every measured case.
    # Auxiliary comparison roles may be unsupported on a library-gap case and
    # do not participate in this ratio.
    pair_problems = []
    for case_id, implementations in sorted(by_case.items()):
        for role in ("naive_library_composition", "expert_dispatch"):
            row = implementations.get(role)
            if row is None:
                pair_problems.append(f"{case_id}/{role}: missing")
                continue
            if row.get("status") != "pass":
                pair_problems.append(
                    f"{case_id}/{role}: {row.get('status')} ({row.get('reason')})"
                )
                continue
            latency = row.get("latency_ms")
            if not isinstance(latency, (int, float)) or latency <= 0:
                pair_problems.append(f"{case_id}/{role}: no positive latency")
    if pair_problems:
        return {
            "decision": "insufficient_data",
            "threshold": threshold,
            "ratios": {},
            "readiness": readiness,
            "reason": "every case needs paired passing naive/expert measurements",
            "problems": pair_problems,
        }

    ratios = {}
    for case_id, implementations in by_case.items():
        naive = implementations.get("naive_library_composition")
        expert = implementations.get("expert_dispatch")
        if not naive or not expert or naive.get("status") != "pass" or expert.get("status") != "pass":
            continue
        if not naive.get("latency_ms") or not expert.get("latency_ms"):
            continue
        ratios[case_id] = naive["latency_ms"] / expert["latency_ms"]
    if not ratios:
        return {"decision": "insufficient_data", "threshold": threshold, "ratios": {}, "reason": "no paired passing naive/expert measurements"}
    geometric_mean = math.exp(sum(math.log(value) for value in ratios.values()) / len(ratios))
    decision = "admit" if geometric_mean >= threshold else "reject"
    return {"decision": decision, "threshold": threshold, "geometric_mean_ratio": geometric_mean,
            "ratios": ratios, "readiness": readiness,
            "reason": "measured naive/expert geometric mean"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate(json.loads(args.baseline.read_text(encoding="utf-8")))
    encoded = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
    return 0 if result["decision"] == "admit" else 2


if __name__ == "__main__":
    raise SystemExit(main())
