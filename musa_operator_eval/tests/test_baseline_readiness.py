"""CPU-only tests for §4.4 baseline roster completeness."""

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "baseline_readiness_test", ROOT / "evaluator" / "baseline_readiness.py"
)
readiness = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(readiness)


def complete_baseline(tier="B_library", cases=("hidden_001", "hidden_002")):
    roles = readiness.required_roles(tier)
    document = {
        "tier": tier,
        "task_release_status": "release_ready",
        "implementations": list(roles),
        "results": [
            {
                "case_id": case_id, "implementation": role, "status": "pass", "latency_ms": 1.0,
                **({"dispatch_trace_passed": True} if role == "expert_dispatch" else {}),
            }
            for case_id in cases for role in roles
        ],
        "scoring": {
            "speedup_denominator": "expert_dispatch" if tier == "B_library" else "upstream_musa",
            "denominator_by_case": {
                case_id: {
                    "implementation": "expert_dispatch" if tier == "B_library" else "upstream_musa",
                    "reason": "frozen unit-test denominator",
                }
                for case_id in cases
            },
        },
    }
    document["readiness"] = readiness.assess(document)
    return document


class BaselineReadinessTests(unittest.TestCase):
    def test_a_complete_b_roster_covers_six_roles_for_every_case(self):
        result = readiness.assess(complete_baseline())
        self.assertTrue(result["evidence_complete"])
        self.assertTrue(result["scoreable"])
        self.assertTrue(result["release_ready"])
        self.assertEqual(len(result["required_roles"]), 6)
        self.assertEqual(result["case_count"], 2)

    def test_a_complete_a_roster_covers_four_roles(self):
        result = readiness.assess(complete_baseline("A_kernel"))
        self.assertTrue(result["evidence_complete"])
        self.assertEqual(len(result["required_roles"]), 4)

    def test_a_mudnn_fallback_requires_exact_semantic_equivalence(self):
        document = complete_baseline("A_kernel")
        selection = document["scoring"]["denominator_by_case"]["hidden_001"]
        selection["implementation"] = "mudnn_fused"
        result = readiness.assess(document)
        self.assertFalse(result["scoreable"])
        self.assertIn(
            "hidden_001: mudnn_fused fallback requires semantic_equivalence='exact'",
            result["scoreability_problems"],
        )

        selection["semantic_equivalence"] = "exact"
        self.assertTrue(readiness.assess(document)["scoreable"])

    def test_a_missing_role_is_incomplete(self):
        document = complete_baseline()
        document["implementations"].remove("mudnn_fused")
        result = readiness.assess(document)
        self.assertFalse(result["evidence_complete"])
        self.assertEqual(result["missing_roles"], ["mudnn_fused"])

    def test_a_missing_case_role_pair_is_incomplete(self):
        document = complete_baseline()
        document["results"] = [
            row for row in document["results"]
            if not (row["case_id"] == "hidden_002" and row["implementation"] == "expert_dispatch")
        ]
        result = readiness.assess(document, ("hidden_001", "hidden_002"))
        self.assertIn("hidden_002/expert_dispatch", result["missing_rows"])

    def test_an_unavailable_comparison_role_is_recorded_and_roster_complete(self):
        document = complete_baseline()
        row = next(row for row in document["results"] if row["implementation"] == "upstream_musa")
        row.update(status="unavailable", latency_ms=None, reason="not buildable on this checkout")
        result = readiness.assess(document)
        self.assertTrue(result["evidence_complete"])
        self.assertIn(f"{row['case_id']}/upstream_musa", result["unavailable_rows"])
        self.assertEqual(result["evidence_problems"], [])

    def test_a_nonpassing_row_is_visible_without_erasing_roster_completeness(self):
        document = complete_baseline()
        row = next(row for row in document["results"] if row["implementation"] == "mudnn_fused")
        row.update(status="fail", reason="output disagreed with the golden")
        result = readiness.assess(document)
        self.assertTrue(result["evidence_complete"])
        self.assertIn(f"{row['case_id']}/mudnn_fused", result["nonpassing_rows"])

    def test_a_stale_declared_summary_is_rejected(self):
        document = complete_baseline()
        document["readiness"]["case_count"] = 99
        errors = readiness.declared_readiness_errors(document)
        self.assertTrue(any("case_count" in error for error in errors), errors)


if __name__ == "__main__":
    unittest.main(verbosity=2)

