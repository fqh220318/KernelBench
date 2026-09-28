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
        "implementations": list(roles),
        "results": [
            {"case_id": case_id, "implementation": role, "status": "pass", "latency_ms": 1.0}
            for case_id in cases for role in roles
        ],
    }
    document["readiness"] = readiness.assess(document)
    return document


class BaselineReadinessTests(unittest.TestCase):
    def test_a_complete_b_roster_covers_six_roles_for_every_case(self):
        result = readiness.assess(complete_baseline())
        self.assertEqual(result["status"], "complete")
        self.assertEqual(len(result["required_roles"]), 6)
        self.assertEqual(result["case_count"], 2)

    def test_a_complete_a_roster_covers_four_roles(self):
        result = readiness.assess(complete_baseline("A_kernel"))
        self.assertEqual(result["status"], "complete")
        self.assertEqual(len(result["required_roles"]), 4)

    def test_a_missing_role_is_incomplete(self):
        document = complete_baseline()
        document["implementations"].remove("mudnn_fused")
        result = readiness.assess(document)
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["missing_roles"], ["mudnn_fused"])

    def test_a_missing_case_role_pair_is_incomplete(self):
        document = complete_baseline()
        document["results"] = [
            row for row in document["results"]
            if not (row["case_id"] == "hidden_002" and row["implementation"] == "expert_dispatch")
        ]
        result = readiness.assess(document, ("hidden_001", "hidden_002"))
        self.assertIn("hidden_002/expert_dispatch", result["missing_rows"])

    def test_an_unavailable_required_role_is_recorded_but_incomplete(self):
        document = complete_baseline()
        row = next(row for row in document["results"] if row["implementation"] == "upstream_musa")
        row.update(status="unavailable", latency_ms=None, reason="not buildable on this checkout")
        result = readiness.assess(document)
        self.assertEqual(result["status"], "incomplete")
        self.assertIn(f"{row['case_id']}/upstream_musa", result["unavailable_rows"])

    def test_a_failing_required_role_is_incomplete(self):
        document = complete_baseline()
        row = next(row for row in document["results"] if row["implementation"] == "mudnn_fused")
        row.update(status="fail")
        result = readiness.assess(document)
        self.assertEqual(result["status"], "incomplete")
        self.assertIn(f"{row['case_id']}/mudnn_fused", result["nonpassing_rows"])

    def test_a_stale_declared_summary_is_rejected(self):
        document = complete_baseline()
        document["readiness"]["case_count"] = 99
        errors = readiness.declared_readiness_errors(document)
        self.assertTrue(any("case_count" in error for error in errors), errors)


if __name__ == "__main__":
    unittest.main(verbosity=2)

