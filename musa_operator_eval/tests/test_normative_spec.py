"""Phase 21 guards for the preserved requirement and its executable decisions."""

from __future__ import annotations

import hashlib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
GUIDE = ROOT / "docs" / "musa_operator_evaluation_guide.md"
SPEC = ROOT / "docs" / "musa_operator_evaluation_normative_spec_v1.md"
MATRIX = ROOT / "docs" / "musa_operator_evaluation_requirement_traceability_matrix.md"

GUIDE_NORMALIZED_SHA256 = "9f87707ed0379c15080bbdf83d4868be8a353fc7818489173224fdd8fdf454c5"
DECISION_IDS = tuple(f"NORM-{number:03d}" for number in range(1, 13))


class NormativeSpecTests(unittest.TestCase):
    def test_preserved_guide_has_not_been_rewritten(self):
        normalized = GUIDE.read_text(encoding="utf-8").replace("\r\n", "\n")
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        self.assertEqual(digest, GUIDE_NORMALIZED_SHA256)

    def test_spec_and_traceability_matrix_exist(self):
        self.assertTrue(SPEC.is_file())
        self.assertTrue(MATRIX.is_file())

    def test_every_normative_decision_is_tracked_once(self):
        matrix = MATRIX.read_text(encoding="utf-8")
        for decision_id in DECISION_IDS:
            with self.subTest(decision_id=decision_id):
                self.assertEqual(matrix.count(f"| {decision_id} |"), 1)

    def test_spec_freezes_the_high_risk_boundaries(self):
        spec = SPEC.read_text(encoding="utf-8")
        required_phrases = (
            "Python `ModelNew`",
            "通过 `torch_musa`",
            "opaque `source_id`",
            "独立、确定性的 CPU 高精度 reference",
            "`atol`、`rtol` 和 `max_abs_error`",
            "`evidence_complete`",
            "`scoreable`",
            "`release_ready`",
            "geomean(naive / expert) >= 1.3",
        )
        for phrase in required_phrases:
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, spec)


if __name__ == "__main__":
    unittest.main()
