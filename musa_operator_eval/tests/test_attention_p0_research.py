"""CPU-only checks for the first three Attention P0 pre-task contracts."""

import ast
import json
import math
import unittest
from pathlib import Path

import numpy as np

from musa_operator_eval.research.attention_p0.cpu_reference import (
    paged_kv_decode_reference,
    per_sequence_causal_reference,
    varlen_packed_reference,
)


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = ROOT.parent
RESEARCH = ROOT / "research" / "attention_p0"
EXPECTED = {
    "attention_per_sequence_causal_dispatch",
    "attention_varlen_packed_forward",
    "paged_kv_attention_decode",
}


class CpuReferenceTests(unittest.TestCase):
    def test_per_sequence_flags_are_not_collapsed_to_one_scalar(self):
        q = np.zeros((2, 2, 1, 1), dtype=np.float32)
        k = np.zeros_like(q)
        v = np.array([[[[1.0]], [[3.0]]], [[[1.0]], [[3.0]]]])

        output, lse = per_sequence_causal_reference(
            q,
            k,
            v,
            q_lens=np.array([2, 2]),
            kv_lens=np.array([2, 2]),
            causal_flags=np.array([False, True]),
        )

        np.testing.assert_allclose(output[0, :, 0, 0], [2.0, 2.0])
        np.testing.assert_allclose(output[1, :, 0, 0], [1.0, 2.0])
        np.testing.assert_allclose(lse[0, :, 0], [math.log(2), math.log(2)])
        np.testing.assert_allclose(lse[1, :, 0], [0.0, math.log(2)])
        self.assertEqual(output.dtype, np.float64)

    def test_per_sequence_flags_must_really_be_boolean(self):
        tensor = np.zeros((1, 1, 1, 1))
        with self.assertRaisesRegex(ValueError, "boolean vector"):
            per_sequence_causal_reference(
                tensor,
                tensor,
                tensor,
                q_lens=[1],
                kv_lens=[1],
                causal_flags=[1],
            )

    def test_varlen_uses_bottom_right_alignment_and_allows_empty_kv(self):
        q = np.zeros((3, 1, 1))
        k = np.zeros((3, 1, 1))
        v = np.array([[[1.0]], [[3.0]], [[5.0]]])

        output, lse = varlen_packed_reference(
            q,
            k,
            v,
            cu_seqlens_q=[0, 2, 3],
            cu_seqlens_kv=[0, 3, 3],
            causal=True,
        )

        np.testing.assert_allclose(output[:, 0, 0], [2.0, 3.0, 0.0])
        np.testing.assert_allclose(lse[:2, 0], [math.log(2), math.log(3)])
        self.assertTrue(np.isneginf(lse[2, 0]))

    def test_paged_decode_honours_table_order_and_duplicate_pages(self):
        q = np.zeros((1, 1, 1, 1))
        key_cache = np.zeros((2, 2, 1, 1))
        value_cache = np.array([[[[1.0]], [[2.0]]], [[[3.0]], [[4.0]]]])

        permuted, _ = paged_kv_decode_reference(
            q,
            key_cache,
            value_cache,
            block_table=np.array([[1, 0]]),
            cache_seqlens=[3],
        )
        duplicated, _ = paged_kv_decode_reference(
            q,
            key_cache,
            value_cache,
            block_table=np.array([[0, 0]]),
            cache_seqlens=[4],
        )

        np.testing.assert_allclose(permuted[0, 0, 0, 0], 8.0 / 3.0)
        np.testing.assert_allclose(duplicated[0, 0, 0, 0], 1.5)

    def test_paged_decode_rejects_only_required_invalid_pages(self):
        q = np.zeros((1, 1, 1, 1))
        cache = np.zeros((1, 2, 1, 1))
        paged_kv_decode_reference(
            q,
            cache,
            cache,
            block_table=np.array([[0, 999]]),
            cache_seqlens=[1],
        )
        with self.assertRaisesRegex(ValueError, "page id is out of range"):
            paged_kv_decode_reference(
                q,
                cache,
                cache,
                block_table=np.array([[1]]),
                cache_seqlens=[1],
            )


class ResearchArtifactTests(unittest.TestCase):
    def test_contract_and_matrix_cover_exactly_the_planned_first_three(self):
        contracts = json.loads((RESEARCH / "semantic_contracts.json").read_text(encoding="utf-8"))
        matrix = json.loads((RESEARCH / "capability_probe_matrix.json").read_text(encoding="utf-8"))

        self.assertEqual({item["candidate_id"] for item in contracts["contracts"]}, EXPECTED)
        self.assertEqual({item["candidate_id"] for item in matrix["candidates"]}, EXPECTED)
        self.assertFalse(contracts["hardware_validated"])
        self.assertFalse(matrix["hardware_validated"])
        self.assertEqual(matrix["status"], "not_run")
        self.assertTrue(all(
            cell["status"] == "not_run"
            for candidate in matrix["candidates"]
            for cell in candidate["cells"]
        ))

    def test_candidate_pool_points_to_the_new_local_evidence(self):
        pool = json.loads(
            (ROOT / "sources" / "attention_operator_candidate_pool.json").read_text(encoding="utf-8")
        )
        by_id = {item["candidate_id"]: item for item in pool["candidates"]}
        expected_paths = {
            "musa_operator_eval/research/attention_p0/cpu_reference.py",
            "musa_operator_eval/research/attention_p0/semantic_contracts.json",
            "musa_operator_eval/research/attention_p0/capability_probe_matrix.json",
        }
        for candidate_id in EXPECTED:
            with self.subTest(candidate=candidate_id):
                paths = {anchor["path"] for anchor in by_id[candidate_id]["local_anchors"]}
                self.assertEqual(paths, expected_paths)
                self.assertTrue(all((REPOSITORY_ROOT / path).is_file() for path in paths))
                self.assertFalse(by_id[candidate_id]["hardware_validated"])

    def test_cpu_reference_has_no_framework_or_device_dependency(self):
        source = (RESEARCH / "cpu_reference.py").read_text(encoding="utf-8")
        imported = set()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".", 1)[0])
        self.assertNotIn("torch", imported)
        self.assertNotIn("torch_musa", imported)


if __name__ == "__main__":
    unittest.main()
