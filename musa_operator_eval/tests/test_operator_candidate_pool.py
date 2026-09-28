import copy
import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = ROOT.parent
POOL_PATH = ROOT / "sources" / "operator_candidate_pool.json"
INDEX_PATH = ROOT / "sets" / "candidates" / "CANDIDATE_POOL.md"
ATTENTION_POOL_PATH = ROOT / "sources" / "attention_operator_candidate_pool.json"
ATTENTION_INDEX_PATH = ROOT / "sets" / "candidates" / "ATTENTION_CANDIDATE_POOL.md"


def load_tool():
    path = ROOT / "evaluator" / "collect_operator_candidate_pool.py"
    spec = importlib.util.spec_from_file_location("collect_operator_candidate_pool_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class OperatorCandidatePoolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tool = load_tool()
        cls.pool = json.loads(POOL_PATH.read_text(encoding="utf-8"))
        cls.attention_pool = json.loads(ATTENTION_POOL_PATH.read_text(encoding="utf-8"))

    def test_pool_is_valid_and_summary_is_derived(self):
        self.assertEqual(self.tool.pool_errors(self.pool), [])
        self.assertEqual(self.pool["summary"]["source_count"], 9)
        self.assertEqual(self.pool["summary"]["family_count"], 6)
        self.assertEqual(self.pool["summary"]["candidate_count"], 19)
        self.assertEqual(self.pool["summary"]["shortlisted"], 12)

    def test_sources_are_immutable_and_licensed(self):
        for source in self.pool["sources"]:
            with self.subTest(source=source["source_id"]):
                self.assertRegex(source["revision"], r"^[0-9a-f]{40}$")
                self.assertTrue(source["license_verified"])
                self.assertTrue(source["evidence_paths"])

    def test_local_anchors_exist(self):
        for candidate in self.pool["candidates"]:
            for anchor in candidate["local_anchors"]:
                with self.subTest(candidate=candidate["candidate_id"], path=anchor["path"]):
                    self.assertTrue((REPOSITORY_ROOT / anchor["path"]).exists())

    def test_research_is_not_claimed_as_hardware_evidence(self):
        self.assertEqual(self.pool["summary"]["hardware_validated"], 0)
        self.assertTrue(all(not row["hardware_validated"] for row in self.pool["candidates"]))
        self.assertNotIn("hardware_validated", {row["validation_status"] for row in self.pool["sources"]})

    def test_b_hypotheses_respect_the_two_category_floor(self):
        for candidate in self.pool["candidates"]:
            b_tier = candidate["b_tier"]
            with self.subTest(candidate=candidate["candidate_id"]):
                if b_tier["status"] == "not_suitable":
                    self.assertEqual(b_tier["gap_hypotheses"], [])
                else:
                    self.assertGreaterEqual(len(set(b_tier["gap_hypotheses"])), 2)

    def test_generated_index_is_current(self):
        expected = self.tool.render(self.pool).replace("\r\n", "\n")
        actual = INDEX_PATH.read_text(encoding="utf-8").replace("\r\n", "\n")
        self.assertEqual(actual, expected)

    def test_attention_pool_is_valid_and_has_the_planned_scope(self):
        self.assertEqual(self.tool.pool_errors(self.attention_pool), [])
        self.assertEqual(
            self.attention_pool["summary"],
            {
                "source_count": 9,
                "family_count": 6,
                "candidate_count": 16,
                "shortlisted": 11,
                "needs_probe": 4,
                "deferred": 1,
                "hardware_validated": 0,
            },
        )
        self.assertEqual(
            {family["family_id"] for family in self.attention_pool["families"]},
            {
                "attention_dense_training",
                "attention_kv_cache",
                "attention_mla",
                "attention_sparse",
                "attention_linear_state",
                "attention_quantized",
            },
        )

    def test_attention_pool_does_not_duplicate_the_existing_dense_forward_set(self):
        candidate_ids = {row["candidate_id"] for row in self.attention_pool["candidates"]}
        self.assertNotIn("sdpa_forward", candidate_ids)
        excluded = {row["scope"] for row in self.attention_pool["exclusions"]}
        self.assertIn("existing dense forward Attention entries", excluded)

    def test_attention_generated_index_is_current(self):
        expected = self.tool.render(self.attention_pool).replace("\r\n", "\n")
        actual = ATTENTION_INDEX_PATH.read_text(encoding="utf-8").replace("\r\n", "\n")
        self.assertEqual(actual, expected)

    def test_pool_candidate_ids_do_not_collide(self):
        general = {row["candidate_id"] for row in self.pool["candidates"]}
        attention = {row["candidate_id"] for row in self.attention_pool["candidates"]}
        self.assertEqual(general.intersection(attention), set())

    def test_unknown_source_is_rejected(self):
        pool = copy.deepcopy(self.pool)
        pool["candidates"][0]["source_refs"].append("unverified_blog")
        errors = self.tool.pool_errors(pool)
        self.assertTrue(any("unknown" in error and "unverified_blog" in error for error in errors), errors)

    def test_stale_summary_is_rejected(self):
        pool = copy.deepcopy(self.pool)
        pool["summary"]["candidate_count"] += 1
        errors = self.tool.pool_errors(pool)
        self.assertTrue(any("summary.candidate_count" in error for error in errors), errors)

    def test_lead_only_record_cannot_be_shortlisted(self):
        pool = copy.deepcopy(self.pool)
        pool["candidates"][0]["decision"] = "shortlisted"
        pool["candidates"][0]["evidence_level"] = "lead_only"
        pool["summary"]["shortlisted"] += 1
        pool["summary"]["needs_probe"] -= 1
        errors = self.tool.pool_errors(pool)
        self.assertTrue(any("lead-only" in error for error in errors), errors)

    def test_candidate_pool_is_not_part_of_agent_task_packages(self):
        pools = (self.pool, self.attention_pool)
        candidate_ids = {
            candidate["candidate_id"]
            for pool in pools
            for candidate in pool["candidates"]
        }
        source_revisions = {
            source["revision"]
            for pool in pools
            for source in pool["sources"]
            if source["authority"] != "local_repository"
        }
        forbidden = candidate_ids | source_revisions | {pool["pool_id"] for pool in pools}
        for path in (ROOT / "tasks").rglob("*"):
            if not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            with self.subTest(path=path.relative_to(ROOT)):
                leaked = sorted(token for token in forbidden if token in text)
                self.assertEqual(leaked, [], f"candidate provenance leaked into {path}")


if __name__ == "__main__":
    unittest.main()
