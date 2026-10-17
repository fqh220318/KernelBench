import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_inventory():
    path = ROOT / "evaluator" / "collect_baseline_inventory.py"
    spec = importlib.util.spec_from_file_location("collect_baseline_inventory_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class BaselineInventoryTests(unittest.TestCase):
    def setUp(self):
        self.tool = load_inventory()

    def make_root(self, baseline=None, manifest=None):
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name) / "musa_operator_eval"
        task_dir = root / "tasks" / "example_package"
        task_dir.mkdir(parents=True)
        task = {
            "id": "example_b_v0",
            "tier": "B_library",
            "admission": {"status": "measured", "decision": "admit"},
            "publication": {"release_status": "hardware_pending"},
        }
        (task_dir / "task.json").write_text(json.dumps(task), encoding="utf-8")
        private = root / "private" / task["id"]
        private.mkdir(parents=True)
        if manifest is not None:
            (private / "cases.private.json").write_text(json.dumps(manifest), encoding="utf-8")
        if baseline is not None:
            (private / "baseline.hidden.json").write_text(json.dumps(baseline), encoding="utf-8")
        self.addCleanup(temporary.cleanup)
        return root

    def complete_baseline(self):
        roles = list(self.tool.required_roles("B_library"))
        baseline = {
            "task_id": "example_b_v0",
            "tier": "B_library",
            "task_release_status": "hardware_pending",
            "implementations": roles,
            "results": [
                {
                    "case_id": "hidden_case_secret", "implementation": role,
                    "status": "pass", "latency_ms": 1.0,
                    **({"dispatch_trace_passed": True} if role == "expert_dispatch" else {}),
                }
                for role in roles
            ],
            "scoring": {
                "speedup_denominator": "expert_dispatch",
                "denominator_by_case": {
                    "hidden_case_secret": {
                        "implementation": "expert_dispatch",
                        "reason": "frozen inventory-test denominator",
                    }
                },
            },
        }
        baseline["readiness"] = self.tool.assess(baseline, ["hidden_case_secret"])
        return baseline

    def test_absent_baseline_becomes_a_collection_item(self):
        root = self.make_root()
        plan = self.tool.build_plan(root)
        item = plan["tasks"][0]
        self.assertEqual(item["collection_status"], "not_collected")
        self.assertEqual(item["collection_priority"], "release_blocker")
        self.assertEqual(item["missing_roles"], list(self.tool.required_roles("B_library")))

    def test_complete_record_is_recognised_without_disclosing_hidden_case_ids(self):
        baseline = self.complete_baseline()
        manifest = {"cases": [{"case_id": "hidden_case_secret"}]}
        root = self.make_root(baseline, manifest)
        plan = self.tool.build_plan(root)
        self.assertEqual(plan["tasks"][0]["collection_status"], "complete")
        self.assertTrue(plan["tasks"][0]["evidence_complete"])
        self.assertTrue(plan["tasks"][0]["scoreable"])
        self.assertFalse(plan["tasks"][0]["release_ready"])
        self.assertNotIn("hidden_case_secret", json.dumps(plan))

    def test_unavailable_role_is_aggregate_only_and_collection_complete(self):
        baseline = self.complete_baseline()
        baseline["results"][0].update(status="unavailable", reason="private measured reason")
        baseline["readiness"] = self.tool.assess(baseline, ["hidden_case_secret"])
        root = self.make_root(baseline, {"cases": [{"case_id": "hidden_case_secret"}]})
        item = self.tool.build_plan(root)["tasks"][0]
        self.assertEqual(item["collection_status"], "complete")
        self.assertIn(baseline["results"][0]["implementation"], item["unavailable_roles"])
        rendered = json.dumps(item)
        self.assertNotIn("hidden_case_secret", rendered)
        self.assertNotIn("private measured reason", rendered)

    def test_command_template_supplies_every_non_automatic_b_role(self):
        root = self.make_root()
        command = self.tool.build_plan(root)["tasks"][0]["command_template"]
        rendered = " ".join(command)
        self.assertIn("--naive", command)
        self.assertIn("--expert", command)
        for role in ("upstream_musa", "mudnn_fused", "torch_musa_sdpa"):
            self.assertIn(f"{role}=<path-to-{role}-ModelNew-module>", command)
        self.assertNotIn("torch_musa_eager=<", rendered)

    def test_command_template_resolves_the_canonical_upstream_answer(self):
        root = self.make_root()
        task_path = next((root / "tasks").glob("*/task.json"))
        task = json.loads(task_path.read_text(encoding="utf-8"))
        task["set_entry"] = "kb_l3_43"
        task_path.write_text(json.dumps(task), encoding="utf-8")
        answer = root.parent / "KernelBench" / "level3_musa" / "problem_43" / "model_new.py"
        answer.parent.mkdir(parents=True)
        answer.write_text("class ModelNew: pass\n", encoding="utf-8")

        item = self.tool.build_plan(root)["tasks"][0]
        self.assertEqual(
            item["resolved_public_implementations"]["upstream_musa"],
            "KernelBench/level3_musa/problem_43/model_new.py",
        )
        self.assertIn(
            "upstream_musa=KernelBench/level3_musa/problem_43/model_new.py",
            item["command_template"],
        )

    def test_a_gap_blocked_b_contract_is_not_scheduled_for_hardware_collection(self):
        root = self.make_root()
        task_path = next((root / "tasks").glob("*/task.json"))
        task = json.loads(task_path.read_text(encoding="utf-8"))
        task["admission"] = {
            "status": "blocked_gap_coverage",
            "gap_reason_coverage": {"state": "blocked"},
        }
        task["lifecycle"] = {
            "decision": "research_only",
            "reason": "the current contract cannot satisfy the admission floor",
        }
        task_path.write_text(json.dumps(task), encoding="utf-8")

        item = self.tool.build_plan(root)["tasks"][0]
        self.assertEqual(item["collection_priority"], "inactive_current_contract")
        self.assertEqual(item["collection_status"], "not_scheduled")
        self.assertEqual(item["lifecycle_decision"], "research_only")
        self.assertEqual(
            item["collection_action"],
            "retain_research_evidence_no_hardware_collection",
        )
        self.assertEqual(item["issue_codes"], ["inactive_current_contract"])

    def test_a_research_only_a_contract_is_not_a_release_blocker(self):
        root = self.make_root()
        task_path = next((root / "tasks").glob("*/task.json"))
        task = json.loads(task_path.read_text(encoding="utf-8"))
        task["tier"] = "A_kernel"
        task["lifecycle"] = {
            "decision": "research_only",
            "reason": "the semantic domain has a single-call equivalent library path",
        }
        task_path.write_text(json.dumps(task), encoding="utf-8")

        item = self.tool.build_plan(root)["tasks"][0]
        self.assertEqual(item["collection_priority"], "inactive_current_contract")
        self.assertEqual(item["collection_status"], "not_scheduled")
        self.assertEqual(item["collection_action"], "retain_research_evidence_no_hardware_collection")


if __name__ == "__main__":
    unittest.main()
