"""Release packaging tests that do not require torch or a MUSA device."""

import importlib.util
import json
import re
import shutil
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_packager():
    path = ROOT / "evaluator" / "package_agent_task.py"
    spec = importlib.util.spec_from_file_location("package_agent_task", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class AgentReleasePackageTests(unittest.TestCase):
    def setUp(self):
        self.packager = load_packager()
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.task = ROOT / "tasks" / "sdpa_forward_pilot"

        def fake_generate(manifest, output_root, _task_dir):
            for case in manifest["cases"]:
                for direction in ("input", "golden"):
                    directory = Path(output_root) / case["case_id"] / direction
                    directory.mkdir(parents=True, exist_ok=True)
                    (directory / "tensors.json").write_text(
                        json.dumps({"case_id": case["case_id"], "tensors": []}) + "\n",
                        encoding="utf-8",
                    )
            return [case["case_id"] for case in manifest["cases"]]

        self.packager.generator.generate_manifest = fake_generate

    def tearDown(self):
        self.temporary.cleanup()

    def release_ready_copy(self):
        migrated = self.root / "migrated_task"
        shutil.copytree(self.task, migrated)
        task_path = migrated / "task.json"
        task = json.loads(task_path.read_text(encoding="utf-8"))
        task.pop("source", None)
        task.pop("covers", None)
        task["source_id"] = "src_0123456789abcdef"
        task["normative_spec"] = "1.0.0"
        task["publication"] = {
            "release_status": "release_ready",
            "reference_origin": "independent_cpu_reference",
        }
        task.get("starter", {}).pop("cpp", None)
        task["starter"]["frozen"] = [
            path for path in task["starter"]["frozen"] if not path.startswith("starter/cpp/")
        ]
        shutil.rmtree(migrated / "starter" / "cpp")
        task_path.write_text(json.dumps(task, indent=2) + "\n", encoding="utf-8")
        for path in migrated.rglob("*"):
            if path.is_file() and path.suffix in {".md", ".py", ".json", ".txt", ".toml"}:
                text = path.read_text(encoding="utf-8")
                text = re.sub(r"kernelbench", "evaluation_framework", text, flags=re.IGNORECASE)
                path.write_text(text, encoding="utf-8")
        return migrated

    def test_current_task_is_refused_during_normative_migration(self):
        with self.assertRaisesRegex(ValueError, "not release-ready"):
            self.packager.package_task(self.task, self.root / "blocked_by_migration")

    def test_release_ready_package_contains_all_public_self_test_manifests(self):
        task_dir = self.release_ready_copy()
        output = self.root / "release"
        result = self.packager.package_task(task_dir, output)
        public = json.loads((task_dir / "public_cases.json").read_text(encoding="utf-8"))
        self.assertEqual(result["public_cases"], len(public["cases"]))
        self.assertTrue(result["audit_passed"])
        self.assertEqual(
            result["deployment_boundary"]["production_evaluator"],
            "private_mount_required",
        )
        self.assertEqual(result["deployment_boundary"]["forbidden_files"], [])
        for case in public["cases"]:
            self.assertTrue((output / "generated" / case["case_id"] / "input" / "tensors.json").is_file())
            self.assertTrue((output / "generated" / case["case_id"] / "golden" / "tensors.json").is_file())

    def test_packager_refuses_to_overwrite_a_release(self):
        output = self.root / "release"
        output.mkdir()
        with self.assertRaises(FileExistsError):
            self.packager.package_task(self.task, output)

    def test_packager_refuses_a_research_only_task(self):
        with self.assertRaisesRegex(ValueError, "lifecycle is research_only"):
            self.packager.package_task(ROOT / "tasks" / "kb_l3_43_b", self.root / "blocked")

    def test_deployment_boundary_rejects_evaluator_or_private_files(self):
        package = self.root / "bad_package"
        (package / "evaluator").mkdir(parents=True)
        (package / "evaluator" / "run_task.py").write_text("# private", encoding="utf-8")
        result = self.packager.deployment_boundary(package)
        self.assertFalse(result["passed"])
        self.assertEqual(result["forbidden_files"], ["evaluator/run_task.py"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

