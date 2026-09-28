"""Release packaging tests that do not require torch or a MUSA device."""

import importlib.util
import json
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

    def test_package_contains_all_public_self_test_manifests(self):
        output = self.root / "release"
        result = self.packager.package_task(self.task, output)
        public = json.loads((self.task / "public_cases.json").read_text(encoding="utf-8"))
        self.assertEqual(result["public_cases"], len(public["cases"]))
        self.assertTrue(result["audit_passed"])
        for case in public["cases"]:
            self.assertTrue((output / "generated" / case["case_id"] / "input" / "tensors.json").is_file())
            self.assertTrue((output / "generated" / case["case_id"] / "golden" / "tensors.json").is_file())

    def test_packager_refuses_to_overwrite_a_release(self):
        output = self.root / "release"
        output.mkdir()
        with self.assertRaises(FileExistsError):
            self.packager.package_task(self.task, output)

    def test_packager_refuses_a_b_task_blocked_on_gap_coverage(self):
        with self.assertRaisesRegex(ValueError, "not scoreable"):
            self.packager.package_task(ROOT / "tasks" / "kb_l3_43_b", self.root / "blocked")


if __name__ == "__main__":
    unittest.main(verbosity=2)

