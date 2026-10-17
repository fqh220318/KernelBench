"""Build one complete agent-visible task package.

The development tree deliberately does not commit generated tensor bytes.  A
release package is different: guide §4.3 promises the agent 3–5 public smoke
cases *with* their inputs and goldens.  This command copies the declared task,
generates those public artifacts into the copy, and runs the disclosure audit
in its strict release mode before publishing the directory.

No hidden manifest is accepted and no file under ``private/`` is read.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


generator = _load("musa_generate_public_cases", ROOT / "tools" / "generate_cases.py")
audit = _load("musa_audit_agent_package", ROOT / "evaluator" / "audit_agent_package.py")


def deployment_boundary(package_root: Path) -> dict:
    """Prove that the hand-off contains no production-control-plane files."""
    package_root = Path(package_root)
    forbidden = []
    for path in package_root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(package_root)
        if any(part in {"evaluator", "private"} for part in relative.parts):
            forbidden.append(relative.as_posix())
    return {
        "passed": not forbidden,
        "package_scope": "single_public_task",
        "production_evaluator": "private_mount_required",
        "public_harness_role": "development_mirror_only",
        "forbidden_files": sorted(forbidden),
    }


def package_task(task_dir: Path, output_dir: Path) -> dict:
    task_dir = Path(task_dir).resolve()
    output_dir = Path(output_dir).resolve()
    public_path = task_dir / "public_cases.json"
    if not (task_dir / "task.json").is_file() or not public_path.is_file():
        raise ValueError(f"{task_dir} is not a task package")
    task = json.loads((task_dir / "task.json").read_text(encoding="utf-8"))
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing release directory: {output_dir}")
    lifecycle = task.get("lifecycle") or {}
    if lifecycle.get("decision") not in (None, "keep"):
        raise ValueError(
            f"the task lifecycle is {lifecycle['decision']}; only keep tasks can be packaged"
        )
    if task.get("tier") == "B_library":
        admission = task.get("admission") or {}
        reasons = (task.get("library_policy") or {}).get("required_gap_reasons") or []
        if admission.get("status") != "measured" or admission.get("decision") != "admit" or len(reasons) < 2:
            raise ValueError(
                "the B-tier task is not scoreable: release requires measured/admit status "
                "and at least two required gap-reason categories"
            )
    publication = task.get("publication") or {}
    if task.get("normative_spec") != "1.0.0" or publication.get("release_status") != "release_ready":
        raise ValueError(
            "the task is not release-ready under normative spec 1.0.0; "
            "complete the Phase 22-24 migration before packaging"
        )
    public = json.loads(public_path.read_text(encoding="utf-8"))
    if public.get("visibility") != "public":
        raise ValueError("the release packager accepts only a public case manifest")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output_dir.name}-", dir=output_dir.parent) as temporary:
        staging = Path(temporary) / output_dir.name
        shutil.copytree(
            task_dir,
            staging,
            ignore=shutil.ignore_patterns("generated", "__pycache__", "*.pyc"),
        )
        generated = staging / "generated"
        generator.generate_manifest(public, generated, task_dir)

        report = audit.audit_package(staging, require_public_data=True, normative=True)
        if not report["passed"]:
            raise ValueError("release disclosure audit failed:\n  " + "\n  ".join(report["findings"]))
        boundary = deployment_boundary(staging)
        if not boundary["passed"]:
            raise ValueError(
                "release deployment boundary failed:\n  "
                + "\n  ".join(boundary["forbidden_files"])
            )

        shutil.move(str(staging), str(output_dir))

    return {
        "task_id": task["id"],
        "output": str(output_dir),
        "public_cases": len(public["cases"]),
        "audit_passed": True,
        "deployment_boundary": boundary,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        result = package_task(args.task_dir, args.output_dir)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"package_agent_task: {error}")
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

