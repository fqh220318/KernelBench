"""Build a redacted, hardware-independent plan for collecting §4.4 baselines.

The plan is safe to commit: it reports task ids, mandatory public role names and
aggregate readiness states, but never copies hidden case ids, shapes, seeds,
goldens, measurements or failure reasons out of ``private/``.

    python musa_operator_eval/evaluator/collect_baseline_inventory.py
    python musa_operator_eval/evaluator/collect_baseline_inventory.py --check
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVALUATOR_DIR = Path(__file__).resolve().parent
if str(EVALUATOR_DIR) not in sys.path:
    sys.path.insert(0, str(EVALUATOR_DIR))

from baseline_readiness import assess, declared_readiness_errors, required_roles  # noqa: E402


DEFAULT_OUTPUT = ROOT / "collection" / "baseline_collection_plan.json"


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.parent.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return value


def _role_from_pair_label(label: str) -> str:
    return label.rsplit("/", 1)[-1]


def _readiness_codes(derived: dict, drift: list[str]) -> list[str]:
    codes = []
    if derived["case_count"] == 0:
        codes.append("no_cases")
    if derived["missing_roles"]:
        codes.append("missing_required_roles")
    if derived["missing_rows"]:
        codes.append("missing_case_role_rows")
    if derived["duplicate_rows"]:
        codes.append("duplicate_case_role_rows")
    if derived["unavailable_rows"]:
        codes.append("unavailable_required_rows")
    if derived["nonpassing_rows"]:
        codes.append("nonpassing_required_rows")
    if drift:
        codes.append("declared_readiness_drift")
    return codes


def _command_template(task: dict, root: Path) -> list[str]:
    task_id = task["id"]
    task_dir = f"musa_operator_eval/tasks/{task_dir_name(task, root)}"
    private_dir = f"musa_operator_eval/private/{task_id}"
    command = [
        "python",
        "musa_operator_eval/evaluator/measure_baseline.py",
        "--task-dir",
        task_dir,
        "--cases",
        f"{private_dir}/cases.private.json",
        "--tier",
        task["tier"],
    ]
    if task["tier"] == "B_library":
        command.extend(
            [
                "--naive",
                f"{private_dir}/naive/model_new.py",
                "--expert",
                f"{private_dir}/expert/model_new.py",
            ]
        )
        implementation_roles = (
            "upstream_musa",
            "mudnn_fused",
            "torch_musa_sdpa",
        )
    else:
        implementation_roles = (
            "upstream_musa",
            "mudnn_fused",
            "torch_musa_sdpa",
        )
    for role in implementation_roles:
        command.extend(["--implementation", f"{role}=<path-to-{role}-ModelNew-module>"])
    command.extend(["--output", f"{private_dir}/baseline.hidden.json"])
    return command


def task_dir_name(task: dict, root: Path) -> str:
    """Resolve the package directory without exposing anything from private/."""
    for path in sorted((root / "tasks").glob("*/task.json")):
        try:
            if _load_json(path).get("id") == task.get("id"):
                return path.parent.name
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    return task["id"]


def inspect_task(task_path: Path, root: Path) -> dict:
    task = _load_json(task_path)
    task_id = task["id"]
    tier = task["tier"]
    baseline_path = root / "private" / task_id / "baseline.hidden.json"
    manifest_path = root / "private" / task_id / "cases.private.json"
    mandatory = list(required_roles(tier))

    admission = task.get("admission") or {}
    admitted = tier == "A_kernel" or (
        admission.get("status") == "measured" and admission.get("decision") == "admit"
    )
    priority = "release_blocker" if admitted else "after_admission_blocker"

    record = {
        "task_id": task_id,
        "tier": tier,
        "package": _relative(task_path.parent, root),
        "required_roles": mandatory,
        "automatic_roles": ["torch_musa_eager"],
        "collection_priority": priority,
        "private_manifest_present": manifest_path.is_file(),
        "baseline_path": _relative(baseline_path, root),
        "baseline_present": baseline_path.is_file(),
        "command_template": _command_template(task, root),
    }

    if not baseline_path.is_file():
        record.update(
            collection_status="not_collected",
            readiness_status="missing",
            missing_roles=mandatory,
            unavailable_roles=[],
            nonpassing_roles=[],
            issue_codes=["baseline_not_present"],
        )
        return record

    try:
        baseline = _load_json(baseline_path)
        expected_case_ids = None
        if manifest_path.is_file():
            manifest = _load_json(manifest_path)
            expected_case_ids = [case["case_id"] for case in manifest.get("cases", [])]
        derived = assess(baseline, expected_case_ids)
        drift = declared_readiness_errors(baseline, expected_case_ids)
        codes = _readiness_codes(derived, drift)
        if baseline.get("task_id") != task_id:
            codes.append("task_id_mismatch")
        if baseline.get("tier") != tier:
            codes.append("tier_mismatch")
        status = "complete" if derived["status"] == "complete" and not codes else "incomplete"
        missing_pair_roles = {_role_from_pair_label(item) for item in derived["missing_rows"]}
        unavailable_roles = {_role_from_pair_label(item) for item in derived["unavailable_rows"]}
        nonpassing_roles = {_role_from_pair_label(item) for item in derived["nonpassing_rows"]}
        record.update(
            collection_status=status,
            readiness_status=derived["status"],
            readiness_record_matches=not drift,
            missing_roles=sorted(set(derived["missing_roles"]) | missing_pair_roles),
            unavailable_roles=sorted(unavailable_roles),
            nonpassing_roles=sorted(nonpassing_roles),
            issue_codes=sorted(set(codes)),
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        record.update(
            collection_status="invalid_record",
            readiness_status="unknown",
            missing_roles=mandatory,
            unavailable_roles=[],
            nonpassing_roles=[],
            issue_codes=["baseline_record_unreadable"],
        )
    return record


def build_plan(root: Path = ROOT) -> dict:
    records = [inspect_task(path, root) for path in sorted((root / "tasks").glob("*/task.json"))]
    status_counts = Counter(record["collection_status"] for record in records)
    priority_counts = Counter(record["collection_priority"] for record in records)
    return {
        "schema_version": "1.0.0",
        "purpose": "redacted §4.4 baseline collection inventory",
        "redaction": (
            "Contains no hidden case ids, shapes, seeds, goldens, measurements or private failure reasons."
        ),
        "hardware_required_to_collect": True,
        "summary": {
            "task_count": len(records),
            "by_collection_status": dict(sorted(status_counts.items())),
            "by_priority": dict(sorted(priority_counts.items())),
        },
        "tasks": records,
    }


def render(plan: dict) -> str:
    return json.dumps(plan, indent=2, ensure_ascii=False) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true", help="fail when the checked-in plan is stale")
    args = parser.parse_args()

    rendered = render(build_plan())
    if args.check:
        try:
            current = args.output.read_text(encoding="utf-8")
        except OSError as error:
            print(f"[baseline-inventory] cannot read {args.output}: {error}", file=sys.stderr)
            return 1
        if current != rendered:
            print(f"[baseline-inventory] stale: regenerate {args.output}", file=sys.stderr)
            return 1
        print(f"[baseline-inventory] current: {args.output}")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
