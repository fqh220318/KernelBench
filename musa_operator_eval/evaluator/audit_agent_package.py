"""§2's publication audit: what the agent is handed holds task definitions only.

§2 splits the evaluation set in two. The task package is handed over; the raw
source snapshot, the hidden cases, the hidden golden tensors, the official baselines and
the expert dispatch are not. §3 puts the package in `tasks/<id>/` and everything
withheld outside it.

That separation is a claim about a directory, so it is checked against the
directory rather than asserted in prose. Four things can go wrong and each of them
is silent on its own:

  * a maintainer-only file is present, which publishes a hidden case list, a
    baseline or the expert dispatch the task exists to measure;
  * a package file names a path under `private/`, which tells a reader where the
    answers are even though it does not contain them;
  * the prompt, the starter or the case list names the upstream project, which
    §2 keeps out of the agent's hands and out of its prompt;
  * a normative release still carries the legacy provenance object or lacks the
    opaque source id and independent-reference declaration required by NORM-004/006;
  * a file in the tree is in neither the freeze list nor the editable list, so the
    inventory §4.6 asks for does not describe the package it claims to describe.

    python musa_operator_eval/evaluator/audit_agent_package.py
    python musa_operator_eval/evaluator/audit_agent_package.py musa_operator_eval/tasks/kb_l3_43_b
    python musa_operator_eval/evaluator/audit_agent_package.py --require-public-data <release-package>
"""

from __future__ import annotations

import argparse
import functools
import json
import re
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]

# Files that exist only on the maintainer side. A package that has one has
# published it, whatever the ignore rules elsewhere say.
MAINTAINER_ONLY_NAMES = {
    "admission.json",
    "baseline.json",
    "baseline.hidden.json",
    "cases.private.json",
    "private_gap_evidence.json",
    "route_probe.json",
    "unavailable.json",
    "viability.md",
    "environment.snapshot.json",
}

MAINTAINER_ONLY_DIRS = {"expert", "naive", "blind"}

# A package file that mentions the maintainer tree tells a reader where the
# answers live, even if it does not contain them.
PRIVATE_REFERENCE = re.compile(r"musa_operator_eval/private/|\.private\.json|/private/")

# Legacy development audits retain the old task.json provenance exception so the
# migration inventory remains inspectable. Normative release audits do not: names are
# matched case-insensitively so a Python import cannot leak what a heading avoids.
LEGACY_UPSTREAM_NAMES = re.compile(r"KernelBench|ScalingIntelligence")
NORMATIVE_UPSTREAM_NAMES = re.compile(r"kernelbench|scalingintelligence", re.IGNORECASE)
UPSTREAM_ENTRY_ID = re.compile(r"\bkb_l[0-9]+_[0-9]+\b", re.IGNORECASE)
OPAQUE_SOURCE_ID = re.compile(r"^src_[0-9a-f]{16}$")
NORMATIVE_STATUSES = {"hardware_pending", "release_ready"}

# The files in a package whose text the agent reads as instructions or as the
# interface, and which therefore must not name the upstream project.
AGENT_FACING_SUFFIXES = (".md",)

TEXT_SUFFIXES = (".md", ".py", ".json", ".txt", ".toml")


def _text_files(package_dir: Path) -> List[Path]:
    return sorted(
        path for path in package_dir.rglob("*")
        if path.is_file() and path.suffix in TEXT_SUFFIXES and "__pycache__" not in path.parts
    )


def _public_case_ids(package_dir: Path) -> set:
    manifest = json.loads((package_dir / "public_cases.json").read_text(encoding="utf-8"))
    return {case["case_id"] for case in manifest["cases"]}


def _is_public_golden(relative: Path, public_case_ids: set) -> bool:
    parts = relative.parts
    return len(parts) >= 4 and parts[0] == "generated" and parts[1] in public_case_ids and parts[2] == "golden"


@functools.lru_cache(maxsize=1)
def _upstream_problem_sources() -> List[tuple]:
    """Known raw problem bodies available to the maintainer-side release audit."""
    repository = ROOT.parent
    rows = []
    for level in ("level1", "level2", "level3"):
        directory = repository / "KernelBench" / level
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.py")):
            body = path.read_text(encoding="utf-8").strip()
            if len(body) >= 500:
                rows.append((path.relative_to(repository).as_posix(), body))
    return rows


def _copied_upstream_problem(package_dir: Path) -> str | None:
    problem = package_dir / "problem.py"
    if not problem.is_file():
        return None
    body = problem.read_text(encoding="utf-8")
    for relative, source in _upstream_problem_sources():
        if source in body:
            return relative
    return None


def audit_package(
    package_dir: Path, require_public_data: bool = False, normative: bool = False
) -> Dict[str, object]:
    """Check one task package against §2 and §4.6.

    Returns:
        {"package": str, "passed": bool, "findings": [...]}
    """
    package_dir = Path(package_dir)
    findings: List[str] = []
    task_path = package_dir / "task.json"
    task = json.loads(task_path.read_text(encoding="utf-8"))
    starter = task.get("starter") or {}
    public_case_ids = _public_case_ids(package_dir)

    if normative:
        if "source" in task:
            findings.append("task.json: legacy provenance object is forbidden in a normative release")
        if not OPAQUE_SOURCE_ID.fullmatch(str(task.get("source_id", ""))):
            findings.append("task.json: normative release requires an opaque source_id")
        if task.get("normative_spec") != "1.0.0":
            findings.append("task.json: normative_spec must be 1.0.0")
        publication = task.get("publication") or {}
        if publication.get("release_status") not in NORMATIVE_STATUSES:
            findings.append(
                "task.json: publication.release_status must be hardware_pending or release_ready"
            )
        if publication.get("reference_origin") != "independent_cpu_reference":
            findings.append("task.json: normative release requires an independent CPU reference")
        oracle = task.get("golden_oracle") or {}
        oracle_path = oracle.get("path")
        if not oracle_path or not (package_dir / oracle_path).is_file():
            findings.append(
                "task.json: normative release must name a shipped golden_oracle file"
            )
        if (task.get("lifecycle") or {}).get("decision") != "keep":
            findings.append("task.json: a normative package must have lifecycle.decision=keep")
        if (
            task.get("tier") == "B_library"
            and (task.get("library_policy") or {}).get("formal_call_surface") != "torch_musa"
        ):
            findings.append("task.json: normative B-tier execution must use the torch_musa call surface")
        if starter.get("cpp"):
            findings.append("task.json: compiled starter is research-only and cannot enter a normative release")
        if any(path.is_file() for path in (package_dir / "starter" / "cpp").glob("**/*")):
            findings.append("starter/cpp: compiled artifacts are research-only and cannot enter a normative package")
        copied_from = _copied_upstream_problem(package_dir)
        if copied_from:
            findings.append(f"problem.py: contains a verbatim upstream problem body ({copied_from})")

    for path in sorted(package_dir.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        relative_path = path.relative_to(package_dir)
        relative = relative_path.as_posix()
        if path.name in MAINTAINER_ONLY_NAMES:
            findings.append(f"{relative}: a maintainer-only file is inside the agent-visible package")
        if set(relative_path.parts[:-1]) & MAINTAINER_ONLY_DIRS:
            findings.append(f"{relative}: it lives in a maintainer-only directory")
        if "golden" in relative_path.parts[:-1] and not _is_public_golden(relative_path, public_case_ids):
            findings.append(f"{relative}: a golden outside a declared public case is maintainer-only")
        if relative_path.parts and relative_path.parts[0] == "generated":
            if len(relative_path.parts) < 2 or relative_path.parts[1] not in public_case_ids:
                findings.append(f"{relative}: generated data belongs to no declared public case")

    if require_public_data:
        for case_id in sorted(public_case_ids):
            case_root = package_dir / "generated" / case_id
            for direction in ("input", "golden"):
                manifest = case_root / direction / "tensors.json"
                if not manifest.is_file():
                    findings.append(
                        f"generated/{case_id}/{direction}/tensors.json: required public self-test data is missing"
                    )

    for path in _text_files(package_dir):
        relative = path.relative_to(package_dir).as_posix()
        text = path.read_text(encoding="utf-8")
        upstream_names = NORMATIVE_UPSTREAM_NAMES if normative else LEGACY_UPSTREAM_NAMES
        if PRIVATE_REFERENCE.search(text):
            findings.append(f"{relative}: names a path under the maintainer tree")
        if upstream_names.search(text) and (normative or path.name != "task.json"):
            findings.append(f"{relative}: names the upstream project")
        if normative and UPSTREAM_ENTRY_ID.search(text):
            findings.append(f"{relative}: exposes an upstream entry identifier")
        if path.suffix in AGENT_FACING_SUFFIXES and upstream_names.search(text):
            findings.append(f"{relative}: names the upstream project in a file the agent reads as instructions")

    # §4.6's inventory has to describe the package it is attached to: a file in
    # neither list is a file nobody declared, and the freeze list is what the
    # audit draws its boundary from.
    editable = {entry["path"] for entry in starter.get("editable", [])} | set(
        (starter.get("cpp") or {}).get("editable", [])
    )
    frozen = set(starter.get("frozen", []))
    for path in sorted(package_dir.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        relative = path.relative_to(package_dir).as_posix()
        if relative in frozen or relative in editable or relative.startswith("generated/"):
            continue
        findings.append(f"{relative}: neither editable nor frozen, so §4.6's inventory does not cover it")

    return {"package": str(package_dir), "passed": not findings, "findings": findings}


def audit_packages(
    packages: List[Path], require_public_data: bool = False, normative: bool = False
) -> Dict[str, object]:
    """Audit several packages and summarise."""
    results = [
        audit_package(package, require_public_data=require_public_data, normative=normative)
        for package in packages
    ]
    findings = [f"{result['package']}: {finding}" for result in results for finding in result["findings"]]
    return {
        "packages": len(results),
        "passed": not findings,
        "findings": findings,
        "results": results,
    }


def all_packages() -> List[Path]:
    return sorted(path for path in (ROOT / "tasks").iterdir() if path.is_dir())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("packages", nargs="*", type=Path, help="packages to audit; defaults to every one in tasks/")
    parser.add_argument("--json", action="store_true", help="emit the report instead of the summary line")
    parser.add_argument(
        "--require-public-data", action="store_true",
        help="require generated input and golden manifests for every public smoke case",
    )
    parser.add_argument(
        "--normative", action="store_true",
        help=(
            "apply Phase 21 public-package rules; hardware_pending packages may pass, "
            "but only release_ready packages can be released"
        ),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    packages = args.packages or all_packages()
    report = audit_packages(
        packages, require_public_data=args.require_public_data, normative=args.normative
    )
    if args.json:
        print(json.dumps(report, indent=2))
    elif report["passed"]:
        print(f"agent package disclosure audit passed ({report['packages']} packages)")
    else:
        print("agent package disclosure audit failed:", file=sys.stderr)
        for finding in report["findings"]:
            print(f"  {finding}", file=sys.stderr)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
