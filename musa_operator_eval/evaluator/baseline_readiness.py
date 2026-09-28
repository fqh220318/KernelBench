"""Assess whether a baseline contains the mandatory §4.4 implementation roster.

An ``unavailable`` row is useful evidence: it says that an implementation was
looked for and why it could not be measured.  It is not, however, a performance
baseline.  This module keeps those two statements separate so admission and
ranking cannot silently treat a documented absence as a completed measurement.
"""

from __future__ import annotations

from typing import Iterable, Optional


COMMON_REQUIRED_ROLES = (
    "upstream_musa",
    "mudnn_fused",
    "torch_musa_sdpa",
    "torch_musa_eager",
)
B_REQUIRED_ROLES = COMMON_REQUIRED_ROLES + (
    "naive_library_composition",
    "expert_dispatch",
)


def required_roles(tier: str) -> tuple[str, ...]:
    if tier == "A_kernel":
        return COMMON_REQUIRED_ROLES
    if tier == "B_library":
        return B_REQUIRED_ROLES
    return ()


def assess(baseline: dict, expected_case_ids: Optional[Iterable[str]] = None) -> dict:
    """Return a derived readiness record; never trust a stored readiness flag."""
    roles = required_roles(baseline.get("tier"))
    declared = set(baseline.get("implementations") or [])
    results = baseline.get("results") or []
    case_ids = sorted(set(expected_case_ids or (row.get("case_id") for row in results if row.get("case_id"))))
    problems: list[str] = []

    if not roles:
        problems.append(f"unknown baseline tier {baseline.get('tier')!r}")
    if not case_ids:
        problems.append("the baseline contains no measured case")

    missing_roles = sorted(set(roles) - declared)
    if missing_roles:
        problems.append(f"mandatory implementations are not declared: {missing_roles}")

    by_pair: dict[tuple[str, str], list[dict]] = {}
    for row in results:
        by_pair.setdefault((row.get("case_id"), row.get("implementation")), []).append(row)

    missing_rows: list[str] = []
    duplicate_rows: list[str] = []
    unavailable_rows: list[str] = []
    nonpassing_rows: list[str] = []
    for case_id in case_ids:
        for role in roles:
            pair = by_pair.get((case_id, role), [])
            label = f"{case_id}/{role}"
            if not pair:
                missing_rows.append(label)
                continue
            if len(pair) > 1:
                duplicate_rows.append(label)
            if any(row.get("status") == "unavailable" for row in pair):
                unavailable_rows.append(label)
            if any(row.get("status") not in ("pass", "unavailable") for row in pair):
                nonpassing_rows.append(label)

    if missing_rows:
        problems.append(f"mandatory case/implementation rows are missing: {missing_rows}")
    if duplicate_rows:
        problems.append(f"case/implementation rows are duplicated: {duplicate_rows}")
    if unavailable_rows:
        problems.append(
            "mandatory implementations are recorded as unavailable rather than measured: "
            f"{unavailable_rows}"
        )
    if nonpassing_rows:
        problems.append(
            "mandatory implementations did not produce a passing measurement: "
            f"{nonpassing_rows}"
        )

    return {
        "status": "complete" if not problems else "incomplete",
        "required_roles": list(roles),
        "case_count": len(case_ids),
        "missing_roles": missing_roles,
        "missing_rows": missing_rows,
        "duplicate_rows": duplicate_rows,
        "unavailable_rows": unavailable_rows,
        "nonpassing_rows": nonpassing_rows,
        "problems": problems,
    }


def declared_readiness_errors(baseline: dict, expected_case_ids=None) -> list[str]:
    """Compare the stored summary with a fresh derivation."""
    derived = assess(baseline, expected_case_ids)
    declared = baseline.get("readiness")
    if not isinstance(declared, dict):
        return ["the baseline has no readiness record"] + derived["problems"]
    errors = []
    for field in ("status", "required_roles", "case_count", "missing_roles", "missing_rows",
                  "duplicate_rows", "unavailable_rows", "nonpassing_rows", "problems"):
        if declared.get(field) != derived[field]:
            errors.append(
                f"readiness.{field} is {declared.get(field)!r}, derived from results it is {derived[field]!r}"
            )
    return errors

