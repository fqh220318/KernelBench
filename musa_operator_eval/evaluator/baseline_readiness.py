"""Assess whether a baseline documents the mandatory §4.4 roster.

Roster completeness and scoreability are deliberately different questions.
Guide §4.4 asks every mandatory role to be recorded with a value *and a status*;
some comparison roles can legitimately be unsupported on the very gap cases a
B-tier task is required to contain.  A documented non-passing row therefore
completes the roster, but never becomes a latency: the candidate gate,
admission recorder and ranker separately require passing rows for the roles
they divide or gate on.

Missing and duplicated case/role rows remain readiness failures.  This keeps a
maintainer from deleting an inconvenient result while avoiding the impossible
rule that a fused library must pass a case selected because that library does
not cover it.
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
    """Derive evidence, scoring and release states from the recorded rows."""
    roles = required_roles(baseline.get("tier"))
    declared = set(baseline.get("implementations") or [])
    results = baseline.get("results") or []
    case_ids = sorted(set(expected_case_ids or (row.get("case_id") for row in results if row.get("case_id"))))
    evidence_problems: list[str] = []

    if not roles:
        evidence_problems.append(f"unknown baseline tier {baseline.get('tier')!r}")
    if not case_ids:
        evidence_problems.append("the baseline contains no measured case")

    missing_roles = sorted(set(roles) - declared)
    if missing_roles:
        evidence_problems.append(f"mandatory implementations are not declared: {missing_roles}")

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
        evidence_problems.append(f"mandatory case/implementation rows are missing: {missing_rows}")
    if duplicate_rows:
        evidence_problems.append(f"case/implementation rows are duplicated: {duplicate_rows}")

    evidence_complete = not evidence_problems
    scoreability_problems: list[str] = []
    denominator_by_case = (baseline.get("scoring") or {}).get("denominator_by_case") or {}
    legacy_denominator = (baseline.get("scoring") or {}).get("speedup_denominator")
    for case_id in case_ids:
        selection = denominator_by_case.get(case_id)
        if isinstance(selection, dict):
            denominator = selection.get("implementation")
            reason = selection.get("reason")
        else:
            denominator = legacy_denominator
            reason = None
        allowed = (
            {"upstream_musa", "mudnn_fused"}
            if baseline.get("tier") == "A_kernel"
            else {"expert_dispatch"}
        )
        if denominator not in allowed:
            scoreability_problems.append(
                f"{case_id}: denominator {denominator!r} is not allowed for {baseline.get('tier')}"
            )
            continue
        if not isinstance(selection, dict) or not isinstance(reason, str) or not reason.strip():
            scoreability_problems.append(
                f"{case_id}: denominator selection and reason are not frozen per scoring case"
            )
        if (
            baseline.get("tier") == "A_kernel"
            and denominator == "mudnn_fused"
            and (selection or {}).get("semantic_equivalence") != "exact"
        ):
            scoreability_problems.append(
                f"{case_id}: mudnn_fused fallback requires semantic_equivalence='exact'"
            )
        pair = by_pair.get((case_id, denominator), [])
        if len(pair) != 1:
            scoreability_problems.append(
                f"{case_id}: expected one {denominator} denominator row, found {len(pair)}"
            )
            continue
        row = pair[0]
        latency = row.get("latency_ms")
        if row.get("status") != "pass" or not isinstance(latency, (int, float)) or latency <= 0:
            scoreability_problems.append(
                f"{case_id}: {denominator} has no passing positive latency"
            )
        if baseline.get("tier") == "B_library" and row.get("dispatch_trace_passed") is not True:
            scoreability_problems.append(
                f"{case_id}: expert_dispatch has no passing dispatch trace"
            )

    scoreable = evidence_complete and not scoreability_problems
    release_blockers: list[str] = []
    if not scoreable:
        release_blockers.append("baseline is not scoreable")
    task_release_status = baseline.get("task_release_status")
    if task_release_status != "release_ready":
        release_blockers.append(
            f"task release status is {task_release_status!r}, not 'release_ready'"
        )
    release_ready = scoreable and not release_blockers
    return {
        "evidence_complete": evidence_complete,
        "scoreable": scoreable,
        "release_ready": release_ready,
        "required_roles": list(roles),
        "case_count": len(case_ids),
        "missing_roles": missing_roles,
        "missing_rows": missing_rows,
        "duplicate_rows": duplicate_rows,
        "unavailable_rows": unavailable_rows,
        "nonpassing_rows": nonpassing_rows,
        "evidence_problems": evidence_problems,
        "scoreability_problems": scoreability_problems,
        "release_blockers": release_blockers,
    }


def declared_readiness_errors(baseline: dict, expected_case_ids=None) -> list[str]:
    """Compare the stored summary with a fresh derivation."""
    derived = assess(baseline, expected_case_ids)
    declared = baseline.get("readiness")
    if not isinstance(declared, dict):
        return (
            ["the baseline has no readiness record"]
            + derived["evidence_problems"]
            + derived["scoreability_problems"]
            + derived["release_blockers"]
        )
    errors = []
    for field in (
        "evidence_complete", "scoreable", "release_ready", "required_roles", "case_count",
        "missing_roles", "missing_rows", "duplicate_rows", "unavailable_rows",
        "nonpassing_rows", "evidence_problems", "scoreability_problems", "release_blockers",
    ):
        if declared.get(field) != derived[field]:
            errors.append(
                f"readiness.{field} is {declared.get(field)!r}, derived from results it is {derived[field]!r}"
            )
    return errors

