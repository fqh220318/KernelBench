# Phase 09 — public smoke release package

## Outcome

The development checkout may keep generated tensor bytes out of Git, but an
agent release can no longer omit the public self-test promised by guide §4.3.

`evaluator/package_agent_task.py` now:

1. copies only one declared task package into a temporary release directory;
2. regenerates the public manifest's inputs and goldens from each case seed;
3. requires `generated/<case_id>/input/tensors.json` and
   `generated/<case_id>/golden/tensors.json` for every public case;
4. runs the disclosure audit in strict release mode;
5. publishes the directory only after all checks pass and refuses overwrite.

The disclosure audit now distinguishes public goldens from hidden goldens. It
accepts a golden only below a case id present in `public_cases.json`; unknown
generated case ids, hidden manifests, official baselines, and goldens elsewhere
remain release failures.

## Verification

```text
AgentPackageAuditTests + AgentReleasePackageTests: 12/12 passed
development-tree disclosure audit: 11/11 packages passed
```

The unit tests stub tensor generation so they run without torch. Producing the
real binary release artifacts still requires running the documented packaging
command in the pinned torch/MUSA environment; failure to import that stack is a
collection-environment failure, not a silently incomplete release.

