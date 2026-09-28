# Phase 08 — integration baseline and source provenance

## Outcome

The compliance work now runs in the isolated local branch
`feat/musa-operator-evaluation-compliance`, based on remote commit
`8d037710795d00fb23b58a8d43d033ea8f44038e`. The older untracked phase 01–07
tree in the original worktree was not overwritten.

Source provenance is now a validation requirement rather than optional prose:

- every task contract carries `revision`, `license`, `snapshot_location`, and
  `snapshot_revision`;
- in-repository KernelBench problems name the commit that last changed the
  reference and the immutable evaluation snapshot that contains it;
- the external MUSA attention archive is pinned to full commit
  `f249b2fcbddc9d9fb13183437ace87d280a02cca`;
- the archive branch name was updated to the currently reachable remote ref,
  `feat/import-musa-attention-set-useless`;
- the generated Attention set index was regenerated from the corrected ledger.

## Verification

```text
validate_schemas.py --all: 34/34 valid
SourceProvenanceTests: 4/4 passed
collect_attention_set.py --check:
  8 entries, 15 measurements, 11 candidates,
  5 archive paths reachable, 5 deliberately absent everywhere
```

## Remaining hardware-independent work

Public smoke goldens still need a release bundle, the generalization probe is
still manifest-bound, four B-tier tasks still declare only one gap category,
and the compiled evaluator still needs stage-aware short-circuiting.

