# Private evaluation assets

Everything below this directory is evaluator-only and is ignored by Git in the
public KernelBench fork, which keeps only this README. Do not place hidden cases,
golden tensors, expert dispatch source, or baseline artifacts in an agent-visible
checkout.

The tree itself is versioned in its own private repository
(`Young-Lord/musa-operator-eval-private`) and mounted into the evaluator, because
containment and durability are different problems: ignoring the tree keeps the
answers out of a public history, and a repository of its own keeps them from being
lost with a working copy. `private/generated/` is deliberately not stored there --
the goldens are regenerated on the target machine with `tools/generate_cases.py`.

The public, redacted backlog is `../collection/baseline_collection_plan.json`.
Regenerate it after mounting this tree with
`python musa_operator_eval/evaluator/collect_baseline_inventory.py`; it exposes
only aggregate readiness and public role names, never hidden case ids, shapes,
seeds, measurements or private failure reasons.
