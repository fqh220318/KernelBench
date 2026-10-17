# MUSA operator evaluation implementation decisions

This document is the implementation decision layer for the preserved requirement
document at `docs/musa_operator_evaluation_guide.md`. It does not rewrite that source.
Phase 21 resolved the source document's internal conflicts in
`docs/musa_operator_evaluation_normative_spec_v1.md`; that normative specification now
takes precedence over this decision log. `CONFORMANCE.md` records whether the current
implementation has migrated to the normative target.

## 0. Normative authority after Phase 21

The preserved guide remains historical input. The normative specification and its
traceability matrix are the only current implementation target. In particular:

- the scored submission is KernelBench-style Python `ModelNew`;
- a B submission reaches the library through `torch_musa`;
- the native binary/direct-muDNN path is maintainer-side `research_only` evidence;
- public task packages carry an opaque `source_id`, not upstream provenance;
- A/B eligibility is decided for a complete semantic domain, not forced per entry.

Existing code is not made compliant merely by this decision. The matrix records each
migration as incomplete until code, packages and tests enforce it.

## 1. Baseline roster completeness and scoreability are separate

Guide §4.4 requires all four A roles or all six B roles to be represented, and asks
each result to carry both a value and a status. Guide §4.3 also requires B-tier hidden
cases that exercise real library gaps. Requiring the fused library to pass every such
case makes those two requirements impossible to satisfy together.

The implementation therefore derives three separate states:

1. `evidence_complete` means every mandatory role/case observation exists exactly
   once. A non-passing row must include a reason and remains visible in
   `unavailable_rows` or `nonpassing_rows`.
2. `scoreable` requires a frozen, passing, positive denominator for every scoring
   Case. B admission also requires passing naive/expert rows and the matching expert
   dispatch trace for every Case.
3. `release_ready` requires scoreability and the task-level final publication state.
   Ranking rejects the complete task if any Case lacks a usable runtime or denominator;
   it never silently computes a partial geomean.

No non-passing row is converted into a latency, and no lookalike implementation may be
recorded under a canonical role name.

## 2. One timing protocol per ratio

The current v0 implementation uses KernelBench's device-event timing on both sides of
every ratio: warmup, per-trial L2 flush, first-trial discard and the mean of the
remaining samples. Samples and spread statistics are retained.

Normative decision NORM-009 selects this KernelBench mean as the scored statistic and
keeps quantiles as diagnostics. Historical measurements still require re-collection
after all Phase 22-24 contract changes; changing one side of a ratio alone remains
forbidden.

## 3. Compiled and Python paths have different roles

The Python `ModelNew` path is the only normative scored path. The compiled path remains
useful for capability discovery, ABI experiments, link inspection and direct library
diagnostics, but is `research_only` under NORM-002 and may not supply a scored result,
admission decision, ranking row or official denominator.

Phase 22 now enforces this separation for all three retained contracts: the B-tier pilot,
the A-tier ReLU-attention task and the A-tier windowed-classifier task. Any research-only
legacy report that treats a compiled or direct-muDNN result as formal evidence remains
non-normative.

## 4. Evaluator visibility is a deployment boundary

The checked-in `evaluator/` directory is explicitly a public development mirror/mock,
not a production secrecy boundary. Production evaluation must mount evaluator code and
`private/` assets from a private repository or access-controlled artifact store. Agent
handoff packages contain exactly one task and are rejected if evaluator/private assets
appear. Repository code and tests therefore satisfy the software boundary; the actual
private mount remains a production deployment responsibility.

## 5. Blocked B tasks stay blocked without real evidence

The four entry-scoped B tasks with only `unsupported_shape` evidence remain
`blocked_gap_coverage`, now carry `admission.decision=reject` and
`lifecycle.decision=research_only`, and are not scheduled in the hardware collection
queue. A second category may not be invented or inferred from a source comment. Any
future formal redesign must use a new versioned task id; the old contract and historical
evidence remain unchanged as research evidence.

## 6. A-tier lifecycle is decided before reference migration

The six legacy A contracts were reclassified against the complete semantic-domain rule.
Standard SDPA, packed multi-head attention, causal softmax attention and the transformer
block whose declared kernel scope is only causal softmax attention each have a single-call
equivalent library path. They are `research_only` and require a new task id for any future
redesign. Swin cosine-window attention and causal ReLU attention remain `keep` because
their declared cores have no such one-call equivalent.

This decision is not a hardware result. It prevents reference migration work from making
an ineligible contract look closer to release.

## 7. A verified golden is the correctness authority

Digest verification alone does not make an independent oracle authoritative. The Python
evaluation path now loads each generated case's serialized inputs and graded golden
tensors, passes those into the device evaluator and records
`correctness_reference=independent_golden`. `problem.py` still defines construction and
diagnostic timing, but its forward result no longer replaces the oracle during formal
correctness. Diagnostic golden tensors that are not part of `outputs.required` remain
available for oracle checks without becoming submission outputs.

## 8. Parameter initialization is data supplied to an independent oracle

The retained windowed classifier has many seeded framework parameters. Reimplementing a
framework initializer is unrelated to operator correctness and would make the oracle
depend on undocumented random-number details. A golden oracle may therefore request
`model_state` context: `generate_cases.py` constructs the public model under the case
seed and exports its state as NumPy arrays in memory. The oracle must perform every
forward numerical operation itself and may not call the public model. The shipped
windowed-classifier oracle follows this rule with NumPy float64 patch embedding,
attention, MLP, merging, pooling and classification.

