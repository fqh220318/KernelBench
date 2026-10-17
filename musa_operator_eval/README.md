# MUSA Operator Evaluation

> **Phase 21 migration notice:** the current executable target is
> `docs/musa_operator_evaluation_normative_spec_v1.md`; the resolution of every
> conflicting source requirement is in
> `docs/musa_operator_evaluation_requirement_traceability_matrix.md`. The existing
> One B-tier pilot and both retained A-tier tasks are `keep / hardware_pending`;
> four rejected B contracts and four A contracts with a single-call equivalent library
> path are explicitly `research_only`. All three keep contracts have completed static
> migration, but none of the 11 tasks is `release_ready`.
> Phases 22-24 are complete on the software/control-plane side. The three retained
> tasks remain `hardware_pending`; Phase 25 must replay them on the target device
> before any baseline or task becomes `release_ready`.

Task contracts, capability evidence and admission tooling for the mutually exclusive
MUSA A/B task classes. Version 0 is scoped to Attention families; other
families must add their own semantic contract instead of reusing
Attention-specific fields.

The numerical execution and timing primitives come from the KernelBench
framework in `src/kernelbench/` (`eval.py`, `timing.py`,
`kernel_static_checker.py`, `musa_extension.py`). This directory adds the parts
the framework has no notion of and drives them end to end: A/B contracts, hidden
case policy, dispatch evidence, complete baseline rosters, compiled submissions,
stage short-circuiting, admission and separate rankings.

Each formal task is classified exactly once as A or B. Historical paired directories
remain only as `research_only` evidence and cannot be packaged, baselined or ranked.

An A-tier candidate starts from the KernelBench problem inventory, but a normative
package is an independently specified operator contract. It exposes an opaque source
identifier, a public execution model and an independent CPU oracle; the private registry
keeps the reverse source mapping. Legacy research-only A packages retain byte-identical
problem copies solely so historical evidence stays reproducible.

A B-tier task needs something the framework has no notion of, because it is graded
on how the answer was reached rather than only on what it computes: a library
whitelist, a dispatch order, and a trace the submission has to emit. Its package
carries that contract. `sdpa_forward_pilot/` is the family-scoped form of it; its
public contract generalises the attention core while the source-to-entry mapping stays
in the private provenance registry.

Legacy research directories are named `<set entry>_a` and `<set entry>_b`. Normative
packages do not expose `set_entry`; maintainer-side tests recover that relation from the
private registry when it is mounted. `tests/test_task_packages.py` checks set coverage,
the isolation of legacy paired research assets and the frozen lifecycle decisions.


The one framework change that lives with this work: a problem file may declare
`TIER` and `LIBRARY_POLICY`, and `eval.py` reads them to pick the matching
source-audit rule set. See `kernel_static_checker.static_audit_kernel`.

The A tier adds one check to the shipped defaults: `attention_entry_point`, which
rejects a call to a fused attention entry point. That is not because every torch
op is forbidden — it is not, and the answers in `KernelBench/level3_musa/` hold
their projections as `nn.Linear` while computing the attention themselves. It is
because standard softmax-attention contracts admit exactly that shortcut. Those contracts
are now `research_only`; the retained A contracts use cosine attention with learned bias
or causal ReLU attention, neither of which has a one-call equivalent. The upstream
`torch_computation_ops` check that would also catch broad framework compute remains a
warning, while each retained task freezes its narrower kernel boundary explicitly.

Where that check cannot reach, each A-tier contract declares its own
`kernel_scope`: what must be in a kernel, and what may stay in the library. An
`nn.Linear` holding weights and one computing a projection are the same source,
so that half of the boundary is stated for a reader rather than enforced by a
pattern.

## Layout

- `tasks/`: agent-visible task packages. Each carries a KernelBench-format
  `problem.py` (`Model`, `get_inputs`, `get_init_inputs`, plus `TIER` and
  `LIBRARY_POLICY` for the B tier), a `semantics.json` contract, a `PROMPT.md`,
  a `starter/` project, the public case list, and a `tensor_contract` naming the
  dtypes it is graded at. A task names the environment it was measured on rather
  than describing it. `problem.py` declares its own tier, so the driver refuses to
  run it at a precision the contract does not permit.
  The contract's `starter` block is §4.6's inventory: which file the agent copies
  and where its marked regions are, which files are frozen, how the project is
  built and run, and the ABI it is built against. The static audit reads `frozen`
  from there, so the boundary it draws is the one the package declares.
- `environments/`: one agent-visible record per machine configuration, named
  after its `snapshot_id`. A task points here; the device model, architecture and
  component versions live here and nowhere else, so a machine is described once
  and two tasks on it cannot disagree about what it is. Full snapshots, which
  keep the serial number and GPU UUID, live under `private/environments/`.
- `sets/`: curated operator sets and maintainer review indexes. `sets/attention/`
  has a hand-written manifest and generated index for the active attention
  family. `sets/candidates/` contains generated maintainer indexes for the
  cross-family pool and the Attention expansion pool; neither is a released
  task set. The attention set records
  which A-tier answer, which measurement and which candidate kernel already
  exist for every entry. A set is a maintainer ledger — it names the A-tier
  answer path and the candidate implementations, so it is never agent-visible.
- `agent_reference/`: sanitized, agent-visible capability descriptions, one file
  per family, named `<family>_capabilities.json`. A capability boundary belongs
  to the library and the operation rather than to one task, so every task in a
  family names the same file; the name is what stops a second family from
  quietly inheriting a first family's boundaries.
- `sources/`: provenance and license records. `attention-source-001.json` covers
  the upstream snapshots a task derives from;
  `attention-kernel-candidates.json` inventories candidate attention kernel
  implementations and which target each can serve. The maintainer-only
  `operator_candidate_pool.json` pins sources, licenses and triage decisions for
  possible new families. `attention_operator_candidate_pool.json` does the same
  for backward, varlen, paged KV, MLA, sparse, linear and quantized Attention
  expansion. Their rows are research leads, not completed tasks or hardware
  evidence.
- `templates/`: schema-valid incomplete A/B baseline templates and a complete
  hidden gap-evidence skeleton. Templates are validation inputs, not evidence of
  a finished device run.
- `collection/`: redacted collection inventories. These contain public task ids
  and role names only; hidden cases and measurement details stay under `private/`.
- `research/`: CPU-only pre-task contracts and probe plans. The first Attention
  P0 bundle fixes per-sequence causal, packed varlen and paged-KV semantics, but
  every hardware matrix cell remains `not_run`; these files are not released
  tasks, admission evidence or performance results.
- `tools/`: the public flow. `generate_cases.py` turns a §4.3 case entry into
  inputs and a golden, and `case_specialization.py` is the convention that turns a
  case into a runnable problem. The evaluator imports that convention from here
  rather than keeping a second copy; nothing here imports the evaluator.
- `evaluator/`: the evaluation service — the §4.7 pipeline, the link whitelist
  check, the admission gate, the publication audit, the environment collector and
  the schema validator. §2 puts the scoring criteria in the task definition and
  the scripts out of the agent's hands; see `evaluator/README.md` for how this
  checkout stands against that and what the production mount looks like.
- `tests/`: CPU-only unit tests for the tools above.
- `progress/`: one Markdown hand-off for each completed compliance phase.
- `CONFORMANCE.md`: the guide, requirement by requirement, with the shortfalls
  this set carries stated rather than left to be found.
- `IMPLEMENTATION_DECISIONS.md`: the current policy layer for ambiguous or
  conflicting implementation choices; it does not rewrite the preserved guide.
- `../docs/musa_operator_evaluation_normative_spec_v1.md`: the sole current executable
  specification after Phase 21 arbitration.
- `../docs/musa_operator_evaluation_requirement_traceability_matrix.md`: source clause,
  normative decision, migration status and hardware dependency for each conflict.
- `private/`: evaluator-only assets; ignored by Git except for its README.

## Tools

```bash
# Capture the MUSA environment: device identity, driver, every toolkit component
# with its commit, and the host side no container image pins
python musa_operator_eval/evaluator/collect_environment.py

# Probe fused-library / composition coverage per route
python musa_operator_eval/evaluator/probe_sdpa.py

# Deterministically regenerate a task's public inputs and golden outputs from
# case.seed. The golden comes from the package's own `golden_oracle` where it
# declares one and from its reference model in float64 on the CPU otherwise. In
# a full run the serialized input/golden pair is also the correctness authority;
# problem.py is not silently substituted for an independent oracle.
python musa_operator_eval/tools/generate_cases.py \
  --task-dir musa_operator_eval/tasks/kb_l3_43_b

# The same generator produces the hidden set, which is why the evaluator's golden
# cross-check has something to compare against. A hidden manifest may not write
# into a task package: that tree is the agent's.
python musa_operator_eval/tools/generate_cases.py \
  --task-dir musa_operator_eval/tasks/kb_l3_43_b \
  --manifest musa_operator_eval/private/mingpt_causal_attention_b_v0/cases.private.json \
  --output-dir musa_operator_eval/private/generated/kb_l3_43_b

# §4.7's B-tier-only link check, standalone
python musa_operator_eval/evaluator/link_whitelist.py \
  --build-dir runs/kb_l3_43_b/build --task-dir musa_operator_eval/tasks/kb_l3_43_b

# §2's publication audit: what the agent is handed holds task definitions only
python musa_operator_eval/evaluator/audit_agent_package.py

# Build the actual agent hand-off. Unlike the development tree, the release
# contains every public smoke input and golden promised by §4.3.
python musa_operator_eval/evaluator/package_agent_task.py \
  --task-dir musa_operator_eval/tasks/sdpa_forward_pilot \
  --output-dir releases/sdpa_forward_pilot

# Build a private hidden-case manifest from measured gap evidence
python musa_operator_eval/evaluator/make_private_manifest.py \
  --evidence <private-gap-evidence.json> \
  --task-dir musa_operator_eval/tasks/<task-id> \
  --output musa_operator_eval/private/<task-id>/cases.private.json

# Produce/check the redacted baseline backlog without a device or private assets
python musa_operator_eval/evaluator/collect_baseline_inventory.py
python musa_operator_eval/evaluator/collect_baseline_inventory.py --check

# Measure the complete B-tier roster in the gate's own shape (needs the device).
# torch_musa_eager is measured automatically from the task reference; every
# other mandatory role is supplied explicitly.
python musa_operator_eval/evaluator/measure_baseline.py \
  --task-dir musa_operator_eval/tasks/kb_l3_43_b \
  --cases musa_operator_eval/private/<task-id>/cases.private.json \
  --tier B_library \
  --naive  musa_operator_eval/private/<task-id>/naive/model_new.py \
  --expert musa_operator_eval/private/<task-id>/expert/model_new.py \
  --implementation upstream_musa=<model-module.py> \
  --implementation mudnn_fused=<model-module.py> \
  --implementation torch_musa_sdpa=<model-module.py> \
  --output musa_operator_eval/private/<task-id>/baseline.hidden.json

# Apply the B-tier naive/expert admission gate
python musa_operator_eval/evaluator/candidate_gate.py <baseline.json>

# Drive a task across all of its cases
python musa_operator_eval/evaluator/run_task.py --submission <model_new.py> --static-only
python musa_operator_eval/evaluator/run_task.py --submission <model_new.py> \
  --backend musa --precision fp16 --output runs/<task>/report.json
# Grade the hidden set, which lives outside the task package
python musa_operator_eval/evaluator/run_task.py --submission <model_new.py> \
  --cases musa_operator_eval/private/<task-id>/cases.private.json \
  --generalization-seed 20260927  # optional: replay a reported probe

# Research-only compiled diagnostic; never produces a formal score or release row
python musa_operator_eval/evaluator/run_binary.py \
  --task-dir musa_operator_eval/tasks/<task-id> --submission <dir with kernel.mu> \
  --cases musa_operator_eval/private/<task-id>/cases.private.json \
  --generated-dir musa_operator_eval/private/generated/<task-id> \
  --generalization-seed 20260927  # optional: replay a reported probe

# §5 step 8: the two tracks, ranked apart and never combined
python musa_operator_eval/evaluator/rank.py --report runs/a/report.json \
  --report runs/b/report.json --baseline 'musa_operator_eval/private/*/baseline.hidden.json'

# Verify a curated set and regenerate its index
python musa_operator_eval/evaluator/collect_attention_set.py
python musa_operator_eval/evaluator/collect_attention_set.py --check   # verify only, for CI

# Validate both maintainer research pools and regenerate/check their indexes
python musa_operator_eval/evaluator/collect_operator_candidate_pool.py
python musa_operator_eval/evaluator/collect_operator_candidate_pool.py --check

# A single pool can be selected during review
python musa_operator_eval/evaluator/collect_operator_candidate_pool.py --pool attention --check

# Verify the CPU-only contract for the first three Attention P0 probes
python -m unittest musa_operator_eval.tests.test_attention_p0_research -v
```

`run_task.py` is the entry point for a whole task. It specializes `problem.py`
per case using the `case_parameters` mapping in `task.json`, evaluates the
submission against the reference, and checks the dispatch trace. `--static-only`
stops after the source audit and the case-override check, so it runs on a
machine with no device, and `--cases` points it at a manifest outside the task
package, which is how the hidden set is graded without being readable from it.
The private manifest stores a `generalization_probe` sampling policy rather than
a fixed probe case. At run time the evaluator draws a seed from system entropy,
rejects shapes and seeds already present in the public/hidden lists, generates
the input and golden, and records the realized seed and shape only in that run's
report. Passing `--generalization-seed` reproduces a reported probe for debugging.

`measure_baseline.py` measures each mandatory implementation per case under the
framework protocol and writes the shape `candidate_gate.py` reads. It requires
all four A roles or all six B roles to be supplied; `torch_musa_eager` is the task
reference and is measured automatically. An implementation that cannot honestly
be built may still be recorded as `unavailable` with a reason. That row can complete
`evidence_complete` but never supplies a latency. `scoreable` additionally requires
every scoring Case to have its frozen, passing, positive denominator (and the B expert
trace); `release_ready` also requires the task's final publication evidence. The B gate
requires passing naive/expert rows for every case, and the ranker rejects the whole task
if any scoring Case lacks a usable runtime or denominator. It never computes a partial
geomean. `collect_baseline_inventory.py` exposes only aggregate states and
unavailable/non-passing roles, without publishing hidden case identities, reasons or
measurements.

A B task is released and scored only when its contract is `measured/admit` and
declares at least two measured gap-reason categories. The four entry-scoped B
packages currently retain their earlier measurements but are marked
`blocked_gap_coverage`; only the family-scoped pilot currently meets the §4.3
category floor. Those four current contracts are `not_scheduled` for new baseline
collection; a redesign uses a new task id, while the existing packages remain as
auditable evidence. The manifest builder, admission recorder, release packager, both
evaluators, and the ranker all enforce that state.

Every implementation is passed in as a module path rather than named in the
tool's source, because reaching a task's attention core is the task's business:
`kb_l3_43_b` takes one input and five init inputs and computes the core behind its
own projections, while the family-scoped pilot takes `q, k, v` directly. The
tool's own contribution is to reproduce the harness's three measurement facts
rather than paraphrase them — weights are aligned by re-seeding and rebuilding as
`kernelbench.eval` does, inputs and parameters are cast by its
`_process_input_tensor`, and correctness is judged against the generated golden under
the selected profile's independent `atol`, `rtol` and `max_abs_error`. A baseline measured under a different protocol from
the submissions it is the denominator for would admit a task for the wrong reason.

`collect_attention_set.py` reads `sets/attention/attention_set.json`, which is
curated by hand, and verifies every claim it makes against the repository: the
reference and A-tier answer files exist, and every recorded measurement still
matches the file it was read from. It then regenerates `sets/attention/SET.md`.
Exit code is non-zero on drift, so a re-run of a baseline cannot silently leave
a stale number in the index.

Public inputs and goldens are regenerated on demand and are not committed to the
development tree. `package_agent_task.py` generates them into the release copy
and refuses to publish unless every public case has both manifests and the
strict disclosure audit passes.
All tools are standard-library-plus-NumPy and are hardware-independent; the
probes and baselines are not, and must run on the target device.

## Tests

```bash
# Case generation, environment and capability tooling (stdlib + NumPy)
python musa_operator_eval/tests/test_environment_tools.py -v
python musa_operator_eval/tests/test_reference_and_cases.py -v

# Task driver: case specialization, static report, CLI (stdlib only)
python musa_operator_eval/tests/test_run_task.py -v

# Task packages: set coverage, A-tier reference identity, contracts, cases, and
# that the driver honours the precision a task's tensor contract permits
python musa_operator_eval/tests/test_task_packages.py -v

# Attention set: measurement drift, generated index well-formedness (stdlib only)
python musa_operator_eval/tests/test_attention_set.py -v

# Admission gate: pairing rules, and that the mean is geometric (stdlib only)
python musa_operator_eval/tests/test_candidate_gate.py -v

# Publication separation: private assets must stay untracked (stdlib only)
python musa_operator_eval/tests/test_private_assets.py -v

# Cross-checks the task's PyTorch reference against the NumPy one (needs torch)
python musa_operator_eval/tests/test_problem_reference.py -v

# Framework-level B-tier support (stdlib only)
python src/kernelbench/unit_tests/test_library_static_checker.py -v
python src/kernelbench/unit_tests/test_dispatch_trace.py -v
```
