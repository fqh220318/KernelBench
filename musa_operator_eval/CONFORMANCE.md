# Conformance with the guide

> **Phase 21 status notice (2026-10-05):** the preserved guide contains conflicting
> architecture and disclosure clauses. The current target is now
> `docs/musa_operator_evaluation_normative_spec_v1.md`, with clause-by-clause status in
> `docs/musa_operator_evaluation_requirement_traceability_matrix.md`. The audit below
> predates that arbitration. Its measurements remain evidence, but any `Met` claim is
> historical until re-audited against the normative specification. Phases 22--24 have
> migrated all three retained contracts to `hardware_pending`, classified eight current
> contracts as `research_only`, and closed the repository-side contract, scoring and
> deployment boundaries.
> Compiled/direct-muDNN scoring and the remaining copied references are not accepted
> compliance.

`docs/musa_operator_evaluation_guide.md` is the preserved specification this set is
built to. This file
records, requirement by requirement, where the repository satisfies it and where it
does not, so that a shortfall is a decision a reader can see rather than something
to be discovered later.

Every claim here is checked by something in `tests/` or by a command whose output is
quoted. Where a requirement is met by a declaration rather than by an enforcement,
that is said in the row.

`IMPLEMENTATION_DECISIONS.md` records subordinate implementation choices without
rewriting the preserved guide or the normative specification. A decision remains a
shortfall until the implementation and its tests enforce the normative target.

## §1 The two tiers

| requirement | where |
|---|---|
| tier on every package | `task.json:tier`, `tests/test_task_packages.py` |
| the two tiers independent in task, API, baseline, metric, audit, ranking | `library_policy` versus `kernel_scope`; `scoring.ranking` says the tracks are never merged; `sets/attention/attention_set.json` records per-entry eligibility per tier |
| the three judgement questions | applied in the ledger's per-entry `tier.*.reason`, each citing a measurement |

## §2 What the set contains

| part | where | given to the agent |
|---|---|---|
| source snapshot, commit, licence | `sources/`, `task.json:source` | the provenance record only; the snapshot is fetched from the archive branch |
| environment snapshot | `environments/<snapshot_id>.public.json` (redacted), full copy under `private/environments/` | the redacted one |
| task contract, semantics, prompt, reference, public cases, starter | `tasks/<id>/` | yes |
| hidden cases, goldens, baselines | `private/<id>/`, `private/generated/<task id>/` | no |
| evaluator | `evaluator/` | not handed over; the scoring criteria are (`task.json:scoring`, `PROMPT.md`, `agent_reference/`) |

Checked by `evaluator/audit_agent_package.py` (all 11 packages) and
`tests/test_private_assets.py`.

**Production boundary.** The evaluator scripts committed here are deliberately a public
development mirror/mock, not the production secret evaluator. Release-package
acceptance rejects both `evaluator/` and `private/`, and records that production must
mount its evaluator from a private repository or access-controlled store. Repository
tests enforce that package boundary; the deployment system remains responsible for
actually supplying the private mount.

## §3 Directory structure and the starter

`sources/`, `tasks/`, `private/`, `evaluator/` all exist. Each package carries
`starter/model_new.py` (the file the agent copies) and `starter/build.py` (the one
build entry point §4.7 runs), and `task.json:starter` is the §4.6 inventory.

The guide's starter table is written for a C++ project with CMake; this set's
submission is a Python module. The roles are realised as: build entry point
(`starter/build.py`, one file in every package, compared by a test), frozen header
and main program (the framework's `problem.py` / `ModelNew` contract), host launch
and device kernel (marked regions), dispatch file (a marked region — see §4.1 below),
tensor IO (the frozen `.bin` + JSON manifest ABI).

## §4.1 Task contract

Every §4.1 field is present, including the B tier's four extras. The dispatch file is
declared as `starter.dispatch`, because a submission here is one module and the
dispatch is a region of it; `library_policy` keeps the whitelist and the trace
requirements.

## §4.2 Semantics contract

Every package records the tensor signature, the dynamic-axis ranges, the ABI, the
determinism rule and §4.2's three pinned items — including the families where a mask
cannot arise, which say `not_applicable` rather than leaving the question open. A
formal contract is classified exactly once as A or B over its complete semantic
domain. The old A/B pairs remain byte-comparable only as legacy research assets;
`tests/test_task_packages.py` checks that every such pair is `research_only`, so it
cannot become two formal tasks again.

## §4.3 Case lists

Public: 3 smoke cases per package with golden paths recorded. Hidden cases are
private and cover correctness, boundary, non-aligned, extreme and performance,
with seeds disjoint from the public list and no repeated shape. The
generalisation probe is not stored in either list: a private policy realizes it
after evaluation starts, and the report records the replay seed. Cases carry
parameters and seeds, not tensors; the generator is `tools/generate_cases.py`.

**Shortfall, now a hard blocker.** §4.3 asks a B-tier hidden set to reach at least two
library-gap reasons. Four of the five B packages reach one (`unsupported_shape`); the
pilot reaches two (`unsupported_shape` and `semantic_mismatch`). Inventing a second
gap would mean grading a configuration the library actually serves.

The four entry-scoped packages are therefore `blocked_gap_coverage`, explicitly
`decision=reject / lifecycle=research_only`, and are neither admitted nor scoreable.
Their old performance numbers remain historical evidence only.
`make_private_manifest.py`, `record_admission.py`, the release packager, both
evaluation drivers and `rank.py` refuse to turn that evidence into a score. The
attention-set ledger also records `scoreable: false`. The private evidence carries an
`exactly_one_reason_category` block whose `why_the_others_are_out_of_reach` accounts
for every category the vocabulary has and is not reached, and
`tests/test_private_assets.py::GapCoverageTests` fails if a single-category task has
no such block or argues the wrong set of categories. The argument is per task because
the lever is per family. For the four non-pilot tasks it is the same one: the fused
path's only boundary is `head_dim <= 128` (the library says so itself: *"FlashAttention
in MUSA backend now requires head_dim to be less equal to 128, but got: 192"*), and an
isolating device probe recorded in each task's `probe_recheck` block shows the boundary
is a *route* rather than an error -- at head_dim 192 and 256 the call is answered on the
math path instead of raising -- and that float16, bfloat16 and float32 are all accepted
at head_dim 64 and 128, so the hypothesis that could have produced a second category
(the bfloat16 gap cases being dtype gaps) is refuted by measurement. What else holds:
the reference passes a frozen `1/sqrt(head_dim)` scale and a plain top-left causal mask
with no window (`semantic_mismatch` unreachable), hands the fused path the transposed
non-contiguous tensors it accepts (`layout_mismatch` unreachable), and asks for one
attention op whose neighbours in the block are elementwise work
(`multi_operator_required` unreachable — and it is a property of the task, not of a
case, so it cannot separate a gap case from an ordinary one).

**Swin split, made explicit.** §4.3 asks the hidden shape range not to overlap the public
one. `swin_transformer_v2_a_v0` cannot vary the image size faithfully: the reference is
built for 224 after four downsampling stages. The batch axis remains legal from 1 through
4, so the public smoke list now occupies B=1/2/3 and reserves B=4 for the hidden shape
domain. The public manifest and semantics contract both state that partition, and a
repository test prevents a future smoke-case edit from consuming the reserved boundary.

## Across the grades: one named tolerance profile per case

Every case selects a named entry from `task.json:tolerance_profiles`. Its `atol`,
`rtol` and `max_abs_error` are independent contract limits: the framework's per-dtype
floor (`get_tolerance_for_precision`) can raise only `atol` and `rtol`, while the
absolute-error ceiling remains exactly the declared `max_abs_error`. A case passes
only when both `allclose(atol, rtol)` and the independent maximum-error ceiling pass.
The Python evaluator, the legacy binary diagnostic and the baseline collector use
the same resolution rule, and reports retain the profile plus all three effective
limits.

This replaced a real inconsistency: the hidden manifests carried a per-case
`atol`/`rtol` (0.01 or 0.02) that only the compiled driver read, while the Python path
graded at 0.05 -- two bars for one task, where a submission at error 0.03 passed one
path and failed the other. The ambiguous scalar field is gone from the schema and all
11 task contracts plus all 35 public cases select explicit profiles;
`additionalProperties: false` makes a leftover key invalid, and every evaluated row
records `tolerance_used`.

## §4.4 Baselines

The schema and collector now make the complete roster mechanical. Every hidden
case must have exactly one status-bearing row for all four A roles (`upstream_musa`,
`mudnn_fused`, `torch_musa_sdpa`, `torch_musa_eager`) or all six B roles (those
four plus `naive_library_composition` and `expert_dispatch`). Missing or duplicate
role/case rows make the derived `evidence_complete` state false.

**Roster completeness is not scoreability.** §4.4 asks each implementation row to
record a value and a status, while §4.3 requires B-tier gap cases where a fused
library may be unsupported by definition. A reasoned `unavailable`, `unsupported`,
`error` or `fail` row therefore completes the observation roster but never becomes
a latency. `candidate_gate.py` separately requires passing, positive-latency
`naive_library_composition` and `expert_dispatch` rows for every case;
`record_admission.py` additionally requires their correctness and the expert trace;
and `rank.py` requires a frozen passing denominator and candidate runtime for every
scoring Case. Any missing Case makes the entire task unranked rather than producing a
partial geomean. This separation prevents
both hiding a failed comparison row and requiring a fused library to pass a case
chosen because the library does not cover it.

This public checkout deliberately does not contain the maintainer tree. The
redacted `collection/baseline_collection_plan.json` separates three keep contracts
from eight `inactive_current_contract/not_scheduled` research-only contracts. Of the
keep set, all three tasks are `hardware_pending` after static migration. The four
entry-scoped B research packages cannot
reach a second gap category, while four A research packages have a single-call equivalent
library path. Any redesign needs a new versioned contract rather than another baseline
run. This remains an inventory, not a claim that measurements are complete.

**One protocol, and it is the framework's.** A baseline's `latency_ms` is a denominator
and a `run_task.py` report's `runtime` is a numerator, so the two have to be the same
kind of number. They were not: the tool carried its own protocol -- warmup 10, five
rounds of a hundred calls, the median of the round means, no cache flush -- while the
reports come from `kernelbench.timing`, whose loop warms up, empties the allocator, then
runs one pass of a hundred trials with an L2 thrash before each (so its numbers are
cold-cache numbers), discards the first trial and reports the **mean**. Every ratio was
consequently a ratio between two protocols, and the fact that they agreed to within a
tenth was luck rather than design.

The tool now calls that loop instead of imitating it:
`kernelbench.timing.time_execution_with_cuda_event` measures, `get_timing_stats`
summarises, and the record's `measurement_protocol` names both. `latency_ms` is the mean
of the trials, `latency_ms_samples` is the trials themselves (renamed from
`latency_ms_rounds`, because the framework's loop has one pass and calling a trial a
round described a structure it does not have), and `latency_ms_stats` keeps the spread
the mean came from. `tests/test_private_assets.py` fails if a record does not name that
loop or if its number is not the mean of its own samples.

**Remaining protocol deviation.** The guide literally names repeated rounds and
a median in the minimum protocol record. KernelBench's scored runtime is one
cold-cache trial sequence summarized by its mean, so this implementation keeps
the baseline denominator on that same protocol instead of silently comparing a
median-of-rounds denominator with a mean numerator. The samples and p10/p90 are
retained, but the median/repeated-round wording is not yet satisfied. Resolving
it requires changing both submission timing and baseline timing together and
re-measuring on hardware; it is not safe to change the denominator alone.

`measure_baseline.py` records the task reference under the canonical
`torch_musa_eager` role and accepts repeatable `--implementation ROLE=PATH`
arguments in both tiers. The B command previously accepted only naive/expert (and
an optional SDPA shortcut), so it was impossible for that command to produce the
required six-role record. That interface defect is fixed; every non-eager role
can now be backed by a real model-level implementation.

The numbers moved when the protocol did, most of all on the small shapes where a flushed
cache costs the most, and no decision moved with them:

| package | before | after |
|---|---|---|
| `scaled_dot_product_attention_b_v0` (level 1, B) | 3.834 | 3.213 |
| `vision_attention_b_v0` | 1.751 | 1.712 |
| `mingpt_causal_attention_b_v0` | 3.691 | 3.447 |
| `mingpt_block_b_v0` | 2.190 | 2.124 |
| `sdpa_forward_b_v0` (the pilot) | 7.811 | 6.152 |

**Historical evidence, not current completion.** Earlier maintainer-side records
measured useful rows, but they predate the current complete role/case roster and the
unified timing protocol. When mounted, they must be re-derived and re-collected;
documented non-passing rows may complete the roster, but only passing rows from the
score-critical roles can enter a gate or ratio. The observations below remain useful
explanations of the backlog; they are not accepted as current scoreable baselines.

The A tier records `mudnn_fused` as `unavailable` for one measured reason: this tier is
graded at float32, and `musa::dnn::ScaledDotProductAttention::RunFlash` refuses it with
the library's own words -- `Unsupport Type FLOAT` -- at every shape tried (head dim
32/64/128/160/192, sequence 1/128/512/1024), while the same call succeeds at float16.
`torch_musa_sdpa` is **measured** on four historical A research records whose source
entries also have a legacy B research record: at float32 the entry point answers, dispatching to
`aten::_scaled_dot_product_attention_math_musa`, and the implementation measured under
that name is the module the legacy B research record's expert is. The two retained A tasks
record why the entry point is not a faithful implementation of their reference at all,
quoting the ledger's measurements: `swin_transformer_v2_a_v0`'s only remaining lever was
a head dimension out of reach of a byte-identical reference, and
`relu_causal_attention_a_v0`'s reference applies ReLU to the scores, which no library
path expresses. `tests/test_private_assets.py` holds that rule against the ledger, so a
historical paired record cannot come back with an unmeasured library row.

The four entry-scoped B packages record `upstream_musa`, `mudnn_fused` and
`torch_musa_eager` as `unavailable`: no upstream answer exists in this checkout for
them, the Python roster reaches muDNN only through torch_musa, and eager PyTorch is
what the reference already is. The measurement tool records an implementation it cannot
honestly build as unavailable with a reason rather than substituting a lookalike; a
baseline under a real implementation's name would be worse than a gap.

Also worth recording: the A tier's upstream implementation refuses most hidden shapes
with `RuntimeError: Tile divisibility` and, for `vision_attention_a_v0`, agrees with
the reference on only four of eight cases. Those rows are in the baseline records.
The hand-written answers are specialised to the shapes they were tuned on, which is
what a hidden set outside the public range is for.

The B tier's gate (naive-to-expert ratio, geometric mean, floor 1.3) is re-measured
and re-recorded by `measure_baseline.py` → `candidate_gate.py` →
`record_admission.py`. The family-scoped pilot meets the two-category prerequisite
and remains admitted. The four entry-scoped measurements pass the performance gate
but no longer claim admission because their hidden sets do not meet §4.3. Per-case
evidence lives in `private/<task id>/admission.json` and the contract keeps only the
summary, because §2 hands the contract to the agent.

## §4.5 Prompts

Seven sections per package, with the tier direction asserted by test: the A tier's
prohibitions name the library entry points, the B tier's requirements say the library
path comes first. No prompt names the upstream project, a library function or a
version string; the budget matches the contract.

## §4.6 Starter inventory and ABI

`task.json:starter` declares the editable file and its marked regions, the frozen
list, the build and self-test commands, the ABI and — for the B tier — the link
whitelist, which a test requires to equal `library_policy`. The static audit reads
`frozen` from there, so the boundary it draws is the declared one.

**Normative resolution.** The Python `ModelNew` path is the sole scored path. A migrated
normative package therefore removes `starter/cpp`; the compiled runner and direct-library
adapter remain maintainer-side diagnostics and cannot supply admission, ranking or an
official denominator. All three keep tasks enforce this boundary now. Legacy
research-only packages still carry the old compiled assets for reproducibility only.

The compiled-path evidence below is historical diagnostic evidence, not a current scored
architecture claim.

**Met on the compiled path for the level-1 family, and verified end to end.**
muDNN's host API is present on this machine and needs no torch: the toolkit ships
`mudnn.h`, `mudnn_base.h`, `mudnn_math.h` and `mudnn_nn.h` under
`/usr/local/musa-3.1.0/include/` (`$MUSA_HOME/include`), and `mudnn_nn.h` declares
`musa::dnn::ScaledDotProductAttention` with `SetHeadsNum`, `SetEmbedDim`, `SetCausal`,
`SetMaskMode`, `SetKeyFormat`, `RunFlash` and `RunMath`. `libmudnn.so` exports 10001
`musa::dnn::*` symbols, and a program that links only `libmudnn` and `libmusart` --
no torch in its dynamic section -- gets `SUCCESS` back from `RunFlash`.

`private/scaled_dot_product_attention_b_v0/compiled/kernel.mu` is that adapter for
this family. It asks the library first and reads the answer rather than predicting
it: on this build the fused kernel answers head dimension up to 160 and refuses above
it (*"Flash Attention 2 Not Support HeadDim > 160 Now"*), so the adapter takes the
fused path or the library's non-fused path and records which one produced the number.

**Four families have a compiled answer, and every one of them was verified end to end.**
The three beyond level-1 are the families whose reference builds weights in `__init__`,
which the staging above is what makes possible at all. Each is a worked submission under
`private/`, each links only `libmudnn` and `libmusart`, and each was driven through
`evaluator/run_binary.py` over its whole hidden set on the device:

| family | result | worst error against the golden | recorded paths |
|---|---|---|---|
| `kb_l1_97_a` (A tier: hand-written kernel) | 9 of 9 | 2.2e-06 | no trace on this tier |
| `kb_l1_97_b` (SDPA through the library) | 10 of 10 | 4.9e-4 / 3.9e-3 | fused 6, composition 4 |
| `kb_l3_31_b` (ViT attention, layer-normed) | 10 of 10 | 1.2e-3 - 3.1e-2 | fused 6, composition 4 |
| `kb_l3_43_b` (minGPT causal attention) | 10 of 10 | 2.4e-4 - 3.9e-3 | fused 6, composition 4 |
| `kb_l3_44_b` (minGPT block: attention + MLP) | 10 of 10 | 2.0e-3 | fused 6, composition 4 |

Every recorded path equals the case's expected one, which is what the trace check
enforces, and every artifact's dynamic section names `libmudnn` and no torch.

**The library's behaviour shaped each answer, and every claim here was measured on the
device rather than read off a header.** `RunWithBiasAdd` at bfloat16 with a 768-wide
output returns SUCCESS and computes the product *without* the bias (768 where 770 was
due) while the same call at 8x32x64 adds it; muDNN's `Binary` add writes the bias alone
when its output and left operand are the same tensor; `RunMath` ignores `SetCausal` (a row
comes out as the mean over every key) and refuses the strided `[B, H, T, D]` view over a
`[B*T, 3C]` buffer that the fused kernel accepts (*"Validate q stride"*); `Permute`
produces the wrong elements -- with `SUCCESS` -- unless both sides are contiguous with the
permutation configured through `ConfigDimStride`, and its three-axis rotation is not its
own inverse. The answers therefore add every bias by a separate broadcast add into its own
buffer, pass the causal mask explicitly to whichever path answers, project into contiguous
buffers, and name each layout change at its call site. `HARDWARE_RUNBOOK.md` records the
same list for anyone writing a submission.

**The state is materialized for the case, not for the problem file's defaults.** The
first version of this tool loaded `problem.py` as written and built the model from
`get_init_inputs()`. That is the wrong model whenever a case re-binds a constant the
construction reads: `kb_l3_43_b`'s cases map `shape.C`/`shape.H`/`shape.S_max` onto
`n_embd`/`n_head`/`max_seqlen`, so eight of its ten cases were handed a state of the
wrong shape -- `c_attn.weight` at 2304x768 where the case needs 192x64 -- reported as
materialized, with a digest of its own to match it. The tool now specializes the source
with the case's own entry, through the same `case_source` helper `run_task.py` and
`generate_cases.py` use, and it takes the case id and the manifest rather than a bare
seed so a call that does not identify the case is a usage error instead of a silent
rebuild. The row records `case_id` and `init_inputs` beside the digest, so which
configuration the state came from is in the report rather than in the command that
produced it.

**The compiled path's input set, and what decides it.** It grades a case out of the
tensors in its input directory -- that is the whole interface -- and eight of the
eleven packages are references that build weights in `__init__`, which a torch-free
submission cannot reconstruct: they are a function of the case's seed and the
construction order, and §4.3 deliberately keeps a case to "parameters and seeds, not
tensors". So the evaluator materializes that state with the framework's own
construction -- seed, `get_init_inputs()`, seed, `Model(*init)`, cast to the case's
dtype, the order `kernelbench.eval` uses -- stages it beside the case's own tensors
under `state_*` names, and records a digest of the whole state on every case row.

The generated tree is never written to: the runner reads a staged copy, so the case's
own inputs stay what they were and the Python path, which builds its own model, reads
nothing new. No golden was regenerated for this, because the state comes from the same
seed stream the golden was produced from. `task.json:starter.cpp.model_state` declares
the arrangement per package (`tensor_prefix`, `materialized_by`, where the seeding
comes from) or is null for a reference that carries nothing.

**The construction arguments are the other half of that.** A case re-binds module
constants (`case_parameters`), so `n_head` decides how the qkv projection's output splits
-- and no tensor's shape carries it, because the projection's width is `3C` either way.
Every contract now declares `starter.cpp.case_configuration` (the names `get_init_inputs()`
returns, extracted from each problem), the evaluator stages the case's own values as
`case.json` beside its tensors, and the driver exports the staged directory as
`KB_CASE_INPUT_DIR`, because the tensors arrive as arguments and the path is not implied by
them. A contract that declares a different number of names than the reference returns is a
staging error rather than a submission's wrong number.

**The scope, as a property of the declarations rather than a backlog.** A compiled answer
is legitimate exactly where the task hands over everything a torch-free process needs: the
case's tensors, the reference's state when it has one, and the construction arguments. Four
of the eleven packages meet that and have an answer (the two level-1 families, the
layer-normed ViT attention and the two minGPT families). What does not is stated with its
reason: `swin_transformer_v2_a_v0` and `relu_causal_attention_a_v0` are exclusively A-tier
tasks, so a library-path answer would be a different contract; and the legacy A research
records of the ViT and minGPT families cannot use library
calls even in principle, so their compiled path would mean hand-writing the projection,
attention, MLP and normalisation kernels rather than dispatching -- a different job from
this one, and one the tier does not require of a submission.
`tests/test_private_assets.py::CompiledAnswerScopeTests` fails if an answer exists where
neither the case's inputs nor a declaration covers it, and if an answer's contract does not
declare both halves of what it reads; `tests/test_materialize_model_state.py` checks on the
device that each declaration matches what its reference actually holds.

The anti-cheating half holds independently of that: the artifact is built, its
dynamic section is read, and a submission linking outside the whitelist is refused
before a single case runs (a device test compiles a program that really links zlib and
asserts the run stops with `EXIT_LINK_WHITELIST_VIOLATION`).

**Met, and checked, on the compiled path.** §4.6 confines library calls to the
dispatch file and the Host launch file. A Python submission is a single module, so
that is a region constraint rather than a file constraint; a compiled one marks its
regions in the source, so `run_binary.audit_compiled_source` reads the markers and
rejects a library call outside the permitted regions — and, on the A tier, any
reference to the upstream stacks at all. The check is what rejected an early
revision of the worked kernel for declaring one region where the inventory
declared two.

## §4.7 Evaluator

Nine stages, short-circuiting, named exit codes: environment, case generation, static
audit, build, link whitelist (B only), dispatch trace (B only, per case), correctness
with the stability re-run, performance, summary. `run_task.py` reports the stages it
ran; a stage that fails is not followed by the next one, which the device runs show
as `skipped` rows.

`evaluator/run_binary.py` can still drive the compiled diagnostic path against the same
goldens, but a normative release neither ships nor scores that path. The Python case loop
now loads each generated input/golden pair and passes it into the framework evaluator;
the report records `correctness_reference=independent_golden`. Golden digest replay is
therefore no longer mistaken for correctness authority while `problem.py` silently
decides the verdict.

**Compiled sub-stage short-circuit, corrected.** The compiled driver formerly ran
every case in one flat loop and chose an exit code afterward. A correctness failure
therefore did not stop boundary or performance cases, despite the top-level report
claiming the same pipeline as the Python path. Both paths now derive the stage from
the fixed case tag vocabulary, execute `correctness → boundary → stability →
performance` regardless of manifest list order, finish all cases in the current
stage, and emit explicit skipped rows for every later stage. This also fixes the
runtime generalisation probe: it is appended to the manifest after materialisation,
but is moved back into correctness instead of accidentally running after performance.
Compiled reports carry an `evaluation_stages` status list, and the exit code comes
from the first failed stage (with dispatch-trace and stability codes retained).

The link check parses the build products' dynamic sections with no dependency beyond
the standard library, and was checked against `readelf -d` and `nm -D` on six real
MUSA extension objects and on system binaries: identical dependency and symbol sets.

**Timing, and what `runtime` means on this path.** A compiled case runs in a process
that starts cold: measured on the target device, the first call in a fresh process pays
237 ms for the device context and roughly as much again for mapping a 2.2 GB muDNN,
the case's files cost 58-470 ms to read and write depending on size, and the attention
call itself takes 0.5 ms the first time and under 0.1 ms thereafter. A wall clock around
that process is therefore 99.9% startup and I/O, and comparing it with the Python
path's `runtime` -- which is measured in a warm process with the tensors already on the
device -- compares two different quantities.

So the frozen runner takes `--warmup W --repeat N` (defaults 3 and 100, the framework's
own call counts): it reads the inputs once, calls `kernel_entry` W times to settle the
process and the device, then takes `repeat + 1` timed calls and reports each one. The
clock is in the frozen half, so a submission cannot report its own grade, and the
statistic is in the driver, where a test can hold it.

**The measurement is the framework's protocol, not one this path invented.** The same
commit that moved the baselines onto `kernelbench.timing` moved the runner onto the same
measurement, so that a compiled `runtime` and a baseline's `latency_ms` divide each other
as like by like: a MUSA **event pair** around each call rather than a host clock, the
**L2 cache flushed before every timed call** (a 256 MB fill, which is what upstream's
`clear_l2_cache` is) so the numbers are cold-cache numbers, the **first timed call
discarded** as `discard_first=1` discards it, and the **mean** of what is left, in the
three-significant-digit format `get_timing_stats` renders. The flush is queued outside
the measured interval, and the fields that describe the protocol travel in the row --
`timer`, `cache`, `l2_thrash_bytes`, `samples` -- so a row cannot claim a measurement it
did not make: a runner that could not allocate the flush buffer records `cache: "warm"`,
and a runner that reports no timings at all still grades, with `runtime` falling back to
the process wall clock and `timing` null, which is what an older starter does.

Two boundaries are stated rather than left to be discovered. The events are recorded on
the device's default stream, so a submission that launches on a stream of its own and
does not synchronize is timed as if it had done nothing; every worked answer and every
skeleton uses the default stream, which is the pattern the ABI documents. And a compiled
case is still only ever ranked against another compiled case, because the forms differ by
their overheads -- the ABI's host-to-device copies are inside this interval by design --
rather than by their clocks.

**What aligning the two clocks cost, measured.** Both compiled rows were re-measured
under the framework's protocol (event pair, per-call L2 flush, first trial discarded,
mean) and both still pass -- the worked A kernel 9 of 9, the B adapter 10 of 10 -- but
they moved by different amounts, and the difference is the interesting part. The worked
kernel's smallest case went from 3.8 ms to 10.7 ms, a factor of 2.8 with a *tight*
distribution (100 samples spanning 10.68-10.84 ms), because that kernel is compute- and
cache-bound and a flushed cache is exactly what it feels. The B adapter barely moved:
0.501 ms to 0.496 ms on the single-token case, 24.356 ms to 23.8 ms on the widest one,
because those rows are the ABI's host-to-device transport rather than cache behaviour
(§4.7's arithmetic: 98% of that case is the copy). A protocol change that had only moved
the small case would have been a change in the timer; the fact that it moved the
cache-sensitive row and left the transport-bound row alone is evidence it did what it
says.

**Defect found and fixed by that run.** The stage selected build products by suffix
(`.so`, `.o`), and a linker's final product is an executable with no suffix at all,
so it was reading exactly the artifacts that have no dynamic section and reporting a
pass. It now identifies products by the ELF magic and keeps the suffix list as a
second signal, so a mangled object is still reported rather than skipped. Three tests
cover it, including that a suffixless executable is checked and its libraries read.

## §4.8 Schema validation

Seven schemas, versioned, validated by value with the tier rules. In this public
checkout, `validate_schemas.py --all --engine stdlib` reports 38/38 valid (the
two baseline templates are included in discovery rather than treated as prose).
Maintainer-side baselines now require a derived readiness record and the complete
§4.4 roster, so the earlier 68/68 device-tree result is historical and must not be
reused: after mounting `private/`, regenerate the collection inventory and run
validation again. Tests that require the maintainer tree skip when it is absent;
the full current test totals must therefore be read from the validation run rather
than copied from an older public-checkout count.

## §5 Build steps

Environment comparison, source provenance, per-family semantics, tiered contracts,
public case generation, evaluator assembly and separate tracks are implemented.
True §4.4 baseline completion is still pending the private mount and target device:
the redacted inventory reports three active `not_collected` records and eight inactive
current contracts in this checkout. No ranking is current until the active records are
`evidence_complete`, `scoreable` and `release_ready`.

**Met.** §5 step 8's "separately split, separately ranked" is enforced in two places:
the scoring blocks and case lists never mix tiers, and `evaluator/rank.py` produces
the two tables -- never a combined figure, because a single number across a device
kernel and a library dispatch is a number about nothing. Each ratio is the tier's
own denominator, frozen in `scoring.denominator_by_case` (`upstream_musa`, or a proved
exact `mudnn_fused` fallback, for A; `expert_dispatch` for B). A ratio must exist for
every scoring Case before their geometric mean is computed. Both sides of a ratio are
milliseconds, and since §4.4's change both
come from the framework's own timing loop, so a ratio is between two implementations
rather than between two protocols. The unit is the one the baseline is in: the
framework's dataclass comment says microseconds and the recorded values say otherwise,
so `HARDWARE_RUNBOOK.md` records the trap.

Historical device readings from before the strict roster/readiness migration
(diagnostic evidence only; they are not a current leaderboard):

| report | ratio | cases compared |
|---|---|---|
| A tier: the worked compiled kernel (fp32, steady state) | `x0.05306` | 2 of 9 |
| B tier: historical entry-scoped Python/compiled runs | excluded | blocked by the two-category prerequisite |

The removed B rows remain useful device diagnostics, but `rank.py` now requires a
`task_admission` snapshot of `measured/admit`; a blocked or legacy report cannot
enter either ranking table.

The compiled answers sit four to twelve times behind their Python counterparts, and the
gap is the form rather than the dispatch decision: the Python path is handed device
tensors and hands them to one library call, while a compiled submission is handed host
bytes, copies them up itself, and pays for the layout changes (`Permute`) that the
framework's own bindings do invisibly. The ABI's host-to-device copies are inside the
timed region by design, which is why a compiled case is only ever ranked against another
compiled case. That explains the historical measurements, but does not override
the gap-coverage prerequisite that now excludes the four entry-scoped B tasks.

Three historical readings from that session are worth keeping as diagnostics. A B-tier Python report now lands
within 0.4% of its own baseline, because the report and the baseline are the same
expert measured by the same loop; the same comparison read `x0.9112` and `x0.892` while
the two sides were measured differently, and that nearness to 1 was luck rather than a
property of the measurement. The compiled adapter reads `x0.08007`, twelve and a half
times behind the expert, which is the host-side staging the ABI asks a submission to do
rather than the library call. And the A tier's worked kernel reads `x0.05187`, about
twenty times slower than the upstream answer -- where the same kernel measured through a
process wall clock had read `x0.002615`, four hundred times slower, because that number
was process startup, device initialisation and disk. The last pair is why a compiled case
is only ever ranked against another compiled case: the forms differ by their overheads,
and the overheads are now visible instead of being the measurement.
