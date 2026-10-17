# The evaluation service

§3 of the guide puts the evaluator beside the task packages rather than inside
them, and §2 splits what the two sides may see: the scoring criteria are part of
the task definition and the agent reads them, while these scripts are not handed
over at all.

    tools/        the public flow: generate_cases.py and the case-specialisation
                  rule it and the evaluator both apply
    evaluator/    this directory: the §4.7 pipeline, the admission gate and the
                  tools that produce the records the task packages cite
    private/      the hidden assets, which even this directory only reads

What a submission is handed is the task package: `tasks/<id>/`. Everything else
in this tree is the maintainer's toolkit, and that is the separation the guide
asks for. The substance of §2's rule -- "公开目录只放任务定义，原始源码与隐藏资产
必须与 Agent 隔离" -- is that no hidden asset and no upstream source is reachable
from `tasks/<id>/`, which `tests/test_private_assets.py` and the task-package
tests check.

## Why the scripts are visible here at all

They are committed to this repository as a **development mirror/mock**, so they are
readable by anyone with the checkout and are not a production secrecy boundary. The
same is true of `private/`, whose README already says what to do about it: in
production evaluator and maintainer tree are mounted from a private repository or an
access-controlled artifact store and given only to the evaluator. This directory
documents the interface that private deployment implements. `package_agent_task.py`
now performs an explicit deployment-boundary acceptance check: the handed-off package
is one public task and contains neither `evaluator/` nor `private/` files.

## What lives here

| file | what it is |
|---|---|
| `run_task.py` | the §4.7 pipeline over a Python-module submission: audit, build, link check, correctness, boundary, stability, performance |
| `run_binary.py` | legacy research-only diagnostics over compiled submissions; it cannot supply a normative score, admission decision, denominator or ranking row |
| `rank.py` | §5 step 8's two tables -- the A-kernel track and the B-library track, ranked apart and never combined |
| `link_whitelist.py` | reads a build product's dynamic section and checks what it links |
| `measure_baseline.py` | the complete §4.4 A/B roster measurement; the B naive/expert rows are also the admission gate's input |
| `baseline_readiness.py` | derives `evidence_complete`, `scoreable` and `release_ready`; non-passing observations stay visible but cannot supply a denominator |
| `collect_baseline_inventory.py` | emits the redacted local/device collection backlog, including lifecycle-based no-collection decisions, without copying hidden case details |
| `candidate_gate.py` | §4.4's geometric-mean gate, including the 1.3 floor |
| `record_admission.py` | writes an admission decision into a contract, and refuses the ones it cannot prove |
| `make_private_manifest.py` | builds a hidden case list from recorded gap evidence |
| `generalization_probe.py` | realizes an unlisted random case from a private runtime policy and emits replay metadata |
| `probe_sdpa.py` | probes which route a configuration resolves to |
| `collect_environment.py` | the environment snapshot and its redacted public half |
| `collect_attention_set.py` | verifies the curated set against the repository and regenerates its index |
| `validate_schemas.py` | §4.8's seven schemas and the tier rules they carry |
| `audit_agent_package.py` | rejects private assets, undeclared files and unknown generated case data; normative mode also enforces opaque provenance, lifecycle keep, independent reference and the Python-only boundary |
| `package_agent_task.py` | creates the hand-off with all public inputs/goldens, then runs the strict disclosure audit |
| `exit_codes.py` | the named exit codes the pipeline returns |
