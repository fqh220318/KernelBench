# Phase 23：合同、容差、计时与计分闭环

日期：2026-10-07

## 完成结论

Phase 23 的无硬件控制面已经完成。11 个任务合同和 35 个公开 Case 已迁移到同一套可执行
Schema；正确性、计时、baseline 状态和跨 Case 排名不再依赖文档约定，而由 evaluator、gate
和回归测试共同执行。

这不等于硬件验证完成。3 个保留任务仍为 `keep / hardware_pending`，8 个不合格当前合同仍为
`research_only`。

## 1. Case 与性能合同

- 每个 Case 强制声明 `attributes`、`distribution` 和 `tolerance_profile`；
- 每个任务强制声明命名 `tolerance_profiles`、正式 `performance_protocol` 与计分策略；
- 正式协议冻结 warmup=3、measurements=100、discard_first=1、L2 每 trial 刷新、mean 主统计量；
- 报告同时保留 min/max/mean/std/p10/p50/p90 和原始 samples，便于诊断但不改变主统计量。

## 2. 三种容差独立执行

Case 选择命名 profile 后，`atol`、`rtol`、`max_abs_error` 分别解析和报告。框架 dtype floor
只可抬高 `atol` 与 `rtol`，不能覆盖独立的最大绝对误差上限。正式通过条件为：

1. `allclose(atol, rtol)` 通过；
2. `observed_max_abs_error <= max_abs_error`。

Python evaluator、legacy binary 诊断和 baseline 采集器使用相同规则。正式裁决仍直接比较固定
输入上的独立 oracle golden；公开 `problem.py:Model` 不再作为正确性裁判。

## 3. Baseline 三状态与分母策略

baseline readiness 已拆成三个互不替代的状态：

- `evidence_complete`：强制角色与全部 Case 行是否齐全；
- `scoreable`：每个评分 Case 是否都有冻结且通过的正分母；
- `release_ready`：在可计分基础上，任务本身是否已经取得发布所需证据。

A 类逐 Case 首选 `upstream_musa`；只有在声明并验证 `semantic_equivalence=exact` 时才能回退到
`mudnn_fused`。B 类只能使用带通过分派轨迹的 `expert_dispatch`。任一评分 Case 缺少有效分母，
整个任务都不排名，禁止静默删 Case 后计算部分 geomean。

## 4. 当前边界

- 软件 Schema、执行器、报告、inventory、gate、admission 和 ranker 已闭环；
- 旧 baseline 不能自动继承到最终合同；
- Phase 25 必须在 MUSA 环境按冻结协议重测三个 `hardware_pending` 任务，之后才可能提升为
  `release_ready`。

## 验证

- Schema 全量验证：38/38 有效；
- `musa_operator_eval/tests` 全量回归：594 项执行，54 项因本机无 MUSA/torch/POSIX 构建环境
  按条件跳过，其余全部通过；
- KernelBench library static checker：49/49 通过；
- baseline inventory、attention set 与两份候选池生成物检查通过；
- `git diff --check` 通过，冻结的原始需求指南无改动。

