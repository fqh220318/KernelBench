# 阶段 12：基线角色完整性与待采集清单

## 本阶段结论

需求文档 §4.4 规定 A 类每个性能 Case 至少覆盖 4 个基线角色，B 类覆盖
6 个角色。本阶段把这个要求从文字说明变成了可重新推导的硬门槛，并生成
不泄露隐藏 Case 的本地采集清单。

当前公开工作树没有挂载 `private/`，因此不能声称完成了真实基线采集。生成的
`collection/baseline_collection_plan.json` 如实记录：

- 共 11 个任务，当前均为 `not_collected`；
- 7 个是当前 release blocker：6 个 A 类任务和已准入的 family-scoped B pilot；
- 4 个 entry-scoped B 任务先受 §4.3 缺口原因覆盖不足阻塞，基线采集优先级为
  `after_admission_blocker`；
- 清单只包含公开 task id、角色名、状态码和命令模板，不包含隐藏 Case id、
  shape、seed、golden、测量值或私有失败原因。

## 回退审查

没有发现需要回退 Phase 08–11 的证据。运行时泛化、B 类双原因准入门槛、发布
阻断和排行榜阻断的前置契约仍成立。

Phase 12 自身发现两个需要局部返工的问题，已在本阶段修正：

1. B 类测量命令原来只能传入 naive/expert（以及可选 SDPA shortcut），无法传入
   `upstream_musa`、`mudnn_fused`、`torch_musa_sdpa`，所以六角色“完整”状态在
   接口上不可达。现在 A/B 两类都可重复使用 `--implementation ROLE=PATH`，B 类
   保留 `--naive` 和 `--expert` 两个专用参数。
2. 初版 readiness 只拒绝 `unavailable`，会错误放过 `fail/error/unsupported`。
   现在所有非通过的必需角色都会令记录不完整；重复行、缺失角色、缺失
   Case/角色组合和陈旧 readiness 同样被拒绝。

此外，`record_admission.py` 现在必须接收私有 Case manifest，并按 manifest 的
完整 Case id 集合复核；从 baseline 中删掉整条隐藏 Case 后重新生成 readiness，
也不能再绕过准入。

## 新增或修改的执行点

- `evaluator/baseline_readiness.py`：按 tier 推导强制角色及每个 Case 的覆盖状态；
  存储的 `readiness` 只作记录，所有消费者都会重新计算。
- `schemas/baseline.schema.json`：要求 readiness、A/B 角色清单及 unavailable 原因。
- `evaluator/measure_baseline.py`：测量完整角色清单；`torch_musa_eager` 由任务
  reference 自动测量；不完整记录会写出证据但返回非零，不能进入后续流程。
- `candidate_gate.py`、`record_admission.py`、`rank.py`：统一拒绝不完整或发生漂移
  的基线。
- `evaluator/collect_baseline_inventory.py`：生成/检查脱敏待采集清单。
- `tests/test_baseline_readiness.py`、`tests/test_baseline_inventory.py`：覆盖角色缺失、
  Case/角色缺失、不可用、非通过、陈旧摘要和脱敏要求。
- `tests/test_private_assets.py`：仅在 maintainer tree 真正挂载时检查隐藏清单与
  完整基线；公开 checkout 不再因预期缺少私有资产而误报失败。

## 验证命令与结果

```text
基线 readiness/inventory/gate/admission/rank/schema 定向测试：58/58 通过
record_admission 完整 Case 集合回归测试：14/14 通过
私有资产公共 checkout 测试：15 项通过，11 项按预期跳过
Repository schema validation（stdlib engine）：34/34 文档有效
baseline collection --check：通过
py_compile 与 git diff --check：通过
```

整套 `musa_operator_eval/tests` 在 Windows 上仍有平台相关失败：测试夹具直接运行
POSIX `.sh`、读取 `/bin/true`、调用 `false` 和写入 `/tmp`。这些失败不来自本阶段
逻辑；定向测试已覆盖本阶段所有改动。设备相关测试仍需在 Linux + MUSA 环境中
执行。

## 仍未完成

1. 在目标 MUSA 环境挂载 maintainer tree，并为 11 个任务提供真实的模型级角色
   实现后运行清单中的命令；禁止用相似实现冒充缺失角色。
2. 对生成的 baseline 运行 schema、readiness、candidate gate（B 类）和 rank
   前置检查，再重新生成本清单；只有 `complete` 才可计分。
3. §4.4 原文还明确写了“重复轮数、中位数”。当前 baseline 与 submission 都沿用
   KernelBench 的单次 cold-cache trial sequence + mean，以保证分子分母协议一致；
   这仍是已记录的协议偏差。必须在硬件上同时修改两侧计时并重测，不能只改变
   baseline 分母。

