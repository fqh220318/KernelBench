# MUSA 算子评测工作进度补充（2026-09-27）

## 1. 文档用途

本文记录 2026-09-27 在已有 MUSA 算子评测工作的基础上新增、修正和验证的内容，
用于与此前的周报、Phase 08–11 阶段总结进行对照。

本次工作的原则是：

- 已有阶段只有在前置契约不成立时才回退；
- 无 MUSA 硬件时完成所有可验证的控制面、schema、调度和测试工作；
- 不使用占位值、历史数据或 `unavailable` 状态冒充真实硬件测量完成；
- 每个阶段完成后生成独立 Markdown 总结。

工作分支与目录：

```text
branch: feat/musa-operator-evaluation-compliance
base:   8d03771
worktree:
C:\Users\冯连雨\KernelBench_github\.worktrees\musa-operator-evaluation-compliance
```

当前改动尚未创建 commit，也未上传 GitHub。

## 2. 与此前进度的对照

| 内容 | 此前状态 | 今天的补充或修正 | 当前结论 |
|---|---|---|---|
| 来源与版本追溯 | Phase 08 已完成 | 复核后未发现需要回退的问题 | 保持完成 |
| Agent public smoke release | Phase 09 已完成 | 复核发布隔离和 11 个 package audit | 保持完成 |
| 运行时泛化探针 | Phase 10 已实现随机采样和 seed 回放 | 修复动态泛化 Case 可能因追加位置而在 perf 之后执行的问题 | 控制面完成，待设备复验 |
| B 类缺口覆盖 | Phase 11 已将 4 个单原因任务改为阻塞 | 确认没有通过修改字段解除 blocker；仍要求第二类真实测量 | 1 个 pilot 可准入，4 个 entry-scoped B 继续阻塞 |
| §4.4 baseline 角色 | 此前存在历史 baseline，但角色不完整、部分为 `unavailable` | 新增 A 四角色、B 六角色的机器可检查完整性规则 | 控制面完成，真实 baseline 待硬件重采 |
| B baseline 测量接口 | 只能传 naive/expert，完整六角色在接口上不可达 | B 类现可通过 `--implementation ROLE=PATH` 提供全部公共角色 | 已修复 |
| Baseline 准入与排名 | 主要检查 naive/expert 比值 | readiness 不完整或漂移时，gate、admission、rank 全部拒绝 | 已形成统一硬门槛 |
| 隐藏 Case 完整性 | 删除整条 Case 后可能仅从 baseline 自身重新推导 | `record_admission.py` 现在必须对照完整私有 manifest | 已修复绕过路径 |
| compiled Case 执行 | 所有 Case 平坦执行，最后才选择退出码 | 按 correctness、boundary、stability、performance 分阶段执行并短路 | 已修复 |
| 模板 | B baseline 和 gap evidence 模板已经漂移 | 新增 A 模板并重写 B/gap 模板，纳入验证 | 已修复 |
| CPU 测试 | Windows、无私有树环境存在大量假失败 | 修复跨平台夹具和预期跳过逻辑 | 全套 CPU 测试通过 |

## 3. 今天完成的 Phase 12：Baseline 完整角色与采集清单

### 3.1 新增基线 readiness 推导

新增 `musa_operator_eval/evaluator/baseline_readiness.py`，从 baseline 的真实结果
重新推导状态，不信任文件中手写的 `readiness.status`。

A 类强制角色：

1. `upstream_musa`
2. `mudnn_fused`
3. `torch_musa_sdpa`
4. `torch_musa_eager`

B 类在上述角色之外增加：

5. `naive_library_composition`
6. `expert_dispatch`

以下任一情况都会得到 `readiness.status=incomplete`：

- 必需角色未声明；
- 某个隐藏 Case 缺少必需角色的结果行；
- 同一个 Case/角色存在重复结果；
- 必需角色为 `unavailable`；
- 必需角色状态为 `fail`、`error` 或 `unsupported`；
- 存储的 readiness 与结果重新推导出的内容不一致。

### 3.2 接入所有后续门禁

以下工具现在统一拒绝不完整 baseline：

- `candidate_gate.py`
- `record_admission.py`
- `rank.py`
- `validate_schemas.py`

`measure_baseline.py` 可以保留带原因的 `unavailable` 行作为证据，但会将记录标记为
不完整并返回非零状态。这样可以保留真实失败原因，同时避免把“记录过缺失”解释为
“已经完成测量”。

### 3.3 修复 B 类测量接口

审查发现，旧 B 类命令只支持：

- `--naive`
- `--expert`
- 可选的 `--blind`

它无法提供 `upstream_musa`、`mudnn_fused` 和 `torch_musa_sdpa`，因此即使有硬件，
也无法生成需求文档规定的六角色完整 baseline。

现在 B 类同样支持重复传入：

```text
--implementation upstream_musa=<module.py>
--implementation mudnn_fused=<module.py>
--implementation torch_musa_sdpa=<module.py>
```

`torch_musa_eager` 由任务 reference 自动测量，禁止再用其他模块冒充。

### 3.4 防止遗漏隐藏 Case

`record_admission.py` 新增必需的 `--cases` 参数。准入审核会读取私有 manifest 的
完整 Case id 集合，再检查 baseline。

因此，即使有人从 baseline 删除一个难例并同步修改 readiness，准入工具仍会发现
该 Case 的六个角色结果全部缺失并拒绝写入 `admit`。

### 3.5 生成脱敏采集清单

新增：

```text
musa_operator_eval/collection/baseline_collection_plan.json
```

当前统计：

```text
task_count: 11
not_collected: 11
release_blocker: 7
after_admission_blocker: 4
```

其中：

- 6 个 A 类任务和 family-scoped B pilot 是直接 release blocker；
- 4 个 entry-scoped B 任务先受 gap 原因不足阻塞；
- 每个任务都附有硬件测量命令模板；
- 文件不包含隐藏 Case id、shape、seed、golden、测量值或私有错误原因。

## 4. 今天完成的 Phase 13：评测阶段顺序与短路

### 4.1 发现的问题

Python 路径已有按 Case tag 划分阶段的逻辑，但依赖 manifest 中的列表顺序。
compiled 路径则把所有 Case 放在一个循环中运行，结束后才决定退出码。

这会导致：

- correctness 已失败，boundary/performance 仍然运行；
- 动态泛化 Case 在运行时追加到 manifest 末尾后，可能在 perf Case 之后才运行；
- 报告宣称“阶段短路”，实际执行并没有短路。

### 4.2 修正后的规则

两条执行路径现在都固定按照以下顺序运行：

```text
correctness -> boundary -> stability -> performance
```

具体行为：

- manifest 顺序不再决定阶段顺序；
- 同一阶段内的全部 Case 都会执行，便于一次发现同阶段的多个错误；
- 当前阶段存在失败时，后续阶段全部生成显式 `skipped` 行；
- dispatch trace 失败继续使用退出码 41；
- stability 失败继续使用退出码 52；
- 其他失败使用所属阶段的专用退出码；
- compiled 报告新增 `evaluation_stages`；
- summary 中 skipped 不再被重复计入 failed。

该修复不需要回退 Phase 10。运行时随机采样、seed 回放和报告字段保持不变，只调整
泛化 Case 进入正确性阶段的执行位置。

## 5. 今天完成的 Phase 14：控制面收尾

### 5.1 修正文档中的历史完成声明

更新 `CONFORMANCE.md`，不再声称：

- 公开 checkout 已拥有 11 份完整 baseline；
- 旧 maintainer tree 仍能保持原来的 schema 全通过；
- 历史设备比值是当前有效排行榜。

旧设备结果仍可作为诊断记录，但在完成新版 readiness 迁移前不能作为当前准入和
排名证据。

### 5.2 修复模板漂移

新增：

```text
musa_operator_eval/templates/baseline.a.template.json
```

更新：

```text
musa_operator_eval/templates/baseline.b.template.json
musa_operator_eval/templates/private_gap_evidence.template.json
```

改动包括：

- B 模板移除错误的 `reference_model`；
- 补充 `torch_musa_sdpa`；
- A/B 模板都包含可重新推导的 incomplete readiness；
- gap evidence 模板包含至少 3 条 gap、至少 2 类原因、1 条 perf Case；
- 泛化部分只保存运行时采样策略，不保存固定泛化 Case；
- 两份 baseline 模板已经加入 `validate_schemas.py --all` 的发现范围。

### 5.3 修复无硬件测试噪声

- 未挂载 maintainer tree 时，私有资产测试按预期跳过；
- Windows 下允许使用 Python fake runner 测试 compiled ABI；
- POSIX shell 和 ELF 专属测试在不满足平台条件时明确跳过；
- 使用 Python 子进程代替仅 Unix 存在的 `false` 命令；
- 测试临时目录不再硬编码 `/tmp`。

这些调整没有把失败改成 pass，而是把“环境能力不足”与“代码逻辑失败”分开。

## 6. 最终验证证据

### 6.1 全套 MUSA 控制面测试

```text
python -m unittest discover -s musa_operator_eval/tests -q

Ran 531 tests
OK (skipped=51)
```

即 480 项实际通过，51 项因未挂载私有树、未安装 torch/MUSA 或不是 POSIX/ELF
环境而按预期跳过。

测试输出中的 attention-set drift 和不存在 submission 路径属于负向测试主动制造的
错误条件；最终 unittest 状态为 `OK`。

### 6.2 KernelBench 相关检查

```text
test_library_static_checker.py: 46/46 passed
test_dispatch_trace.py:         31/31 passed
```

### 6.3 Schema 与清单检查

```text
validate_schemas.py --all --engine stdlib: 36/36 valid
collect_attention_set.py --check:
  8 entries, 15 measurements, 11 candidates verified
collect_baseline_inventory.py --check: current
audit_agent_package.py: 11 packages passed
compileall: passed
git diff --check: passed
```

## 7. 回退审核结论

本次没有回退 Phase 08–11，原因如下：

- 来源追溯与版本固定仍然成立；
- public smoke release 与 Agent disclosure audit 仍然成立；
- 运行时随机泛化设计仍然成立，仅修正调度顺序；
- B 类两种 gap 原因的硬门槛仍然成立；
- 四个不满足 gap 覆盖的 B 任务继续保持阻塞，没有恢复历史 `admit`。

发生的返工仅限于 Phase 12 自身：补齐 B baseline 输入接口、非通过状态判断和
manifest 完整性复核。

## 8. 当前仍未完成的内容

以下内容不能仅凭本地代码认定完成：

1. 11 个任务的新版完整 baseline 尚未在真实 MUSA 环境采集；
2. 四个 entry-scoped B 任务仍缺第二类真实 gap；
3. compiled runner、真实分派 trace、稳定性和性能需要在 Linux + MUSA 上复验；
4. §4.4 写明“重复轮数、中位数”，而当前 KernelBench 分子和 baseline 分母都使用
   单次 cold-cache trial sequence 的 mean；如需改变，必须同时修改两侧并重测；
5. Python submission 路径仍与 libtorch/ATen 同进程，强隔离只由 compiled path
   满足；
6. evaluator 源码仍在公开仓库中，生产环境应像私有资产一样从受控仓库挂载；
7. `swin_transformer_v2_a_v0` 的合法 shape 空间限制仍构成已记录例外。

## 9. 后续硬件执行顺序

1. 挂载 maintainer tree，重新采集环境快照并确认 snapshot 一致；
2. 优先为四个 blocked B 任务寻找第二类真实 gap；
3. 重新生成并校验 private manifests 和 hidden goldens；
4. 按 baseline collection plan 提供所有真实 ModelNew 实现并测量完整角色；
5. 重新生成 inventory，确认 baseline 为 `complete`；
6. 运行 candidate gate 和带 `--cases` 的 admission recorder；
7. 分别运行 Python/compiled evaluator；
8. 分别生成 A/B 排行并保存设备日志；
9. 在私有树挂载状态下重新执行 schema 和 private-asset tests。

详细命令参见：

- `musa_operator_eval/HARDWARE_RUNBOOK.md`
- `musa_operator_eval/collection/baseline_collection_plan.json`
- `musa_operator_eval/progress/phase_12_baseline_roster_and_collection_inventory.md`
- `musa_operator_eval/progress/phase_13_compiled_stage_short_circuit.md`
- `musa_operator_eval/progress/phase_14_control_plane_closeout_and_hardware_handoff.md`

## 10. 可用于汇报的简短结论

本次工作没有新增虚假的设备成绩，而是完成了评测控制面的闭环：基线角色完整性、
隐藏 Case 完整性、B 类准入、分阶段短路、运行时泛化顺序、模板和跨平台测试均已
机械化验证。当前 531 项控制面测试执行完成，36 份 schema 文档有效，11 个任务
均通过 Agent disclosure audit。剩余任务已经收敛为可执行的硬件清单，且系统会
阻止不完整 baseline、单原因 B 任务和旧格式报告进入发布或排名。

