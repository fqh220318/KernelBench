# 阶段 14：控制面收尾与硬件交接

## 本阶段结果

完成一次从需求文档到当前实现的收尾审计。无硬件环境中还能可靠完成的控制面
工作已补齐；剩余项均需要真实私有资产、MUSA 设备测量，或需要改变部署边界，
没有用占位数据伪造完成状态。

本阶段修复了三类收尾问题：

1. 文档中的旧完成声明：`CONFORMANCE.md` 不再声称公开 checkout 已有 11 份完整
   baseline，也不再把旧设备排名写成当前排行榜；当前真实状态与脱敏 inventory
   保持一致。
2. 已漂移的模板：旧 B baseline 模板缺 `torch_musa_sdpa`、仍含
   `reference_model`，也没有 readiness；gap evidence 模板只有一条不完整记录。
   现在 A/B baseline 模板都是 schema-valid 的显式 `incomplete` 记录，gap 模板
   提供 3 条、2 类原因、perf Case 和运行时泛化策略的完整结构起点。
3. 无硬件验证噪声：公开 checkout 缺少私有树时按预期跳过；Windows 可运行
   Python fake compiled runner；POSIX shell/ELF 专属夹具明确跳过；失败命令与
   临时目录测试改为跨平台写法。全套 CPU 测试由红转绿。

## 当前可验证状态

```text
musa_operator_eval/tests：531 项执行完成，480 项通过，51 项按环境能力跳过
KernelBench library static checker：46/46 通过
KernelBench dispatch trace：31/31 通过
Repository schema validation：36/36 文档有效
Attention set check：8 entries、15 measurements、11 candidates 通过
Baseline inventory check：通过
Agent package disclosure audit：11 packages 通过
compileall 与 git diff --check：通过
```

其中跳过项有明确边界：未挂载 maintainer tree、未安装 torch/MUSA，或当前主机
不是 POSIX/ELF 构建环境。它们不是 pass，也没有计入真实设备完成度。

## 硬件交接顺序

在 Linux + MUSA 环境挂载私有树后，按以下顺序执行，避免无效重测：

1. 运行 environment collector，确认 snapshot 与任务合同一致。
2. 先处理四个 entry-scoped B 任务的第二类真实 gap；无法找到时保持
   `blocked_gap_coverage`，不采信历史 admit 数值。
3. 重新生成/校验 private manifests 与隐藏 goldens。
4. 按 `collection/baseline_collection_plan.json` 提供所有非 eager 的真实
   ModelNew 实现，采集 4 角色 A baseline 和 6 角色 B baseline。
5. 重新运行 inventory；只有 `complete` 的记录才进入 gate/rank。
6. 对 B 类运行 candidate gate，再用包含 `--cases` 的 `record_admission.py`
   写入准入。
7. 分别运行 Python 与 compiled evaluator，检查 `evaluation_stages`、分派 trace、
   稳定性和性能；最后分别生成 A/B 排行。
8. 在私有树挂载状态下重新运行 schema 与 private-asset tests，保存终端输出作为
   设备证据。

具体命令已写入 `HARDWARE_RUNBOOK.md` 和每个任务的 inventory command template。

## 仍不能认定完成的需求

- 11 个任务的新版完整 baseline 尚未在本 checkout 采集；inventory 为
  `not_collected`。
- 四个 entry-scoped B 任务仍缺第二类实测 gap，不能发布、评分或排名。
- §4.4 的“重复轮数、中位数”与 KernelBench 当前 one-sequence + mean 协议仍有
  明确偏差；必须同步修改 submission 与 baseline 两侧并在硬件上重测。
- Python submission 路径仍在含 libtorch/ATen 的 evaluator 进程内运行；§4.6 的
  强隔离只由 compiled path 满足。
- evaluator 源码仍提交在公共仓库中；生产环境需要像 `private/` 一样从受控仓库
  或制品存储挂载，才能严格满足 Agent 不可见。
- `swin_transformer_v2_a_v0` 的合法 shape 空间限制使隐藏 shape range 与公开范围
  完全分离仍是已记录例外。

这些项目需要真实测量、任务范围调整或部署权限边界决策，继续在本地填字段会
制造虚假证据，因此本阶段不自动“补齐”。

