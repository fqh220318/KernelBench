# Phase 22（进行中）：规范发布硬门槛

日期：2026-10-05

## 已完成的发布门禁

Phase 21 已确认当前 11 个任务仍携带旧 provenance、上游 reference 和 compiled Starter，不能在
迁移期间继续被 release packager 当成正式任务发布。本阶段先关闭这个风险窗口：

- task Schema 同时识别 legacy `source` 和新 opaque `source_id`；
- `publication.release_status=hardware_pending/release_ready` 时强制使用规范 1.0.0、独立 CPU
  reference、Python submission、`torch_musa` 正式调用面，并禁止正式 compiled Starter；
- disclosure audit 新增 `--normative` 模式，禁止 provenance、上游项目名、非 opaque source id、
  上游条目编号、逐字上游 problem、非独立 reference 和 compiled Starter；
- release packager 永远启用 normative audit；
- release packager 只接受 `release_ready`，因此静态迁移完成但缺硬件证据的任务仍然 fail closed。

普通 audit 暂时保留 legacy 模式，用于检查和迁移现有 11 个包；它通过不代表可以发布。

## 首个静态迁移试点

`sdpa_forward_b_v0` 已完成不依赖硬件的迁移并进入 `hardware_pending`：

- 公开 `source` 和上游 coverage 映射已移除，只保留随机 opaque `source_id`；
- 完整来源反向映射写入私有 `provenance_registry.json`，该文件受 `.gitignore` 保护并由私有仓库保存；
- 已有 blocked NumPy FP64 oracle 被确认为独立 CPU reference；
- `starter/cpp` 和正式 link-whitelist 声明已从 Agent 包移除；
- B 类合同显式声明 `formal_call_surface=torch_musa`，静态检查拒绝直接后端库 import、
  `torch.ops` 旁路和直接动态加载；
- PROMPT、task note 和公开文件已清除上游项目名与上游条目编号；
- normative package audit 已通过，release packager 仍按预期拒绝该任务。

这里新增 `hardware_pending` 是为了表达“静态迁移完成但尚无最终硬件证据”，避免把软件整改完成
误报成 `release_ready`。该状态已经写入规范、Schema 和测试，不是隐含口径。

## 第二批：其余 B 类合同处置

其余四个 entry-scoped B 合同均已有相同的充分证据：历史 geomean 超过 1.3，但当前语义域
只能产生 `unsupported_shape` 一个 gap 类别，无法满足至少两个类别的硬门槛。因此本阶段没有
把它们伪迁移成 `hardware_pending`，而是统一固化为：

- `admission.decision=reject`；
- `lifecycle.decision=research_only`；
- `lifecycle.successor_policy=new_task_id_required`；
- baseline inventory 为 `inactive_current_contract / not_scheduled`；
- release packager 在 admission 检查前即按 lifecycle 拒绝。

涉及的合同为 `scaled_dot_product_attention_b_v0`、`vision_attention_b_v0`、
`mingpt_causal_attention_b_v0` 和 `mingpt_block_b_v0`。历史测量继续保留为失败准入证据；如需
扩展语义寻找第二类真实 gap，必须创建新 ID，不能继承旧 admission。

## A 类重新判级与首个迁移试点

六个旧 A 合同已按“完整语义域内不得存在一次等价库调用”重新判级：

- `swin_transformer_v2_a_v0` 与 `relu_causal_attention_a_v0` 保留 `keep`；前者包含 cosine
  normalization、learned logit scale 与 relative-position bias，后者以 ReLU 替代 softmax，
  均不存在一次等价库调用；
- `scaled_dot_product_attention_a_v0`、`vision_attention_a_v0`、
  `mingpt_causal_attention_a_v0`、`mingpt_block_a_v0` 改为 `research_only`。最后一个虽然
  外层是 transformer block，但其声明的 kernel scope 仅为标准 causal softmax attention；
  四者如需扩展语义重新进入正式评测，必须使用新 task id。

`relu_causal_attention_a_v0` 已完成首个 A 类无硬件静态迁移并进入 `hardware_pending`：

- public provenance 改为随机 opaque `source_id`，完整 entry/source 反向映射进入私有 registry；
- 上游源码副本替换为独立编写的 public execution model；
- 新增 NumPy FP64 CPU oracle，框架只用于按 seed 重建公开 ABI 声明的 projection 参数；
- 正式任务移除 compiled Starter，并通过 normative disclosure audit；
- release packager 按预期继续拒绝该任务，直到最终 MUSA 正确性和计时证据完成。

`swin_transformer_v2_a_v0` 随后完成第二个 A 类静态迁移并进入 `hardware_pending`：

- public provenance 改为随机 opaque `source_id`，完整 entry/source 反向映射进入私有 registry；
- 逐字上游副本替换为独立组织的 public torch execution model；
- 新增完整 NumPy FP64 CPU oracle，覆盖 patch embedding、12 个 window block、patch merging、
  pooling 与 classifier，而不是只验证其中一个 attention 切片；
- case 生成器新增受 schema 约束的 `model_state` oracle context：框架只负责按 case seed
  初始化公开构造器参数，全部 forward 数值计算由 NumPy oracle 完成；
- 修正 semantics 中把 `(B, 1000)` 分类输出误写为 BCHW 的旧错误；
- 移除 compiled Starter，并通过 normative disclosure audit。

## Golden 权威路径修正

迁移 A 试点前发现，原 Python 评测只验证 golden 完整性与可重算性，实际正确性仍比较
`problem.py:Model` 与提交结果。这样即使存在独立 oracle，它也没有裁决权。当前正式 case
路径已改为读取生成后的固定输入和 `outputs.required` 对应的 golden 张量，并传入 device
evaluator；报告记录 `correctness_reference=independent_golden`。不属于提交 ABI 的诊断 golden
（例如 SDPA 的 LSE）仍保留用于 oracle 自检，但不要求提交返回。

## 尚未完成

- 8 个 research-only 合同仍保留 legacy provenance/reference/compiled 研究资产，但已被
  release packager 与 baseline 队列隔离；
- ReLU 与 Swin oracle 和 public execution model 的数值交叉核对在当前无 torch 环境中只能运行
  NumPy 单测；需在有 torch 的软件环境先生成/重放全部 public 与 hidden golden；
- 三个 `hardware_pending` 任务仍需在最终合同下重新取得正确性、分派轨迹（仅 B）和计时
  证据，才能提升为 `release_ready`。

这些项目完成前，Phase 22 保持“进行中”，Phase 25 硬件 baseline 不启动。
