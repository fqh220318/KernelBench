# 阶段 11：B 级缺口覆盖与准入硬门槛

## 本阶段结果

需求文档 §4.3 要求 B 级隐藏集至少包含 3 条库缺口，并覆盖至少 2 类原因。此前实现把“记录为什么只能达到一类”当成了豁免，导致四个只有 `unsupported_shape` 的任务仍标记为 `admit`。本阶段取消了该豁免，并保留真实历史测量而不伪造第二类缺口。

当前状态：

- `sdpa_forward_b_v0`（family-scoped pilot）有 `unsupported_shape` 与 `semantic_mismatch` 两类，保持 `measured/admit`。
- `scaled_dot_product_attention_b_v0`、`vision_attention_b_v0`、`mingpt_causal_attention_b_v0`、`mingpt_block_b_v0` 只有一类实测缺口，改为 `blocked_gap_coverage`，移除 `decision: admit`。
- 四个任务原有几何平均性能值仍保留，明确作为历史测量，不能作为当前准入结论。
- Attention set 对这四项增加 `scoreable: false` 与具体 blocker，生成的 `SET.md` 显示为“阻塞”。

## 机械执行点

以下路径现在都会执行同一硬门槛：B 级必须为 `measured/admit`，且至少有两类缺口原因。

1. `make_private_manifest.py`：一类原因无条件拒绝，说明文字不能豁免。
2. `record_admission.py`：一类原因不能写入 `admit`。
3. `package_agent_task.py`：阻塞任务不能生成 Agent release。
4. `run_task.py` 与 `run_binary.py`：阻塞任务在生成 Case 或访问设备之前退出，返回 `EXIT_TASK_NOT_ADMITTED`。
5. `rank.py`：B 报告必须携带本次任务的 `task_admission=measured/admit` 快照；阻塞或旧格式报告不进入排行榜。
6. `task_contract.schema.json`：一类原因只允许作为显式 `blocked_gap_coverage` 候选存在，不能声明 admitted。

## 验证结果

```text
缺口清单、schema、录取、发布、Python/编译入口定向测试：45 项通过，1 项因无 torch 跳过
LibraryPolicyTests + GapReasonVocabularyTests：9/9 通过
Attention set + package/ledger 一致性：61/61 通过
Rank tests：10/10 通过
Schema tests：55 项通过，1 项因未安装可选 jsonschema 跳过
Repository schema validation：34/34 文档有效
Attention collection：8 entries、15 measurements、11 candidates 验证通过
py_compile 与 git diff --check：通过
```

## 后续解除 blocker 的条件

不能通过改字段人工解除。必须在固定 MUSA 环境中找到并记录第二类真实缺口，重新生成包含至少两类原因的私有隐藏集，再运行 baseline、candidate gate 与 `record_admission.py`。如果当前语义合同确实无法产生第二类缺口，则这些 entry-scoped B 任务应继续不计分，只有 pilot 进入 B 级发布和排行榜。

