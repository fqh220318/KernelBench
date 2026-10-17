# 阶段 18：B 轨处置与硬件采集队列收敛

## 本阶段结果

将“已有任务包”和“当前应上硬件采集的任务”分离。正式任务包仍保留 6 个 A 类、
5 个 B 类，共 11 个；但四个 entry-scoped B 包已经有充分证据表明当前语义只能产生
`unsupported_shape` 一类 gap，无法满足需求文档要求的至少两类原因。

这些包继续保留 `blocked_gap_coverage`、历史测量和复核依据，但不再列为
`after_admission_blocker` 硬件待办。采集计划现在标记为：

```text
release_blocker：7
inactive_current_contract：4

not_collected：7
not_scheduled：4
```

七个 active release blocker 为六个 A 类任务和 family-scoped
`sdpa_forward_b_v0` pilot。

## 为什么不是继续寻找第二类 gap

四个任务的现有合同已经逐类记录：

- `semantic_mismatch` 在当前固定 scale/mask/window 语义中不可达；
- `layout_mismatch` 已被历史探针反驳；
- `multi_operator_required` 是整个任务的静态属性，不能区分普通 Case 与 gap Case；
- 剩余可复现边界只有 `unsupported_shape`。

继续扫描相同语义只会重复第一类 gap，或者诱导人为给 Case 贴上不真实的第二类标签。

## 后续处置规则

1. 历史任务包不删除，继续作为失败准入的可审计证据；
2. 不为当前合同采集新版完整 baseline，也不进入 release/rank；
3. 若决定扩展语义，创建新的版本化任务 ID，重新冻结 A/B 共用语义、Case 和来源；
4. 新任务从 `pending_device_measurement` 开始，不能继承旧任务的 admission；
5. 如果没有值得扩展的真实语义，正式撤销该 entry 的 B 轨，只保留 A 轨。

## baseline 实现盘点

当前公开工作树可自动解析 10 个 entry-scoped 包的 `upstream_musa` 实现。没有发现可
直接复用的 task-level `mudnn_fused` 或 `torch_musa_sdpa` ModelNew 模块；已有内容是
来源记录、能力探针和历史设备说明，不能冒充可执行 baseline。

因此 active 队列剩余实现准备为：

- 六个 A 任务：每个已解析 `upstream_musa`，仍需处理 `mudnn_fused` 和
  `torch_musa_sdpa`；
- B pilot：仍需提供 `upstream_musa`、`mudnn_fused`、`torch_musa_sdpa`，以及私有
  naive/expert 实现；
- 无法构建的比较角色必须在设备侧以带原因的非通过记录保存，不能在本地预填。

## 验证

本阶段新增测试保证 `blocked_gap_coverage` 任务得到
`inactive_current_contract/not_scheduled`，并要求下一动作固定为新 ID 重设计或撤销
B 轨。生成后的脱敏 inventory 不包含隐藏 Case、shape、seed、失败原因或测量值。

