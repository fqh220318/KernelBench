# Phase 21：规范仲裁与执行口径冻结

日期：2026-10-05

## 本阶段解决的问题

原始需求同时保留了 2026-09-20 的 KernelBench Python 实现说明和更早的 C++ 裸二进制/
独立 evaluator 要求；来源字段、A/B 配对、ATen、计时和 Agent 可见性也存在互相冲突的
条款。继续补实现或采集硬件数据会让同一结果在两套规则下得到不同结论。

## 本阶段产物

- `docs/musa_operator_evaluation_normative_spec_v1.md`：当前唯一执行口径；
- `docs/musa_operator_evaluation_requirement_traceability_matrix.md`：12 条规范决策及迁移状态；
- `tests/test_normative_spec.py`：保护原始指南并校验决策覆盖；
- 总入口文档的 Phase 21 状态提示和历史资料优先级说明。

原始 `docs/musa_operator_evaluation_guide.md` 未改写。其规范化换行 SHA-256 为：

```text
9f87707ed0379c15080bbdf83d4868be8a353fc7818489173224fdd8fdf454c5
```

## 已冻结的关键结论

1. 正式评分统一使用 KernelBench Python `ModelNew`。
2. B 类正式路径经 `torch_musa` 使用库；C++/直接 muDNN 降为 `research_only`。
3. 公开任务只保留 opaque `source_id`，完整 provenance 私有。
4. A/B 按完整语义域排他判级，不再为了双轨强制配对。
5. 正式任务使用独立 CPU 高精度 reference。
6. 采用 KernelBench mean 作为正式延迟，所有 ratio 必须同协议。
7. 容差保持 atol、rtol、max_abs_error 三个独立判据。
8. baseline 拆成 evidence_complete、scoreable、release_ready。
9. geomean 只在全部评分 Case 都有有效分母时生成。

## 当前边界

本阶段只冻结规范，没有宣称现有 11 个任务已经迁移完成，也没有生成新的硬件结论。当前任务
在完成 Phase 22～24 前都不是 `release_ready`。

## 下一阶段

Phase 22 处理不需要硬件的发布边界和 reference 迁移：

1. 拆分公开 task contract 与私有 provenance；
2. 清除 Agent 包中的上游标识和源码副本；
3. 把 compiled/direct-muDNN 路径从正式评分降为研究诊断；
4. 为后续独立 CPU reference 建立统一接口和迁移清单。

