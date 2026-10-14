# MUSA 算子评测需求追踪矩阵

> 生效日期：2026-10-05  
> 当前规范：`musa_operator_evaluation_normative_spec_v1.md`  
> 原始需求：`musa_operator_evaluation_guide.md`

本矩阵不修改原始需求，负责记录冲突如何裁决、现有实现受什么影响，以及哪些结论仍需硬件。

| 决策 ID | 原始条款 | 冲突或缺口 | 当前规范决策 | 当前实现状态 | 后续动作 | 硬件 |
|---|---|---|---|---|---|---|
| NORM-001 | 全文 | 原始指南、实现说明和后续旧章节没有优先级 | 原始指南冻结；本规范是唯一执行口径；决策和符合性文档从属于本规范 | 本阶段完成 | 所有入口文档链接本规范，旧阶段标记为快照 | 否 |
| NORM-002 | §3 实现说明、§4.6、§4.7 | KernelBench Python 路径与 C++ 裸二进制路径冲突 | 正式路径为 Python `ModelNew`；C++/直接 muDNN 仅 `research_only` | 已符合：3 个 keep 合同均已移除 compiled 正式路径；8 个 research-only 合同被 packager/baseline 隔离 | 保持 packager 和审计回归；legacy compiled 资产只保留诊断角色 | 否 |
| NORM-003 | §2、§4.7 | evaluator 脚本应对 Agent 不可见，但当前公开提交 | 公开规则和 mock，生产 evaluator 私有挂载 | 已符合（软件边界）：release 包验收拒绝 evaluator/private 资产，公开 harness 明确仅为开发镜像，生产要求私有挂载 | 在生产部署流程中继续执行私有挂载验收 | 否 |
| NORM-004 | §2、§4.1、§4.5 | 来源字段要求与“不向 Agent 暴露来源”冲突 | 公开包只保留 opaque `source_id`；完整 provenance 私有 | 已符合：3 个 keep 合同使用 opaque ID 和私有反向映射；8 个 research-only 合同不可发布；normative 泄漏审计通过 | 保持私有 registry 与任务绑定的一致性检查 | 否 |
| NORM-005 | §1、§4.2、§5 | A/B 排他判级与“同一 entry 双轨”冲突 | 以完整语义域判级；一个合同只能是 A 或 B | 已符合：11 个合同排他收口为 A 2 keep/4 research-only、B 1 keep/4 research-only；legacy 配对全部被强制隔离为研究资产 | 保持 lifecycle、packager、inventory 和排名回归 | 否 |
| NORM-006 | §2、§5 步骤 3 | 公开 reference 与禁止上游源码、独立 CPU reference 冲突 | 使用独立 CPU 高精度 reference；禁止上游源码副本 | 公开软件侧已符合：3 个 keep 合同均有独立 NumPy FP64 oracle 和显式状态策略；2 个 A oracle 已与公开执行模型交叉检查，公开 golden 两轮生成一致；research-only 副本不可发布 | 私有资产挂载后重生并核验 hidden golden | 否 |
| NORM-007 | §4.3 | Case 的属性、分布、容差和性能协议未被完整强制 | 每个 Case 强制 attributes、distribution、tolerance_profile；协议显式冻结 | 已符合：11 个 task contract 和 35 个公开 Case 已迁移，Schema 强制字段与正式计时协议 | 保持 schema 全量验证；Phase 25 使用同一合同重测 | 否 |
| NORM-008 | §4.1、§4.3 | 三个容差字段被合并成一个阈值 | 分别执行 atol、rtol、max_abs_error，报告三个实际阈值 | 已符合：Python evaluator、binary 诊断、baseline collector 和 report schema 均分别执行并记录三个阈值 | 保持 framework floor 不得覆盖独立 max_abs_error 的回归 | 否 |
| NORM-009 | §3 实现说明、§4.4 | KernelBench mean 与原文重复轮次 median 冲突 | 正式采用 KernelBench 同源 mean；分位数作为诊断 | 软件协议已冻结；warmup、测量次数、首次舍弃、cache 策略、mean 及诊断统计已入 Schema 和报告；旧硬件记录需重测 | Phase 25 按冻结协议重测全部有效 baseline | 是 |
| NORM-010 | §4.3、§4.4 | gap Case 中库失败与 baseline“全部通过”不可同时成立 | 拆分 evidence_complete、scoreable、release_ready | 已符合：Schema、readiness derivation、inventory、gate 和 admission 使用三个独立状态 | 保持非通过对比行可见，但不得把其当作可用 latency | 否 |
| NORM-011 | §4.4、§5 | 原文未定义跨 Case 聚合和缺分母处理 | 使用 geomean；正式评分要求全部评分 Case 有有效分母 | 已符合：ranker 对任一缺失/失败分母整任务 fail closed；A 逐 Case 分母与 B expert 分母已冻结 | Phase 25 采集时为每个评分 Case 生成可审计分母选择 | 是 |
| NORM-012 | §1、§4.3 | 不合格任务是继续排队、重设计还是退役不明确 | 只允许 keep、redesign_new_id、retire、research_only | 已符合：11 个合同均已判定；共 3 个 keep、8 个 research-only；Schema、packager 和采集清单执行该状态 | 保持回归测试；任何语义扩展使用新 task id | 是 |

## 当前发布判断

- 软件控制面和研究资产可以继续维护。
- Phase 22～24 的仓库内软件控制面已完成。
- 当前 3 个 keep 合同均为 `hardware_pending`，8 个合同为 `research_only`；仍没有任务是 `release_ready`。
- Phase 25 的公开 oracle/golden 软件重放及模型状态合同已完成；私有 hidden golden 仍待维护者资产挂载。
- 下一步先完成私有软件重放，再只为 3 个 `hardware_pending` 任务采集新的正式硬件 baseline。

