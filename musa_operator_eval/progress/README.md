# MUSA operator evaluation progress

本目录记录当前合规分支中已经完成并验证的阶段。阶段文档是当时的交接快照；
当前总状态以 Phase 24、可执行规范、需求追踪矩阵和脱敏 collection inventory 为准。

| 阶段 | 内容 | 当前状态 |
|---|---|---|
| [08](phase_08_provenance_and_integration.md) | 来源追溯与隔离工作树集成 | 已完成 |
| [09](phase_09_public_smoke_release.md) | Agent 可见 public smoke release | 已完成 |
| [10](phase_10_runtime_generalization_probe.md) | 运行时随机泛化探针与回放 | 已完成 |
| [11](phase_11_b_tier_gap_admission.md) | B 类 ≥2 缺口原因硬门槛 | 控制面完成；4 个任务保持阻塞 |
| [12](phase_12_baseline_roster_and_collection_inventory.md) | §4.4 完整角色与脱敏采集清单 | 控制面完成；真实采集待硬件 |
| [13](phase_13_compiled_stage_short_circuit.md) | compiled 分阶段执行与短路 | 已完成，待设备复验 |
| [14](phase_14_control_plane_closeout_and_hardware_handoff.md) | 全量 CPU 验证、模板和硬件交接 | 已完成 |
| [15](phase_15_operator_candidate_pool_and_family_expansion.md) | 跨项目候选算子池与新 family 预研 | 本地研究与控制面完成；硬件筛选待执行 |
| [16](phase_16_attention_candidate_expansion.md) | Attention 算子级候选扩充与来源核验 | 本地研究与控制面完成；能力探针待硬件 |
| [17](phase_17_no_hardware_compliance_freeze.md) | baseline 完整性/可计分性拆分与实现决策冻结 | 第一批完成；548 项全量测试通过/按环境跳过 |
| [18](phase_18_b_tier_disposition_and_collection_queue.md) | 冻结四个单一 gap B 合同并收敛硬件队列 | 7 个 active blocker；4 个当前合同不排期 |
| [19](phase_19_swin_hidden_shape_partition.md) | Swin 公开/隐藏 shape 分域 | public B=1/2/3；B=4 保留给 hidden |
| [20](phase_20_attention_p0_cpu_contracts.md) | 前三个 Attention P0 的 CPU 合同与探针矩阵 | CPU 语义冻结；硬件 cell 全部未运行 |
| [21](phase_21_normative_spec_arbitration.md) | 原始需求冲突仲裁与唯一执行口径冻结 | 已完成；现有 11 个任务进入迁移状态，尚未 release-ready |
| [22](phase_22_release_boundary_gate.md) | 公开/私有边界、A/B lifecycle 与 reference 迁移 | 静态迁移完成；3 个 keep/hardware-pending，8 个 research-only；待软件数值重放与硬件证据 |
| [23](phase_23_contract_scoring_closure.md) | Case、三重容差、计时协议、baseline 三状态与全 Case 计分闭环 | 无硬件控制面已完成；正式 baseline 待 Phase 25 重测 |
| [24](phase_24_production_boundary_and_legacy_isolation.md) | 私有生产 evaluator 边界、A/B 排他分类与 legacy 研究资产隔离 | 仓库内边界和验收已完成；生产私有挂载由部署流程执行 |
| [25-1](phase_25_software_replay_environment.md) | 无硬件软件重放环境与 oracle 验收 | 环境完成；当时发现的 2 项 model_state 阻塞已由 Phase 25-2 收口 |
| [25-2](phase_25_model_state_and_public_golden_replay.md) | 模型状态合同与公开 golden 软件重放 | 已完成；3 个 keep 状态策略已冻结，ReLU/Swin 公开 Case 两轮生成 SHA-256 一致；私有资产与硬件仍待执行 |

当前最直接的机器可读入口：

- `../collection/baseline_collection_plan.json`：11 个任务的 baseline 待办与命令模板；
- `../CONFORMANCE.md`：逐条需求符合性及明确偏差；
- `../HARDWARE_RUNBOOK.md`：Linux + MUSA 上的执行顺序和命令。
- `../sources/operator_candidate_pool.json`：带来源 Commit、许可证、A/B 初筛和下一证据的机器可读候选池；
- `../sets/candidates/CANDIDATE_POOL.md`：由候选池生成的维护者评审索引。
- `../sources/attention_operator_candidate_pool.json`：Attention backward、varlen、KV cache、MLA、稀疏、线性和量化方向的机器可读候选池；
- `../sets/candidates/ATTENTION_CANDIDATE_POOL.md`：Attention 扩充候选的生成式维护者索引。
- `../research/attention_p0/`：前三个 Attention P0 的 CPU reference、语义表与未运行探针矩阵。
- `../IMPLEMENTATION_DECISIONS.md`：不改写原始需求文档的实现决策层与显式偏差。
- `../../docs/musa_operator_evaluation_normative_spec_v1.md`：Phase 21 后唯一执行规范；
- `../../docs/musa_operator_evaluation_requirement_traceability_matrix.md`：12 条决策的来源、现状、动作和硬件依赖。

