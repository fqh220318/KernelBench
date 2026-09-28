# MUSA operator evaluation progress

本目录记录当前合规分支中已经完成并验证的阶段。阶段文档是当时的交接快照；
当前总状态以 Phase 16、`CONFORMANCE.md`、候选算子池和脱敏 collection inventory 为准。

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

当前最直接的机器可读入口：

- `../collection/baseline_collection_plan.json`：11 个任务的 baseline 待办与命令模板；
- `../CONFORMANCE.md`：逐条需求符合性及明确偏差；
- `../HARDWARE_RUNBOOK.md`：Linux + MUSA 上的执行顺序和命令。
- `../sources/operator_candidate_pool.json`：带来源 Commit、许可证、A/B 初筛和下一证据的机器可读候选池；
- `../sets/candidates/CANDIDATE_POOL.md`：由候选池生成的维护者评审索引。
- `../sources/attention_operator_candidate_pool.json`：Attention backward、varlen、KV cache、MLA、稀疏、线性和量化方向的机器可读候选池；
- `../sets/candidates/ATTENTION_CANDIDATE_POOL.md`：Attention 扩充候选的生成式维护者索引。

