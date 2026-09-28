# MUSA 两轮算子收集成果汇总（2026-09-27）

## 1. 文档用途

本文合并总结以下两轮算子收集工作：

1. 第一轮：建立跨 family 的通用候选算子池，并扩展 Attention 之外的新 family；
2. 第二轮：继续扩充 Attention 算子，覆盖训练、KV cache、MLA、稀疏、线性状态和量化方向。

本文面向组会汇报、后续硬件实验排期和 GitHub 提交说明。逐候选的机器可读事实以
以下两个文件为准：

- [`musa_operator_eval/sources/operator_candidate_pool.json`](../musa_operator_eval/sources/operator_candidate_pool.json)
- [`musa_operator_eval/sources/attention_operator_candidate_pool.json`](../musa_operator_eval/sources/attention_operator_candidate_pool.json)

逐算子特征、来源和推进状态可在配套工作簿中筛选与排序：

- [`musa_operator_candidate_inventory_2026-09-27.xlsx`](../outputs/musa_operator_collection_20260927/musa_operator_candidate_inventory_2026-09-27.xlsx)

当前工作位于分支 `feat/musa-operator-evaluation-compliance` 的隔离 worktree 中，尚未
由本文档创建 commit 或上传 GitHub。

## 2. 两轮工作的总体结果

两轮共形成 **18 条来源记录、11 个不重复上游仓库、129 条源码或测试证据路径、
12 个候选 family、35 个候选算子**。

| 指标 | 第一轮：跨 family | 第二轮：Attention | 合计 |
|---|---:|---:|---:|
| 来源记录 | 9 | 9 | 18 |
| 源码/测试证据路径 | 66 | 63 | 129 |
| family | 6 | 6 | 12 |
| 候选算子 | 19 | 16 | 35 |
| `shortlisted` | 12 | 11 | 23 |
| `needs_probe` | 5 | 4 | 9 |
| `deferred` | 2 | 1 | 3 |
| `hardware_validated` | 0 | 0 | 0 |

候选优先级分布为：

| 优先级 | 数量 | 解释 |
|---|---:|---|
| P0 | 16 | 第一批语义冻结或设备能力探针对象 |
| P1 | 15 | 在 P0 之后推进，或需要先补充边界证据 |
| P2 | 4 | 当前条件下暂缓，保留研究线索 |

其中有 **15 个候选同时满足 P0 和 `shortlisted`**。它们是后续真实硬件阶段最直接的
实验入口。

## 3. 第一轮成果：跨 Family 候选池

第一轮从本地 KernelBench 归档、Moore Threads 官方仓库和可信社区项目中整理了
6 个新 family、19 个候选算子。

| family | 候选数 | 入围 | 待探测 | 暂缓 | P0 入围 |
|---|---:|---:|---:|---:|---:|
| `gemm_linear` | 4 | 3 | 1 | 0 | 2 |
| `convolution` | 4 | 3 | 0 | 1 | 3 |
| `normalization` | 3 | 1 | 2 | 0 | 1 |
| `reduction_scan_selection` | 3 | 2 | 0 | 1 | 0 |
| `moe_routing` | 2 | 2 | 0 | 0 | 2 |
| `transformer_fusions` | 3 | 1 | 2 | 0 | 0 |
| **合计** | **19** | **12** | **5** | **2** | **8** |

这一轮的主要价值不是罗列普通算子名称，而是把候选按可共享的语义契约归类，并记录
动态 shape、layout、stride、量化规则、融合顺序、潜在库缺口和后续证据需求。

### 3.1 第一轮 P0 入围候选

| 候选 ID | 名称 | family | 主要研究点 |
|---|---|---|---|
| `batched_grouped_gemm` | Batched/Grouped/Ragged GEMM | `gemm_linear` | grouped/ragged offsets、batch broadcast 与分派边界 |
| `quantized_fp8_w4a8_gemm` | FP8/W4A8 Quantized GEMM | `gemm_linear` | scale 粒度、K 对齐、layout 与量化格式 |
| `conv2d_shape_matrix` | Conv2D Shape/Layout 分派矩阵 | `convolution` | layout、groups、stride、dilation 和非对齐 shape |
| `depthwise_pointwise_conv2d` | Depthwise/Pointwise/Separable Conv2D | `convolution` | 单算子、组合路径与真正融合路径的区分 |
| `conv_transpose_1d_2d` | ConvTranspose1D/2D 动态参数分派 | `convolution` | output padding、groups、dilation 与升维路径 |
| `fused_add_rmsnorm` | Residual Add + RMSNorm 融合 | `normalization` | 残差写回、epsilon、累加 dtype 与别名规则 |
| `grouped_topk_router` | MoE Grouped TopK Router | `moe_routing` | group 限制、tie-break、权重归一化与 token 路由 |
| `mixed_dtype_moe_gemm` | Mixed-dtype MoE Expert GEMM | `moe_routing` | ragged expert、量化 scale 和混合精度组合 |

### 3.2 第一轮结论

- 普通 GEMM、Conv、RMSNorm、TopK 等已有等价库调用，不能通过简单禁用库调用伪装为
  A 类题；
- 融合、ragged、量化、动态布局和运行时分派方向更符合需求文档对区分度的要求；
- B 类候选只记录 gap 假设，仍需真实设备证明至少 3 条 gap、至少 2 类原因以及
  `naive/expert >= 1.3`；
- `conv3d_transpose3d` 和 `segmented_reduction` 因现阶段证据或优先级不足而暂缓。

## 4. 第二轮成果：Attention 专项扩充

第二轮没有直接修改现有 8 个 Attention set entry，而是新增独立的、维护者可见的
Attention 算子级候选池。这样可以避免把“找到上游源码”误写成“已经具备 reference、
Case、baseline 和准入状态”。

| family | 候选数 | 入围 | 待探测 | 暂缓 | P0 入围 |
|---|---:|---:|---:|---:|---:|
| `attention_dense_training` | 4 | 3 | 1 | 0 | 3 |
| `attention_kv_cache` | 3 | 2 | 1 | 0 | 2 |
| `attention_mla` | 3 | 3 | 0 | 0 | 2 |
| `attention_sparse` | 2 | 1 | 1 | 0 | 0 |
| `attention_linear_state` | 2 | 1 | 0 | 1 | 0 |
| `attention_quantized` | 2 | 1 | 1 | 0 | 0 |
| **合计** | **16** | **11** | **4** | **1** | **7** |

### 4.1 第二轮 P0 入围候选

| 候选 ID | 名称 | family | 主要研究点 |
|---|---|---|---|
| `attention_sdpa_backward_dynamic` | 动态 Shape SDPA/FlashAttention Backward | `attention_dense_training` | forward/backward 状态、梯度语义和支持矩阵 |
| `attention_varlen_packed_forward` | Varlen/Packed MHA-GQA Forward | `attention_dense_training` | cu_seqlens、GQA 和非等长 causal |
| `attention_per_sequence_causal_dispatch` | Per-sequence Dynamic Causal Attention Dispatch | `attention_dense_training` | 标量 causal 与逐序列 causal 的真实分派边界 |
| `paged_kv_attention_decode` | Paged KV-cache MHA/GQA Decode | `attention_kv_cache` | page table、page size、cache layout 和 GQA 比例 |
| `kv_cache_append_rope_attention` | KV-cache Append + RoPE + Attention | `attention_kv_cache` | cache 副作用、position、RoPE 与写后读顺序 |
| `dense_mla_decode` | Dense MLA Paged Decode | `attention_mla` | latent/rope 维度、scheduler metadata 与 paged decode |
| `sparse_mla_prefill_decode` | Sparse MLA Prefill/Decode | `attention_mla` | top-k indices、无效索引、prefill/decode 与稀疏边界 |

### 4.2 第二轮结论

- Attention 扩展已经从 dense forward 延伸到 backward、varlen、KV cache、MLA、
  sparse、linear/stateful 和 quantized 路径；
- per-sequence causal 已发现具体的上游 backend 分派线索，适合优先做 B 类能力探针；
- paged decode、dense MLA、FP8 FMHA、KDA 等已有专用融合入口，普通等价任务应优先
  作为 B 类集成与分派评测，而不是 A 类重写题；
- MLA indexer、block sparse 等复合语义可以继续研究 A 类，但必须先冻结独立数学定义、
  CPU reference 和边界行为；
- `linear_attention_forward_backward` 因算法定义过宽且缺少统一 MUSA 库入口而暂缓。

## 5. 来源与可信度

两轮共记录 18 条来源记录，对应 11 个不重复仓库：

| 来源 | 类型 | 两轮中的主要用途 |
|---|---|---|
| [KernelBench](https://github.com/Young-Lord/KernelBench) | 本地可复核仓库 | 现有题目、历史实现、Attention set 和历史测量线索 |
| [torch_musa](https://github.com/MooreThreads/torch_musa) | 厂商官方 | 框架算子、SDPA、benchmark、单元测试和后端选择 |
| [MUTLASS](https://github.com/MooreThreads/mutlass) | 厂商官方 | GEMM、FP8、卷积、FMHA、paged FMHA 和 MLA |
| [MATE](https://github.com/MooreThreads/mate) | 厂商官方 | GEMM、MoE、Norm、FMHA、KV cache、MLA、KDA 和量化 Attention |
| [MT-flashMLA](https://github.com/MooreThreads/MT-flashMLA) | 厂商官方 | MP31/S5000 MLA decode 与 metadata 路径 |
| [vLLM-MUSA](https://github.com/MooreThreads/vllm-musa) | 厂商官方 | 推理分派、MoE、RoPE、KV cache、MLA 和 sparse Attention |
| [Paddle-MUSA](https://github.com/MooreThreads/paddle_musa) | 厂商官方 | FlashAttention、梯度和 KV-cache 集成 |
| [ONNX Runtime MUSA](https://github.com/MooreThreads/onnxruntime-musa) | 厂商官方 | 算子覆盖、融合规则和运行时 kernel |
| [MT-TransformerEngine](https://github.com/MooreThreads/MT-TransformerEngine) | 厂商官方 | FP8、GEMM、RMSNorm 和 Transformer 融合 |
| [TileLang-MUSA](https://github.com/tile-ai/tilelang-musa) | 社区项目 | 多类 kernel 示例和 Attention 变体的交叉证据 |
| [FlagGems](https://github.com/flagos-ai/FlagGems) | 社区项目 | 多后端融合算子和 MThreads 专用实现的交叉证据 |

可信度控制规则如下：

- 每个外部来源固定到完整 40 位 Commit，并记录分支和许可证；
- 来源记录保存具体源码、测试或 benchmark 路径，而不是只保存项目主页；
- 厂商官方、社区项目和本地归档通过 `authority` 字段明确区分；
- 搜索摘要、博客、论坛和 issue 只作为发现线索，不作为实现或性能结论；
- “源码存在”只证明值得研究，不等于已经在本地 MUSA 设备上构建、运行或达到性能门槛；
- 相关上游信息保留在 `maintainer_only` 候选池，不泄露到 Agent 可见任务包。

## 6. A/B 类初筛汇总

### 6.1 A 类状态

| A 类判断 | 数量 | 含义 |
|---|---:|---|
| `promising` | 11 | 复合语义可能形成独立 kernel，但仍需正式语义契约和 reference |
| `needs_redesign` | 9 | 当前定义过于接近已有库调用，需要重新划定语义边界 |
| `not_suitable` | 15 | 普通等价库调用已存在，不适合通过禁库方式构造 A 类题 |

### 6.2 B 类状态

| B 类判断 | 数量 | 含义 |
|---|---:|---|
| `promising` | 22 | 已有专用库或融合路径，存在合理分派和兜底价值 |
| `needs_gap_probe` | 11 | 有源码线索，但尚未证明可复现的设备能力缺口 |
| `not_suitable` | 2 | 当前没有足够统一的库入口，无法构造可信的 B 类分派题 |

证据等级分布为：7 个 `local_measured_plus_source`、26 个 `multi_source_code`、2 个
`official_source_code`。这些等级描述当前证据来源，不替代正式硬件准入结论。

## 7. 工程化产物

本次算子收集不是独立的人工笔记，而是已经接入仓库控制面的可校验资产：

| 文件或目录 | 作用 |
|---|---|
| `musa_operator_eval/sources/operator_candidate_pool.json` | 第一轮通用候选事实源 |
| `musa_operator_eval/sources/attention_operator_candidate_pool.json` | 第二轮 Attention 候选事实源 |
| `musa_operator_eval/schemas/operator_candidate_pool.schema.json` | 两类候选池共用的 JSON Schema 2020-12 约束 |
| `musa_operator_eval/evaluator/collect_operator_candidate_pool.py` | 校验来源、交叉引用和本地锚点，并生成评审索引 |
| `musa_operator_eval/sets/candidates/CANDIDATE_POOL.md` | 第一轮维护者可读索引 |
| `musa_operator_eval/sets/candidates/ATTENTION_CANDIDATE_POOL.md` | 第二轮维护者可读索引 |
| `musa_operator_eval/tests/test_operator_candidate_pool.py` | 数量、family、来源、隔离和生成物漂移测试 |
| `musa_operator_eval/progress/phase_15_operator_candidate_pool_and_family_expansion.md` | 第一轮阶段记录 |
| `musa_operator_eval/progress/phase_16_attention_candidate_expansion.md` | 第二轮阶段记录 |

候选池记录的逐算子字段包括：候选 ID、名称、family、优先级、当前决策、证据等级、
A/B 类判断与理由、B 类 gap 假设、语义范围、动态维度、目标 profile、来源引用、
本地锚点、风险、下一步证据以及硬件验证状态。

## 8. 验证结果

第二轮完成后，对两轮产物统一执行了控制面验证：

```text
Attention set check：8 entries、15 measurements、11 implementation candidates，通过
General candidate pool check：19 candidates / 6 families，通过
Attention candidate pool check：16 candidates / 6 families，通过
Attention source tree audit：9 个固定 Commit、63 条证据路径，通过
Repository schema validation：38/38 有效
Candidate pool 专项测试：14/14 通过
musa_operator_eval 全量 CPU 测试：545 项执行，494 项通过，51 项按环境能力跳过
compileall：通过
git diff --check：通过
```

可复核命令：

```powershell
python musa_operator_eval/evaluator/collect_operator_candidate_pool.py --check
python musa_operator_eval/evaluator/collect_operator_candidate_pool.py --pool attention --check
python musa_operator_eval/tests/test_operator_candidate_pool.py -v
python musa_operator_eval/evaluator/validate_schemas.py --all --engine stdlib
python -m unittest discover -s musa_operator_eval/tests -v
python -m compileall -q musa_operator_eval
git diff --check
```

51 个 skip 来自既有环境边界，例如无 MUSA 运行栈、无私有资产或当前 Windows 主机
不能执行 POSIX/ELF 专属流程；它们没有被计为硬件验证通过。

## 9. 当前成果边界

已经完成：

- 可信来源筛选、Commit 固定、许可证与证据路径记录；
- 两轮候选发现、去重、family 归类、优先级和状态初筛；
- A/B 类适配性分析、gap 假设、风险和下一步证据记录；
- schema、生成工具、维护者索引、自动化测试与 Agent 可见性隔离；
- 不依赖 MUSA 设备的控制面验证。

尚未完成：

- 在 S4000/S5000 上构建和运行 35 个候选；
- 形成真实 dtype、shape、layout 和语义支持矩阵；
- 为 B 类候选证明至少 3 条、至少 2 类可复现 gap；
- 采集同机 naive、expert、框架和融合库 baseline；
- 验证 `naive/expert >= 1.3`；
- 为晋级候选编写正式语义契约、CPU reference、public/private/generalization Case；
- 生成正式 Agent task package、执行准入和排名。

因此，35 个候选的 `hardware_validated` 均保持为 `false`。`shortlisted` 表示值得进入
下一阶段，不表示正式任务或性能结论已经完成。

## 10. 后续建议顺序

建议先处理证据最具体、最容易形成可验证分派边界的候选，再扩展到训练梯度和更复杂
量化语义：

1. `attention_per_sequence_causal_dispatch`；
2. `attention_varlen_packed_forward`；
3. `paged_kv_attention_decode`；
4. `batched_grouped_gemm`；
5. `conv2d_shape_matrix`；
6. `fused_add_rmsnorm`；
7. `grouped_topk_router`；
8. `mixed_dtype_moe_gemm`；
9. `dense_mla_decode`；
10. `sparse_mla_prefill_decode`；
11. `attention_sdpa_backward_dynamic`；
12. 其余 P1/P2 候选。

每个候选按以下顺序推进：

```text
冻结最小语义与 CPU reference
  -> 编写最小能力探针
  -> 采集 MUSA 支持矩阵与 dispatch trace
  -> 根据真实结果重新判定 A/B
  -> 设计 public/private/generalization Case
  -> 采集完整 baseline
  -> 执行准入、任务包发布与排名
```

## 11. 组会汇报简述

本周完成了两轮 MUSA 候选算子收集与工程化接入。第一轮从 Attention 之外扩展了
GEMM、卷积、归一化、归约/选择、MoE 和 Transformer 融合 6 个 family，共 19 个
候选；第二轮围绕 Attention 新增训练、KV cache、MLA、稀疏、线性状态和量化 6 个
子 family，共 16 个候选。两轮累计形成 35 个候选，其中 23 个入围、9 个待探测、
3 个暂缓，15 个属于 P0 入围项。所有来源均记录固定 Commit、许可证和具体证据路径，
候选池已接入 schema、生成工具和自动化测试。当前控制面验证通过，但未把源码存在
冒充为硬件结论；下一阶段需要在 S4000/S5000 上完成能力探针、支持矩阵和 baseline，
再决定哪些候选正式进入任务集。
