# MUSA 候选算子池与新 Family 扩展工作总结（2026-09-27）

## 1. 文档用途

本文总结本轮围绕以下两个方向完成的工作：

1. 建立可追溯、可校验的候选算子池；
2. 在现有 Attention set 之外，预研并规划新的算子 family。

本文可用于组会汇报、后续硬件实验交接及 GitHub 提交说明。详细的逐候选信息以
[机器可读候选池](../musa_operator_eval/sources/operator_candidate_pool.json)和
[候选评审索引](../musa_operator_eval/sets/candidates/CANDIDATE_POOL.md)为准。

工作目录与分支：

```text
branch: feat/musa-operator-evaluation-compliance
base:   8d037710795d00fb23b58a8d43d033ea8f44038e
worktree:
C:\Users\冯连雨\KernelBench_github\.worktrees\musa-operator-evaluation-compliance
```

当前改动尚未创建 commit，也未上传 GitHub。

## 2. 本轮工作结论

本轮共整理 **9 个可信来源、6 个候选 family、19 个候选算子**，并把来源、分类、
A/B 类初筛、风险、下一步证据和当前验证状态写入机器可校验的数据文件。

当前候选状态为：

| 状态 | 数量 | 含义 |
|---|---:|---|
| `shortlisted` | 12 | 值得进入语义契约和硬件能力探测阶段 |
| `needs_probe` | 5 | 证据方向合理，但必须先验证库覆盖或语义边界 |
| `deferred` | 2 | 当前区分度、工作量或证据不足，暂不优先推进 |
| `hardware_validated` | 0 | 本轮没有使用真实 MUSA 硬件，不宣称完成设备验证 |

因此，本轮完成的是候选发现、证据整理、family 归类、A/B 初筛和工程化接入，
不是 19 个正式任务的发布，也不是新 family 的最终性能验收。

## 3. 来源收集与可信度控制

### 3.1 纳入的来源

| 类型 | 来源 | 主要用途 |
|---|---|---|
| 本地可复核归档 | KernelBench、S4000 历史评测归档 | 提供现有题目、历史实现和性能差距线索 |
| 厂商官方 | [torch_musa](https://github.com/MooreThreads/torch_musa) | 框架算子、benchmark、单元测试和 MUSA 后端实现 |
| 厂商官方 | [MUTLASS](https://github.com/MooreThreads/mutlass) | GEMM、FP8 GEMM 和卷积相关实现 |
| 厂商官方 | [MATE](https://github.com/MooreThreads/mate) | 高性能 GEMM、MoE、TopK、Norm 和融合算子 |
| 厂商官方 | [vLLM-MUSA](https://github.com/MooreThreads/vllm-musa) | 推理场景下的 RoPE、MoE、RMSNorm 和状态算子 |
| 厂商官方 | [ONNX Runtime MUSA](https://github.com/MooreThreads/onnxruntime-musa) | 算子覆盖、融合规则和运行时 kernel |
| 厂商官方 | [MT-TransformerEngine](https://github.com/MooreThreads/MT-TransformerEngine) | FP8、GEMM、RMSNorm 和 Transformer 融合实现 |
| 社区项目 | [TileLang-MUSA](https://github.com/tile-ai/tilelang-musa) | GEMM、卷积、Norm、TopK、MoE 等示例的交叉证据 |
| 社区项目 | [FlagGems](https://github.com/flagos-ai/FlagGems) | 多后端 Triton 算子和融合实现的交叉证据 |

### 3.2 可信度规则

为避免把搜索结果误当作已经验证的实现，本轮采用以下规则：

- 优先使用本地可复核归档和厂商官方仓库；
- 社区仓库必须具有明确许可证，并尽量与官方实现交叉核对；
- 每个外部仓库固定到 40 位 Commit，而不是只记录会继续移动的默认分支；
- 每个来源记录许可证、分支和具体源码或测试路径；
- 搜索摘要、博客、论坛帖子和 issue 评论仅作为发现线索，不作为实现或性能证据；
- “仓库中存在代码”只表示值得研究，不等于已经在本项目环境中成功构建和运行；
- 官方来源与社区来源通过 `authority` 字段明确区分。

这些信息已经写入候选池的 `sources` 部分，后续可以据此重新获取同一版本源码并
复核结论。

## 4. 新增的候选 Family

| family | 候选数 | 入围 | 待探测 | 暂缓 | 主要评测价值 |
|---|---:|---:|---:|---:|---|
| `gemm_linear` | 4 | 3 | 1 | 0 | shape 分派、grouped/ragged GEMM、量化和 epilogue 融合 |
| `convolution` | 4 | 3 | 0 | 1 | layout、group、stride、dilation 和转置卷积边界 |
| `normalization` | 3 | 1 | 2 | 0 | 动态归约轴、累加精度及残差融合 |
| `reduction_scan_selection` | 3 | 2 | 0 | 1 | 动态轴、非对齐长度、scan 方向和 TopK tie-break |
| `moe_routing` | 2 | 2 | 0 | 0 | 路由、Grouped TopK、ragged expert 和混合精度 GEMM |
| `transformer_fusions` | 3 | 1 | 2 | 0 | RoPE、FP8 量化、cache 写回和状态更新融合 |
| **合计** | **19** | **12** | **5** | **2** | — |

### 4.1 已入围的 12 个候选

- `batched_grouped_gemm`
- `gemm_bias_activation_epilogue`
- `quantized_fp8_w4a8_gemm`
- `conv2d_shape_matrix`
- `depthwise_pointwise_conv2d`
- `conv_transpose_1d_2d`
- `fused_add_rmsnorm`
- `masked_cumsum_scan`
- `dynamic_topk`
- `grouped_topk_router`
- `mixed_dtype_moe_gemm`
- `rope_quantize_fp8_fusion`

### 4.2 需要先做能力探测的 5 个候选

- `dense_gemm_shape_dispatch`
- `rmsnorm_dynamic`
- `layer_group_norm_dynamic`
- `rope_dynamic`
- `causal_conv1d_state_scan`

### 4.3 当前暂缓的 2 个候选

- `conv3d_transpose3d`：硬件和内存成本较高，当前优先级低于 Conv2D 系列；
- `segmented_reduction`：缺少本地 KernelBench reference 和足够的性能区分证据。

普通逐元素激活、普通 Softmax/LogSoftmax 和整模型 Transformer workload 没有纳入
候选池：前两者目前缺少足够的 A/B 区分度，整模型 workload 则无法隔离单一算子
语义和性能。

## 5. 与需求文档的对应关系

### 5.1 不是按算子名称简单堆积

候选首先按语义 family 归类，再分别记录：

- shape 和动态维度；
- layout、stride、group 和广播规则；
- 输入、输出及累加 dtype；
- 量化 scale、舍入和饱和规则；
- 融合顺序和中间结果写回；
- 极值、空张量、非连续张量和非对齐边界。

这使后续任务可以共享 family 级语义契约，同时避免直接复用 Attention 专属字段。

### 5.2 A 类初筛

普通 GEMM、Conv、RMSNorm、TopK 等通常已经有等价库调用，因此不能通过“禁止库
调用后要求重写”直接变成 A 类题目。只有不存在单次等价库调用、具有独立复合语义
或能合理形成单个 device kernel 的候选，才标记为 A 类 `promising`。

### 5.3 B 类初筛

候选池中的 B 类 gap 目前全部是待验证假设，不是实测结论。每个继续研究的 B 类
候选至少记录两类可能的 gap，包括：

- `unsupported_shape`
- `semantic_mismatch`
- `layout_mismatch`
- `multi_operator_required`

晋级为正式 B 类任务时，仍必须在真实硬件上满足需求文档的硬门槛：至少 3 条实测
gap、至少 2 类 gap 原因，并且 naive/expert 延迟比不低于 1.3。

### 5.4 Agent 可见性隔离

候选池包含上游项目名、Commit、实现路径和研究判断，因此定位为
`maintainer_only`。专项测试检查候选 ID、外部源码 Commit 和候选池 ID 没有进入
现有 Agent 任务包。只有完成语义脱敏、公开 Case 设计和准入测量后，候选才可以被
转换为 Agent 可见的正式 task package。

## 6. 本地工程化产物

| 文件 | 作用 |
|---|---|
| [`musa_operator_eval/sources/operator_candidate_pool.json`](../musa_operator_eval/sources/operator_candidate_pool.json) | 候选来源、family、决策和证据的机器可读事实源 |
| [`musa_operator_eval/schemas/operator_candidate_pool.schema.json`](../musa_operator_eval/schemas/operator_candidate_pool.schema.json) | 候选池 JSON Schema 2020-12 约束 |
| [`musa_operator_eval/evaluator/collect_operator_candidate_pool.py`](../musa_operator_eval/evaluator/collect_operator_candidate_pool.py) | 校验交叉引用、本地锚点并生成评审索引 |
| [`musa_operator_eval/sets/candidates/CANDIDATE_POOL.md`](../musa_operator_eval/sets/candidates/CANDIDATE_POOL.md) | 面向维护者的可读候选清单 |
| [`musa_operator_eval/tests/test_operator_candidate_pool.py`](../musa_operator_eval/tests/test_operator_candidate_pool.py) | 来源、统计、隔离、B 类假设和生成物漂移测试 |
| [`musa_operator_eval/progress/phase_15_operator_candidate_pool_and_family_expansion.md`](../musa_operator_eval/progress/phase_15_operator_candidate_pool_and_family_expansion.md) | Phase 15 阶段性交接记录 |

同时更新了总 README、schema 验证器、schema 总数测试和 progress 索引，使候选池进入
仓库现有的自动校验流程。

## 7. 验证结果与证据

本轮实际验证结果如下：

```text
Repository schema validation：37/37 有效
Candidate pool generated-index check：19 candidates / 6 families，通过
Candidate pool 专项测试：10/10 通过
musa_operator_eval 全量 CPU 测试：541 项执行，490 项通过，51 项按环境能力跳过
compileall：通过
git diff --check：通过
```

复核命令：

```powershell
python musa_operator_eval/evaluator/validate_schemas.py --kind operator_candidate_pool musa_operator_eval/sources/operator_candidate_pool.json --engine stdlib
python musa_operator_eval/evaluator/collect_operator_candidate_pool.py --check
python musa_operator_eval/tests/test_operator_candidate_pool.py -v
python musa_operator_eval/evaluator/validate_schemas.py --all --engine stdlib
python -m unittest discover -s musa_operator_eval/tests -v
python -m compileall -q musa_operator_eval
git diff --check
```

51 个跳过项来自既有环境边界，例如没有挂载私有资产、没有 MUSA/torch 运行栈，
或者当前 Windows 主机无法执行 POSIX/ELF 专属流程。它们没有被记作硬件验证通过。

## 8. 当前尚未完成的内容

以下内容仍需要真实 MUSA 环境，不能根据源码存在情况直接宣布完成：

- 在 S4000/S5000 上构建和运行候选实现；
- 验证 dtype、shape、layout 和语义支持矩阵；
- 形成 B 类至少 3 条、至少 2 类真实库缺口证据；
- 采集同机 naive、expert、框架和融合库 baseline；
- 验证 naive/expert 性能比是否达到 1.3；
- 为通过探测的 family 编写正式语义契约、CPU reference 和 public/private Case；
- 生成、发布和排名新的 task package。

这也是候选池将 `hardware_validated` 全部保留为 `false` 的原因。

## 9. 下一阶段建议顺序

建议优先处理以下 P0 方向：

1. `batched_grouped_gemm`：先验证 grouped/ragged shape 和 batch broadcast；
2. `conv2d_shape_matrix`：覆盖 layout、groups、stride、dilation 和非对齐 shape；
3. `fused_add_rmsnorm`：冻结残差写回、epsilon、归约轴和累加 dtype；
4. `grouped_topk_router`：冻结 group 限制、tie-break 和权重归一化语义；
5. `mixed_dtype_moe_gemm`：验证 expert offsets、FP8/W4A8 scale 和 crossover。

每个方向按以下顺序推进：

```text
最小能力探针
  -> MUSA 支持矩阵与 dispatch trace
  -> A/B 重新判级
  -> family 语义契约和 CPU reference
  -> public/private/generalization Case
  -> 完整 baseline
  -> 准入、正式 task package 和排名
```

如果能力探针只发现普通性能波动，而没有可复现的语义、shape 或 layout 缺口，则该
候选不应进入 B 类正式任务。

## 10. 可用于汇报的简要表述

本轮完成了 MUSA 算子评测从单一 Attention family 向多 family 扩展的前期工程。
通过检查 9 个本地、官方和社区开源来源，建立了包含 6 个 family、19 个候选的
机器可读候选池，其中 12 个入围下一轮、5 个需要先做能力探测、2 个暂缓。候选池
已经接入 schema、生成工具和自动化测试，能够检查来源 Commit、许可证、交叉引用、
B 类缺口假设和 Agent 可见性隔离。当前控制面验证全部通过，但没有把源码存在或
历史性能数据冒充为新硬件结果；下一阶段需要在 S4000/S5000 上完成能力探针和
baseline 收集，再决定哪些候选能够正式扩展为新的算子 family 和任务包。
