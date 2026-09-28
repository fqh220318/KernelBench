# 阶段 15：候选算子池与新 family 扩展预研

## 本阶段结果

围绕“建立候选算子池”和“扩展新的算子 family”完成一轮可追溯收集，并将结果
接入现有校验体系。当前形成 9 个可信来源、6 个候选 family、19 个候选算子：

| family | 候选数 | 入围 | 待探测 | 暂缓 |
|---|---:|---:|---:|---:|
| `gemm_linear` | 4 | 3 | 1 | 0 |
| `convolution` | 4 | 3 | 0 | 1 |
| `normalization` | 3 | 1 | 2 | 0 |
| `reduction_scan_selection` | 3 | 2 | 0 | 1 |
| `moe_routing` | 2 | 2 | 0 | 0 |
| `transformer_fusions` | 3 | 1 | 2 | 0 |
| **合计** | **19** | **12** | **5** | **2** |

`shortlisted` 表示值得进入语义契约和硬件探测阶段，不表示已经成为正式题目；
本轮没有 MUSA 设备执行，因此 19 个候选的 `hardware_validated` 均为 `false`。

## 收集与可信度策略

来源优先级为：本地可复核归档、摩尔线程官方仓库、具有明确许可证且能由多个实现
交叉核对的社区仓库。最终纳入：KernelBench 本地归档、torch_musa、MUTLASS、
MATE、vLLM-MUSA、ONNX Runtime MUSA、MT-TransformerEngine，以及社区维护的
TileLang-MUSA 和 FlagGems。

每个外部来源都记录了 40 位 Commit、分支、许可证和具体源码/测试路径。搜索摘要、
博客转述和 issue 评论只允许作为发现线索，没有作为实现存在或性能结论的证据。
官方仓库与社区仓库在数据中通过 `authority` 明确区分。

## 与需求文档的贴合方式

1. 先按语义划分 family，再记录 shape、layout、dtype、边界和融合顺序，避免按单个
   文件名机械拆题。
2. 每个候选分别做 A/B 初筛。普通 GEMM、Conv、RMSNorm、TopK 等存在等价库调用，
   不会仅通过“禁库后重写”伪装成 A 类；融合或扩展语义才可能进入 A 类。
3. B 类目前只记录缺口假设。一个可研究的 B 候选至少列出两类待测 gap，但必须在
   真实硬件上满足至少 3 条 gap、至少 2 类原因及 naive/expert 延迟比不低于 1.3，
   才能晋级。
4. 候选池位于 maintainer-only 来源区，测试保证其来源名和候选 ID 不进入现有
   Agent 任务包。
5. 本地 S4000 历史归档只用于排序，例如 GEMM、Conv2D 和 ConvTranspose 的明显
   路径差距；旧数字不是当前协议下的新 baseline。

## 新增资产

- `sources/operator_candidate_pool.json`：机器可读候选池，是事实源。
- `schemas/operator_candidate_pool.schema.json`：候选池结构、枚举与证据字段约束。
- `evaluator/collect_operator_candidate_pool.py`：校验交叉引用、本地锚点并生成索引。
- `sets/candidates/CANDIDATE_POOL.md`：面向维护者的逐 family 评审索引。
- `tests/test_operator_candidate_pool.py`：来源固定、许可证、统计、B 类假设、隔离和
  生成物漂移测试。

## 可复核结果

执行以下命令即可在无硬件环境复核本阶段产物：

```bash
python musa_operator_eval/evaluator/validate_schemas.py \
  --kind operator_candidate_pool \
  musa_operator_eval/sources/operator_candidate_pool.json \
  --engine stdlib
python musa_operator_eval/evaluator/collect_operator_candidate_pool.py --check
python musa_operator_eval/tests/test_operator_candidate_pool.py -v
python musa_operator_eval/evaluator/validate_schemas.py --all --engine stdlib
python -m unittest discover -s musa_operator_eval/tests -v
```

本次实际结果：

```text
Repository schema validation：37/37 有效
Candidate pool generated-index check：19 candidates / 6 families，通过
Candidate pool 专项测试：10/10 通过
musa_operator_eval 全量 CPU 测试：541 项执行，490 项通过，51 项按环境能力跳过
compileall 与 git diff --check：通过
```

51 个跳过项仍是此前记录的环境边界，例如没有挂载私有资产、没有 MUSA/torch
运行栈或当前主机不是 POSIX/ELF 环境；它们没有被算作设备验证通过。

## 下一阶段顺序

1. 先为 P0 候选建立最小能力探针，优先顺序为 grouped/ragged GEMM、Conv2D
   shape matrix、fused add RMSNorm、MoE grouped TopK 和 mixed-dtype expert GEMM。
2. 在 S4000/S5000 上记录支持矩阵、错误类型、dispatch trace 和同机延迟，淘汰
   只有性能波动却没有真实库缺口的 B 候选。
3. 对通过探针的 family 编写独立语义契约与 CPU reference，再设计 public、hidden
   和运行时泛化 Case。
4. 按现有准入流程采齐 A 的 4 个或 B 的 6 个 baseline 角色；只有 readiness
   complete 的候选才转成正式 task package。

本阶段完成的是“可信候选收集、归类、初筛和可重复校验”，没有声称完成任何新
family 的正式任务发布、设备正确性、性能基线或排名。
