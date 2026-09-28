# 阶段 16：Attention 候选算子扩充

## 本阶段结果

在不修改现有 8 个 Attention set entry 和 11 个实现来源库存的前提下，新增一份
算子语义级候选池，用于承接“继续扩充 Attention 算子”的预研结果。

本轮从 9 个本地、厂商官方和社区来源中整理出 6 个 Attention 子 family、16 个
算子候选：

| family | 候选数 | 入围 | 待探测 | 暂缓 |
|---|---:|---:|---:|---:|
| `attention_dense_training` | 4 | 3 | 1 | 0 |
| `attention_kv_cache` | 3 | 2 | 1 | 0 |
| `attention_mla` | 3 | 3 | 0 | 0 |
| `attention_sparse` | 2 | 1 | 1 | 0 |
| `attention_linear_state` | 2 | 1 | 0 | 1 |
| `attention_quantized` | 2 | 1 | 1 | 0 |
| **合计** | **16** | **11** | **4** | **1** |

16 个候选的 `hardware_validated` 均为 `false`。`shortlisted` 只表示值得进入语义
契约和设备能力探测阶段，不表示已经成为正式任务。

## 为什么没有直接扩写现有 Attention set

现有 `sets/attention/attention_set.json` 描述的是已经有 KernelBench reference、
历史答案或任务包关系的 dense forward Attention entry；
`sources/attention-kernel-candidates.json` 描述的是可以服务这些 entry 的实现来源。

本轮发现的 backward、varlen、paged KV、MLA、稀疏、线性和量化路径是新的算子
语义边界。若直接塞进旧 set，会把“发现源码”误写成“已经具备 reference、Case、
baseline 和准入状态”。因此新增独立的 maintainer-only 候选池，待通过硬件筛选后
再晋级为正式 set entry。

## 来源与可信度

纳入的 9 个来源为：

- 本地 KernelBench Attention set；
- MooreThreads/torch_musa；
- MooreThreads/MATE；
- MooreThreads/MUTLASS；
- MooreThreads/MT-flashMLA；
- MooreThreads/vLLM-MUSA；
- MooreThreads/Paddle-MUSA；
- tile-ai/TileLang-MUSA；
- FlagOS/FlagGems。

每个来源均记录完整 40 位 Commit、分支、许可证和具体源码/测试路径。厂商官方与
社区项目通过 `authority` 字段区分；搜索摘要、issue 和论坛内容只用于发现线索，
没有作为实现存在或性能结论的依据。

本轮核验出的代表性事实包括：

- MATE 的兼容面覆盖 dense/varlen FMHA、KV-cache、paged KV、FP8、MLA、MSA、
  KDA 和 SageAttention，但这些仍是上游声明，尚未在本地目标机复验；
- vLLM-MUSA 明确拒绝把 per-sequence causal tensor 传给只接受标量 causal 的
  FA3 路径，并分派到其他 backend，这是值得做 B 类探针的语义缺口线索；
- MT-flashMLA 的公开范围是 MP31 上的变长 MLA decode、FP16/BF16 和 page size 64；
- FlagGems 包含 MThreads 专用 sparse Attention kernel，可用于交叉核对稀疏语义，
  但不能代替 MUSA 设备测量。

## 与需求文档的贴合

每个候选都分别回答需求文档的 A/B 判定问题：

1. 已存在一次等价融合调用的普通算子，例如标准 paged decode、dense MLA、FP8
   FMHA 和 KDA，不会通过“禁用库”伪装成 A 类。
2. per-sequence causal、MLA indexer 和 block sparse Attention 等复合语义可继续
   研究 A 类，但必须先冻结独立 reference 和完整语义。
3. B 类候选目前只记录至少两类 gap 假设；只有真实硬件证明至少 3 条 gap、至少
   2 类原因且 naive/expert 比值不低于 1.3，才能晋级。
4. causal 对齐、window 方向、全 mask 行、GQA head mapping、LSE、dropout seed、
   cache 更新副作用和量化规则均列为必须写死的合同项。
5. 上游仓库名、Commit 和源码路径只保留在 maintainer-only 候选池，不进入 Agent
   可见任务包。

## 新增与修改的资产

- `sources/attention_operator_candidate_pool.json`：Attention 算子级候选事实源；
- `sets/candidates/ATTENTION_CANDIDATE_POOL.md`：自动生成的逐候选评审索引；
- `schemas/operator_candidate_pool.schema.json`：增加公共 `title` 字段；
- `evaluator/collect_operator_candidate_pool.py`：支持 general/attention 两个候选池，
  默认同时生成或检查，也可用 `--pool attention` 单独检查；
- `evaluator/validate_schemas.py`：自动发现所有
  `*operator_candidate_pool.json`；
- `tests/test_operator_candidate_pool.py`：增加 Attention 数量、family、去重、生成物
  和 Agent 隔离测试；
- `README.md` 与 progress 索引：增加 Attention 候选池入口。

## 可复核命令

```powershell
python musa_operator_eval/evaluator/collect_operator_candidate_pool.py --pool attention --check
python musa_operator_eval/evaluator/collect_operator_candidate_pool.py --check
python musa_operator_eval/tests/test_operator_candidate_pool.py -v
python musa_operator_eval/evaluator/validate_schemas.py --all --engine stdlib
python -m unittest discover -s musa_operator_eval/tests -v
python -m compileall -q musa_operator_eval
git diff --check
```

本次实际结果：

```text
Attention set check：8 entries、15 measurements、11 implementation candidates，通过
General candidate pool check：19 candidates / 6 families，通过
Attention candidate pool check：16 candidates / 6 families，通过
Source tree audit：9 个固定 Commit、63 条源码/测试证据路径，通过
Repository schema validation：38/38 有效
Candidate pool 专项测试：14/14 通过
musa_operator_eval 全量 CPU 测试：545 项执行，494 项通过，51 项按环境能力跳过
compileall 与 git diff --check：通过
```

51 个 skip 来自既有环境边界，例如无 MUSA、无私有资产或当前主机不是 POSIX/ELF
环境；它们没有被算作硬件验证通过。

## 下一阶段优先顺序

第一批 P0 能力探针建议按以下顺序执行：

1. `attention_per_sequence_causal_dispatch`：语义缺口证据最具体；
2. `attention_varlen_packed_forward`：覆盖 packed layout、GQA 和 causal 对齐；
3. `paged_kv_attention_decode`：覆盖 page table、page size 和 cache layout；
4. `dense_mla_decode`：对照 MATE、MT-flashMLA 和 MUTLASS；
5. `sparse_mla_prefill_decode`：先固定外部提供 indices 的最小核心；
6. `attention_sdpa_backward_dynamic`：最后进入训练梯度和确定性测试。

每个候选按“最小 CPU reference → 能力探针 → dispatch trace → A/B 重新判级 →
public/private/generalization Case → 完整 baseline → 准入”的顺序推进。

本阶段没有创建新的 Agent 任务包，也没有修改现有 Attention 排名或历史 baseline。
