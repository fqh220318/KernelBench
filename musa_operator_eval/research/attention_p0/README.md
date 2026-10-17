# Attention P0 前三项预任务资产

这里固定 Phase 16 排序中前三个候选的最小语义，不把它们提前包装成正式任务：

| candidate | CPU reference | 已冻结的关键语义 | 当前状态 |
|---|---|---|---|
| `attention_per_sequence_causal_dispatch` | `per_sequence_causal_reference` | BSHD、逐序列 bool、bottom-right causal、GQA、零 KV | 仅 CPU 合同 |
| `attention_varlen_packed_forward` | `varlen_packed_reference` | THD packed、cu_seqlens、bottom-right causal、空 segment、GQA | 仅 CPU 合同 |
| `paged_kv_attention_decode` | `paged_kv_decode_reference` | 只读 paged cache、重复 page 合法、必需 page 越界报错、GQA | 仅 CPU 合同 |

共同规则：默认 scale 为 `1/sqrt(head_dim)`，累积与输出使用 float64；全 mask 行输出
零，LSE 为负无穷；不包含 dropout、隐式 bias 或未声明的 cache 写副作用。

`semantic_contracts.json` 是机器可读语义表，`capability_probe_matrix.json` 是目标机待执行
矩阵。矩阵中的每个 cell 都是 `not_run`，不能当成支持性、缺口或性能证据。目标机适配器
必须先解析到具体版本化入口，设备记录还必须包含环境快照、dispatch path、正确性、异常
原因和重复次数，之后才能重新做 A/B 判级。

CPU 侧复核：

```powershell
python -m unittest musa_operator_eval.tests.test_attention_p0_research -v
```
