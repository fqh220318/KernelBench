# 阶段 20：前三个 Attention P0 的 CPU 合同与探针矩阵

## 本阶段结果

完成 Phase 16 排序中前三个候选的无硬件准备：

1. `attention_per_sequence_causal_dispatch`；
2. `attention_varlen_packed_forward`；
3. `paged_kv_attention_decode`。

本阶段只建立预任务语义，不创建正式 set entry、Agent task、public/private case 或 baseline，
也不改变候选的 `hardware_validated: false`。

## CPU reference

`research/attention_p0/cpu_reference.py` 提供三个 NumPy-only float64 入口。共同规则为：

- 默认 scale 是 `1/sqrt(head_dim)`；
- causal 采用 bottom-right 对齐，覆盖 `S_q != S_kv`；
- MQA/GQA 要求 `H_q` 是 `H_kv` 的整数倍，并按连续 query-head group 映射；
- 全 mask 行输出零，LSE 为负无穷；
- 不包含 dropout、隐式 bias 或 cache 写副作用。

三个入口额外固定了逐序列 bool 不得折叠、packed 空段、paged table 重排/重复 page/必需
page 越界等候选核心语义。

## 语义表与探针矩阵

- `semantic_contracts.json`：记录输入/输出 layout、边界行为、排除项和当前 A/B 假设；
- `capability_probe_matrix.json`：记录 dtype、shape、causal、GQA、stride、page size 和
  block-table 等待测轴，以及每个重点 cell；
- 所有矩阵 cell 均为 `not_run`；计划本身不是支持性、缺口或性能证据；
- 每条未来设备记录必须带环境快照、适配器版本、真实 dispatch path、正确性、异常或
  mismatch 原因和重复次数。

Attention 候选事实源已为这三项增加 CPU reference、语义表和矩阵本地锚点，并把
`next_evidence` 更新为目标机执行项。生成式维护者索引已同步。

## CPU 防回归

专项测试覆盖：

- mixed per-sequence causal flag 不能被折叠；
- bottom-right varlen 对齐和空 KV；
- paged KV 的物理页重排、重复页和必需页越界；
- 合同、矩阵、候选池三个集合严格一致；
- 所有待测 cell 保持 `not_run`，CPU reference 不导入 torch 或设备运行时。

## 硬件侧下一步

1. 将矩阵中的逻辑 backend 名解析为目标环境中的具体版本化入口；
2. 先跑正确性与 dispatch trace，不先跑性能；
3. 根据真实结果归类 gap，并检查 B 类是否达到至少两类原因；
4. 只有通过语义和准入门槛的候选，才进入正式 task/case/baseline 设计。

## 验证

```text
Attention P0 CPU 专项测试：8/8 通过
Repository schema validation：38/38 有效
General candidate pool：19 candidates / 6 families，通过
Attention candidate pool：16 candidates / 6 families，通过，生成索引一致
Baseline collection inventory：与生成文件一致
musa_operator_eval 全量 CPU 测试：559 项执行，507 项通过，52 项按环境能力跳过
compileall 与 git diff --check：通过
```

52 个 skip 仍来自无 MUSA、维护者私有树未挂载或当前平台不是 POSIX/ELF 等环境边界；
它们没有被计为硬件验证通过。
