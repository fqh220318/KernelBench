# MUSA 算子评测集可执行规范 v1

> 状态：Phase 21 冻结的当前执行口径  
> 生效日期：2026-10-05  
> 原始需求来源：`docs/musa_operator_evaluation_guide.md`  
> 追踪矩阵：`docs/musa_operator_evaluation_requirement_traceability_matrix.md`

## 1. 文档权威与变更规则

原始指南是需求来源和历史基线，保持原文，不在其中消除冲突。本规范负责把原始指南中
互相冲突、已被后续实现说明覆盖或缺少机器判定口径的条款收敛成唯一执行规则。

从本规范生效起：

1. 新代码、任务、Schema、评测报告和发布结论以本规范为准。
2. `IMPLEMENTATION_DECISIONS.md` 解释实现选择，但不得覆盖本规范。
3. `CONFORMANCE.md` 只报告事实状态；不能用测试通过替代规范符合性。
4. `progress/phase_*.md` 是阶段快照。新阶段可以覆盖旧结论，但必须明确记录替代关系。
5. 任何改变评分含义的修改都必须增加规范版本，并重新评估受影响的硬件证据。

## 2. 评测目的

评测集用于分别衡量两种 MUSA 能力：

- A 类：在声明的完整语义域内，没有一次等价库调用时，自行实现 Device kernel。
- B 类：在声明的完整语义域内，库覆盖大部分配置时，正确探测能力、选择库路径并处理真实缺口。

两类任务独立准入、独立基线、独立评分和独立排名。候选池、能力研究和诊断程序不是正式
任务或硬件成绩。

## 3. 判级单位和 A/B 准入

### 3.1 完整语义域

判级单位不是 family 名，而是以下字段共同构成的完整语义域：

- 数学定义和输出；
- 输入、输出与累加 dtype；
- layout、stride 和动态 shape 范围；
- mask、窗口、缩放、GQA 等属性；
- 确定性、种子和容差；
- ABI 和评分 Case 域。

只要其中任一字段不同，就是不同的任务合同。不同合同可以属于同一个 family，但不能声称
共用逐字节相同的语义、Case 或 golden。

### 3.2 A 类

完整语义域内同时满足以下条件才能准入 A 类：

1. 不存在一次调用即可给出完全相同语义的允许库路径；
2. 禁止库调用后仍存在可行的高质量 Device kernel；
3. kernel 核心计算边界可以被契约明确说明并接受审计。

存在一次等价库调用的合同不得仅为了保留双轨而建立 A 任务。

### 3.3 B 类

完整语义域内同时满足以下条件才能准入 B 类：

1. 允许库覆盖大部分目标配置；
2. 隐藏集中至少有 3 条可复现库缺口，覆盖至少 2 类原因；
3. naive 与 expert 在全部评分 Case 上正确且有有效延迟；
4. expert 分派轨迹逐 Case 匹配期望路径；
5. `geomean(naive / expert) >= 1.3`。

达不到条件的 B 合同只能标记为 `research_only`、`redesign_new_id` 或 `retire`，不能进入
正式采集和排名。

## 4. 唯一正式执行架构

### 4.1 提交接口

正式评分提交统一为 KernelBench 风格的 Python `ModelNew`。自定义 Device kernel 使用
`musa_extension.load_inline` 编译 `.mu` 源码。

A 类不得用 muDNN、muBLAS、SDPA、ATen 或其他库路径完成合同声明的 kernel 核心计算。

B 类通过 `torch_musa` 使用声明白名单内的库能力。正式评分路径不得通过独立 C++ runner
直接链接并调用 muDNN 来绕过 `torch_musa`。

### 4.2 诊断路径

现有裸二进制 ABI、无 libtorch runner 和直接 muDNN adapter 可以保留为维护者侧：

- capability probe；
- 库边界定位；
- ABI/链接实验；
- 与正式路径分离的性能诊断。

这类结果必须标记为 `research_only`，不得作为正式提交、正式 baseline 分母、admission
证据或榜单成绩。

### 4.3 evaluator 边界

Agent 可见：评分公式、退出类别、公开 self-test、Schema 和接口契约。

Agent 不可见：生产 evaluator 源码、隐藏 Case、隐藏 golden、官方 baseline、expert 实现和
gap 原始证据。公开仓库中的 evaluator 只能作为开发镜像或 mock，生产版本必须由私有仓库
或受控挂载提供。

## 5. 公开任务与私有溯源

### 5.1 公开任务包

公开任务包只允许包含：

- 脱敏任务契约；
- 语义契约；
- 题面；
- 独立 reference；
- 3～5 个公开 smoke Case 及其输入、golden；
- Starter 和公开 self-test；
- 不可反查上游的 opaque `source_id`。

不得包含仓库名、仓库 URL、Commit URL、上游文件路径、原始源码副本、隐藏资产位置或
维护者答案。公开题面也不得通过模块名、命名空间或示例代码泄漏上游项目名。

### 5.2 私有 provenance

仓库、Commit、LICENSE、快照位置、原始实现和测试全部进入维护者私有 provenance。完整
环境快照、隐藏资产、baseline 与 evaluator 使用同一访问边界。

## 6. Reference 与 Golden

正式算子任务使用独立、确定性的 CPU 高精度 reference。优先使用 FP64；若算子或框架无法
可靠执行 FP64，必须声明替代精度、误差分析和原因。

完整网络或大模型模块只有在被测边界仍能代表单个算子或明确融合子图时才能成为任务；否则
必须提取核心算子。上游实现只用于溯源和 baseline，不得作为 Agent 可见 reference 的逐字
源码副本。

公开与隐藏 golden 必须由同一 reference 和同一 Case 生成规则产生。reference、合同或容差
改变后，所有受影响 golden 失效并必须重新生成。

## 7. Case 合同

每个 Case 必须记录：

- `case_id`、`tag`、`seed`、`dtype`、`shape`；
- `attributes`，无额外属性时使用空对象；
- `distribution`；
- `tolerance_profile`，特殊 Case 可声明 override；
- `golden` 路径。

公开 Case 与隐藏 Case 不共用 seed，评分 shape 域不得重叠。泛化探针在运行时由熵源生成，
不预先写入公开或隐藏清单；报告必须记录可回放 seed。

性能协议在任务或 baseline 中统一冻结。只有确需不同协议的 Case 才允许显式覆盖，不能依赖
未记录的 evaluator 默认值。

B 类公开 Case 一律期望融合路径且不披露 gap。B 类隐藏 Case 才能记录期望路径和结构化 gap
原因。

## 8. 数值容差

任务按 dtype 声明独立的 `atol`、`rtol` 和 `max_abs_error`。正确性同时满足：

```text
allclose(actual, expected, atol=effective_atol, rtol=effective_rtol)
and max_abs_diff <= max_abs_error
```

若复用框架容差下限，必须显式计算：

```text
effective_atol = max(declared_atol, framework_atol_floor)
effective_rtol = max(declared_rtol, framework_rtol_floor)
```

`max_abs_error` 不得同时替代 `atol` 和 `rtol`。报告必须记录实际使用的三个阈值和观测到的
最大绝对误差。

## 9. 计时、baseline 与排名

### 9.1 正式计时协议

正式 numerator 和 denominator 统一复用 KernelBench 的 device-event 计时循环：预热、逐 trial
L2 flush、丢弃首个 trial，以剩余 trial 的 mean 作为评分延迟。样本、p10、p50、p90、标准差、
最小值和最大值作为诊断数据保存。

任何一方改变协议都必须让 numerator 和 denominator 同时重测；不同协议的值不得计算 ratio。

### 9.2 baseline 三层状态

- `evidence_complete`：所有必需 role/Case 观察存在且唯一；非通过行必须给出原因。
- `scoreable`：所有评分关键 role 在全部评分 Case 上通过并有有效延迟，必要轨迹匹配。
- `hardware_pending`：公开边界、独立 reference、正式执行接口和 Schema 已完成静态迁移，但
  尚未按最终合同重新取得硬件正确性、分派与计时证据；不得发布或计分。
- `release_ready`：任务准入、公开隔离、Schema、golden 和硬件证据全部通过。

非通过行可以完成证据清单，但永远不能充当延迟或分母。

### 9.3 A 类分母

A 类在发布前冻结每个评分 Case 的 denominator policy：首选 `upstream_musa`；仅当它在冻结语义
域内退化或不覆盖，且 `mudnn_fused` 语义完全等价时，才选择 `mudnn_fused`。选择和原因必须
进入私有 baseline 与公开评分口径，发布后不得按提交动态切换。

### 9.4 B 类门禁和排名

naive 与 expert 必须覆盖全部评分 Case。任务区分度和最终 speedup 都使用逐 Case ratio 的几何
平均，并保留全部逐 Case ratio。缺少任一评分 Case 的有效分母时，任务不可计分，而不是静默
缩小排名集合。

## 10. 任务生命周期和发布门禁

正式任务状态只允许：

- `keep`：合同满足准入条件，继续进入发布；
- `redesign_new_id`：改变语义域或 Case 后用新 ID 重建；
- `retire`：不再作为正式任务；
- `research_only`：只保留研究和能力证据。

`research_only`、`redesign_new_id` 和 `retire` 状态的合同不得进入 release packager、正式
baseline 队列或排名。研究资产可以保留在开发树中，但不能因此被视为 Agent 发布包；若其
语义重新进入正式评测，必须按 lifecycle 决策使用新任务 ID 或重新完成准入。

正式发布必须同时通过：

1. 规范与 Schema 校验；
2. 公开包泄漏审计；
3. 独立 reference 与 golden 重放；
4. A/B 准入；
5. 全部评分 Case 的正确性和 denominator 覆盖；
6. 同环境、同协议 baseline；
7. A/B 分榜生成。

当前 11 个任务在完成重新判级、公开边界整改和 reference 迁移前均不视为 `release_ready`。

