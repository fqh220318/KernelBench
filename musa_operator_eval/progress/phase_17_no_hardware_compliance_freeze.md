# 阶段 17：无硬件合规冻结（第一批）

## 本阶段目标

在继续占用 MUSA 设备之前，先消除会让硬件结果永久无法准入的控制面矛盾，并把
当前工作树整理为可继续验证的状态。本阶段不生成、修改或补写任何设备测量。

原始需求文档 `docs/musa_operator_evaluation_guide.md` 保持不变；实现层决策集中记录
在 `IMPLEMENTATION_DECISIONS.md`，合规差异继续由 `CONFORMANCE.md` 跟踪。

## 已完成

### 1. 拆分 baseline 清单完整性与可计分性

旧规则要求 A 四角色/B 六角色在每个 Case 上全部 `pass` 才能得到
`readiness.status=complete`。这与 B 类必须包含真实库缺口 Case 的要求冲突：融合库
在缺口 Case 上不应被强迫产生一个伪造的通过结果。

新规则为：

- readiness 只判断所有强制 role/Case 观察是否存在且唯一；
- `fail`、`error`、`unsupported`、`unavailable` 必须携带原因；
- 非通过行继续出现在 `unavailable_rows`/`nonpassing_rows`，但不再被当作缺失行；
- B gate 要求每个 Case 的 naive/expert 都通过且有正延迟；
- admission 仍要求隐藏正确性和 expert dispatch trace；
- rank 只使用声明分母中实际通过的 Case；没有任何通过分母时明确不排名。

因此，完整清单不再等同于获得分数，也不会让非通过记录进入比值。

### 2. 补强 Schema 与测试

- baseline Schema 现在要求所有非 `pass` 状态填写非空原因；
- baseline Schema 与新生成记录版本提升到 `1.1.0`，明确本次 readiness 语义变更；
- 新增或调整测试，覆盖辅助角色 unavailable、naive/expert 失败、分母部分覆盖和
  完全没有可用分母；
- baseline inventory 将非通过角色作为脱敏观察项保留，而不是重新列为未采集。

### 3. 固定实现决策入口

新增 `IMPLEMENTATION_DECISIONS.md`，明确：

- 原始需求文档不在本阶段改写；
- 当前统一使用 KernelBench mean 计时，median 偏差继续显式保留；
- compiled path 是强隔离路径，Python path 是框架绑定与诊断路径；
- evaluator 公共可见仍是部署短板，不能用 Agent 导出包掩盖；
- 四个单一 gap 原因的 B 任务继续阻塞，不能人工补证据。

### 4. 工作区保护

已确认 `.tmp/` 是本地临时输出，不属于后续提交范围；当前没有删除其中内容，也没有
执行 Git commit 或 push。提交整理阶段应显式排除该目录。

### 5. 解析现有 upstream baseline 实现

baseline inventory 现在会根据公开 `set_entry` 自动查找
`KernelBench/level*_musa/problem_*/model_new.py`。当前 10 个 entry-scoped A/B 包可直接
得到规范的 `upstream_musa` 路径，不再为这一角色保留占位符；family-scoped pilot
以及 muDNN/torch_musa 专用实现仍保持显式未解析，等待维护者提供真实模块或不可用
证据。

## 本阶段验证

定向回归：

```text
baseline readiness / inventory / candidate gate / rank / schema：101 项执行
结果：100 项通过，1 项因本机未安装 jsonschema 按预期跳过
```

全量回归：

```text
Repository schema validation：38/38 有效
General candidate pool：19 candidates / 6 families，通过
Attention candidate pool：16 candidates / 6 families，通过
Baseline collection inventory：与生成文件一致
musa_operator_eval 全量测试：548 项执行，497 项通过，51 项按环境能力跳过
compileall 与 git diff --check：通过
```

51 个 skip 仍来自无 MUSA、无私有资产或非 POSIX/ELF 环境，没有计为硬件通过。

## 仍待完成的无硬件工作

1. 将当前大量已修改和未跟踪文件按“控制面、候选池、文档”拆分为可审查提交；
2. 为 7 个 release blocker 补齐 muDNN/torch_musa 等剩余实现路径，消除对应占位；
3. 决定四个 entry-scoped B 任务是重设计还是撤销 B 轨；
4. 处理 `swin_transformer_v2_a_v0` 公开/隐藏 shape 范围重叠；
5. 准备前三个 Attention P0 候选的最小 CPU reference、语义表和能力探针矩阵；
6. 将 evaluator/private 的生产挂载方式落实为可部署配置。

## 硬件阶段入口条件

只有在以下条件完成后才进入正式采集：

- baseline readiness/scoreability 规则不再变化；
- 计时协议已冻结且分子分母一致；
- 私有 manifest/golden 的版本和挂载位置已锁定；
- 每个待测 role 都有真实实现路径或明确的不可用记录方案；
- 当前控制面全量测试通过。

