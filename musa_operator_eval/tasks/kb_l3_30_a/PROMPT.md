# Swin Transformer V2（A 类：算子实现）

## 目标

完整的 Swin Transformer V2 分类器：4 个 stage 的窗口注意力，交替的循环位移、-100 位移 mask、余弦注意力、可学习 logit scale，以及由 MLP 生成的连续相对位置偏置。
自己写 Device kernel 完成核心计算。禁止调用任何库算子。

## 交付物

一个 Python 模块，定义 `ModelNew`，构造参数与输入与公开执行接口一致。Device kernel 可用评测环境提供的中性 `musa_extension.load_inline` JIT 编译 `.mu` 源码。

## 必须做

- 把窗口内的 QK^T、Q/K 的 L2 归一化、learned logit scale、连续相对位置偏置、位移 mask、Softmax 和 P·V 放进 Device kernel。
- QKV 与输出投影可以保留为 `nn.Linear`；不得用库调用替代上一条声明的注意力核心。
- 循环位移与位移 mask 必须与公开语义合同逐元素一致。
- 相对位置偏置的生成路径（坐标表 -> MLP -> 16*sigmoid）必须复现。

## 禁止做

- 调用 muDNN / muBLAS / SDPA / ATen 完成合同声明的注意力核心。
- 把数据拷回 Host 计算。
- 只在非位移窗口上正确、对位移窗口走别的路径。

## 工程提示

- 窗口只有 49 个 token，整个窗口可以一次性 stage 住，不要每个 query 行重算一遍 K/V。
- 余弦注意力要求先做 L2 归一化，注意除零保护与公开语义合同一致。
- 位移 mask 是加性 -100 而不是 -inf，照抄这个常数，它会影响 Softmax 的数值结果。
- 整个模型有 4 个 stage，注意力只是其中一部分，patch merging 与 MLP 同样占时间。

## 评分口径

```text
speedup = upstream_musa_baseline_latency / submission_latency
```

A 类结果单独排名，不与 B 类合并或求平均。

## 尝试预算

最多 20 次提交尝试；单次评测最长 14400 秒。
