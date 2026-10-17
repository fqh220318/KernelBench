# Phase 25-2：模型状态合同与公开 golden 软件重放

日期：2026-10-12

## 完成结论

Phase 25 的无硬件优先项已经收口：3 个 keep 任务现在显式声明独立 oracle 如何获得模型状态，
两个有状态 A 类任务已与公开执行模型完成数值交叉检查，公开 Case 的 golden 生成也已完成两轮
字节级确定性复核。当前没有引入 profiling 工具，因为它不解除正确性、私有资产或正式硬件
baseline 门禁。

本阶段只形成 CPU 软件证据，不形成 MUSA 设备正确性、分派轨迹或性能证据。

## 1. 模型状态合同收口

`golden_oracle.state_handling` 固定为三种互斥策略：

- `stateless`：模型没有参数或 buffer；
- `explicit_model_state`：框架构造并导出确定状态，oracle 直接接收状态；
- `deterministic_constructor_replay`：oracle 只根据 `case_seed` 与 `init_inputs` 重建构造过程。

3 个 keep 任务的最终声明如下：

| 任务 | 状态策略 | oracle context | 结果 |
|---|---|---|---|
| `sdpa_forward_b_v0` | `stateless` | 无 | 无模型状态 |
| `relu_causal_attention_a_v0` | `explicit_model_state` | `init_inputs`, `model_state` | 3 个状态张量由框架显式导出 |
| `swin_transformer_v2_a_v0` | `explicit_model_state` | `model_state` | 250 个状态张量由框架显式导出 |

Schema 要求每个正式 `golden_oracle` 声明策略；无 torch 的标准验证器还会检查策略与 context
是否一致。torch 状态物化测试继续对照公开 `Model.state_dict()` 检查声明和实际状态。

ReLU causal attention 的独立 oracle 不再导入 torch 或按 PyTorch RNG 重建投影参数，而是直接
消费框架导出的 `c_attn.weight`、`c_attn.bias` 和 `causal_mask`，再用 NumPy float64 计算。
这样避免了 golden 对 PyTorch 版本和初始化器实现细节的隐式依赖。

## 2. 公开 Case golden 重放

在 `musa-phase25-cpu` 环境中，分别对 ReLU causal attention 和 Swin Transformer V2 的
3 个公开 smoke Case 运行正式 `musa_operator_eval.tools.generate_cases` 入口，并在新的输出目录
重复生成一次：

| 任务 | 公开 Case | 每轮文件数 | 两轮 SHA-256 不一致 |
|---|---:|---:|---:|
| `relu_causal_attention_a_v0` | 3 | 12 | 0 |
| `swin_transformer_v2_a_v0` | 3 | 12 | 0 |

生成物只保存在系统临时目录，没有写回公开任务包。该结果证明公开输入、显式模型状态和 golden
在当前 CPU 重放环境中可重复；它不替代私有 hidden Case 的重新生成与核验。

## 3. 验证结果

- `musa_operator_eval/tests` 全量运行 599 项，结果为 `OK`；其中 28 项私有资产或 MUSA/POSIX 工具链相关测试按边界跳过；
- 独立 oracle 与公开 torch 模型的定向交叉检查通过；
- 模型状态物化测试通过，未挂载的私有 manifest 改为明确 skip，而不是环境错误；
- KernelBench library static checker 49 项测试通过；
- `pip check` 未发现损坏依赖；
- `git diff --check` 无 whitespace error。

## 4. 尚未解除的门禁

当前工作树仍未挂载 `musa_operator_eval/private/<task-id>/cases.private.json`，因此不能生成或核验
hidden golden，也不能声称私有软件重放完成。CPU 环境也没有 `torch_musa`、MUSA Toolkit、
驱动或目标加速卡，因此 3 个 keep 任务仍为 `hardware_pending`，没有任务是 `release_ready`。

## 5. 后续顺序

1. 在维护者环境挂载私有 Case 和 provenance，重生 hidden golden，并逐 Case 核对 manifest 与 digest；
2. 软件证据完整后，在目标 Linux + MUSA 环境对 3 个 keep 任务执行正确性、稳定性、分派轨迹和正式 baseline；
3. 只有在正式硬件执行需要定位性能瓶颈或补充可审计证据时，再接入 MUSA 官方 profiling 工具。

