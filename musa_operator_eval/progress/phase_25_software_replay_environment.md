# Phase 25（前半）：软件重放环境配置

日期：2026-10-08

## 完成结论

已建立隔离的 CPU 环境 `musa-phase25-cpu`，用于 Phase 25 在无 MUSA 硬件条件下完成公开模型与
独立 oracle 的交叉重放，以及在私有资产挂载后生成/复核 golden。该环境不包含
`torch_musa`、MUSA Toolkit、驱动、muDNN 或 muBLAS，因此不能提供任何设备正确性、分派轨迹或
性能证据。

环境定义位于
`musa_operator_eval/environments/phase25_cpu_replay.environment.yml`。当前机器上的实际解释器为
`D:\miniconda3\envs\musa-phase25-cpu\python.exe`。

## 1. 已安装的软件栈

| 项目 | 当前值 | 与正式目标的关系 |
|---|---|---|
| Python | 3.10.20 | 匹配 3.10 minor；目标快照为 Linux Python 3.10.8 |
| NumPy | 1.26.4 | 与目标快照一致 |
| packaging | 24.2 | 与目标快照一致 |
| PyTorch | 2.2.0 CPU（官方 Conda 构建） | 兼容重放；不等同于目标的 `2.2.0a0+git8ac9b20` MUSA 构建 |
| torchada | 0.1.74 | 与目标快照一致；本机不启用 MUSA 路由 |
| torch_musa | 未安装 | 留给 Phase 25 后半的目标硬件环境 |

最初尝试的 pip CPU PyTorch 与 Conda MKL NumPy 会重复加载 `libiomp5md.dll`。已改用
`pytorch` Conda 渠道的 CPU 构建，未设置不安全的 `KMP_DUPLICATE_LIB_OK` 绕过项。

根项目没有安装为 editable 包：其当前 `pyproject.toml` 要求 `torch>=2.9.0`，与本阶段必须使用的
PyTorch 2.2 基线冲突。评测工具按现有实现直接将仓库 `src/` 加入 `sys.path`。

## 2. 验收结果

| 检查 | 结果 |
|---|---|
| `python -m pip check` | 通过，`No broken requirements found` |
| `kernelbench.eval` 导入 | 通过 |
| CPU-only 边界 | `torch.cuda.is_available() == False`，且无 `torch.musa` |
| `tests/test_problem_reference.py` | 12/12 通过 |
| 独立 ReLU causal oracle 对公开模型 | 通过 |
| 独立 Swin oracle 对公开模型 | 通过 |

终端对中文用户目录的显示存在 mojibake，但 Python 能解析工作树并成功导入模块；这属于终端输出
编码现象，不是仓库路径缺失。

## 3. 软件重放前发现的合同阻塞

启用 torch 后，状态物化声明检查首次实际执行并发现两项非环境错误：

- `swin_transformer_v2_a_v0` 声明 `model_state=false`，公开参考模型实际含 250 个状态张量；
- `relu_causal_attention_a_v0` 声明 `model_state=false`，公开参考模型实际含 3 个状态张量。

因此不能把“环境已配置”表述为“软件重放已完成”。应先决定这两个 A 类合同是否需要声明并物化
模型状态，再重跑 `test_materialize_model_state.py`。本次没有擅自修改合同。

## 4. 私有挂载边界

当前工作树没有挂载 `musa_operator_eval/private/<task-id>/cases.private.json`，这与公开仓库的发布
边界一致。隐藏 Case 与正式 golden 不应从公开 Case 推测或伪造。私有资产挂载后，才能执行完整
状态物化、hidden golden 重生和逐 Case digest 对比。

## 5. 使用方式

```powershell
conda activate musa-phase25-cpu
python musa_operator_eval/tests/test_problem_reference.py -v
python -m pip check
```

从环境文件重建：

```powershell
conda env create -f musa_operator_eval/environments/phase25_cpu_replay.environment.yml
```

下一步不是进入硬件计时，而是先收口第 3 节的两个 `model_state` 合同，再挂载私有 Case 并执行
golden 软件重放。全部软件重放通过后，才能切换到 Phase 25 后半的 Linux + MUSA 硬件环境。

## 6. 后续状态（2026-10-12）

第 3 节的两个合同阻塞及公开 golden 重放已在
[Phase 25-2](phase_25_model_state_and_public_golden_replay.md) 收口。当前下一门禁是维护者私有资产
挂载与 hidden golden 重放；完成后才进入目标 MUSA 硬件验证。
