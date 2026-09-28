# 阶段 13：编译路径的分阶段执行与失败短路

## 本阶段结果

需求文档 §4.7 要求任一评测环节失败后立即短路，不进入后续环节。审查发现
Python 路径已有 Case 子阶段短路，但编译路径 `run_binary.py` 仍把所有 Case 放在
一个平坦循环中执行，最后才根据失败行选择退出码。因此 correctness 失败后仍会
运行 boundary、stability 或 performance，和文档及报告声明不一致。

本阶段已统一两条路径的 Case 调度规则：

1. 固定执行顺序为 `correctness → boundary → stability → performance`；
2. 同一阶段内的所有 Case 都运行，避免第一个错误遮蔽同阶段的其他错误；
3. 当前阶段有任一失败后，后续阶段不执行，并为每个未执行 Case 写入显式
   `skipped` 行及原因；
4. manifest 的 JSON 列表顺序不再决定评测顺序；
5. 分派轨迹失败仍返回专用退出码 41，稳定性失败仍返回 52，其余失败返回该
   Case 所属阶段的退出码；
6. 编译报告新增 `evaluation_stages`，逐阶段记录 `not_present/passed/failed/skipped`
   和 Case 数量；summary 不再把 skipped 行重复计入 failed。

## 审查中修复的隐蔽问题

运行时泛化 Case 是在评测启动后追加到 manifest 的。如果直接按列表顺序运行，
它可能位于已有 perf Case 之后，即使其 tag 属于 correctness，也会在性能测量后
才执行。现在调度器按 tag 重排阶段，因此动态生成的位置不会改变安全顺序。

这属于 Phase 13 的局部修复，不需要回退 Phase 10 的运行时泛化设计：随机采样、
回放 seed 和报告内容都保持不变，只修正了它进入流水线的时机。

## 代码与验证

- `evaluator/run_task.py`：新增稳定的 stage-order 排序，保留阶段内 manifest 顺序。
- `evaluator/run_binary.py`：新增可独立测试的 compiled stage scheduler，并接入真实
  staging/runner 回调。
- `schemas/evaluation_report.schema.json`：声明 `evaluation_stages` 结构和闭合枚举。
- `tests/test_run_task.py`：增加“perf 写在 manifest 前面也不能先运行”的回归测试。
- `tests/test_run_binary.py`：CPU-only 验证同阶段全跑、后续阶段跳过、boundary
  退出码和 dispatch 专用退出码。
- `tests/test_schemas.py`：验证阶段状态结构及未知阶段拒绝。

```text
编译调度、Python 短路和 report schema 定向测试：14/14 通过
Repository schema validation（stdlib engine）：34/34 文档有效
py_compile 与 git diff --check：通过
```

编译 runner 的端到端假设备测试使用 POSIX 可执行脚本，在当前 Windows 主机上会
触发 WinError 193；这与阶段调度的 CPU-only 测试分开记录。真实 `mcc` 构建、MUSA
runner、分派 trace 与性能阶段仍需在 Linux + MUSA 环境复验。

