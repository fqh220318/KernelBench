# 阶段 10：运行时泛化探针

## 本阶段结果

已将泛化探针从可预先读取的固定 Case，改为评测开始后才实例化的随机 Case，落实需求文档 §4.3“随机 shape 不在任何公开或隐藏清单里”的要求。

具体完成：

1. 新增 `evaluator/generalization_probe.py`，从私有采样策略生成随机 dtype、shape 和 case seed，并排除公开/隐藏清单中已有的 shape 与 seed。
2. `case_manifest.schema.json` 不再允许静态 Case 使用 `generalization` tag；私有清单必须携带 `generalization_probe` 策略，公开清单禁止携带该策略。
3. B 级策略必须提供 `expected_path_rules`，使运行时 Case 仍可核对分派路径；规则必须有默认分支。
4. `make_private_manifest.py` 拒绝固定 probe，改为验证和写入运行时策略。
5. Python 评测入口 `run_task.py` 和编译评测入口 `run_binary.py` 均会在运行时生成输入与 golden，将 probe 追加到本次内存清单，并在报告中记录实际 runtime seed、case seed、dtype、shape 和期望路径。
6. 编译入口仅用临时文件向模型状态生成器传递本次运行清单，评测结束后清理，不把随机 Case 固化为新的隐藏清单。
7. 两个入口都新增 `--generalization-seed`，可按报告中的 seed 精确复现失败 Case；不传时使用系统熵。

## 验证结果

```text
PrivateManifestGoldenPathTests + GeneralizationProbeTests +
PrivateManifestSchemaTests: 23 项通过，1 项因本机无 torch 跳过

RuntimeGeneralizationIntegrationTests + CaseManifestRuleTests +
GeneralizationProbeTests + PrivateManifestSchemaTests: 23/23 通过

test_schemas: 54 项通过，1 项因未安装可选 jsonschema 跳过
validate_schemas.py --all --engine stdlib: 34/34 文档有效
py_compile: generalization_probe.py、make_private_manifest.py、
run_task.py、run_binary.py 均通过
```

## 当前边界

本阶段验证的是策略、随机实例化、清单隔离、可复现性和两条评测入口的控制逻辑。真实输入/golden 的生成以及在 MUSA 上执行该随机 Case，仍需在固定 torch/MUSA 环境中完成；实际采样结果会由那次运行的评测报告收集，而不是提前伪造到仓库。

