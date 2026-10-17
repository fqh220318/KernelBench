# 阶段 19：Swin 公开/隐藏 shape 分域

## 本阶段结果

关闭 `swin_transformer_v2_a_v0` 的公开与隐藏 shape 范围重叠问题，同时保持参考模型、
输入图像和最大批量边界不变。

该参考实现只能可靠使用 `image=224`：输入生成器读取模块级 `image_size`，模型构造却使用
默认值 224，并且四个 stage 的窗口划分也依赖该尺寸。因此不能靠改变图像大小制造公开与
隐藏分域。

批量轴的合同范围原本就是 `B=[1,4]`。现在将其明确划分为：

- public smoke：`B=1,2,3`；
- hidden：保留 `B=4`；
- `image=224` 在两侧保持固定。

这项修正把第三个公开 smoke case 从 `B=4` 改为 `B=3`，没有提高显存上限，也没有扩展
算子语义。私有 manifest 未挂载，因此本阶段不生成、不猜测也不写入隐藏 case；维护者挂载
私有资产后，应让该任务的所有隐藏 case 使用保留域 `B=4`，并重新生成对应 golden。

## 防回归约束

`tests/test_task_packages.py` 新增仓库级检查，固定以下关系：

1. 三个公开批量值必须为 `1,2,3`；
2. 语义合同仍声明合法范围 `[1,4]`；
3. public manifest 与 semantics 都必须写明 `B=4` 是隐藏保留域。

`tests/test_private_assets.py` 也新增维护者侧检查：私有树挂载时，Swin 的每个隐藏 case
都必须恰好使用 `B=4,image=224`。这样后续修改任一侧时，都不会无意中破坏分域。

## 硬件侧后续

本阶段不需要 MUSA 硬件。挂载私有资产后还需要：

1. 让 `cases.private.json` 的 Swin shape 全部落在 `B=4,image=224`，并通过维护者侧检查；
2. 重新生成或核对这些 case 的 golden；
3. 按新的隐藏 case 集采集完整 baseline roster；
4. 用目标卡复验内存占用、正确性和计时稳定性。

## 验证结果

```text
Swin/public 与 maintainer-side 定向测试：109 项执行，97 项通过，12 项因私有树未挂载跳过
Repository schema validation：38/38 有效
General candidate pool：19 candidates / 6 families，通过
Attention candidate pool：16 candidates / 6 families，通过
Baseline collection inventory：与生成文件一致
musa_operator_eval 全量 CPU 测试：551 项执行，499 项通过，52 项按环境能力跳过
compileall 与 git diff --check：通过
```

52 个 skip 包括新增的 Swin 私有 shape 检查；它只会在维护者私有树挂载后执行，不能计为
隐藏分域已经在设备侧验证。
