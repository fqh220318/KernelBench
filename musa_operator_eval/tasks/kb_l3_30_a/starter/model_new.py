"""窗口余弦注意力分类器的 A 类算子实现 Starter。

将本文件复制为 ``model_new.py``，填写标记区域后提交。接口为
``ModelNew()`` 和 ``forward(x)``，其中 ``x`` 是形状 ``(B, 3, 224, 224)``
的张量，返回 ``(B, 1000)`` 分类结果。

常用命令（从任务包目录执行）：

    python ../../tools/generate_cases.py --task-dir .
    python ../../evaluator/run_task.py --task-dir . --submission model_new.py --static-only
    python ../../evaluator/run_task.py --task-dir . --submission model_new.py --backend musa --precision fp32

允许使用评测环境提供的中性 ``musa_extension.load_inline`` 编译 ``.mu``
源码。冻结边界、可编辑区域和正式 ABI 以 ``task.json`` 为准。
"""

import torch
import torch.nn as nn


class ModelNew(nn.Module):
    """在三个标记区域中实现完整模型和声明的 attention 核心。"""

    def __init__(self):
        super().__init__()
        # --- BEGIN state ---
        # 按公开执行接口的构造顺序建立参数。评测端会对 reference 与提交物
        # 分别重置为同一个 case seed，因此无需读取任何冻结文件的 state_dict。
        # --- END state ---

    def forward(self, x):
        # --- BEGIN device_kernel ---
        # QK^T、Q/K L2 归一化、learned logit scale、连续相对位置偏置、
        # 位移 mask、Softmax 和 P·V 必须由 Device kernel 完成。
        # --- END device_kernel ---

        # --- BEGIN host_launch ---
        # 在这里配置并启动 kernel；QKV/输出投影以及 task.json 明确列出的
        # 周边模型组件可以保留为框架操作。
        # --- END host_launch ---
        raise NotImplementedError("fill the marked regions above")
