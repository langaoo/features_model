# Policy 文件 vs Train 脚本：职责划分说明

## 问题

为什么有 `features_common/dp_rgb_policy_multitask.py` 和 `tools/train_dp_rgb_single_task_4models.py`？它们分别做什么？会不会重复/混淆？

---

## 答案：一个是"模型库"，一个是"训练入口"

### 1. Policy 文件（模型库）

**位置**：`features_common/dp_rgb_policy_*.py`

**职责**：定义 PyTorch nn.Module，实现：
- `forward()`：模型计算图（obs → action）
- `compute_loss()`：训练时的 loss 计算
- `predict_action()`：推理时的采样逻辑

**不包含**：
- 数据加载（Dataset/DataLoader）
- 训练循环（optimizer/scheduler/checkpoint）
- 命令行参数解析
- 日志/保存逻辑

**文件清单**：
- `dp_rgb_policy_single.py`：单任务 head 实现（`HeadSpec` + `DiffusionRGBHead`）
- `dp_rgb_policy_multitask.py`：多任务包装（共享 encoder + 多个 head）

**类比**：这就像 `torch.nn.Transformer` 或 `torchvision.models.resnet50`——它们只是模型定义，不管怎么训练。

---

### 2. Train 脚本（训练入口）

**位置**：`tools/train_dp_rgb_single_task_4models.py`

**职责**：
- 解析命令行参数（task/batch_size/epochs/lr/save_dir/...）
- 构建 Dataset + DataLoader
- 实例化 Policy（从 `features_common` import）
- 实例化 Optimizer/Scheduler
- 训练循环（epoch/batch/loss/backward/step）
- 保存 checkpoint

**不包含**：
- 模型定义细节（交给 `features_common/dp_rgb_policy_*.py`）

**文件清单**：
- `tools/train_dp_rgb_single_task_4models.py`：单任务训练入口（4模型 + 冻结对齐 encoder + 单 head）
- `tools/train_dp_rgb_dual_arm.py`：双臂训练入口（可能有特殊逻辑）

**类比**：这就像 `train.py` / `main.py`——它负责"怎么训"，而不是"模型长什么样"。

---

## 3. 为什么要分开？

| 维度 | 分开的好处 |
|-----|-----------|
| **复用性** | Policy 可以被多个 train 脚本共用（单任务/多任务/在线训练...） |
| **测试性** | Policy 可以单独测试 forward/loss，不依赖训练循环 |
| **可读性** | Train 脚本只关心"训练流程"，不被模型细节干扰 |
| **扩展性** | 新增一个训练策略（比如 meta-learning）只需加 train 脚本，不动 Policy |

---

## 4. 实际使用流程

### 训练时

```bash
# 1. Train 脚本负责"调度"
python tools/train_dp_rgb_single_task_4models.py --task beat_block_hammer --epochs 10

# 内部：
# 2. Train 脚本 import Policy
from features_common.dp_rgb_policy_single import HeadSpec, DiffusionRGBHead

# 3. Train 脚本实例化 Policy
head = DiffusionRGBHead(spec=HeadSpec(...))

# 4. Train 脚本跑训练循环
for epoch in range(epochs):
    for batch in dataloader:
        loss = head.compute_loss(obs, action, ...)
        loss.backward()
        ...
```

### 推理时

```bash
# 1. Infer 脚本负责"加载 + 预测"
python tools/infer_dp_rgb_4models.py --head_ckpt outputs/.../final_head.pt

# 内部：
# 2. Infer 脚本 import Policy
from features_common.dp_rgb_policy_single import DiffusionRGBHead

# 3. Infer 脚本加载 Policy
head = DiffusionRGBHead(...)
head.load_state_dict(ckpt['head_state'])

# 4. Infer 脚本调用 predict_action
action = head.predict_action(obs_features, ...)
```

---

## 5. 对你的影响

**如果你想改模型架构**（比如换成 Transformer head）：
- 只改 `features_common/dp_rgb_policy_single.py`
- Train/Infer 脚本**不用动**（只要接口一致）

**如果你想改训练策略**（比如加 curriculum learning）：
- 只改 `tools/train_dp_rgb_single_task_4models.py`
- Policy 文件**不用动**

---

## 6. 总结

| 文件 | 职责 | 类比 |
|-----|------|------|
| `features_common/dp_rgb_policy_single.py` | 模型定义（nn.Module） | `torch.nn.Transformer` |
| `tools/train_dp_rgb_single_task_4models.py` | 训练入口（Dataset + Optimizer + 循环） | `train.py` |

**记住：Policy 是"零件"，Train 是"组装车间"。你可以用同一个"零件"（Policy）在不同"车间"（Train 脚本）里生产不同产品（单任务/多任务/在线训练）。**
