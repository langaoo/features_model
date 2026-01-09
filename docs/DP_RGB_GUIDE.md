# DP RGB 特征训练使用指南

## 概述

本指南介绍如何将你的 RGB 蒸馏特征接入 Diffusion Policy (DP) head 进行端到端训练。

## 架构流程

```
原始 RGB 图像 
  ↓
视觉模型 (CroCo/DA3/DinoV3/VGGt)
  ↓
zarr 特征 [T, Hf, Wf, C]
  ↓
RGB2PC 蒸馏模型 (已训练)
  ↓
融合特征 [T, D]
  ↓
Diffusion Policy
  ↓
动作预测 [Ta, action_dim]
```

## 关于不同 action_dim 的处理策略

### 方案 1：每个 action_dim 单独训练（推荐）

**优点：**
- 模型更专注，收敛更快
- 每个任务的超参数可以单独调优
- 便于调试和评估

**缺点：**
- 需要训练多个模型
- 无法利用跨任务的知识迁移

**实施方式：**
```bash
# 为 action_dim=7 的任务训练
python tools/train_dp_rgb.py --config configs/train_dp_rgb_arm7.yaml

# 为 action_dim=14 的任务训练
python tools/train_dp_rgb.py --config configs/train_dp_rgb_arm14.yaml
```

### 方案 2：统一 action space（进阶）

**策略：**
- 使用 action_dim = max(所有任务的 action_dim)
- 对于维度不足的任务，pad 0 或 mask 掉
- 在 policy 中添加 task embedding

**优点：**
- 单一模型处理多任务
- 可能学到跨任务的通用策略

**缺点：**
- 实现复杂度高
- 可能影响单任务性能

## 快速开始

### 1. 检查数据

首先确认你的数据结构：

```bash
# 检查 RGB 特征
ls /home/gl/features_model/rgb_dataset/features_*_zarr/

# 检查轨迹数据
ls /home/gl/features_model/raw_data/dump_bin_bigbin/demo_randomized/_traj_data/

# 查看一个 episode 的 action 维度
python -c "
import pickle
import numpy as np
data = pickle.load(open('/home/gl/features_model/raw_data/dump_bin_bigbin/demo_randomized/_traj_data/episode0.pkl', 'rb'))
left = data.get('left_joint_path', [])
print(f'Left arm steps: {len(left)}')
if len(left) > 0:
    action = np.array([left[0][0] if isinstance(left[0], (list, tuple)) else left[0]]).flatten()
    print(f'Action dim: {len(action)}')
"
```

### 2. 按 action_dim 分组任务

创建不同 action_dim 的任务列表：

```python
# tools/check_action_dims.py
import pickle
from pathlib import Path
from collections import defaultdict

traj_root = Path("/home/gl/features_model/raw_data")
tasks = [
    "dump_bin_bigbin-demo_randomized-20_head_camera",
    "beat_block_hammer-demo_randomized-20_head_camera",
    "place_burger_fries-demo_randomized-20_head_camera",
    "shake_bottle-demo_randomized-20_head_camera",
    "stack_bowls_two-demo_randomized-20_head_camera",
]

action_dim_tasks = defaultdict(list)

for task in tasks:
    # 找到第一个 episode
    task_base = task.split('-')[0]
    pkl_path = traj_root / task_base / "demo_randomized" / "_traj_data" / "episode0.pkl"
    
    if pkl_path.exists():
        data = pickle.load(open(pkl_path, 'rb'))
        left = data.get('left_joint_path', [])
        if len(left) > 0:
            if isinstance(left[0], (list, tuple)):
                joints = left[0][0] if len(left[0]) > 0 else []
                gripper = [left[0][1]] if len(left[0]) > 1 else []
                action_dim = len(joints) + len(gripper)
            else:
                action_dim = 1
            action_dim_tasks[action_dim].append(task)
            print(f"{task}: action_dim={action_dim}")

print("\n按 action_dim 分组:")
for dim, tasks in action_dim_tasks.items():
    print(f"  action_dim={dim}: {len(tasks)} tasks")
    for t in tasks:
        print(f"    - {t}")
```

### 3. 为每个 action_dim 创建配置

例如，如果你有 action_dim=7 和 action_dim=14 两组：

**configs/train_dp_rgb_arm7.yaml:**
```yaml
# 继承 default 配置
<<: *default

tasks:
  - task_with_action_dim_7_a
  - task_with_action_dim_7_b

save_dir: /home/gl/features_model/outputs/dp_rgb_runs/arm7
wandb_run_name: "dp_rgb_arm7"
```

**configs/train_dp_rgb_arm14.yaml:**
```yaml
<<: *default

tasks:
  - task_with_action_dim_14_a
  - task_with_action_dim_14_b

save_dir: /home/gl/features_model/outputs/dp_rgb_runs/arm14
wandb_run_name: "dp_rgb_arm14"
```

### 4. 训练

```bash
# 训练 action_dim=7 的模型
python tools/train_dp_rgb.py \
  --config configs/train_dp_rgb_arm7.yaml \
  --epochs 100 \
  --batch_size 32 \
  --amp

# 训练 action_dim=14 的模型
python tools/train_dp_rgb.py \
  --config configs/train_dp_rgb_arm14.yaml \
  --epochs 100 \
  --batch_size 32 \
  --amp
```

### 5. 调试

如果遇到问题，可以先用小规模测试：

```bash
# 单任务、小 batch、少 epoch 测试
python tools/train_dp_rgb.py \
  --config configs/train_dp_rgb_default.yaml \
  --tasks dump_bin_bigbin-demo_randomized-20_head_camera \
  --epochs 5 \
  --batch_size 4 \
  --num_workers 0 \
  --tqdm
```

## 训练阶段策略

### 阶段 1: 冻结 RGB 编码器 (推荐先做)

```yaml
freeze_encoder: true
lr: 1.0e-4
epochs: 50
```

只训练 Diffusion Policy head，保持 RGB 特征提取器不变。

### 阶段 2: 联合微调（可选）

```yaml
freeze_encoder: false
lr: 1.0e-5  # 使用更小的学习率
epochs: 50
```

解冻 RGB 编码器，进行端到端微调。

## 推理

训练完成后，使用模型进行推理：

```python
# tools/eval_dp_rgb.py
import torch
from features_common.dp_rgb_policy import DiffusionRGBPolicy

# 加载模型
ckpt = torch.load("outputs/dp_rgb_runs/arm7/final_policy.pt")
policy = DiffusionRGBPolicy(
    obs_dim=ckpt['config']['obs_dim'],
    action_dim=ckpt['config']['action_dim'],
    **ckpt['config']
)
policy.load_state_dict(ckpt['policy_state'])
policy.set_normalizer(ckpt['normalizer'])
policy.eval()

# 推理
with torch.no_grad():
    obs = ...  # [1, To, C] 从 zarr 加载
    result = policy.predict_action({'obs': obs})
    action = result['action']  # [1, Ta, A]
```

## 常见问题

### Q1: 数据集返回 "No valid samples found"

**解决方法：**
- 检查 `rgb_zarr_roots` 和 `traj_root` 路径是否正确
- 确认任务名称匹配（zarr 和 pkl 的任务名要一致）
- 查看 pkl 文件中是否有 `left_joint_path` 或 `right_joint_path`

### Q2: Action 维度不匹配

**解决方法：**
- 确保同一批任务的 action_dim 一致
- 检查 `use_left_arm`、`use_right_arm`、`fuse_arms` 设置
- 运行 `tools/check_action_dims.py` 查看每个任务的 action_dim

### Q3: 显存不足

**解决方法：**
- 减小 `batch_size`
- 增加 `grad_accum_steps`（保持有效 batch size = batch_size * grad_accum_steps）
- 使用 `amp: true` 混合精度训练
- 设置 `freeze_encoder: true`

### Q4: 训练不收敛

**解决方法：**
- 检查数据归一化是否正确
- 降低学习率
- 增加 `horizon` 和 `n_obs_steps`
- 检查 RGB 特征质量（是否正确加载了蒸馏模型）

## 高级配置

### 使用双臂

如果你的任务需要双臂协同：

```yaml
use_left_arm: true
use_right_arm: true
fuse_arms: true  # concat 双臂动作
```

此时 `action_dim = left_dim + right_dim`。

### 自定义网络结构

在 `features_common/dp_rgb_policy.py` 中修改：

```python
# 更大的 Diffusion 模型
down_dims: [512, 1024, 2048]

# 更深的 encoder
obs_encoder_dim: 512
```

## 总结

**推荐工作流程：**

1. ✅ 使用 `tools/check_action_dims.py` 查看所有任务的 action_dim
2. ✅ 为每个 action_dim 创建独立配置文件
3. ✅ 先用 `freeze_encoder=true` 训练
4. ✅ 评估性能，必要时解冻微调
5. ✅ 在真实环境中测试

这样可以最大化利用你已经训练好的 RGB2PC 蒸馏特征，同时避免不同 action_dim 带来的复杂性。
