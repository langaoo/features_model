# rgb2pc_aligned_encoder_4models.py 文件说明

## 核心作用
**RGB特征到点云语义空间的对齐编码器**

将4个视觉模型提取的异构RGB特征，通过对齐训练学到的映射关系，转换为与ULIP点云特征语义一致的统一表示。

---

## 在整个链路中的定位

```
阶段1: 特征提取
RGB图像 → [CroCo(1024), VGGT(2048), DINOv3(768), DA3(2048)]

阶段2: 对齐训练 (train_rgb2pc_distill.py)
4个RGB特征 + ULIP点云特征 → 训练对齐编码器 → checkpoint

阶段3: DP动作头训练 (本文件被调用)
4个RGB特征 → rgb2pc_aligned_encoder_4models.py → 对齐后特征 → DP Head → 动作

阶段4: 推理部署
实时RGB → 4个视觉模型 → rgb2pc_aligned_encoder_4models.py → DP Head → 机械臂执行
```

**定位**：衔接RGB特征提取和DP训练的桥梁模块

---

## 与其他文件的调用关系

### 1. 被调用方（谁使用它）

| 文件 | 调用位置 | 用途 |
|------|---------|------|
| `tools/train_online_multigpu_hdf5.py` | Line 112 | DP头训练时加载编码器 |
| `tools/train_online_from_config.py` | Line 145 | 配置化训练脚本 |
| `tools/train_dp_rgb_single_task_4models.py` | - | 离线特征训练 |
| `tools/infer_dp_rgb_complete.py` | - | 推理部署 |

### 2. 调用的依赖

| 模块 | 用途 |
|------|------|
| `features_common/fusion.py` | WeightedFusion, MoEFusion（特征融合策略） |
| PyTorch `nn.Module` | 基础神经网络模块 |

### 3. 数据流依赖

**上游**：
- 训练checkpoint：`outputs/train_rgb2pc_runs/run_best_bs32/ckpt_step_0010000.pt`
  - 来自 `train_rgb2pc_distill.py` 训练产物

**下游**：
- 输入给DP Head进行动作预测

---

## 输入输出详解

### 输入
```python
x: torch.Tensor  # [B, To, M, C]
```
- **B**: Batch size（批大小）
- **To**: 观测步数（temporal obs steps，通常为2）
- **M**: 模型数量（固定为4）
- **C**: 各模型特征维度（异构）
  - CroCo: 1024
  - VGGT: 2048
  - DINOv3: 768
  - DA3: 2048

**示例**：
```python
# Batch=4, To=2, 4个模型, 最大维度2048
obs = torch.randn(4, 2, 4, 2048).cuda()
```

### 输出
```python
z: torch.Tensor  # [B, To, D]
```
- **B**: Batch size（与输入相同）
- **To**: 观测步数（与输入相同）
- **D**: 融合后的统一维度（fuse_dim=1280）

**示例**：
```python
z = encoder(obs)  # [4, 2, 1280]
```

---

## 内部处理流程

```
输入 [B, To, 4, C_hetero]
    ↓
1. Adapters (4个独立MLP)
   - CroCo(1024) → Adapter0 → 2048
   - VGGT(2048) → Adapter1 → 2048
   - DINOv3(768) → Adapter2 → 2048
   - DA3(2048) → Adapter3 → 2048
    ↓
2. Fusion (WeightedFusion / MoEFusion)
   - 输入: 4 x [B*To, 2048]
   - 输出: [B*To, 2048]
    ↓
3. Context Encoder (Transformer, 如果启用)
   - 输入: [To, B, 2048]
   - Positional Encoding
   - TransformerEncoder (2 layers)
   - 输出: [To, B, 2048]
    ↓
4. proj_student (MLP投影)
   - 输入: [B*To, 2048]
   - 输出: [B*To, 1280]
    ↓
输出 [B, To, 1280]
```

---

## 关键设计点

### 1. 异构特征处理
通过独立的Adapter将不同维度映射到统一空间（2048）

### 2. 融合策略
- **WeightedFusion**: 学习权重加权（简单高效）
- **MoEFusion**: 专家混合（更强表达力）

### 3. Context增强（新增）
- 使用Transformer捕捉时序依赖
- Positional Encoding编码时间信息
- 仅在训练时启用（checkpoint中有相关权重时）

### 4. 冻结使用
DP训练时通常冻结此编码器（freeze=True），仅训练DP Head

---

## 使用示例

### 加载编码器
```python
from features_common.rgb2pc_aligned_encoder_4models import RGB2PCAlignedEncoder4Models

encoder = RGB2PCAlignedEncoder4Models.from_checkpoint(
    "outputs/train_rgb2pc_runs/run_best_bs32/ckpt_step_0010000.pt",
    freeze=True
)
encoder = encoder.cuda().eval()
```

### 前向推理
```python
# 输入: 4个模型的特征
obs = torch.randn(4, 2, 4, 2048).cuda()  # [B=4, To=2, M=4, C=2048]

# 前向
with torch.no_grad():
    aligned_feat = encoder(obs)  # [4, 2, 1280]

# 传给DP Head
action = dp_head(aligned_feat)  # [4, 8, 7]
```

---

## 为什么需要这个文件？

### 问题背景
1. **点云特征很准确但推理慢**：ULIP提取点云特征效果好，但实时推理慢
2. **RGB特征快但与点云不对齐**：4个视觉模型快速，但语义空间不同

### 解决方案
通过对齐训练，让RGB特征学习点云特征的语义表示：
- 训练时：RGB特征 + 点云特征 → InfoNCE对比学习
- 推理时：仅用RGB特征，效果接近点云

### 核心价值
**实现"无点云推理，有点云效果"的关键桥梁**

---

## 与train_rgb2pc_distill.py的关系

| 文件 | 作用 | 时机 |
|------|------|------|
| `train_rgb2pc_distill.py` | **训练**对齐编码器 | 阶段2 |
| `rgb2pc_aligned_encoder_4models.py` | **加载并使用**训练好的编码器 | 阶段3/4 |

**流程**：
1. train_rgb2pc_distill.py 训练并保存 → checkpoint
2. rgb2pc_aligned_encoder_4models.py 加载checkpoint → 冻结使用

---

## GPU使用说明

**当前代码中的gpu_ids[0]**：
```python
encoder = encoder.to(f'cuda:{gpu_ids[0]}')
obs = batch['obs'].to(f'cuda:{gpu_ids[0]}')
```

**含义**：
- `gpu_ids` 是配置中的GPU列表，如 `[0, 1, 2, 3]`
- `gpu_ids[0]` 表示第一张GPU（通常是CUDA:0）
- **只用一张GPU训练DP Head**，其他GPU用于特征提取

**为什么不用多卡训练**：
1. 在线训练的瓶颈在特征提取（4个大模型），已分配到多GPU
2. DP Head本身较小，单卡足够
3. 避免DDP通信开销

**如果要多卡训练DP**：
需要使用 `torch.nn.DataParallel` 或 `DistributedDataParallel` 包装policy
