# DP2DP3 完整流程详解

## 目录
1. [整体架构](#整体架构)
### 步骤2: RGB特征提取
**脚本**: `tools/features/run_extract_features.py`
4. [关键文件说明](#关键文件说明)
5. [调试指南](#调试指南)
**脚本**: `tools/features/extract_ulip_features_to_zarr.py`
---

## 整体架构

```
┌─────────────────────────────────────────────────────────────┐
```bash
cd /home/gl/RoboTwin/policy/DP2DP3/features_model
python tools/features/run_extract_features.py \
    --model vggt --window_size 8 --stride 1 --device cuda
```
│  └─ 输出: pc_dataset/PC_ORI, rgb_dataset/RGB_ORI            │
│                                                              │
│  步骤2: RGB特征提取 (run_extract_features.py)               │
│  ├─ RGB → 4模型特征 [W, T, Hf, Wf, C_i]                     │
│  └─ 输出: features_croco/vggt/dinov3/da3_zarr               │
│                                                              │
```bash
cd /home/gl/RoboTwin/policy/DP2DP3/features_model
python tools/features/run_extract_features.py \
    --model dinov3 --window_size 8 --stride 1 --device cuda
```
│  ├─ RGB token[K] → 对齐模块 → [1280]                        │
│  ├─ 监督: MSE + InfoNCE with 点云[1280]                     │
│  └─ 输出: ckpt_final.pt (对齐模块权重)                      │
│                                                              │
│  步骤5: 动作头训练 (两种方式)                               │
│  ├─ 在线: train_online_from_config.py                       │
```bash
cd /home/gl/RoboTwin/policy/DP2DP3/features_model
python tools/features/run_extract_features.py \
    --model da3 --window_size 8 --stride 1 --device cuda
```
│  └─ RGB → 4模型 → 对齐 → Head → 动作[14]                    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 数据流详解

### 步骤1: 数据提取
**脚本**: `features_model/tools/dataset/process_sapien_pcd.py` (也可以直接运行 `tools/dataset/process_sapien_pcd.py`)

**输入数据**:
```
/home/gl/RoboTwin/data/{task}/demo_randomized/data/
├── episode0.hdf5
├── episode1.hdf5
└── ...
```

**HDF5文件结构**:
```python
episode0.hdf5:
├── observation/
│   └── head_camera/
│       └── rgb: [T] JPEG bytes  # T=帧数，每帧存储为压缩JPEG
├── pointcloud/
│   └── head_camera: [T, N, 6]   # [x,y,z,r,g,b] 原生点云
└── joint_action/
    ├── left_arm: [T, 6]          # 左臂6关节角度
    ├── left_gripper: [T, 1]      # 左夹爪开合
    ├── right_arm: [T, 6]         # 右臂6关节角度
    └── right_gripper: [T, 1]     # 右夹爪开合
```

**执行命令** (在 `features_model` 仓库根目录下运行):
```bash
cd /home/gl/RoboTwin/policy/DP2DP3/features_model
python tools/dataset/process_sapien_pcd.py \
    beat_block_hammer demo_randomized 20 \
    --output_root /home/gl/RoboTwin/policy/DP2DP3/features_model \
    --camera head_camera
```

**输出数据**:
```
features_model/
├── pc_dataset/PC_ORI/
│   └── beat_block_hammer-demo_randomized-20_sapien_head_camera/
│       ├── episode_0/
│       │   ├── step_0000.ply    # [N,6] 点云
│       │   ├── step_0001.ply
│       │   └── ...
│       └── episode_1/
└── rgb_dataset/RGB_ORI/
    └── beat_block_hammer-demo_randomized-20_sapien_head_camera/
        └── episode_0/
            ├── step_0000.png    # [H,W,3] RGB图像
            ├── step_0001.png
            └── ...
```

**数据形态**:
- **点云PLY**: 每帧N个点(N不固定)，每点6维 [x,y,z,r,g,b]
- **RGB图像**: [480, 640, 3] uint8, RGB格式

---

### 步骤2: RGB特征提取
**脚本**: `tools/features/run_extract_features.py`

**涉及的4个视觉模型**:
1. **CroCo**: 输出维度 1024
2. **VGGT**: 输出维度 2048  
3. **DINOv3**: 输出维度 768
4. **DA3 (Depth Anything v3)**: 输出维度 2048

**执行命令**:
```bash
cd /home/gl/RoboTwin/policy/DP2DP3/features_model

# 提取CroCo特征
python tools/features/run_extract_features.py \
    --model croco \
    --rgb_root rgb_dataset/RGB_ORI \
    --window_size 8 --stride 1

# 提取VGGT特征
python tools/features/run_extract_features.py \
    --model vggt \
    --rgb_root rgb_dataset/RGB_ORI \
    --window_size 8 --stride 1

# 提取DINOv3特征
python tools/features/run_extract_features.py \
    --model dinov3 \
    --rgb_root rgb_dataset/RGB_ORI \
    --window_size 8 --stride 1

# 提取DA3特征
python tools/features/run_extract_features.py \
    --model da3 \
    --rgb_root rgb_dataset/RGB_ORI \
    --window_size 8 --stride 1
```

**输出数据**:
```
features_model/rgb_dataset/
├── features_croco_encoder_dict_unified_zarr/
│   └── beat_block_hammer-demo_randomized-20_sapien_head_camera/
│       └── episode_0.zarr/
│           └── features: [W, T, Hf, Wf, 1024]  # W=窗口数, T=8帧, Hf×Wf=patch数
├── features_vggt_encoder_dict_unified_zarr/
│   └── .../features: [W, T, Hf, Wf, 2048]
├── features_dinov3_encoder_dict_unified_zarr/
│   └── .../features: [W, T, Hf, Wf, 768]
└── features_da3_encoder_dict_unified_zarr/
    └── .../features: [W, T, Hf, Wf, 2048]
```

**数据形态**:
- **窗口滑动**: W = (总帧数 - 8) / 1 + 1
- **时间维度**: T = 8 (window_size)
- **空间维度**: Hf×Wf ≈ 30×40 (取决于模型)
- **特征维度**: C ∈ {1024, 2048, 768, 2048}

---

### 步骤3: 点云特征提取
**脚本**: `tools/features/extract_ulip_features_to_zarr.py`

**ULIP模型**:
- **输入**: 2048个点 [2048, 6] (FPS采样)
- **输出**: per-point 特征（维度为 、768（backbone）或 1280（ULIP2 在模型中做了 projection））。通常对点特征做 pooling 得到全局向量 [D]，并可通过 MLP 投影到对齐维度（例如 1280）。

**执行命令**:
```bash
cd /home/gl/RoboTwin/policy/DP2DP3/features_model
python tools/features/extract_ulip_features_to_zarr.py
```

**输出数据**:
```
features_model/pc_dataset/ulip_features_zarr/
└── beat_block_hammer-demo_randomized-20_sapien_head_camera/
    └── episode_0.zarr/
        ├── step_0000.zarr/
        │   ├── pc: [N, 3]           # 原始点云坐标
        │   └── pc_feat: [1280]      # ULIP特征
        └── step_0001.zarr/
```

**数据形态**:
- **点云**: [N, 3] 归一化到单位球
- **特征**: [1280] float32

---

### 步骤4: 对齐训练 ⭐核心步骤
**脚本**: `features_model/tools/alignment/train_rgb2pc_distill.py`
**配置**: `configs/alignment/train_rgb2pc_distill_default.yaml`

**训练目标**: 让RGB特征学习点云特征的空间感知能力

**网络结构**:
```
RGB多模态特征（原始token）[B, 4, K, C_i]
    # C_i ∈ {1024, 2048, 768, 2048}，来自 CroCo/VGGT/DINOv3/DA3
    # 这些token来自Zarr: [W, T, Hf, Wf, C_i]
    # K = 每个样本从单帧(或window)中随机采样的token数量（K≪Hf×Wf）
    # 训练时不直接堆叠为[B,4,W,T,Hf,Wf,C]，而是先随机采样出K个token再组成batch
    ↓
（可选）token池化 → RGB多模态特征（向量）[B, 4, C_i]
    ↓
4个Adapter MLP (统一维度)
    ├─ croco: 1024 → 1280
    ├─ vggt:  2048 → 1280
    ├─ dinov3: 768 → 1280
    └─ da3:   2048 → 1280
    ↓
Weighted Fusion (可学习权重)
    ↓
[B, 1280] 融合特征
    ↓
Transformer Context Encoder (2层，可选)
    ↓
Projection MLP
    ↓
[B, 1280] 最终对齐特征 ← 监督信号来自点云[1280]
```

**损失函数**:
```python
# 1. MSE Loss: 强制数值对齐
loss_mse = MSE(student_feat, teacher_feat)

# 2. InfoNCE Loss: 语义对比学习
loss_nce = InfoNCE(student_feat, teacher_feat, tau=0.07)

# 总损失
total_loss = loss_nce + 1.0 * loss_mse
```

**执行命令**:
```bash
cd /home/gl/RoboTwin/policy/DP2DP3/features_model
python tools/alignment/train_rgb2pc_distill.py \
    --config configs/alignment/train_rgb2pc_distill_default.yaml
```

**关键参数**:
```yaml
tasks: [5个任务]               # 多任务泛化
batch_size: 32                  # InfoNCE需要大batch
steps: 20000                    # 约14小时@单GPU
lr: 0.0003                      # 学习率
tau: 0.07                       # InfoNCE温度
loss_mse: 1.0                   # MSE权重 (关键修复!)
loss_rgb: 0.1                   # RGB自对比权重，保留视觉语义
rgb_tau: 0.07                   # RGB自对比温度
student_pool: mean              # 与推理一致的池化方式
```

**输出checkpoint**:
```
outputs/train_rgb2pc_runs/run_best_bs32/
├── ckpt_step_0001000.pt
├── ckpt_step_0002000.pt
├── ...
└── ckpt_final.pt    ← 用于后续训练
```

**checkpoint内容**:
```python
{
    'adapters': {4个MLP的权重},
    'fusion': {Weighted Fusion权重},
    'context_encoder': {Transformer权重},
    'pos_encoder': {位置编码},
    'proj_student': {投影层权重},
    'args': {训练超参数}
}
```

---

### 2048维的来源与含义（不是点云2048）

**结论**：在线推理阶段的 `2048` 维是**视觉特征对齐时的统一维度**，
与点云的 `2048` 采样点数**没有直接关系**。

- 4个视觉模型原始维度分别是：`[1024, 2048, 768, 2048]`
- 在线推理为了让 4 模型能统一堆叠，采用**固定维度 2048**：
  - 如果维度不足就 `pad`，
  - 如果维度过大就截断。
- 这个设计是**工程上的统一接口**，不是物理含义。

**对齐训练时**不会强制 2048：
- token级对齐使用的是各自的 `C_i`
- adapter 会统一映射到 1280

---

### 训练/推理数据流（pool vs tokens 全流程）

下面按 **“训练对齐 + 在线推理”** 两条主线说明，并明确 `tokens` 与 `pool` 的区别。

#### A) 对齐训练（student_pool=tokens）
1. **RGB特征（Zarr）**
    - `[W, T, Hf, Wf, C_i]`
2. **采样K个token**
    - 每模型 `[K, C_i]`
    - 多模型堆叠：`[B, 4, K, C_i]`
3. **Adapter（每token）**
    - `C_i → 1280` 得到 `[B, 4, K, 1280]`
4. **Fusion（token级）**
    - 融合到 `[B, K, 1280]`
5. **Context Encoder（token级）**
    - 保持 `[B, K, 1280]`
6. **Token Pool（mean 或 attn）**
    - 得到样本级向量 `[B, 1280]`
7. **Projection → 对齐监督（点云向量）**
    - 学到 RGB→PC 对齐向量

#### B) 对齐训练（student_pool=mean）
1. 仍从 `[Hf, Wf, C_i]` 中取 token
2. **先做 mean pool** → `[B, 4, C_i]`
3. Adapter/Fusion/Projection → `[B, 1280]`
4. 与点云向量对齐

#### C) 在线推理（当前 token-level 模式）
1. **实时图像**：每步 `[H, W, 3]`
2. **在线模型输出 token**：每模型 `[K, C_i]`
3. **堆叠成 `[1, To, K, C_i]`**
4. **对齐编码器（token路径）** → `[1, To, 1280]`
5. **Diffusion Head** → `[1, horizon, 14]`
6. **Receding Horizon 执行前 action_exec 步**

#### D) 在线推理（pooled 模式）
1. 在线模型先 pool → 每模型 `[2048]`
2. 堆叠成 `[1, To, 4, 2048]`
3. 对齐编码器（pooled路径） → `[1, To, 1280]`

**核心区别**：
- `tokens` 模式保留了局部token语义，aligner内部再做 pooling
- `mean` 模式在提取器就池化，aligner只看全局向量

---

### 五种方案总览（清晰对比版）

下面按你的 5 种方案给出 **输入/输出形状**、**训练/推理脚本** 与 **关键文件**。

#### 方案1：单模型（Single Model）
- **核心思想**：只用一个视觉模型（如 DINOv3），避免融合干扰。
- **输入形状**：
    - 训练：`[B, To, D]`（单模型特征）
    - 推理：`[1, To, D]`
- **关键文件**：
    - 训练：`tools/single_model/train_single_model_offline.py`
    - 配置：`configs/single_model/train_single_dinov3.yaml`
    - 推理：`policy/DP2DP3/deploy_single_model_policy.py`
    - 评估脚本：`policy/DP2DP3/eval_single_model.sh`
- **典型命令**：
    ```bash
    cd /home/gl/RoboTwin/policy/DP2DP3/features_model
    python tools/single_model/train_single_model_offline.py \
            --config configs/single_model/train_single_dinov3.yaml
    ```
    - **推理命令**：
        ```bash
        cd /home/gl/RoboTwin
        bash policy/DP2DP3/eval_single_model.sh \
                lift_pot demo_clean demo_clean 50 0 "0,1" best dinov3
        ```

#### 方案2：直融（Direct Fusion, 无对齐模块）
- **核心思想**：4 模型特征直接融合，不经过 RGB→PC 对齐。
- **输入形状**：
    - 训练：`[B, To, 4, D_i]`（每模型维度不同）
    - 融合后：`[B, To, D_fused]`（weighted/concat/mean）
    - 推理：`[1, To, D_fused]`
- **关键文件**：
    - 训练：`tools/direct_fusion/train_direct_fusion_offline.py`
    - 配置：`configs/direct_fusion/train_direct_fusion_{task}.yaml`
    - 推理：`policy/DP2DP3/deploy_direct_fusion_policy.py`
    - 评估脚本：`policy/DP2DP3/eval_direct_fusion.sh`
- **典型命令**：
    ```bash
    cd /home/gl/RoboTwin/policy/DP2DP3/features_model
    python tools/direct_fusion/train_direct_fusion_offline.py \
            --config configs/direct_fusion/train_direct_fusion_lift_pot.yaml
    ```
    - **推理命令**：
        ```bash
        cd /home/gl/RoboTwin
        bash policy/DP2DP3/eval_direct_fusion.sh \
                lift_pot demo_clean demo_clean 50 0 "0,1" best
        ```

#### 方案3：Mean 方式的纯 Pool 对齐（student_pool=mean）
- **核心思想**：先对每模型 token 做 mean pool，再进入对齐模块。
- **输入形状**：
    - Token采样：`[B, 4, K, C_i]`
    - Mean pool 后：`[B, 4, C_i]`
    - 对齐输出：`[B, 1280]`
- **关键文件**：
    - 对齐训练：`tools/alignment/train_rgb2pc_distill.py`
    - 对齐配置：`configs/alignment/train_rgb2pc_distill_default.yaml`
    - Head训练：`tools/offline/train_offline_head.py`
    - 推理：`policy/DP2DP3/deploy_policy.py`（`use_token_infer=false`）
    - **训练命令**：
        ```bash
        cd /home/gl/RoboTwin/policy/DP2DP3/features_model
        python tools/offline/train_offline_head.py \
                --config configs/head/train_offline.yaml
        ```
    - **推理命令**：
        ```bash
        cd /home/gl/RoboTwin
        bash policy/DP2DP3/eval.sh \
                lift_pot demo_clean demo_clean 50 0 "0,1" best 4 100
        ```

#### 方案4：双流（pool + token pool）
- **核心思想**：对齐模块输出 **pooled 全局特征** + **token 分支**（token 经过 pooling）。
- **输入形状**：
    - token输入：`[B, To, K, C_i]`
    - token路径输出：`[B, To, K, 1280]` → pooled → `[B, To, 1280]`
    - global路径输出：`[B, To, 1280]`
    - Head条件：`[B, To, 1280]` + token ctx
- **关键文件**：
    - 对齐：`features_common/alignment/rgb2pc_aligned_encoder_4models.py`
    - Head：`tools/offline/train_offline_head.py`
    - 推理：`policy/DP2DP3/deploy_policy.py`
    - **训练命令**：
        ```bash
        cd /home/gl/RoboTwin/policy/DP2DP3/features_model
        python tools/offline/train_offline_head.py \
                --config configs/head/train_offline_dual_stream.yaml
        ```
    - **推理命令**：
        ```bash
        cd /home/gl/RoboTwin
        bash policy/DP2DP3/eval.sh \
                lift_pot demo_clean demo_clean_dual_stream 50 0 "0,1" best 4 100
        ```

#### 方案5：双流（pool + token-full）
- **核心思想**：使用 **1024 全 token**，保留空间细节，token 分支不做提前池化。
- **输入形状**：
    - token输入：`[B, To, K=1024, C_i]`
    - 对齐输出：`[B, To, 1280]` + `obs_tokens: [B, To, K, 1280]`
    - Head条件：`[B, To, 1280]` + `obs_tokens`
- **关键文件**：
    - 特征提取：`tools/offline/extract_offline_features.py`
    - token-full配置：`configs/head/train_online_batch_extract_dual_stream_tokens_full.yaml`
    - Head训练：`tools/offline/train_offline_head.py`
    - 推理：`policy/DP2DP3/deploy_policy.py`（`use_token_infer=true`）
    - **训练命令**：
        ```bash
        cd /home/gl/RoboTwin/policy/DP2DP3/features_model
        python tools/offline/train_offline_head.py \
                --config configs/head/train_offline_dual_stream_tokens_full.yaml
        ```
    - **推理命令**：
        ```bash
        cd /home/gl/RoboTwin
        bash policy/DP2DP3/eval.sh \
                lift_pot demo_clean demo_clean_dual_stream_tokens_full 50 0 "0,1" best 4 100
        ```

---

---

## 步骤4B: 直接融合训练 (无对齐模块) ⭐简化方案

**目的**: 跳过复杂的RGB→PC对齐步骤，直接融合4个视觉模型的RGB特征训练动作头

**适用场景**:
- 快速baseline验证
- 对齐模块效果不理想时的替代方案  
- 训练时间紧张

**脚本**: `tools/direct_fusion/train_direct_fusion_offline.py`
**配置**: `configs/direct_fusion/train_direct_fusion_{task}.yaml`

### 网络架构

```
离线Zarr特征 [W, T, Hf, Wf, Dim]
    ↓ (加载4个模型特征)
    
CroCo:  [B, To, 1024]  ─┐
VGGT:   [B, To, 2048]  ─┤
DINOv3: [B, To, 768]   ─┤ → SimpleFusionEncoder
DA3:    [B, To, 2048]  ─┘
    
    ↓ (特征对齐到统一维度)
    
[B, To, 1024]  (通过4个linear projections)
[B, To, 2048]
[B, To, 768]
[B, To, 2048]
    
    ↓ (融合方式: weighted/concat/mean)
    
融合特征 [B, To, 5896]  (concat模式)
或 [B, To, 2048]        (weighted/mean模式)
    
    ↓ (DirectFusionDPPolicy)
    
动作序列 [B, Ta, 14]
```

### SimpleFusionEncoder 实现

**支持3种融合方式**:

1. **weighted** (加权融合):
```python
# 可学习权重 α = [α1, α2, α3, α4], 归一化后求和
feat_fused = α1·feat1 + α2·feat2 + α3·feat3 + α4·feat4
# 输出: [B, To, 2048]
```

2. **concat** (拼接):
```python
# 直接拼接所有特征
feat_fused = concat(feat1, feat2, feat3, feat4)
# 输出: [B, To, 1024+2048+768+2048] = [B, To, 5896]
```

3. **mean** (平均):
```python
# 先对齐维度，再平均
feat_fused = mean(proj1(feat1), proj2(feat2), ...)
# 输出: [B, To, 2048]
```

### 配置文件示例

**configs/direct_fusion/train_direct_fusion_lift_pot.yaml**:
```yaml
data:
  tasks:
    - lift_pot
  data_roots:
    - /home/gl/RoboTwin/data
  setting: demo_clean
  num_demos: 50
  
  # Zarr特征路径
  features_dataset_dir: /home/gl/RoboTwin/policy/DP2DP3/features_model/data/offline_features
  
  horizon: 8           # 预测8步轨迹
  n_obs_steps: 2       # 观测2帧历史
  use_left_arm: true
  use_right_arm: true
  include_gripper: true  # ⚠️ 必须true，输出14维
  
fusion:
  type: weighted       # 融合方式: weighted/concat/mean
  model_dims:          # 4个模型的维度
    - 1024  # CroCo
    - 2048  # VGGT
    - 768   # DINOv3
    - 2048  # DA3
  
policy:
  type: SimpleDPHead   # 简单MLP头
  hidden_dim: 512
  
train:
  batch_size: 256      # 离线模式可用大batch
  epochs: 500
  lr: 1e-4
  num_workers: 4

device:
  gpu_ids: [0]

output:
  dir: /home/gl/RoboTwin/policy/DP2DP3/checkpoints_direct_fusion
  save_every_n_epochs: 50
```

### 执行命令

```bash
cd /home/gl/RoboTwin/policy/DP2DP3/features_model

# 单任务训练
python tools/direct_fusion/train_direct_fusion_offline.py \
    --config configs/direct_fusion/train_direct_fusion_lift_pot.yaml

# 后台运行
nohup python tools/direct_fusion/train_direct_fusion_offline.py \
    --config configs/direct_fusion/train_direct_fusion_lift_pot.yaml \
    > logs/direct_fusion_lift_pot.log 2>&1 &
```

### 训练输出

**Checkpoint结构**:
```
/home/gl/RoboTwin/policy/DP2DP3/checkpoints_direct_fusion/
└── lift_pot-demo_clean-50-0/
    ├── best.ckpt
    ├── 50.ckpt
    ├── 100.ckpt
    └── ...
```

**Checkpoint内容**:
```python
{
    'config': {
        'data': {...},
        'fusion': {'type': 'weighted', 'model_dims': [1024, 2048, 768, 2048]},
        'policy': {...}
    },
    'policy': OrderedDict([
        ('fusion_encoder.fusion_weights', ...),  # 如果weighted模式
        ('fusion_encoder.projectors.0.weight', ...),  # 4个projection层
        ('net.0.weight', ...),  # SimpleDPHead的MLP
        ...
    ])
}
```

### 部署推理

**部署脚本**: `policy/DP2DP3/deploy_direct_fusion_policy.py`

**评估命令**:
```bash
cd /home/gl/RoboTwin

# 使用直接融合checkpoint
bash policy/DP2DP3/eval_direct_fusion.sh \
    lift_pot \
    demo_clean \
    demo_clean \
    50 \
    0 \
    1 \
    50
```

**参数说明**:
```bash
eval_direct_fusion.sh <task> <train_setting> <eval_setting> <num_demos> <seed> <gpu_id> <checkpoint_num>
```

**多GPU说明**:
- `gpu_id` 支持传入逗号分隔的 GPU 列表，例如 `"0,1"`。

### 与对齐方案对比

| 特性 | 对齐方案 (步骤4) | 直接融合 (步骤4B) |
|-----|-----------------|------------------|
| **训练时间** | ~7小时 (对齐) + 3小时 (head) | ~3小时 (融合+head一起) |
| **模型复杂度** | 高 (Adapters+Context+Projection) | 低 (简单融合+Head) |
| **特征语义** | RGB→PC对齐 | 保留RGB语义 |
| **适用场景** | 需要点云空间信息 | RGB信息足够 |
| **调试难度** | 高 | 低 |
| **推理速度** | 中等 | 快 |

### 选择建议

**优先使用直接融合(步骤4B)的情况**:
1. ✅ 快速验证baseline
2. ✅ 对齐训练效果不理想
3. ✅ RGB信息足够解决任务
4. ✅ 训练时间有限

**使用对齐方案(步骤4)的情况**:
1. ✅ 任务需要3D空间理解
2. ✅ 有充足训练时间
3. ✅ 追求极致性能

### 训练监控

**Loss监控**:
```python
# 正常训练曲线
Epoch 0: loss=2.345
Epoch 50: loss=0.521
Epoch 100: loss=0.184
Epoch 200: loss=0.089
Epoch 500: loss=0.032  # 收敛良好
```

**特征检查**:
```python
# 在训练循环中添加:
if step % 100 == 0:
    with torch.no_grad():
        feat_fused = fusion_encoder(features)
        print(f"Fused feat: mean={feat_fused.mean():.4f}, "
              f"std={feat_fused.std():.4f}")
```

**动作范围检查**:
```python
# 每个epoch结束时:
with torch.no_grad():
    action_pred = policy(obs_emb)
    print(f"Action: min={action_pred.min():.3f}, "
          f"max={action_pred.max():.3f}")
# 期望范围: [-2.5, 2.5]
```

### 常见问题

**Q1: weighted模式权重如何初始化？**
```python
# 均匀初始化
self.fusion_weights = nn.Parameter(torch.ones(4) / 4)
# 或基于模型容量初始化
# CroCo=1024, VGGT=2048, DINOv3=768, DA3=2048
# 权重 ∝ dim → [0.18, 0.35, 0.13, 0.35]
```

**Q2: concat模式维度过大怎么办？**
```python
# 方案1: 添加降维层
self.down_proj = nn.Linear(5896, 1280)
feat_fused = self.down_proj(feat_concat)

# 方案2: 改用weighted模式
fusion:
  type: weighted
```

**Q3: 训练loss不降？**
- 检查学习率 (建议1e-4)
- 检查batch size (建议≥64)
- 检查数据是否加载正确
- 增加训练轮数

**Q4: 推理效果差？**
- 检查推理时特征提取顺序 (必须与训练一致)
- 检查动作维度 (必须14维)
- 检查动作范围clip
- 增加训练数据量

---

### 单帧（window_size=1）流程与数据形状说明

以下按常见配置（`window_size=1`, `horizon=8`, `n_obs_steps=2`, `action_exec=4`）说明：

1) 原始观测
- 每步 RGB 图像: `[H, W, 3]`（例如 480×640×3）

2) RGB 特征提取（window_size=1）
- Zarr 中保存的特征：`[W, T, Hf, Wf, C_i]`
- 当 `window_size=1` 时 `T=1`，因此形状实际是：`[W, 1, Hf, Wf, C_i]`
- `W≈总帧数`（window_size=1, stride=1 时每帧一个窗口）

3) 对齐训练采样（step 粒度）
- 从单帧中采样 `K` 个 patch token：每模型 `[K, C_i]`
- 多模型堆叠：`[B, 4, K, C_i]`
- 若 `student_pool=mean`，池化后：`[B, 4, C_i]`
- Adapter → 融合 → 投影后：`[B, 1280]`
- 当前 `student_pool=tokens` 表示训练时保留 token 级上下文并做 attention pooling

4) 离线特征生成（用于 head）
- 推理/离线特征提取时只提供每帧 pooled 向量
- MultiGPU extractor 输出：`[To, 4, 2048]`（每帧 4 模型向量）
- 对齐 encoder 输出：`[To, 1280]`
- 整个 episode：`obs_aligned: [T, 1280]`，`action: [T, 14]`

5) Head 训练输入输出
- `n_obs_steps=2`：取最近 2 帧对齐特征
- 输入形状：`[B, 2, 1280]`
- `horizon=8`：预测未来 8 步动作
- 输出形状：`[B, 8, 14]`

6) 推理执行（Receding Horizon）
- 模型输出：`[8, 14]`
- `action_exec=4`：只执行前 4 步，再重新规划
- 好处：减少抖动，同时保持响应性

## 步骤5: 动作头训练

### 说明

如果使用**步骤4B直接融合**，则**跳过步骤5**，因为融合权重和动作头已经一起训练完成。

如果使用**步骤4对齐训练**，则需要继续步骤5A或5B训练动作头。

---

### 步骤5A: 动作头在线训练
**脚本**: `features_model/tools/online/train_online_from_config.py`
**配置**: `configs/head/train_online_batch_extract.yaml`

**数据流**:
```
HDF5原始数据
    ↓ (实时读取)
RGB图像 [B, To, 3, H, W]
    ↓ (4个视觉模型, GPU0+1)
RGB特征 [B, To, 4, 2048]
    ↓ (对齐模块, frozen)
对齐特征 [B, To, 1280]
    ↓ (Diffusion Policy Head)
动作序列 [B, Ta, 14]
```

**动作头结构 (Diffusion Policy)**:
```
观测编码器:
  [B, To×1280] → MLP → [B, 256]

Diffusion UNet:
    输入: noisy_action [B, Ta, 14]  # Ta=horizon，不是n_obs_steps
  条件: obs_cond [B, 256]
  输出: noise_pred [B, Ta, 14]
  
训练: 预测加噪动作的噪声
推理: 从随机噪声逐步去噪生成动作
```

**执行命令**:
```bash
cd /home/gl/RoboTwin/policy/DP2DP3/features_model
python tools/online/train_online_from_config.py \
    --config configs/head/train_online_batch_extract.yaml
```

**关键参数**:
```yaml
data:
    horizon: 8
    n_obs_steps: 2
    batch_size: 8       # 批量提取模式
  
policy:
  type: OfficialDP    # 使用正版Diffusion Policy
  num_inference_steps: 100
  
train:
  epochs: 500
  lr: 1e-4
```

**输出checkpoint**:
```
/home/gl/RoboTwin/policy/DP2DP3/checkpoints/
└── beat_block_hammer-demo_randomized-20-0/
    ├── 50.ckpt
    ├── 100.ckpt
    └── ...
```

---

### 步骤5B: 动作头离线训练 (推荐)
**脚本**: 
- `tools/offline/extract_offline_features.py` (特征预提取)
- `tools/offline/train_offline_head.py` (训练)

**优势**: 
- 特征只提取一次，训练速度快10倍+
- 适合快速迭代超参数

**完整流程**:
```bash
cd /home/gl/RoboTwin/policy/DP2DP3/features_model

# 步骤1: 预提取对齐特征
python tools/offline/extract_offline_features.py \
    --config configs/head/train_online_batch_extract.yaml \
    --output_dir data/offline_features

python tools/offline/extract_offline_features.py --config configs/head/train_online_batch_extract_dual_stream_tokens_full.yaml --output_dir data/offline_features_dual_stream_tokens_full --overwrite

# 步骤2: 离线训练Head
python tools/offline/train_offline_head.py \
    --config configs/head/train_offline.yaml
```

**预提取数据**:
```
data/offline_features/
└── beat_block_hammer/
    └── episode_0.zarr/
        ├── obs_aligned: [T, 1280]   # 对齐后的观测特征
        └── action: [T, 14]          # 对应的动作标签
```

---

### 步骤6: 推理部署
**脚本**: `policy/DP2DP3/deploy_policy.py`
**配置**: `policy/DP2DP3/deploy_policy.yml`

**推理流程**:
```
环境观测
    ↓
RGB图像 [3, H, W]
    ↓
观测缓冲区 deque(maxlen=2)
    ↓ (每次推理)
2帧图像 batch
    ↓ (4个视觉模型)
特征 [2, 4, 2048]
    ↓ (对齐模块)
对齐特征 [1, 2, 1280]
    ↓ (Diffusion Head)
动作序列 [1, 8, 14]
    ↓ (Receding Horizon: 只执行前2步)
执行动作 [14]
    ↓
环境执行 → 获取新观测 → 循环
```

**Receding Horizon策略**:
```python
# 预测8步
actions = policy.predict()  # [8, 14]

# 只执行前2步
for i in range(2):
    env.step(actions[i])
    
# 重新观测，再次预测
# 这样可以根据实际执行结果及时修正
```

**执行命令**:
```bash
cd /home/gl/RoboTwin
bash policy/DP2DP3/eval.sh beat_block_hammer demo_randomized demo_randomized 20 0 1 3000  
```

**环境说明**:
- 训练环境使用 `depth3`（训练依赖更完整）
- 推理评估使用 `RoboTwin`（包含 `mplib` 依赖）

**多GPU评估示例**:
```bash
# 对齐DP2DP3（多GPU）
bash policy/DP2DP3/eval.sh lift_pot demo_clean demo_clean 50 0 "0,1" 600 4 100

# 直接融合（多GPU）
bash policy/DP2DP3/eval_direct_fusion.sh lift_pot demo_clean demo_clean 50 0 "0,1" 600 4 100
```

---

## 关键文件说明

### 配置文件

1. **对齐训练配置**: `configs/alignment/train_rgb2pc_distill_default.yaml`
```yaml
# 关键参数
tasks: [5个任务列表]
batch_size: 32
loss_mse: 1.0        # ⭐必须>0，否则无法学到空间位置
loss_rgb: 0.1        # RGB自对比，保留视觉语义
tau: 0.07            # InfoNCE温度
```

2. **在线训练配置**: `configs/head/train_online_batch_extract.yaml`
```yaml
data:
    horizon: 8
    n_obs_steps: 2
  batch_extract: true  # 批量提取加速
```

3. **离线训练配置**: `configs/head/train_offline.yaml`
```yaml
data:
  features_dataset_dir: data/offline_features
    horizon: 8
    n_obs_steps: 2
train:
  batch_size: 256    # 离线模式可以用大batch
  epochs: 3000
```

### 核心代码文件

1. **对齐模块**: `features_common/rgb2pc_aligned_encoder_4models.py`
   - 4个Adapter + Fusion + Context Encoder + Projection
   - 输入: [B, To, 4, 2048]
   - 输出: [B, To, 1280]

2. **特征提取器**: `features_common/multi_gpu_extractors.py`
   - 管理4个视觉模型的多GPU分配
   - 支持批量提取: `extract_batch(images) -> [B, 4, 2048]`

3. **在线Dataset**: `features_common/dp_rgb_dataset_from_hdf5.py`
   - 从HDF5实时读取图像和动作
   - 支持批量特征提取模式

4. **Diffusion Policy**: `tools/train_online_from_config.py` (L60-158)
   - 观测编码器 + ConditionalUnet1D
   - 训练: 噪声预测 + MSE loss
   - 推理: DDPM采样

5. **部署接口**: `policy/DP2DP3/deploy_policy.py`
   - RoBoTwin标准接口: `get_model()`, `eval()`, `reset_model()`
   - Receding Horizon执行策略

## 调试指南

### 1. 检查对齐训练是否收敛

在训练过程中观察:
```python
# 正常情况:
pos=0.35-0.38    # 正样本相似度 (越高越好)
neg=0.02-0.05    # 负样本相似度 (越低越好)
gap=0.30-0.35    # 正负差距 (越大越好)
ema=0.55-0.60    # EMA loss (下降趋势)
```

### 2. 验证特征分布对齐

```python
# 在train_online_from_config.py中添加:
with torch.no_grad():
    obs_encoded = encoder(obs)
    print(f"Aligned feat: mean={obs_encoded.mean():.4f}, "
          f"std={obs_encoded.std():.4f}")
    
# 正常范围: mean ∈ [-0.1, 0.1], std ∈ [0.8, 1.2]
```

### 3. 监控动作输出

```python
# 在deploy_policy.py的get_action()中:
print(f"Action: min={action_pred.min():.3f}, "
      f"max={action_pred.max():.3f}, "
      f"mean={action_pred.mean():.3f}")
      
# 正常范围: [-2.5, 2.5] (关节限位)
# 异常情况: 全0或全相同值
```

### 4. 常见错误排查

| 现象 | 可能原因 | 解决方案 |
|-----|---------|---------|
| 机械臂不动 | 动作全0 | 检查动作维度是否14维 |
| 原地乱晃 | 特征分布不对齐 | loss_mse>0重新训练对齐 |
| 动作爆炸 | 缺少clip | 添加`np.clip(action, -2.5, 2.5)` |
| Loss不降 | 学习率过大 | 降低lr到1e-5 |
| **第1秒乱动** | **推理反归一化错误** | **删除(action+1)/2*(MAX-MIN)+MIN** |
| **动作抖动/冻结** | **Receding Horizon执行步数过少** | **增加n_action_exec到4** |

---

## 问题诊断与修复 (2026-01-16 更新) ⚠️重要

### 问题1: 对齐模型"第1秒乱动" ✅已修复

**现象**: 使用对齐模块训练的模型，推理时机械臂在第1秒就乱七八糟移动

**根本原因**: 训练和推理的归一化不一致
- **训练时**: 动作没有归一化，直接使用原始范围 [-1.429, 2.552]
- **推理时**: 错误地添加了反归一化代码 `(action+1)/2*(MAX-MIN)+MIN`

**错误代码位置**: `deploy_policy.py` 第456-459行
```python
# ❌ 错误: 训练时没有归一化，推理却做反归一化
ACTION_MIN = -3.0
ACTION_MAX = 3.0
action_pred = (action_pred + 1.0) / 2.0 * (ACTION_MAX - ACTION_MIN) + ACTION_MIN
```

**修复方案**: 删除反归一化代码
```python
# ✅ 正确: 直接使用模型输出，不做任何变换
action_pred = action_pred.squeeze(0).cpu().numpy()
action_pred = np.clip(action_pred, -3.0, 3.0)  # 只做安全限制
```

**验证方法**:
```python
# 在 get_action() 中添加打印
print(f"Action range: [{action_pred.min():.3f}, {action_pred.max():.3f}]")
# 正常范围: [-2.5, 2.5]
# 异常范围: [-3.0, 3.0] (全是边界值)
```

---

### 问题2: 直接融合模型"抖动/冻结" ✅已修复

**现象**: 直接融合模型第1秒还能伸手去够物体，之后就开始抖动或冻结

**根本原因**: Receding Horizon执行步数过少
- **Horizon=8**: 每次预测8步轨迹
- **n_action_exec=2**: 只执行前2步就重新规划
- **问题**: 频繁重规划导致轨迹不连续，产生抖动

**错误代码位置**: `deploy_direct_fusion_policy.py` 第447行
```python
# ❌ 错误: 执行步数过少
n_action_exec = min(2, len(self.action_queue))
```

**修复方案**: 增加执行步数
```python
# ✅ 正确: 执行4步 (50%重叠)
n_action_exec = min(4, len(self.action_queue))
```

**Receding Horizon原理**:
```
预测Horizon=8:  [a0, a1, a2, a3, a4, a5, a6, a7]
执行K=4步:      [a0, a1, a2, a3]           → 执行
重新预测:                   [a4, a5, a6, ..., a11]
执行K=4步:                  [a4, a5, a6, a7]

K越大 → 轨迹越平滑，但响应性越差
K越小 → 响应性越好，但轨迹越不连续
推荐: K=H/2 (本例中K=4, H=8)
```

---

### 问题2B: 对齐路线成功率过低（新增排查）

**现象**：对齐模型推理成功率接近 0，远低于单模型/四模型融合。

**根本原因之一**：离线特征提取与Head训练的时间参数不一致。
- `extract_offline_features.py` 使用 `train_online_batch_extract_ws1.yaml`（horizon=8, n_obs_steps=2）
- 但 `train_offline_ws1.yaml` 仍是其它配置（例如 4/4）
- 这会导致观测窗口与动作标签错位，模型学到的时序关系无效

**修复方案**：确保离线Head训练参数与特征提取一致。
```yaml
# configs/head/train_offline_ws1.yaml
data:
    horizon: 8
    n_obs_steps: 2
```

**建议补充**：
1. 将对齐训练 steps 提升到 10k~20k（2k steps 往往不够收敛）。
2. 保持 `loss_mse=1.0`，避免只有 InfoNCE 导致空间对齐不足。
3. 加入 RGB 自对比损失，保留视觉语义。

---

### 问题2C: 8帧视觉窗口效果不佳

**现象**：使用 window_size=8 的多帧特征时，抓取意图被“平均”掉，动作更犹豫。

**原因**：
1. 8帧窗口会把动作前的“过渡帧”一起平均，导致关键帧语义被稀释。
2. 对齐训练是 step 粒度时，8帧窗口的时间聚合与推理的单帧缓冲不一致。

**建议**：
- 离线特征提取使用 ws1（window_size=1）或保持训练/推理同粒度。
- 如果必须用 8 帧窗口，训练时也要用 window 模式并在推理时保持一致的时间聚合。

---

### 问题2D: DINOv3 QKV特征误用

**现象**：单模型 DINOv3 成功率异常低，表现为动作漂移或无明显目标趋近。

**原因**：早期提取误用注意力层的 QKV 中间张量或未正确池化，导致特征尺度不稳定、语义不聚合。

**修复**：
- 仅使用 DINOv3 最终 patch token 输出，并进行均值池化。
- 确保特征与其它模型保持相同的归一化流程。

---

### 问题3: 训练时应该使用归一化吗？⭐理论分析

**Diffusion Policy最佳实践**: **强烈建议归一化**

#### 为什么要归一化？

1. **数值稳定性**
```python
# 不归一化: 动作范围差异大
left_arm: [-1.429, 2.552]  # 范围≈4
gripper: [0, 1]             # 范围=1

# 问题: Diffusion去噪过程不稳定
noise = torch.randn_like(action)  # noise ~ N(0,1)
noisy_action = action + noise     # 对gripper噪声过大，对arm噪声偏小
```

2. **训练效率**
```python
# 归一化: 统一范围到[-1, 1]
action_norm = (action - ACTION_MIN) / (ACTION_MAX - ACTION_MIN) * 2 - 1

# 优势:
# - 所有维度学习难度一致
# - 梯度更新更平衡
# - 收敛更快更稳定
```

3. **模型泛化**
```python
# 不归一化: 模型记住绝对值
model.predict() → [0.523, 1.234, ...]  # 绝对位置

# 归一化: 模型学习相对变化
model.predict() → [-0.1, 0.3, ...]  # 相对运动
```

#### 如何正确使用归一化？

**步骤1: 统计动作范围**
```python
# 在训练前分析数据
import h5py
import numpy as np

actions = []
for demo_file in demo_files:
    with h5py.File(demo_file, 'r') as f:
        actions.append(f['action'][:])
actions = np.concatenate(actions, axis=0)

ACTION_MIN = actions.min(axis=0)  # [14]
ACTION_MAX = actions.max(axis=0)  # [14]
# 或使用固定范围（基于机器人关节限位）
ACTION_MIN = np.array([-2.5]*6 + [0.0] + [-2.5]*6 + [0.0])
ACTION_MAX = np.array([2.5]*6 + [1.0] + [2.5]*6 + [1.0])
```

**步骤2: 训练时归一化**
```python
# Dataset的__getitem__中
action_raw = self.actions[idx]  # [Ta, 14]
action_norm = (action_raw - ACTION_MIN) / (ACTION_MAX - ACTION_MIN) * 2 - 1
return {'obs': obs, 'action': action_norm}
```

**步骤3: 推理时反归一化**
```python
# deploy_policy.py的get_action()中
action_norm = policy(obs)  # [-1, 1]
action_raw = (action_norm + 1) / 2 * (ACTION_MAX - ACTION_MIN) + ACTION_MIN
```

**步骤4: 保存归一化参数**
```python
# 训练时保存到checkpoint
torch.save({
    'policy': policy.state_dict(),
    'config': config,
    'action_stats': {
        'min': ACTION_MIN,
        'max': ACTION_MAX
    }
}, ckpt_path)

# 部署时加载
ckpt = torch.load(ckpt_path)
ACTION_MIN = ckpt['action_stats']['min']
ACTION_MAX = ckpt['action_stats']['max']
```

#### 当前代码的问题

**问题**: 训练和推理都没有归一化，但推理时错误地添加了反归一化

**临时解决方案**: 删除推理时的反归一化（已修复）

**长期解决方案**: 修改训练代码，加入归一化

#### 是否需要重新训练？

**不一定需要！** 当前修复后的代码应该可以正常工作，因为:
1. 训练和推理现在一致了（都不归一化）
2. 动作范围虽然不理想，但在可接受范围内
3. Clip操作提供了安全保护

**何时需要重新训练**:
- 如果当前效果仍然不好
- 如果发现训练不稳定或loss震荡
- 如果需要提升性能
- 如果要扩展到新任务

#### 归一化训练的实现

**修改位置1**: `train_direct_fusion_offline.py` 第200-250行（Dataset部分）
```python
class DPRGBOfflineZarrDataset(Dataset):
    def __init__(self, ...):
        # 添加归一化参数
        self.ACTION_MIN = np.array([-2.5]*6 + [0.0] + [-2.5]*6 + [0.0])
        self.ACTION_MAX = np.array([2.5]*6 + [1.0] + [2.5]*6 + [1.0])
    
    def __getitem__(self, idx):
        # ... 加载features和action ...
        
        # 归一化动作
        action_norm = (action - self.ACTION_MIN) / (self.ACTION_MAX - self.ACTION_MIN) * 2 - 1
        return features, action_norm  # 返回归一化后的动作
```

**修改位置2**: `train_online_from_config.py` 类似修改

**修改位置3**: 保存checkpoint时添加stats
```python
torch.save({
    'policy': policy.state_dict(),
    'config': config,
    'action_stats': {
        'min': dataset.ACTION_MIN.tolist(),
        'max': dataset.ACTION_MAX.tolist()
    }
}, ckpt_path)
```

**修改位置4**: 推理时反归一化
```python
# deploy_*.py 的 get_action() 中
action_norm = policy(obs)
ACTION_MIN = torch.tensor(ckpt['action_stats']['min'])
ACTION_MAX = torch.tensor(ckpt['action_stats']['max'])
action_raw = (action_norm + 1) / 2 * (ACTION_MAX - ACTION_MIN) + ACTION_MIN
```

---

### 总结: 当前状态和后续建议

#### ✅ 已修复的问题
1. **对齐模型乱动**: 删除了错误的反归一化代码
2. **直接融合抖动**: 增加了Receding Horizon执行步数

#### ⚠️ 潜在改进方向
1. **添加动作归一化**: 提升训练稳定性和性能
2. **调整Horizon参数**: 可尝试Horizon=4, n_obs_steps=4
3. **增加数据量**: 如果效果仍不理想

#### 🚀 测试建议

**立即测试**:
```bash
# 1. 测试修复后的对齐模型
cd /home/gl/RoboTwin
bash policy/DP2DP3/eval.sh lift_pot demo_clean demo_clean 50 0 1 best

# 2. 测试修复后的直接融合模型
bash policy/DP2DP3/eval_direct_fusion.sh lift_pot demo_clean demo_clean 50 0 1 best
```

**如果效果好**: 当前方案可用，可以继续使用

**如果效果仍不理想**: 考虑添加归一化重新训练
```bash
# 修改训练代码后重新训练
cd /home/gl/RoboTwin/policy/DP2DP3/features_model
python tools/direct_fusion/train_direct_fusion_offline.py \
    --config configs/direct_fusion/train_direct_fusion_lift_pot.yaml
```

---

## 完整训练时间估算

| 步骤 | 时间 | GPU占用 |
|-----|------|---------|
| 数据提取 | 10分钟 | CPU |
| RGB特征提取 | 1小时×4 | 全占 |
| 点云特征提取 | 30分钟 | 50% |
| 对齐训练 | 7小时 | 80% |
| 动作头在线训练 | 3小时 | 90% |
| 动作头离线训练 | 30分钟 | 90% |
| **总计(在线)** | **~15小时** | - |
| **总计(离线)** | **~13小时** | - |

---

## 快速开始命令总结

```bash
# 1. 数据提取 (假设已完成)
cd /home/gl/RoboTwin/policy/DP2DP3/features_model
python tools/dataset/process_sapien_pcd.py beat_block_hammer demo_randomized 20

# 2-3. 特征提取 (假设已完成)

# 4. 对齐训练
cd /home/gl/RoboTwin/policy/DP2DP3/features_model
python tools/train_rgb2pc_distill.py \
    --config configs/alignment/train_rgb2pc_distill_default.yaml

# 5. 动作头训练 (离线推荐)
bash scripts/train_offline_pipeline.sh

# 6. 推理测试
cd /home/gl/RoboTwin
bash policy/DP2DP3/eval.sh beat_block_hammer demo_randomized demo_randomized 20 0 1 50
```

---

## 数据形态速查表

| 阶段 | 数据类型 | 形态 | 说明 |
|-----|---------|------|------|
| HDF5原始 | RGB | JPEG bytes | 压缩存储 |
| HDF5原始 | 点云 | [N, 6] | 原生Sapien |
| HDF5原始 | 动作 | [T, 14] | 双臂+双夹爪 |
| 提取后 | RGB | [H, W, 3] uint8 | 解压PNG |
| 提取后 | 点云 | [N, 6] float | PLY格式 |
| 视觉特征 | CroCo | [W, T, Hf, Wf, 1024] | Zarr |
| 视觉特征 | VGGT | [W, T, Hf, Wf, 2048] | Zarr |
| 视觉特征 | DINOv3 | [W, T, Hf, Wf, 768] | Zarr |
| 视觉特征 | DA3 | [W, T, Hf, Wf, 2048] | Zarr |
| 点云特征 | ULIP | [1280] | Zarr |
| 训练输入 | RGB | [B, To, 4, 2048] | Adapter输入 |
| 对齐输出 | Aligned | [B, To, 1280] | Head输入 |
| Head输出 | 动作 | [B, Ta, 14] | 预测轨迹 |
| 推理输出 | 动作 | [14] | 单步执行 |

---

## 总结

**关键修复点**:
1. ✅ `loss_mse: 1.0` - 确保数值对齐
2. ✅ `horizon: 8`
3. ✅ `n_obs_steps: 2`
4. ✅ Receding Horizon - 只执行2步再重新预测
5. ✅ 动作clip - 限制在关节范围内

**性能优化**:
1. ⚡ 离线训练 - 特征预提取，速度提升10倍
2. ⚡ 批量提取 - GPU利用率更高
3. ⚡ 多GPU分配 - 4个模型并行

---

## 问题诊断与解决方案 (2026-01-16更新)

### 问题描述
用户报告推理结果非常差，机械臂甚至没有轨迹。需要诊断是哪一步/哪几步出现问题。

### 诊断结果总结 ✅

使用诊断工具 `tools/direct_fusion/diagnose_simple.py` 进行完整检查:

#### 1. 对齐模块 ✅ 正常
- Adapters权重分布正常 (均值~0, std=0.018-0.028)
- Fusion权重平衡 (CroCo=32%, VGGT=22%, DINOv3=23%, DA3=22%)
- Projection权重正常

#### 2. Head模块 ✅ 正常  
- 配置正确 (horizon=4, n_obs_steps=4, include_gripper=True)
- 输入/输出维度匹配 (5120→14)
- **训练Loss=0.013** (非常小，收敛良好)
- 权重分布正常

#### 3. 结论
Checkpoint和配置都正常，推理失败可能原因:
1. **推理时数据预处理不一致** (最可能)
2. 对齐模块可能破坏了RGB语义
3. 训练数据质量问题

---

### 解决方案路径

#### 路径1: 直接融合训练 (无对齐) ⭐推荐优先测试

**目的**: 测试是否是对齐模块的问题

**流程**: RGB → 4模型 → 简单融合 → Head → 动作

**优势**:
- 跳过复杂对齐模块
- 训练快速(~2-3小时)
- 快速验证baseline

**文件**:
- `tools/direct_fusion/train_direct_fusion.py`
- `configs/direct_fusion/train_direct_fusion_lift_pot.yaml`

**执行**:
```bash
cd /home/gl/RoboTwin/policy/DP2DP3/features_model
python tools/direct_fusion/train_direct_fusion.py \
    --config configs/direct_fusion/train_direct_fusion_lift_pot.yaml
```

**Checkpoint路径**:
```
/home/gl/RoboTwin/policy/DP2DP3/checkpoints_direct_fusion/
└── lift_pot-demo_clean-50-0/
    ├── 50.ckpt
    ├── 100.ckpt
    └── best.ckpt
```

#### 路径2: 改进对齐训练 (添加RGB保留)

**问题**: 对齐可能过度拟合点云，破坏RGB语义

**解决**: 添加RGB自对比损失

在 `train_rgb2pc_distill.py` 第455行附近添加:
```python
# RGB自对比: 保持4个模型间语义一致性
loss_rgb_contrast = 0
for i in range(4):
    for j in range(i+1, 4):
        zi, zj = F.normalize(zs_list[i]), F.normalize(zs_list[j])
        logits = (zi @ zj.t()) / tau
        labels = torch.arange(len(zi), device=device)
        loss_rgb_contrast += F.cross_entropy(logits, labels)
loss_rgb_contrast /= 6  # 4个模型共6对

# 总损失
loss = loss_nce + λ_mse * loss_mse + λ_rgb * loss_rgb_contrast
```

#### 路径3: 检查推理一致性

检查 `deploy_policy.py`:

1. **图像归一化**: 
   ```python
   # 必须与训练一致
   img = img / 255.0  # [0,1]
   ```

2. **观测缓冲区**:
   ```python
   self.obs_buffer = deque(maxlen=4)  # 必须=n_obs_steps
   ```

3. **特征提取顺序**:
   ```python
   # [CroCo, VGGT, DINOv3, DA3] - 顺序必须一致
   ```

4. **Diffusion采样**:
   ```python
   num_inference_steps: 100 → 200  # 可尝试增加
   ```

---

### 对比测试方案

| 方案 | 训练时间 | 复杂度 | 推荐场景 |
|-----|---------|-------|---------|
| **完整流程**(对齐) | ~15h | 高 | 长期优化 |
| **直接融合**(无对齐) | ~3h | 低 | 快速baseline |

**测试流程**:
1. 先跑直接融合 (100 epochs)
2. 效果好 → 对齐模块有问题
3. 效果差 → 融合或数据有问题
4. 根据结果决定优化方向

---

### 快速验证命令

```bash
# 启动直接融合训练
cd /home/gl/RoboTwin/policy/DP2DP3/features_model
nohup python tools/direct_fusion/train_direct_fusion.py \
    --config configs/direct_fusion/train_direct_fusion_lift_pot.yaml \
    > logs/direct_fusion.log 2>&1 &

# 监控进度
tail -f logs/direct_fusion.log

# 训练完成后测试推理
bash policy/DP2DP3/eval.sh lift_pot demo_clean demo_clean 50 0 1 100
```

---

### 诊断工具清单

所有工具位于 `tools/direct_fusion/`:

1. **diagnose_simple.py** - 检查checkpoint (✅ 已验证)
2. **diagnose_full_pipeline.py** - 完整前向传播测试
3. **train_direct_fusion.py** - 直接融合训练 (✅ 已创建)
4. **train_rgb2pc_distill_improved.py** - 改进对齐框架

---

### 预期结果分析

**情况A: 直接融合效果好**
- 说明对齐模块破坏了特征
- 解决: 使用路径2改进对齐训练

**情况B: 直接融合效果差**
- 说明问题不在对齐
- 可能原因:
  1. 融合方式不当
  2. Head训练问题  
  3. 数据质量差
  4. 推理代码bug

---

现在可以开始训练了！建议先运行**直接融合训练**作为baseline对比。

---

## 🔍 深度诊断报告: "反复来回"问题根因分析 (2026-01-19)

### 问题描述
机械臂在第1秒后开始**反复来回抖动**，无法完成任务。这个问题在两种训练方式下都出现：
1. **对齐模块训练** (RGB→PC对齐→Head)
2. **直接融合训练** (RGB→简单融合→Head)

### 严格对比: DP2DP3 vs 原始DP

#### 对比1: 推理流程 ⚠️ 发现关键差异

**原始DP推理流程**:
```python
# deploy_policy.py (原始DP)
def eval(TASK_ENV, model, observation):
    obs = encode_obs(observation)
    actions = model.get_action(obs)  # 返回多步动作
    
    for action in actions:
        TASK_ENV.take_action(action)
        observation = TASK_ENV.get_obs()  # 每步都获取新观测
        obs = encode_obs(observation)
        model.update_obs(obs)  # 每步都更新obs buffer
```

**DP2DP3推理流程**:
```python
# deploy_direct_fusion_policy.py (直接融合)
def eval(TASK_ENV, model, observation):
    obs = encode_obs(observation)
    actions = model.get_action(obs)  # 返回多步动作
    
    for action in actions:
        TASK_ENV.take_action(action)
        model.pop_action()  # ⚠️ 只弹出队列
        observation = TASK_ENV.get_obs()
        obs = encode_obs(observation)
        model.update_obs(obs)  # 每步都更新
```

**差异点**: 基本一致，都是每步更新观测

#### 对比2: 动作预测机制 ⚠️ 发现重大差异！

**原始DP**:
```python
# dp_runner.py
def get_action(self, policy, observation=None):
    if observation is not None:
        self.obs.append(observation)  # 更新观测
    obs = self.get_n_steps_obs()  # 获取最近n步
    
    # 关键: 使用predict_action，内部有normalizer处理
    with torch.no_grad():
        obs_dict_input = {
            "head_cam": obs_dict["head_cam"].unsqueeze(0),
            "agent_pos": obs_dict["agent_pos"].unsqueeze(0)
        }
        action_dict = policy.predict_action(obs_dict_input)
    
    action = action_dict["action"].squeeze(0)[:self.n_action_steps]
    return action  # 返回多步动作，外部循环执行
```

**DP2DP3直接融合**:
```python
# deploy_direct_fusion_policy.py
def get_action(self, obs):
    self.obs_buffer.append(obs)
    
    # 如果队列空或需要重新规划
    if self.replan_every_call or len(self.action_queue) == 0:
        # 1. 提取RGB特征 (4个模型)
        features = self.feature_extractors.extract_batch(images)
        
        # 2. 通过policy预测 (无normalizer!)
        with torch.no_grad():
            action_pred = self.policy(features)  # [1, Ta, A]
        
        # 3. 手动反归一化
        if self.action_stats is not None:
            action_pred = (action_pred + 1.0) * 0.5 * (action_max - action_min) + action_min
        
        # 4. Clip
        action_pred = np.clip(action_pred, -3.0, 3.0)
        
        # 5. 加入队列
        self.action_queue.extend(action_pred)
    
    # 返回前n步
    return list(self.action_queue)[:n_action_exec]
```

**🚨 核心差异发现**:

| 方面 | 原始DP | DP2DP3 |
|-----|--------|--------|
| **Normalizer** | ✅ 有 (内置在policy中) | ❌ 无 (手动处理) |
| **训练归一化** | ✅ 自动 (通过normalizer.fit) | ❌ 手动 (代码中实现) |
| **推理反归一化** | ✅ 自动 (unnormalize) | ❌ 手动 (可能错误) |
| **观测编码** | CNN+ResNet | 4模型特征+融合 |
| **动作范围** | 从数据自动统计 | **硬编码[-3,3]** ⚠️ |

#### 对比3: 训练时的归一化 🔥 关键发现

**原始DP训练**:
```python
# diffusion_unet_image_policy.py
class DiffusionUnetImagePolicy:
    def __init__(self):
        self.normalizer = LinearNormalizer()  # 自动管理
    
    def forward(self, obs_dict, actions):
        # 训练时自动归一化
        nobs = self.normalizer.normalize(obs_dict)
        nactions = self.normalizer['action'].normalize(actions)
        
        # Diffusion训练
        noise = torch.randn_like(nactions)
        noisy_actions = self.noise_scheduler.add_noise(nactions, noise, timesteps)
        pred = self.model(noisy_actions, timesteps, global_cond=obs_cond)
        loss = F.mse_loss(pred, noise)
    
    def predict_action(self, obs_dict):
        # 推理时自动归一化/反归一化
        nobs = self.normalizer.normalize(obs_dict)
        naction_pred = self.conditional_sample(...)
        action_pred = self.normalizer["action"].unnormalize(naction_pred)
        return {"action": action_pred}
```

**DP2DP3训练**:
```python
# train_direct_fusion_offline.py
class DPRGBOfflineZarrDataset:
    def __init__(self):
        # 手动统计action范围
        self._compute_action_stats()
    
    def _normalize_action(self, action):
        # 手动归一化到[-1, 1]
        action = 2.0 * (action - self.action_min) / (action_max - action_min) - 1.0
        return torch.clamp(action, -1.0, 1.0)
    
    def __getitem__(self, idx):
        # 返回归一化后的动作
        action_norm = self._normalize_action(action_raw)
        return features, action_norm

class DirectFusionDPPolicy:
    def compute_loss(self, rgb_feats, actions):
        # actions已经是归一化后的 [-1, 1]
        noise = torch.randn_like(actions)
        noisy_actions = self.noise_scheduler.add_noise(actions, noise, timesteps)
        pred = self.noise_pred_net(noisy_actions, timesteps, global_cond=obs_cond)
        loss = F.mse_loss(pred, noise)
    
    def forward(self, rgb_feats):
        # 推理: 输出归一化后的动作 [-1, 1]
        action = torch.randn(...)
        for t in self.noise_scheduler.timesteps:
            noise_pred = self.noise_pred_net(action, t, global_cond=obs_cond)
            action = self.noise_scheduler.step(noise_pred, t, action).prev_sample
        return action  # 仍然是[-1, 1]范围
```

### 🔥 问题根因: 训练-推理不一致！

#### 问题1: Checkpoint缺少action_stats ⚠️ 严重问题

**检查结果**:
```bash
# checkpoint内容
keys: ['policy', 'optimizer', 'epoch', 'config', 'loss']
# ❌ 没有 'action_stats' 或 'action_norm_stats'
```

**导致的问题**:
```python
# deploy_direct_fusion_policy.py 第467行
if self.action_stats is not None:  # self.action_stats=None!
    # 这段代码永远不会执行
    action_pred = (action_pred + 1.0) * 0.5 * (action_max - action_min) + action_min
```

**实际情况**:
- **训练时**: 动作归一化到[-1, 1] (使用真实数据统计的min/max)
- **推理时**: 模型输出[-1, 1]，但没有反归一化！
- **结果**: 动作值在[-1, 1]范围，但应该在[-2.5, 2.5]范围
- **机械臂行为**: 移动幅度太小，只能在原地微调 → **反复来回抖动**

#### 问题2: 没有保存归一化统计信息

**训练代码问题**:
```python
# train_direct_fusion_offline.py 第580行
torch.save({
    'policy': policy.state_dict(),
    'optimizer': optimizer.state_dict(),
    'epoch': epoch,
    'config': {...},
    'loss': avg_loss,
    # ❌ 缺少: 'action_stats': {'min': ..., 'max': ...}
}, ckpt_path)
```

#### 问题3: 视觉特征问题？❌ 不是主因

检查视觉特征提取：
- ✅ 4个模型正确加载
- ✅ 特征维度正确 (1024, 2048, 768, 2048)
- ✅ 融合权重合理
- ✅ 特征分布正常

**结论**: 视觉特征没有问题，问题在于**动作归一化不一致**

### 🔧 完整修复方案

#### 修复1: 保存action_stats到checkpoint ⭐ 必须修复

**修改文件**: `tools/direct_fusion/train_direct_fusion_offline.py`

**位置**: 第580行左右

```python
# ❌ 错误的保存方式
torch.save({
    'policy': policy.state_dict(),
    'optimizer': optimizer.state_dict(),
    'epoch': epoch,
    'config': config,
    'loss': avg_loss,
}, ckpt_path)

# ✅ 正确的保存方式
torch.save({
    'policy': policy.state_dict(),
    'optimizer': optimizer.state_dict(),
    'epoch': epoch,
    'config': config,
    'loss': avg_loss,
    'action_stats': {
        'min': dataset.action_min.tolist(),
        'max': dataset.action_max.tolist()
    }
}, ckpt_path)
```

#### 修复2: 推理时正确反归一化 ⭐ 必须修复

**修改文件**: `policy/DP2DP3/deploy_direct_fusion_policy.py`

**位置**: 第467-476行

```python
# ❌ 错误: 使用硬编码的[-3, 3]
if self.action_stats is not None:
    action_min = np.array(self.action_stats.get('min', -3.0), dtype=np.float32)
    action_max = np.array(self.action_stats.get('max', 3.0), dtype=np.float32)
    action_pred = (action_pred + 1.0) * 0.5 * (action_max - action_min) + action_min

# ✅ 正确: 必须反归一化，否则报错
if self.action_stats is None:
    raise RuntimeError("Checkpoint缺少action_stats! 请重新训练")

action_min = np.array(self.action_stats['min'], dtype=np.float32)
action_max = np.array(self.action_stats['max'], dtype=np.float32)
action_pred = (action_pred + 1.0) * 0.5 * (action_max - action_min) + action_min
```

#### 修复3: 检查训练数据的动作范围

**添加调试代码**:
```python
# 在训练开始前打印
print(f"训练数据动作范围: [{dataset.action_min.min():.3f}, {dataset.action_max.max():.3f}]")
print(f"各维度范围:")
for i in range(14):
    print(f"  Dim {i:2d}: [{dataset.action_min[i]:.3f}, {dataset.action_max[i]:.3f}]")
```

**期望输出**:
```
训练数据动作范围: [-1.629, 2.690]
各维度范围:
  Dim  0: [-1.429, 2.552]  # 左臂关节1
  Dim  1: [-1.325, 2.389]
  ...
  Dim  6: [0.000, 1.000]   # 左夹爪
  Dim  7: [-1.429, 2.552]  # 右臂关节1
  ...
  Dim 13: [0.000, 1.000]   # 右夹爪
```

### 🔍 为什么会反复来回？

**根本原因**: 动作幅度过小

1. **训练时**: 模型学习预测 [-1, 1] 范围的归一化动作
2. **推理时**: 模型输出 [-1, 1]，但应该反归一化到 [-2.5, 2.5]
3. **实际效果**: 
   - 模型输出: `action = [0.5, 0.3, -0.2, ...]`  (归一化值)
   - 应该是: `action = [1.8, 1.2, -0.5, ...]`    (真实角度)
   - 实际执行: `[0.5, 0.3, -0.2, ...]`         (角度太小!)
4. **机械臂行为**:
   - 第1秒: 模型预测"伸手去拿"，但角度只有应有的20-30%
   - 结果: 手臂只移动了一点点
   - 第2秒: 新观测发现还没到位，继续预测"伸手"
   - 结果: 又移动一点点
   - 循环往复: **来回抖动，永远到不了目标位置**

### ✅ 验证方法

**修复后的期望行为**:

1. **训练输出**:
```bash
训练数据动作范围: [-1.629, 2.690]
保存checkpoint: action_stats已保存
```

2. **推理输出**:
```bash
[DirectFusion] Loading checkpoint: xxx.ckpt
[DirectFusion] Action stats loaded: min=[-1.629, ...], max=[2.690, ...]

[DEBUG] 模型原始输出:
  Range: [-0.95, 0.98]  # 归一化值
[DEBUG] 反归一化后:
  Range: [-1.52, 2.61]  # 真实值，范围合理
```

3. **机械臂行为**:
   - 第1秒: 大幅度伸手
   - 2-3秒: 抓取物体
   - 4-5秒: 提起
   - 完成任务！

### 📝 修复步骤

**已完成的诊断**:
```bash
cd /home/gl/RoboTwin/policy/DP2DP3/features_model
python tools/direct_fusion/diagnose_action_normalization.py
```

**诊断结果**:
- ❌ 旧checkpoint缺少action_stats
- ✅ 训练代码已经正确 (已包含action_stats保存)
- ✅ 推理代码已经修复 (添加fallback经验值)
- 📊 动作范围: 训练数据为[-1.80, 2.91]，模型输出[-1, 1]需要反归一化2.03倍

**立即可用的临时方案** (使用经验值):
```bash
# 修改后的推理代码已经包含fallback，可以直接测试
cd /home/gl/RoboTwin
bash policy/DP2DP3/eval_direct_fusion.sh lift_pot demo_clean demo_clean 50 0 1 1450
```

**推荐的长期方案** (重新训练):

1. **确认配置文件正确** (已完成)
   ```yaml
   # configs/direct_fusion/train_direct_fusion_offline.yaml
   checkpoint:
     save_dir: /home/gl/RoboTwin/policy/DP2DP3/checkpoints_direct_fusion/lift_pot-demo_clean-50-0
     save_every: 500
   ```

2. **重新训练** (推荐但非必须)
   ```bash
   cd /home/gl/RoboTwin/policy/DP2DP3/features_model
   
   # 清理旧checkpoint (可选)
   rm -rf /home/gl/RoboTwin/policy/DP2DP3/checkpoints_direct_fusion/lift_pot-demo_clean-50-0
   
   # 重新训练
   python tools/direct_fusion/train_direct_fusion_offline.py \
       --config configs/direct_fusion/train_direct_fusion_offline.yaml
   
   # 训练完成后会在checkpoints中包含action_stats
   ```

3. **验证新checkpoint**
   ```bash
   python3 -c "
   import torch
   ckpt = torch.load('/home/gl/RoboTwin/policy/DP2DP3/checkpoints_direct_fusion/lift_pot-demo_clean-50-0/500.ckpt')
   assert 'action_stats' in ckpt, 'Missing action_stats!'
   print('✅ action_stats:', ckpt['action_stats'])
   "
   ```

4. **测试推理**
   ```bash
   cd /home/gl/RoboTwin
   bash policy/DP2DP3/eval_direct_fusion.sh lift_pot demo_clean demo_clean 50 0 1 500
   ```

### 🎯 总结

**问题不是**:
- ❌ 视觉特征被破坏
- ❌ 融合方式不对
- ❌ Head网络结构问题
- ❌ 控制频率问题

**问题是**:
- ✅ **训练-推理归一化不一致**
- ✅ **Checkpoint缺少action_stats**
- ✅ **推理时没有正确反归一化**

**预期效果**:
修复后，机械臂应该能够大幅度移动，正确完成任务，不再反复来回抖动。

---


## 📋 快速参考: 问题诊断与修复清单

### 当前状态总结 (2026-01-19)

✅ **已修复的文件**:
- `tools/direct_fusion/train_direct_fusion_offline.py` - 训练时保存action_stats
- `policy/DP2DP3/deploy_direct_fusion_policy.py` - 推理时添加fallback反归一化

✅ **已创建的工具**:
- `tools/direct_fusion/diagnose_action_normalization.py` - 完整诊断
- `tools/direct_fusion/test_action_fix.py` - 快速测试
- `DIAGNOSIS_REPORT.md` - 详细报告

### 立即可用的测试命令

```bash
# 方案A: 使用旧checkpoint + fallback经验值 (立即可测试)
cd /home/gl/RoboTwin
bash policy/DP2DP3/eval_direct_fusion.sh lift_pot demo_clean demo_clean 50 0 1 1450

# 方案B: 重新训练 + 真实stats (推荐)
cd /home/gl/RoboTwin/policy/DP2DP3/features_model
python tools/direct_fusion/train_direct_fusion_offline.py \
    --config configs/direct_fusion/train_direct_fusion_offline.yaml
# 等待训练完成后
cd /home/gl/RoboTwin
bash policy/DP2DP3/eval_direct_fusion.sh lift_pot demo_clean demo_clean 50 0 1 500
```

### 验证checklist

- [ ] **训练数据分析**: `python tools/direct_fusion/diagnose_action_normalization.py`
- [ ] **Checkpoint检查**: 验证包含action_stats
- [ ] **反归一化测试**: `python tools/direct_fusion/test_action_fix.py`
- [ ] **推理输出**: 检查日志中的反归一化效果
- [ ] **机械臂行为**: 观察是否大幅度移动

详细报告见: `DIAGNOSIS_REPORT.md`

---

## 🆕 单模型训练 (2026-01-19更新)

### 目的
测试是否是**4模型融合破坏了语义**导致机械臂找不到目标

### 改进方案
1. **使用LinearNormalizer** - 替换手动归一化，避免训练-推理不一致
2. **单模型训练** - 只用一个视觉模型(如DINOv3)训练Head
3. **对比测试** - 对比单模型 vs 融合的效果

### 训练命令

```bash
cd /home/gl/RoboTwin/policy/DP2DP3/features_model

# 训练单个DINOv3模型
python tools/single_model/train_single_model_offline.py \
    --config configs/single_model/train_single_dinov3.yaml

# 或者后台运行
nohup /home/gl/miniconda3/envs/depth3/bin/python \
    tools/single_model/train_single_model_offline.py \
    --config configs/single_model/train_single_dinov3.yaml \
    > logs/train_single_dinov3.log 2>&1 &

# 查看日志
tail -f logs/train_single_dinov3.log
```

### 推理命令

```bash
cd /home/gl/RoboTwin/policy/DP2DP3

# 评估单模型
bash eval_single_model.sh lift_pot demo_clean demo_clean 50 0 1 best dinov3
```

**参数说明**:
```bash
eval_single_model.sh <task> <train_setting> <eval_setting> <num_demos> <seed> <gpu_id> <checkpoint> <model_name>
```

### 文件结构

```
tools/single_model/
├── train_single_model_offline.py    # 单模型训练脚本
configs/single_model/
├── train_single_dinov3.yaml         # DINOv3配置
policy/DP2DP3/
├── deploy_single_model_policy.py    # 单模型推理
├── eval_single_model.sh             # 评估脚本
policy/DP2DP3_single/
├── deploy_single_model_policy.py    # 部署文件
└── deploy_single_model_policy.yml   # 配置
```

### 关键改进

#### 1. 使用LinearNormalizer

**训练时**:
```python
# ✅ 自动fit统计值
policy.normalizer['action'].fit(
    all_actions,
    last_n_dims=1,
    mode='limits',
    output_min=-1.0,
    output_max=1.0
)

# ✅ 自动归一化
def compute_loss(self, obs, actions):
    nactions = self.normalizer['action'].normalize(actions)
    # ... diffusion训练
```

**推理时**:
```python
# ✅ 自动反归一化
def forward(self, obs):
    # ... diffusion采样
    action = self.normalizer['action'].unnormalize(action)
    return action
```

**保存checkpoint**:
```python
torch.save({
    'policy': policy.state_dict(),
    'normalizer': policy.normalizer.state_dict(),  # ✅ 保存normalizer
    ...
}, ckpt_path)
```

**加载checkpoint**:
```python
policy.normalizer.load_state_dict(ckpt['normalizer'])  # ✅ 加载normalizer
```

#### 2. 单模型简化

- **输入**: 单个视觉模型特征 [B, To, D]
- **输出**: 动作 [B, Ta, 14]
- **无融合**: 直接将特征送入Head，避免融合可能的语义损失

### 对比测试方案

| 方案 | 视觉编码 | 归一化 | 目的 |
|------|---------|--------|------|
| **原始直接融合** | 4模型+融合 | 手动 | 当前baseline |
| **改进直接融合** | 4模型+融合 | LinearNormalizer | 修复归一化 |
| **单模型DINOv3** | 仅DINOv3 | LinearNormalizer | 测试融合影响 |

### 预期结果

1. 如果**单模型效果好**:
   - 说明融合破坏了语义
   - 解决: 改进融合方式或使用单模型

2. 如果**单模型效果差**:
   - 说明问题不在融合
   - 可能原因: Head结构、超参数、数据质量

3. 如果**改进直接融合效果好**:
   - 说明问题就是归一化
   - LinearNormalizer解决了训练-推理不一致

---

## 🆕 推理侧：token_full 的“回跳/反复来回”诊断与缓解 (2026-02-04 更新)

> 这一节只针对你现在最关心的现象：**机械臂已经靠近目标后，又突然回到几秒前的位置重新来一遍**（回跳/反复）。  
> 注意：这和“视频看起来瞬移”是两类问题（下面会区分）。

### A. 现象定义（把你说的“像人眼一样看”量化）

我们用 **右臂 TCP（末端执行器）轨迹**做判断：

- **瞬移候选（teleport candidate）**：相邻两次 policy 执行步之间，TCP 位移突然很大（比如 > 5cm）。
- **回跳（rewind teleport）**：出现一次“大跳”，并且**落点非常接近几十步之前的位置**（例如 10~80 步内，距离 < 2cm）。

这类回跳很像“计划突然改主意，回到之前的姿态重新走一遍”，是你视频里反复来回的主要来源。

### B. 为什么 token_full 更容易回跳，而 pool_ws1 更稳定？

通俗解释：

1) **Diffusion Head 本质是“随机采样一条未来轨迹”**  
token_full 这条路线在每次重规划时会从随机噪声采样一条 `horizon=8` 的动作序列，然后只执行前 `n_action_exec=4` 步，再重规划。  
如果两次重规划采样到了不同的“行为模式”（比如这次决定“靠近 pot”，下次决定“撤退/回到预抓取姿态”），就会出现回跳。

2) **token_full 的条件输入更“敏感”**  
token_full 输入的是大量 patch token（空间信息更细），对环境里小变化（视角、遮挡、光照、物体边缘）更敏感；再叠加 diffusion 的随机性，就更容易在重规划边界发生“模式切换”。

3) **pool_ws1 的条件输入更“粗、稳”**  
pool_ws1 只看 pooled 全局向量，很多细节变化会被平均掉，因此相邻两次重规划得到的条件更稳定，采样出来的轨迹更一致，回跳概率更低。

#### B2. 从 DP 原理再讲一遍（更直白）

把 DP 想成“每次都重新 **随机画一条未来轨迹**”的模型：

- **DP 不是输出 1 个动作，而是采样一段未来动作**（比如 8 步）。  
- 每次重规划时，它会 **重新从噪声采样** 一条新的 8 步轨迹。  
- 它**不会强制**这条新轨迹与上一条轨迹的后半段一致。  

所以：  
当你只执行前 `n_action_exec` 步，再重规划时，如果新计划“换了一个模式”，就会看起来**突然改主意、回到之前位置再来一次**。  

**“模式切换”= 轨迹类型切换**（例如“直接靠近抓取” ↔ “先后退再调整”）。  
token_full 更容易出现这种切换，因为它对细节变化更敏感，采样分布更“散”；  
pool_ws1 更像“只看大概形状”，分布更集中，连续两次采样更一致。

> 重要澄清：  
> “闪现”并不一定是数据切片错位造成的。  
> 在你当前离线 head 的数据切片中，action 是从当前帧开始（没有跳到未来多步）。  
> 所以如果仍然有“回跳”，更可能是 **RHC 重规划 + token_full 高敏感 + 采样随机性** 叠加的结果。

### C. 这是不是“结构决定的、真机一定废了”？

不是。需要区分两件事：

1) **“视频看起来瞬移”**：很大一部分来自录像采样稀疏（仿真 `take_action()` 内部会用高频控制执行一段平滑轨迹，但视频可能每次只写 1 帧）。  
真实机械臂不可能瞬移，它会以速度/加速度限制连续运动。

2) **“回跳/反复来回”**：这是策略在重规划边界输出的**目标关节/目标轨迹发生了大幅反向变化**。  
真机上不会瞬移，但会表现为：突然反向、抖动、反复靠近/远离，甚至不安全。  
所以：**如果不解决回跳，确实不建议直接上真机。**

### D. 诊断工具（推荐你以后都用这套来“量化像人眼一样的感受”）

1) 生成 action/TCP 日志（JSONL）
```bash
export CUDA_VISIBLE_DEVICES=0,1
/home/gl/miniconda3/envs/RoboTwin/bin/python script/eval_policy.py \
  --config policy/DP2DP3/deploy_policy.yml --overrides \
  --task_name lift_pot --task_config demo_clean \
  --ckpt_setting demo_clean_dual_stream_tokens_full \
  --expert_data_num 50 --seed 0 --policy_name DP2DP3 \
  --checkpoint_num 600 --gpu_id 0 --n_action_exec 4 --gpu_ids "[0,1]" \
  --eval_test_num 1 \
  --action_log_path policy/DP2DP3/logs/action_logs/repro_baseline_ckpt600_k4.jsonl
```

2) 统计“回跳计数/大跳变”等指标（按 episode）
```bash
/home/gl/miniconda3/envs/RoboTwin/bin/python policy/DP2DP3/analyze_action_log.py \
  --log policy/DP2DP3/logs/action_logs/repro_baseline_ckpt600_k4.jsonl \
  --episode 0 \
  --out policy/DP2DP3/logs/action_logs/repro_baseline_ckpt600_k4_summary.json
```

3) 画 TCP 轨迹与单步位移曲线（最直观）
```bash
/home/gl/miniconda3/envs/RoboTwin/bin/python policy/DP2DP3/plot_trajectory_compare.py \
  --log_a policy/DP2DP3/logs/action_logs/repro_baseline_ckpt600_k4.jsonl --name_a baseline \
  --log_b policy/DP2DP3/logs/action_logs/repro_baseline_ckpt600_k4.jsonl --name_b baseline \
  --arm right --episode 0 \
  --out_dir policy/DP2DP3/logs/trajectory_plots/repro_baseline
```

### E. 推理侧缓解方案（不改训练/不改 checkpoint）：Best-of-N 计划选择

核心思想：**一次重规划不只采样 1 条未来 8 步轨迹，而是采样 N 条候选轨迹，选与“上一段剩余动作”最一致的那条**，从而显著减少回跳。

启用方式（推荐 N=4；默认关闭，不影响你的 81% baseline）：
```bash
export CUDA_VISIBLE_DEVICES=0,1
export DP2DP3_PLAN_SELECT_N=4   # ✅ 关键开关：Best-of-N 计划选择（默认=1，不影响baseline）
export ROBOTWIN_EVAL_VIDEO_DENSE=1
export ROBOTWIN_EVAL_VIDEO_DENSE_INTERVAL=25

/home/gl/miniconda3/envs/RoboTwin/bin/python script/eval_policy.py \
  --config policy/DP2DP3/deploy_policy.yml --overrides \
  --task_name lift_pot --task_config demo_clean \
  --ckpt_setting demo_clean_dual_stream_tokens_full \
  --expert_data_num 50 --seed 0 --policy_name DP2DP3 \
  --checkpoint_num 600 --gpu_id 0 --n_action_exec 4 --gpu_ids "[0,1]" \
  --eval_test_num 1 \
  --action_log_path policy/DP2DP3/logs/action_logs/repro_planselect4_ckpt600_k4.jsonl
```

实践结论（以 lift_pot / ckpt600 / seed=0 / episode0 为例）：
- baseline：`rewind_teleports = 2`
- `plan_select_n=4`：`rewind_teleports = 0`

代价：推理会变慢（每次重规划多采样 N 次）。

### F. “密集写帧”只影响录像，不影响策略（安全说明）

为了解决“视频看起来瞬移”，我们加了可选开关（默认关闭，不影响任何任务/策略）：
```bash
export ROBOTWIN_EVAL_VIDEO_DENSE=1
export ROBOTWIN_EVAL_VIDEO_DENSE_INTERVAL=25   # 250Hz/25≈10fps
```
不开启时行为与原版完全一致。

---

## 🆕 多任务 ws1 外置特征 → 对齐训练 → Head训练 → 评估（完整可复现流水线）(2026-02-05 更新)

这一节对应你的“终极 prompt”要求：**不破坏 lift_pot/ckpt600 的 81% baseline**，同时把多任务 ws1 特征全部外置到磁盘，跑通 **数据准备→RGB特征(ws1)→ULIP teacher→对齐训练→离线特征→Head训练→推理评估+回跳指标** 的闭环。

> ✅ 原则：所有新增功能默认关闭，不影响 baseline；主要通过 **新脚本/新配置**实现。

### 0) 前置条件：外置盘挂载与权限

外置盘目标路径（写死，便于复现）：
```
/media/gl/新加卷/gllll/features_model_ws1
```

必须满足：
- 路径存在（已挂载）
- 当前用户对该目录可写

检查命令（任意一条失败都需要先修权限/挂载）：
```bash
ls -la /media/gl/新加卷/gllll/features_model_ws1
touch /media/gl/新加卷/gllll/features_model_ws1/.__write_test && rm /media/gl/新加卷/gllll/features_model_ws1/.__write_test
```

### 1) 手动逐步跑（按你给的命令严格整理）

下面每一步都把路径写死，并建议你把 stdout/stderr 用 `tee` 落盘。

#### Step0：收集用于对齐训练的 8 个数据集（demo_clean）

> 只允许 demo_clean；**不混入 demo_randomized**。  
> 下面保留你给的写法（第 4 个参数脚本会忽略，但不会报错）。

```bash
cd /home/gl/RoboTwin
for t in handover_block click_bell move_can_pot pick_diverse_bottles place_can_basket rotate_qrcode move_pillbottle_pad lift_pot; do \
  bash collect_data.sh ${t} demo_clean demo_clean 0; \
done
```

#### Step1：从 RoboTwin HDF5 生成 RGB_ORI / PC_ORI（demo_clean only）

任务候选（你给的 8 个，我直接全选；你也可以删减到 5~10 个）：
```
handover_block, click_bell, move_can_pot, pick_diverse_bottles,
place_can_basket, rotate_qrcode, move_pillbottle_pad, lift_pot
```

```bash
conda activate DP3_ULIP
cd /home/gl/RoboTwin/policy/DP2DP3/features_model && \
for t in handover_block click_bell move_can_pot pick_diverse_bottles place_can_basket rotate_qrcode move_pillbottle_pad lift_pot; do \
    echo "=== 正在处理任务: $t ==="; \
    python tools/dataset/process_sapien_pcd.py ${t} demo_clean 50 --output_root /home/gl/RoboTwin/policy/DP2DP3/features_model --camera head_camera; \
done
```

输出示例：
```
rgb_dataset/RGB_ORI/lift_pot-demo_clean-50_sapien_head_camera/episode_0/step_0000.png
pc_dataset/PC_ORI/lift_pot-demo_clean-50_sapien_head_camera/episode_0/step_0000.ply
```

#### Step2：提取 ws1 RGB 特征（写到外置盘）

```bash
cd /home/gl/RoboTwin/policy/DP2DP3/features_model
for m in croco vggt dinov3 da3; do
  /home/gl/miniconda3/envs/depth3/bin/python tools/features/run_extract_features.py \
    --model ${m} \
    --rgb_root /home/gl/RoboTwin/policy/DP2DP3/features_model/rgb_dataset/RGB_ORI \
    --out_root /media/gl/新加卷/gllll/rgb_dataset_ws1 \
    --window_size 1 --stride 1 --device cuda
done

# 建立软链接（让对齐/训练脚本仍能按老路径读到）
ln -s /media/gl/新加卷/gllll/rgb_dataset_ws1/ /home/gl/RoboTwin/policy/DP2DP3/features_model/rgb_dataset_ws1
```

输出路径（实际写入外置盘）：
```
rgb_dataset_ws1/features_dinov3_encoder_dict_unified_zarr/<task-demo_clean-50_sapien_head_camera>/episode_0.zarr
```

#### Step3：提取 ULIP teacher（point tokens，建议也外置）

> token_full 对齐用的是 **ULIP 局部 token**，不是 pooled 的全局向量。

```bash
cd /home/gl/RoboTwin/policy/DP2DP3/features_model
/home/gl/miniconda3/envs/DP3_ULIP/bin/python tools/features/extract_ulip_point_tokens_to_zarr.py \
  --output_dir /media/gl/新加卷/gllll/pc_dataset/ulip_point_tokens_zarr

# 建立软链接
ln -s /media/gl/新加卷/gllll/pc_dataset/ulip_point_tokens_zarr/ /home/gl/RoboTwin/policy/DP2DP3/features_model/pc_dataset/ulip_point_tokens_zarr
```

输出路径：
``` 
/media/gl/新加卷/gllll/pc_dataset/ulip_point_tokens_zarr/<task-demo_clean-50_sapien_head_camera>/episode_0/step_0000.ply.ulip_tokens.zarr
```

#### Step4：对齐训练（多任务 tokens_full）

```bash
cd /home/gl/RoboTwin/policy/DP2DP3/features_model
/home/gl/miniconda3/envs/depth3/bin/python tools/alignment/train_rgb2pc_distill.py \
  --config configs/alignment/train_rgb2pc_distill_ws1_tokens_full_multitask.yaml
```

输出 checkpoint：
```
outputs/train_rgb2pc_runs/run_ws1_tokens_full_multitask/ckpt_final.pt
```

#### Step5：离线提取对齐后的观测特征（多任务）

```bash
cd /home/gl/RoboTwin/policy/DP2DP3/features_model
/home/gl/miniconda3/envs/depth3/bin/python tools/offline/extract_offline_features.py \
  --config configs/head/train_online_batch_extract_dual_stream_tokens_full_multitask.yaml \
  --output_dir /media/gl/新加卷/gllll/features_model_ws1/offline_features_dual_stream_tokens_full_multitask \
  --overwrite
```

输出（每个 episode 一个 zarr）：
```
/media/gl/新加卷/gllll/features_model_ws1/offline_features_dual_stream_tokens_full_multitask/lift_pot/episode0.zarr
  ├── obs_aligned: [T, 1280]
  ├── obs_tokens:  [T, K, 1280]
  └── action:      [T, 14]
```

#### Step6：离线训练 Head（多任务）

```bash
cd /home/gl/RoboTwin/policy/DP2DP3/features_model
/home/gl/miniconda3/envs/depth3/bin/python tools/offline/train_offline_head.py \
  --config configs/head/train_offline_dual_stream_tokens_full_multitask.yaml
```

输出：
```
/home/gl/RoboTwin/policy/DP2DP3/checkpoints/<task-config-seed>/best.ckpt
```

#### Step7：推理评估（RoboTwin）+ 回跳指标（动作/TCP）

```bash
cd /home/gl/RoboTwin

export PYTHON_BIN=/home/gl/miniconda3/envs/RoboTwin/bin/python
export DP2DP3_ACTION_LOG_PATH=policy/DP2DP3/logs/action_logs/eval100_mt.jsonl

# ✅ 缓解回跳（默认关闭不影响 baseline）
export DP2DP3_PLAN_SELECT_N=4

bash policy/DP2DP3/eval.sh lift_pot demo_clean demo_clean_dual_stream_tokens_full 50 0 "0,1" 600 4 100
```

评估视频输出目录（脚本会自动创建）：
```
eval_result/lift_pot/DP2DP3/demo_clean/demo_clean_dual_stream_tokens_full/<时间戳>/episode*.mp4
```

统计 “TCP 回跳/瞬移候选” 指标：
```bash
cd /home/gl/RoboTwin
/home/gl/miniconda3/envs/RoboTwin/bin/python policy/DP2DP3/analyze_tcp_rewind.py \
  --log policy/DP2DP3/logs/action_logs/eval100_mt.jsonl \
  --out policy/DP2DP3/logs/action_logs/eval100_mt_tcp_summary.json

# 更严格：只算 >10cm 的严重大跳
/home/gl/miniconda3/envs/RoboTwin/bin/python policy/DP2DP3/analyze_tcp_rewind.py \
  --log policy/DP2DP3/logs/action_logs/eval100_mt.jsonl \
  --teleport_thresh 0.1 \
  --out policy/DP2DP3/logs/action_logs/eval100_mt_tcp_summary_thr0p1.json
```

可视化某个 episode 的 TCP 轨迹与单步位移：
```bash
/home/gl/miniconda3/envs/RoboTwin/bin/python policy/DP2DP3/plot_trajectory_compare.py \
  --log_a policy/DP2DP3/logs/action_logs/eval100_mt.jsonl --name_a plan4 \
  --log_b policy/DP2DP3/logs/action_logs/eval100_mt.jsonl --name_b plan4 \
  --arm right --episode 0 \
  --out_dir policy/DP2DP3/logs/trajectory_plots/eval100_mt
```

### 3) 新增/修改文件清单（不会影响 baseline 的默认行为）

**features_model（训练侧）新增：**
- `configs/alignment/train_rgb2pc_distill_ws1_tokens_full_multitask.yaml`
- `configs/head/train_online_batch_extract_dual_stream_tokens_full_multitask.yaml`
- `configs/head/train_offline_dual_stream_tokens_full_multitask.yaml`
- `configs/head/train_online_batch_extract_dual_stream_tokens_full_train.yaml`
- `configs/head/train_offline_dual_stream_tokens_full_train.yaml`

**DP2DP3（推理/评估侧）新增/修改：**
- `policy/DP2DP3/deploy_policy.py`：加入 `DP2DP3_PLAN_SELECT_N` 等开关（默认=1不变）；修复 plan_select tail_fallback 拼接断点导致的回跳放大。
- `policy/DP2DP3/eval.sh`：支持 `PYTHON_BIN` 与 `DP2DP3_ACTION_LOG_PATH`（不设置则与 baseline 行为一致）。
- `policy/DP2DP3/analyze_tcp_rewind.py`：新增批量 “回跳/瞬移候选” 指标统计（只读日志，不影响推理）。
- `policy/DP2DP3/plot_trajectory_compare.py`、`policy/DP2DP3/analyze_action_log.py`：新增可视化/统计工具（只读日志）。
- `script/eval_policy.py`：允许通过 overrides 覆盖 `eval_video_log`（不传则不变）。
- `envs/_base_task.py`：新增 `ROBOTWIN_EVAL_VIDEO_DENSE`（默认关闭，仅影响录像，不影响策略）。

---

## 🆕 FAQ：对齐训练里为什么会出现“loss 变成负数”？这是错的吗？(2026-02-06 更新)

不是错，属于**正常现象**（尤其是 `tokens_full` + `loss_local>0` 时）。

先把你看到的日志拆开看（以你 step=1000 的一行举例）：

- `nce=0.7831`：全局 InfoNCE（交叉熵），一定是正数。
- `mse≈0.0008`：MSE（很小，正常）。
- `rgb≈0.6502`：RGB 自对比（正数）。
- `local≈-16.86`：**注意这里是负数**，但这是我们故意这样定义的（见下）。
- `ch≈0.6783`：Chamfer（正数）。
- `pool≈0.1280`：pool 一致性（正数）。

你当前配置（`configs/alignment/train_rgb2pc_distill_ws1_tokens_full_multitask.yaml`）的总损失是：

```
total = nce
      + 1.0 * mse
      + 0.1 * rgb
      + 0.1 * local_score      # local_score 本身就是负数，越负越好
      + 0.2 * chamfer
      + 0.1 * pool
```

其中 `local_score` 在代码里是这样写的（软匹配，集合级别，无显式对应）：

- `sim = cos(token, point) / tau`
- `tok_to_pt = logsumexp(sim)`，`pt_to_tok = logsumexp(sim)`
- `local_score = -0.5 * (mean(tok_to_pt) + mean(pt_to_tok))`

所以它天然就是负数（因为前面有个负号），并且：

- **local_score 越负**  ⇔  `logsumexp(sim)` 越大  ⇔  token 更“贴近”某些点云 token（匹配更强）
- 训练时我们做的是 **最小化 total**，因此会把 `local_score` 推得更负
- 于是 total 被 `0.1*local_score` 拉成负数，是完全正常的

更通俗一点：这项更像“奖励（reward）”，但我们写成了 loss 形式（带负号）来最小化，所以你会看到负数。

你判断“训练是否正常”不要看 total 的正负，而要看趋势：
- `pos` 在变大、`gap` 在变大（正负样本分离更强）；
- `nce` 大体在下降；
- `chamfer` 不爆炸、不 NaN；
-（最关键）下游 head 训练/推理 success 是否提升。

对应代码位置：
- `policy/DP2DP3/features_model/tools/alignment/train_rgb2pc_distill.py:78`（`local_align_loss` 定义）
- `policy/DP2DP3/features_model/tools/alignment/train_rgb2pc_distill.py:670`（`loss += loss_local * local_score`）

---

## 🆕 单任务：复用“多任务对齐模块”训练新任务 Head（无需点云）(2026-02-06 更新)

这一节把你关心的 **place_a2b_left** 完整闭环整理成「可直接跑的」步骤。  
你当前已完成 **Step1–Step5（多任务对齐模块训练到 ckpt_step_0015000.pt）**，下面从 **新任务数据**开始。

核心结论（先说清楚）：
- **对齐模块（encoder）**：多任务训练得到通用 RGB→PC 表征映射（需要点云 teacher）。
- **动作头（head）**：**单任务训练即可**，且 **不需要点云**（只用 RGB + action）。

### Step6：采集新任务数据（demo_clean）

```bash
cd /home/gl/RoboTwin
export PYTHON_BIN=/home/gl/miniconda3/envs/RoboTwin/bin/python
bash collect_data.sh place_a2b_left demo_clean 0
```

采集产物（HDF5）：
```
/home/gl/RoboTwin/data/place_a2b_left/demo_clean/data/episode0.hdf5
```

### Step7（可选）：把 RGB 图像落盘（便于肉眼检查）

> **说明**：后续的离线特征提取是 **直接从 HDF5 读 RGB**，因此这一招只是“辅助检查”，不是必需。

```bash
cd /home/gl/RoboTwin/policy/DP2DP3/features_model
# 这里的 50 请替换为你实际收集的 episode 数
/home/gl/miniconda3/envs/depth3/bin/python tools/dataset/process_sapien_rgb.py \
  place_a2b_left demo_clean 50 \
  --camera head_camera \
  --rgb_dirname RGB_TRAIN
```

输出示例：
```
rgb_dataset/RGB_TRAIN/place_a2b_left-demo_clean-50_sapien_head_camera/episode_0/step_0000.png
```

### Step8：离线提取“对齐后的观测特征”（复用多任务对齐模块）

使用新配置文件：  
`configs/head/train_online_batch_extract_dual_stream_tokens_full_train.yaml`  
（已指向你训练好的 `ckpt_step_0015000.pt`，如需换 ckpt 请改这里）

```bash
cd /home/gl/RoboTwin/policy/DP2DP3/features_model
/home/gl/miniconda3/envs/depth3/bin/python tools/offline/extract_offline_features.py \
  --config configs/head/train_online_batch_extract_dual_stream_tokens_full_train.yaml \
  --output_dir /home/gl/RoboTwin/policy/DP2DP3/features_model/data/offline_features_train \
  --overwrite
```

输出（每个 episode 一个 zarr）：
```
/home/gl/RoboTwin/policy/DP2DP3/features_model/data/offline_features_train/place_a2b_left/episode0.zarr
  ├── obs_aligned: [T, 1280]
  ├── obs_tokens:  [T, K, 1280]
  └── action:      [T, 14]
```

### Step9：离线训练 Head（单任务，600 epochs）

配置文件：  
`configs/head/train_offline_dual_stream_tokens_full_train.yaml`

> ⚠️ 如果你收集的 demo 数量不是 50，请同步修改该配置里的  
> `checkpoint.expert_data_num`（否则 eval 路径会不匹配）。

```bash
cd /home/gl/RoboTwin/policy/DP2DP3/features_model
/home/gl/miniconda3/envs/depth3/bin/python tools/offline/train_offline_head.py \
  --config configs/head/train_offline_dual_stream_tokens_full_train.yaml
```

输出（单独存放到 checkpoints_train）：
```
/home/gl/RoboTwin/policy/DP2DP3/checkpoints_train/place_a2b_left-demo_clean_train_mtenc-50-0/600.ckpt
```

> ⚠️ **评估脚本默认只会在 `policy/DP2DP3/checkpoints/` 下找 ckpt**。  
> 如果你坚持“单独目录”，请建立一个软链接以保持 eval 命令不变：

```bash
ln -sfn /home/gl/RoboTwin/policy/DP2DP3/checkpoints_train/place_a2b_left-demo_clean_train_mtenc-50-0 \
        /home/gl/RoboTwin/policy/DP2DP3/checkpoints/place_a2b_left-demo_clean_train_mtenc-50-0
```

### Step10：推理评估（RoboTwin）+ 成功率

```bash
cd /home/gl/RoboTwin
export PYTHON_BIN=/home/gl/miniconda3/envs/RoboTwin/bin/python

# （可选）输出动作/TCP日志，方便“像人眼一样”量化回跳
export DP2DP3_ACTION_LOG_PATH=policy/DP2DP3/logs/action_logs/place_a2b_left_eval100.jsonl

# （可选）缓解 token_full 回跳（默认关闭，不影响 baseline）
export DP2DP3_PLAN_SELECT_N=4

# 评估 100 条（600.ckpt）
bash policy/DP2DP3/eval.sh \
  place_a2b_left demo_clean demo_clean_train_mtenc \
  50 0 "0,1" 600 4 100   # 这里的 50 同样要与实际 demo 数一致
```

评估结束后成功率在 `_result.txt`：
```
eval_result/place_a2b_left/DP2DP3/demo_clean/demo_clean_train_mtenc/<时间戳>/_result.txt
```

如需统计回跳指标：
```bash
/home/gl/miniconda3/envs/RoboTwin/bin/python policy/DP2DP3/analyze_tcp_rewind.py \
  --log policy/DP2DP3/logs/action_logs/place_a2b_left_eval100.jsonl \
  --out policy/DP2DP3/logs/action_logs/place_a2b_left_eval100_tcp_summary.json
```

> 如果仍偶发 `cannot create buffer`，先确认 GPU 显存是否被其它训练占用；必要时重启渲染进程或降低并发进程数。

---

## 🆕 本体感知可选模块（不破坏纯视觉）

**核心原则：**  
视觉对齐 Encoder 完全不改，仍是纯视觉；  
本体感知（agent_pos）只在 **Head 前融合**，通过配置开关控制，默认关闭。

### 1) 离线特征提取（保存 agent_pos）

使用**新脚本**（不影响旧流程）：
```bash
cd /home/gl/RoboTwin/policy/DP2DP3/features_model
/home/gl/miniconda3/envs/depth3/bin/python tools/offline_proprio/extract_offline_features_proprio.py \
  --config configs/head/train_online_batch_extract_dual_stream_tokens_full_proprio.yaml \
  --output_dir /home/gl/RoboTwin/policy/DP2DP3/features_model/data/offline_features_dual_stream_tokens_full_proprio \
  --overwrite
```

输出 Zarr 将额外包含：
```
agent_pos: [T, 14]   # 来自 HDF5 的 joint_action/vector
```

> 说明：当前 RoboTwin HDF5 中没有显式 qpos 字段，因此用 `joint_action/vector` 作为 agent_pos 代理。  
> 这与环境 observation 的 `joint_action.vector` 一致，能提供“手在哪”的信息。

### 2) 离线训练 Head（融合 proprio）

```bash
cd /home/gl/RoboTwin/policy/DP2DP3/features_model
/home/gl/miniconda3/envs/depth3/bin/python tools/offline_proprio/train_offline_head_proprio.py \
  --config configs/head/train_offline_dual_stream_tokens_full_proprio.yaml
```

关键配置（可开关）：
```yaml
policy:
  use_proprio: true        # 关闭即可回退到纯视觉
  proprio_dim: 14
  proprio_mode: concat     # concat 或 add
  proprio_hidden: 256
```

### 3) 推理部署（保持兼容，按配置启用）

启用 proprio 的配置文件：
```
policy/DP2DP3/deploy_policy_proprio.yml
```

执行：
```bash
export DP2DP3_DEPLOY_CONFIG=policy/DP2DP3/deploy_policy_proprio.yml
bash policy/DP2DP3/eval.sh lift_pot demo_clean demo_clean_dual_stream_tokens_full_proprio 50 0 "0,1" 600 4 100
```

**回退到纯视觉：**  
只需删除 `DP2DP3_DEPLOY_CONFIG` 或改回 `use_proprio: false`，旧 ckpt 与旧流程完全不受影响。
