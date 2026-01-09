# DP RGB（4模型 + 冻结对齐Encoder + 单任务Head）流水线说明

这份文档专门解释你现在这条“闭环”到底是怎么接起来的：

> RGB（或视频帧） → 4个视觉backbone特征 → **RGB→PC 对齐encoder（冻结）** → **Diffusion Policy head（训练/推理）** → 动作序列（horizon=8）

如果你只想看最少入口：
- 训练：`tools/train_dp_rgb_single_task_4models.py`
- 离线推理：`tools/infer_dp_rgb_4models.py`
- 在线推理（需要时先提特征）：`tools/infer_dp_rgb_4models_online.py`

---

## 1. 关键“契约”（不要猜，按这个来）

### 1.1 训练/推理时的张量形状

- **4模型输入特征（离线zarr读出来）**：
  - `obs_raw`: `FloatTensor[B, To, 4, 2048]`
  - 其中：
    - `B` batch
    - `To = n_obs_steps`（默认 2）
    - `4` 对应 `(croco, vggt, dinov3, da3)`
    - `2048` 是为了能 stack：对每个模型先 slice/pad 到各自 in_dim，再 pad 到 2048

- **对齐 encoder 输出**（冻结）：
  - `obs_features`: `FloatTensor[B, To, fuse_dim]`
  - `fuse_dim` 来自对齐 ckpt 的配置（你当前是 1280）

- **DP head 输出动作序列**：
  - `action_pred`: `FloatTensor[B, horizon, action_dim]`
  - `horizon` 默认 8
  - `action_dim` 由轨迹 pkl 里的 `left_joint_path/right_joint_path` 解析得到：
    - 单臂：6
    - 双臂：`6(left)+6(right)=12`（当 `--use_left_arm --use_right_arm --fuse_arms`）

### 1.2 “4模型输入维度异构”怎么处理

对齐 ckpt 里的 4 个 adapter 的输入维度不一样：

- CroCo: 1024
- VGGT:  2048
- DINOv3: 768
- DA3:   2048

所以 dataset 会做两层处理：
1) 对每个模型 `f` 先 mean pool 成 `[C]`
2) `f` slice/pad 到该模型的 `in_dim`
3) 再 pad 到 `max_dim=2048` 以便能 `stack([4,max_dim])`
4) encoder 里只取每个模型前 `in_dim` 那一段喂对应 adapter

---

## 2. 代码里到底怎么“接 head”的（逐步走一遍）

### 2.1 Dataset：把四个 zarr + traj pkl 变成训练样本

文件：`features_common/dp_rgb_dataset_4models.py`

输出一个 `DP4Sample`：
- `obs`: `[To,4,2048]`
- `action`: `[horizon, action_dim]`

关键点：
- `task` 目录命名不一致时，traj 侧会用 `base_task = task.split('-demo_randomized')[0]` 做兜底路径查找。
- episode 比 horizon 短：允许存在，`action` 会用最后一帧 repeat padding。
- 双臂长度不一致：允许存在，先对短的一侧 repeat padding，再 concat。

### 2.2 冻结对齐 encoder：RGB(4模型) → fused feature

文件：`features_common/rgb2pc_aligned_encoder_4models.py`

核心类：`RGB2PCAlignedEncoder4Models`

- `from_checkpoint(ckpt_path, freeze=True)`：
  - 从 `tools/train_rgb2pc_distill.py` 产出的 ckpt 里加载：
    - `adapters` (4个)
    - `fusion` (weighted 或 moe)
    - `proj_student`
  - 自动从 checkpoint的权重 shape 推断 `in_dims`
  - `freeze=True` 会 `requires_grad=False` 且 `eval()`

- `forward(x)`：
  - 输入 `x: [B,To,4,2048]`
  - 对第 `mi` 个模型：取 `x[..., :in_dim[mi]]` → adapter → 得到 `[B*To, fuse_dim]`
  - 融合 `fusion(zs)` → `proj_student` → reshape 回 `[B,To,fuse_dim]`

### 2.3 DP Head：只训练 head，不更新 encoder

文件：`features_common/dp_rgb_policy_multitask.py`

我们复用这里的单 head 实现：
- `HeadSpec`：定义 head 的所有关键超参（action_dim/horizon/n_obs_steps/.../obs_feature_dim）
- `DiffusionRGBHead`：
  - `compute_loss(obs_features, action, normalizer_obs, normalizer_action)`：训练
  - `predict_action(obs_features, normalizer_obs, normalizer_action)`：推理采样

关键点：
- 这个 head 本质是 `ConditionalUnet1D` + `DDPMScheduler`（diffusers缺失时有最小fallback scheduler）。
- 训练时 loss 是 noise-pred MSE（标准 DDPM）。

### 2.4 训练入口：把 dataset + frozen encoder + head 串起来

文件：`tools/train_dp_rgb_single_task_4models.py`

训练流程：
1) 构建 `DPRGB4ModelDataset`（obs/action）
2) 加载 `RGB2PCAlignedEncoder4Models.from_checkpoint(..., freeze=True)`
3) 构建 `DiffusionRGBHead(spec=HeadSpec(... obs_feature_dim=encoder.spec.fuse_dim ...))`
4) normalizer：
   - `obs`：**identity normalizer**（因为 obs_features 是 encoder 输出，不是 raw 4-model feature）
   - `action`：从 dataset actions 拟合 `SingleFieldLinearNormalizer`
5) 训练时对 encoder 使用 `torch.no_grad()`：
   - `z = encoder(obs_raw)`
   - `loss = head.compute_loss(obs_features=z, action=action, ...)`

产物 `final_head.pt` 包含：
- `head_state`
- `normalizer`（obs + action）
- `config`
- `encoder_ckpt`（对齐 encoder 的 ckpt 路径，推理时会重新加载并冻结）
- `encoder_spec`（ in_dims/fuse_dim/fusion ）
- `action_dim`、`obs_c`（log用途）

---

## 3. 推理与部署执行（receding horizon）

### 3.1 离线推理：从 zarr 取 obs → encoder → head

文件：`tools/infer_dp_rgb_4models.py`

- 从 `final_head.pt` 恢复 head + normalizer
- 再从 `payload['encoder_ckpt']` 恢复 frozen encoder
- 加载 `n_obs_steps` 观测：`load_obs_4(...) -> [1,To,4,2048]`
- `z = encoder(obs)`
- `action_pred = head.predict_action(... )['action_pred']  # [1,H,A]`
- 根据 `--exec_steps K` 返回 `action_exec = action_pred[:K]`

### 3.2 机器人端怎么用（K=1/2最常见）

你最终在真实环境里一般会这样滚动执行：

- 每个控制周期：
  1) 取最新 `To=n_obs_steps` 帧
  2) 预测 `horizon=8` 步动作
  3) **只执行前 `K` 步**（推荐 K=1；或低频控制时 K=2）
  4) 下个周期再预测一次（receding horizon）

这能显著提高鲁棒性：因为环境会偏离离线数据分布，必须每步重规划。

---

## 4. 文件清单（你现在应该看哪些入口）

### 4.1 4模型 DP 必看

- `features_common/dp_rgb_dataset_4models.py`
  - 4模型 zarr + traj pkl → `obs [To,4,2048]` + `action [H,A]`

- `features_common/rgb2pc_aligned_encoder_4models.py`
  - 4模型对齐 encoder（从 rgb2pc ckpt 还原）

- `features_common/dp_rgb_policy_multitask.py`
  - `DiffusionRGBHead`（训练loss + 推理采样）

- `tools/train_dp_rgb_single_task_4models.py`
  - 单任务 head 训练入口（冻结 encoder）

- `tools/infer_dp_rgb_4models.py`
  - 离线推理入口（读 zarr）

- `tools/infer_dp_rgb_4models_online.py`
  - 在线 wrapper：缺特征就先调用 `tools/run_extract_features.py` 生成 zarr，再调用离线推理

---

## 5. 常见坑（你之前踩过的都在这）

- **GPU device mismatch**：DP 采样时 `x/tt` 必须在 `obs_features.device` 上创建。
  - 相关修复在 `DiffusionRGBHead.predict_action()`。

- **action_dim 不一致/双臂丢失**：
  - 默认只用左臂是为了兼容单臂任务；双臂任务请显式传：
    - `--use_left_arm --use_right_arm --fuse_arms`

- **短 episode**：
  - 现在允许 `< horizon`，会在 dataset 内 padding。

---

## 6. 我建议你接下来只做两件事

1) 先只盯单任务（例如 beat_block_hammer），跑通：
   - 提特征（或用已有 zarr）
   - 训练 1-5 epoch
   - 推理输出动作

2) 再做“真在线”（相机流式逐帧提特征 + 缓存最近 To 帧），这会涉及：
   - 每个 backbone 的 streaming 版本
   - 性能/延迟优化（比如 fp16、batch=1、异步队列）

