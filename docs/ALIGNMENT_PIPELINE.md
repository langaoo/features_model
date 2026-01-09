# Alignment / RGB↔PointCloud 对齐训练流水线（阶段1）

更新时间：2025-12-29

这份文档把你当前仓库里已经实现的“阶段1：离线对齐训练”相关内容捋顺：

- 数据与特征文件的约定
- 四个 RGB 编码器的特征提取命令
- 点云（ULIP）特征/UV 的要求
- `tools/train_alignment_skeleton.py` 当前实现了什么、缺了什么
- 为什么之前训练特别慢 & 已做的修复
- 下一步（阶段2/3）怎么接

> 术语：
> - window/time-slice：连续 8 帧组成的时间片
> - pack：每个 episode 保存的 .pt 文件（包含 per_frame_features/frame_paths/meta）

---

## 1. 项目核心目标（你描述的目标）

目标是用 **2D RGB 的视觉特征**（来自多个视觉编码器并融合）在机器人操作任务上达到接近 **3D 点云特征**（ULIP/pointcloud encoder）的效果。

阶段拆解：
1) 对齐训练：RGB fused feature ↔ 点云 feature 做 InfoNCE 对齐
2) 下游评估：把 RGB aligned feature 与原始点云 feature 分别喂同一个 action head 比性能
3) 蒸馏/部署：把 RGB 教师蒸馏成可部署的小模型

---

## 2. 数据与文件结构约定（当前仓库已经在用）

### 2.1 RGB 数据

- 根目录：`rgb_dataset/RGB/<task>/<episode>/*.png|jpg`
- 每个 episode 是一段连续帧序列

### 2.2 RGB 特征（四模型）

每个模型导出一个特征根目录：

- CroCo: `rgb_dataset/features_croco_v2_encoder_dict_unified/<task>/<episode>.pt`
- VGGT:  `rgb_dataset/features_vggt_encoder_dict_unified/<task>/<episode>.pt`
- DINOv3:`rgb_dataset/features_dinov3_encoder_dict_unified_safe/<task>/<episode>.pt`
- DA3:   `rgb_dataset/features_da3_encoder_dict_unified/<task>/<episode>.pt`

每个 `.pt` 都应是 dict（`features_common/feature_pack.py` 会做兼容）：

- `per_frame_features`: `Tensor[W,T,Hf,Wf,C]`
  - `W` = windows 数（由帧数、window_size=8、stride 决定）
  - `T` = 8（默认）
  - `Hf,Wf` = patch 特征图分辨率（与模型预处理有关）
  - `C` = 模型通道数
- `frame_paths`: `List[List[str]]`，长度 W，每个窗口包含 T 个帧路径；用于严格对齐 `step_XXXX`
- `meta`: dict（必须包含足够的信息让 uv 映射到 patch grid）

### 2.3 点云（ULIP 特征 + 对应 uv）

训练脚本默认走：

- `pc_dataset/PC/ULIP_FEAT_PT_POINT/<task>/<episode>/step_XXXX.ply.ulip_*.pt`

其中每个 pt 形状约定：
- `pc`: `[N,6]` (xyzrgb)
- `uv`: `[N,2]` float32（原图像素坐标）
- `pc_feat`: `[N,256]` float32（ULIP 的 per-point feature）

> 关键：要做严格的 point↔patch 对齐，必须有 `uv`。

---

## 3. 特征提取命令（阶段1已实现）

统一入口：`tools/run_extract_features.py`

它会调用各子项目的提特征脚本（尽量保持一致的输出 schema）。

### 3.1 CroCo

- wrapper：`croco/extract_multi_frame_croco_features_unified.py`

### 3.2 VGGT

- `vggt/extract_multi_frame_vggt_features_wrapper.py`

### 3.3 DINOv3

- `dinov3/extract_multi_frame_dinov3_features_local.py`

### 3.4 Depth-Anything-3 (DA3)

- `Depth-Anything-3/extract_multi_frame_depthanything3_features.py`

### 3.5 校验保存的特征

- `tools/verify_saved_features.py`

用于快速检查：shape/dtype、NaN/Inf、相邻窗口/相邻帧相似度等。

---

## 4. 对齐训练脚本：`tools/train_alignment_skeleton.py`

### 4.1 它现在“做到了什么”（阶段1对齐训练已经可跑）

- 多任务训练：从 `--tasks / --pc_root / --vis_roots` 派生实际数据路径
- 多模型视觉融合：
  - 每个模型一个 Adapter：`C -> fuse_dim(默认256)`（MLP）
  - 融合：`WeightedFusion`（可学习全局权重，softmax）
- 点云侧投影：
  - 如果 ULIP `pc_feat` 存在：`[N,256] -> [N,256]`（MLP）
  - 否则 fallback：`xyz[N,3] -> [N,256]`（MLP）
- 对齐损失：
  - 从视觉 feature map 按 `uv` 采样每点对应 patch 特征 `z_v`
  - 对点云特征投影得到 `z_p`
  - 用 batch InfoNCE: `info_nce(z_p, z_v, tau)`

### 4.2 它当前“没做的/属于骨架”的部分（你说的“还没有训练对齐模块”的点）

严格来说：它已经在训练 adapter/fusion/point projector 了（也就是对齐模块的核心）。

---

## 5. RGB→PointCloud 蒸馏训练（路线2，推理 RGB-only）

当你希望**推理阶段完全不用点云/uv**，而是仅输入 RGB 侧特征并输出一个“像点云特征一样好用”的 embedding，可以使用蒸馏训练脚本：`tools/train_rgb2pc_distill.py`。

### 5.1 样本与监督信号

- Teacher（点云 ULIP）：`pc_dataset/PC/ULIP_FEAT_PT_POINT/<task>/<episode>/step_*.ply.ulip_*.pt`
  - `pc_feat`: `[N,256]`
- Student（视觉特征，多模型可选，推荐 zarr）：
  - `rgb_dataset/<features_...>/<task>/episode_x.zarr`
  - zarr 的好处：训练时随机访问某个 window/frame，不需要 `torch.load` 整个 episode。

训练目标是集合级别对齐：
- 从 teacher 的 `pc_feat` 采样 `K_t` 个点并 pool 得到 teacher embedding
- 从每个视觉模型的 `[Hf,Wf,C]` token 网格采样固定 `K` 个 tokens，经 adapter + pooling 得到每模型一个 `[D]`
- 多模型融合（Weighted 或 MoE）得到 student embedding
- 用 CLIP-style batch InfoNCE 对齐 teacher/student embedding

### 5.2 性能设计：DataLoader + 向量化前向

为了避免“训练 loop 里同步采样 + 读盘 + Python 循环”导致 GPU 吃不满：

- 数据读取与采样放到 `features_common/rgb2pc_distill_dataset.py`：worker 里缓存 zarr pack 句柄与 step 索引
- DataLoader 多进程预取：`--num_workers/--prefetch_factor/--persistent_workers/--pin_memory`
- student 前向向量化：tokens 组装成 `[B,K,C]`，一次 reshape 为 `[B*K,C]` 喂 adapter，再 reshape/pool

### 5.3 AMP 与 nonfinite 梯度

启用 `--amp` 后，若出现 `nonfinite_grads>0` 且开启 `--skip_nonfinite`：

- 本步会跳过 `optimizer.step()`（`stepped=0`）
- 仍会调用 `GradScaler.update()` 让 scale 降下来，通常后续步会恢复 `stepped=1`

如果你要强制纯 fp32 排查数值问题：不要传 `--amp` 即可。

### 5.4 推荐命令

window 模式（默认/更稳定）：

```bash
python tools/train_rgb2pc_distill.py \
  --config configs/train_rgb2pc_distill_default.yaml \
  --sample_unit window \
  --amp --skip_nonfinite \
  --num_workers 4 --prefetch_factor 2 --persistent_workers --pin_memory \
  --tqdm --print_every 50
```

step 模式：

```bash
python tools/train_rgb2pc_distill.py \
  --config configs/train_rgb2pc_distill_default.yaml \
  --sample_unit step \
  --amp --skip_nonfinite \
  --num_workers 4 --prefetch_factor 2 --persistent_workers --pin_memory \
  --tqdm --print_every 50
```
但它还是骨架，主要缺：

1) **更合理的时间维使用**：
   - 目前是对一个 window 只取 1 帧做点↔patch 对齐（anchor frame）。
   - 未来可以：
     - 对 window 内多帧做一致性约束
     - 或把 T 帧视觉特征先做 temporal pooling（你已经在 `features_common/adapters.py` 做了更完整的时间聚合模块，但训练脚本还没用它）

2) **更完整的负样本设计**：
   - 当前 InfoNCE 是“同一批次内”负样本
   - 可以扩展成更强的跨任务/跨 episode memory bank 或 queue

3) **更标准的 dataset/dataloader**：
   - 当前训练循环是纯 Python 采样 + torch.load，没用 torch DataLoader
   - 这对多进程预取不友好

---

## 5. 训练为什么之前特别慢？（根因与修复）

### 5.1 根因（已确认）

你之前的 config 默认包含两个非常大的视觉特征文件：
- VGGT `episode_0.pt` ~ 5.4GB
- DA3  `episode_0.pt` ~ 5.1GB

训练脚本在多任务模式下**每一步都会 torch.load 这些大文件**（即使 LRU cache 命中，也会在 episode 切换时频繁 load）。

此外，脚本里原本有一段循环：

- 对同一个 window 的 `step_indices` 做 `for k in range(S)`
- 但后续真正训练只用到循环结束后的最后一个 `ply_path/step_stem`

这等价于“做了 S 次无意义的 Python 循环开销”。

### 5.2 已做的修复（2025-12-29）

我已经修改了 `tools/train_alignment_skeleton.py`：

- 明确每个 window 只选择一帧作为 anchor（默认 `middle`）
- 新增参数 `--anchor_in_window {middle,random,first,last}`
- 删除无效的 step 循环

效果：同样配置下，单步耗时从 **约 100 秒/step（3 steps 用时 ~309s）** 降到 **约 11 秒/step**。

---

## 6. 当前推荐的“阶段1跑通”流程

1) 确认四个 RGB 特征都已导出（每个 task/episode 都有 .pt）
2) 确认点云 ULIP pt 存在且包含 uv + pc_feat
3) 先 smoke 跑训练 1-10 steps，确认 loss 能下降或至少稳定

训练命令（使用你的 default config）：

```bash
python /home/gl/features_model/tools/train_alignment_skeleton.py \
  --config /home/gl/features_model/configs/train_alignment_default.yaml
```

想更稳的可复现实验：

```bash
python /home/gl/features_model/tools/train_alignment_skeleton.py \
  --config /home/gl/features_model/configs/train_alignment_default.yaml \
  --start_step_mode zero \
  --anchor_in_window middle
```

---

## 7. 接下来要实现什么（阶段2/3）

### 7.1 阶段1（更完整的对齐训练）建议

优先级从高到低：

1) **把训练数据管线换成 DataLoader**
   - 让 episode pack 在 worker 进程里预取/缓存
   - 避免主进程频繁 torch.load 大文件

2) **用 `features_common/adapters.py` 的 TemporalGatedPooling**
   - 先通道对齐，再在时间维 T 上聚合
   - 输出 [W,Hf,Wf,D]，然后按 uv 采样

3) **更强的融合方式**
   - 先保持 WeightedFusion
   - 后续可升级到 token-wise gate / MoE

4) **更严谨的评估指标**
   - 除了 loss：做 RGB/PC 特征的检索 top-k accuracy
   - 采样若干 batch，算正对的相似度分布与负对分布

### 7.2 阶段2：接动作头评估

你需要：
- 一个 action head 网络（输入是特征序列或聚合后的特征）
- 两条输入支路：
  - RGB aligned fused feature
  - 原始 pointcloud feature

策略：
- 保持 action head 完全相同，只换输入特征
- 在相同训练/评估协议下比较性能

### 7.3 阶段3：蒸馏

- 教师：对齐后的 RGB encoder 或对齐后输出特征
- 学生：轻量 CNN/ViT（例如小 ResNet）
- 蒸馏目标：
  - feature regression（MSE/cosine）
  - 或 logit 蒸馏（如果 action head 输出动作分布）

---

## 8. 关键模块速查（代码入口）

- 特征pack统一读取：`features_common/feature_pack.py::load_feature_pack`
- UV 映射：`features_common/uv_mapping.py`
- 点云读取（含 uv）：`features_common/pointcloud.py::read_ascii_ply_xyz_rgb`
- 对齐训练：`tools/train_alignment_skeleton.py`
- 提取特征统一入口：`tools/run_extract_features.py`
- 特征保存校验：`tools/verify_saved_features.py`

