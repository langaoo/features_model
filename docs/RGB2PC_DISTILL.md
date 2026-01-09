# RGB→点云表征蒸馏（路线2）

你选择的路线2目标是：

- **推理时只输入 RGB**（或 RGB 预先提取的视觉特征）
- 输出一个 embedding，使它在下游任务上**尽量“像点云特征（ULIP pc_feat）一样好用”**

这与几何对齐（point↔patch，需要 per-point uv）不同：路线2不要求几何对应，而是用点云特征做 teacher 做跨模态蒸馏。

## 训练脚本

脚本：`tools/train_rgb2pc_distill.py`

Teacher（监督）：
- 从 `pc_dataset/PC/ULIP_FEAT_PT_POINT/<task>/<episode>/step_*.ply.ulip_*.pt` 读取 `pc_feat: [N,256]`

Student（学习）：
- 从四个视觉特征 root 中读取 `<task>/<episode>.pt`，在每个 pack 内从 token 平面随机采样 K 个 token
- 对每个模型分别用一个小 MLP adapter 投影到统一维度 D
- 融合后做 mean-pooling 得到该样本的 student embedding

Loss：
- 主损失：batch 级 CLIP-style InfoNCE（teacher embedding 与 student embedding 对齐）
- 可选：`--loss_mse` 开启额外的归一化 MSE 蒸馏项

## 四模型特征如何“良好地用起来”

两种融合方式：

1) `--fusion weighted`（默认，推荐先用）
- **全局权重**：学习一个长度为 M 的权重向量，对所有样本共享。
- 优点：稳定、训练最不容易炸、参数少。
- 缺点：无法“按样本/局部内容”自适应选择模型。

2) `--fusion moe`
- **门控融合（MoE gating）**：对每个 token/样本学习一个 [N,M] 的门控权重。
- 优点：表达更强，可能在 domain shift/局部纹理上更占优。
- 缺点：更吃算、更难训稳；建议先用 weighted 收敛，再试 moe。

经验建议：
- 先用 `weighted` 把流程跑通 + 收敛；收敛后再试 `moe` 看下游指标是否提升。

## “再蒸馏到 ResNet”算不算蒸馏后的蒸馏？

是的，它本质上是 **学生压缩/部署蒸馏**，但这是非常常见且合理的工程路径：

1) 第一次蒸馏：点云 teacher → 多视觉特征融合 student（得到强表征，但推理依赖 4 个特征提取器的输出）
2) 第二次蒸馏：强 student → 轻量 student（例如 ResNet/ConvNeXt-T/ViT-Tiny）

第二次蒸馏的目的通常是：
- 线上推理速度/显存/成本
- 统一部署（不依赖四个大模型）

注意：第二次蒸馏仍然需要训练数据，但不一定需要点云了——你可以用第一阶段 student 的输出作为新 teacher。

## 推理阶段的输入到底是什么？

严格来说路线2有两种推理形态：

1) **RGB→embedding（端到端）**
- 需要你把视觉 encoder（croco/vggt/dino/da3）也接入推理图里，或训练一个最终单模型（第二次蒸馏）。

2) **视觉特征→embedding（更现实的第一步）**
- 你可以离线/在线先提取视觉特征 pack，然后用 `train_rgb2pc_distill.py` 训练出的 adapter+fusion+projector 做快速映射。

本仓库当前实现的是第 2) 种：输入是“预先提取好的视觉特征 pack”。

## 几何对齐（你现在的实现）最后能干嘛？

几何对齐（point↔patch，依赖 per-point uv）的价值主要在于：

- 跨模态对应：给定一个点，找到对应的像素/patch；或反过来（2D-3D grounding）
- 2D/3D 检索与标注传播：用 2D 语义把标签/分割传播到 3D，或用 3D 帮 2D 稳定
- 3D 感知融合：把多视觉特征“投影”到点云上作为点特征增强
- 几何一致性约束：对重建/跟踪/姿态估计等任务可作为额外监督

它的核心是“有对应关系就学更精确的局部对齐”。而路线2是“没有对应也要学到像点云一样的语义空间”。

## 时间维（8 帧 window）怎么用？

视觉特征通常是 `per_frame_features[W,T,Hf,Wf,C]`，其中 `T=8` 对应机器人任务里一个 window 的 8 帧。
蒸馏训练时，时间维有两种常见用法：

### A. 逐帧严格配对（推荐默认）

**做法**：

- teacher：随机采样一个 ULIP step 文件（例如 `step_0042.ply.ulip_*.pt`）
- student：在每个视觉 pack 的 `frame_paths` 里找到同名的 `step_0042` 帧，读取该帧特征 `[Hf,Wf,C]`
- 从该帧的 token grid 随机采样 $K$ 个 token，经过 adapter + fusion + projector 得到 student embedding

**优点**：

- 监督最“干净”：teacher 与 student 对应同一时刻
- 对机器人任务很自然：点云 teacher 往往也是逐 step 采集

**实现**：`tools/train_rgb2pc_distill.py` 支持 `--strict_pairing`。

当某些视觉 pack 缺失该 step 时，用 `--pairing_fallback` 控制处理方式：

- `random`：回退到随机帧（更稳，不会卡住，但监督更噪）
- `skip`：跳过该样本并重采（更干净，但会降低有效 batch）
- `error`：直接报错（用于数据一致性排查）

### B. window 聚合（适合弱对齐/弱时序场景）

**做法**：

- teacher：可选把一个 window 里多个 step 的 teacher embedding 做平均，得到 window-level teacher
- student：从 window 内多个帧采样 token 或先对时间维做 pooling，再得到一个 window-level student embedding

**优点**：

- 当时间戳不完全对齐、或某些 step 缺帧时，更鲁棒

**缺点**：

- 监督会更“软”，往往需要更大 batch/更久训练才能达到同等效果

### 机器人任务的推荐

- 如果你能保证 `frame_paths` 与 ULIP step 命名一致（例如都是 `step_XXXX`），优先用 **A：逐帧严格配对**。
- 如果你发现某些模型特征偶尔缺 step（比如抽帧策略不一致），先用 `--pairing_fallback error` 或 `skip` 排查，再决定是否回退 `random`。
