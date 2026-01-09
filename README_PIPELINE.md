# Features Model Pipeline# RGB-Only 蒸馏对齐训练全流程文档



本项目实现了从**多视角RGB/深度**到**机器人动作**的端到端流水线。本文档详细记录了从原始数据到最终对齐模型的完整训练流程（Route 2）。该流程的目标是训练一个 Student 模型（由 4 个视觉模型融合而成），使其仅通过 RGB 图像就能输出与 Teacher 模型（ULIP 点云特征）对齐的语义特征。



## 快速导航## 1. 核心代码文件说明



### 核心文档（按阅读顺序）| 文件路径 | 用途 | 状态 |

| :--- | :--- | :--- |

1. **[DP RGB 4模型流水线](docs/DP_RGB_4MODELS_PIPELINE.md)** ⭐ **必读**| `tools/run_extract_features.py` | **RGB特征提取入口**。统一调用 CroCo/VGGT/DINOv3/DA3 提取特征并直接保存为 `.zarr`。 | ✅ 核心 |

   - 4模型特征提取 → 冻结对齐encoder → Diffusion Policy head| `tools/extract_ulip_features_to_zarr.py` | **点云特征提取**。使用 ULIP 模型提取点云特征并直接保存为 `.zarr`。 | ✅ 核心 |

   - 包含训练/推理入口、shape契约、常见坑| `tools/train_rgb2pc_distill.py` | **训练脚本**。执行 Student (RGB) 到 Teacher (PC) 的蒸馏训练。 | ✅ 核心 |

| `configs/train_rgb2pc_distill_default.yaml` | **训练配置**。定义超参数、数据路径等。 | ✅ 核心 |

2. **[To（n_obs_steps）取值建议](docs/TO_SELECTION_GUIDE.md)**| `features_common/fusion.py` | **融合模块**。实现 WeightedFusion (加权) 和 MoEFusion (门控) 算法。 | ✅ 核心 |

   - 推荐 To=2（默认）或 To=1/4| `features_common/rgb2pc_distill_dataset.py` | **数据加载器**。负责从 Zarr 读取数据、采样 Token、配对 RGB 与点云。 | ✅ 核心 |

   - 解释与对齐训练 8 帧无关

---

3. **[Policy 文件 vs Train 脚本职责划分](docs/POLICY_VS_TRAIN_SCRIPT.md)**

   - 模型库（`features_common/dp_rgb_policy_*.py`）vs 训练入口（`tools/train_*.py`）## 2. 详细复现步骤



4. **[在线 vs 离线定义](docs/ONLINE_VS_OFFLINE.md)**### 第一步：RGB 特征提取与保存

   - 判断标准：是否闭环（不是是否落盘）

   - "新数据不落盘直接串联" = 在线推理我们需要分别提取 4 个视觉模型的特征。



5. **[对齐训练与泛化问题](docs/ALIGNMENT_TRAINING_AND_GENERALIZATION.md)****1.1 提取特征 (直接保存为 Zarr)**

   - 对齐 8 帧训练是否合理？合理（数据增强）

   - 新数据泛化：同任务可以，跨任务需重新训练使用 `tools/run_extract_features.py`。该脚本会调用各子目录下的提取脚本，直接输出 Zarr 格式。



### 补充文档```bash

# 示例：提取 CroCo 特征

- **[对齐流水线](docs/ALIGNMENT_PIPELINE.md)**：RGB→PC 对齐模块训练python tools/run_extract_features.py \

- **[DP RGB 单模型指南](docs/DP_RGB_GUIDE.md)**：旧单模型版本（已被 4模型替代）  --model croco \

- **[RGB→PC 蒸馏](docs/RGB2PC_DISTILL.md)**：对齐模块蒸馏细节  --rgb_root /path/to/rgb_images \

  --out_root /path/to/save/zarr_files \

---  --device cuda \

  --tasks task1 task2 ...

## 主要流水线```



### 1. RGB特征提取（离线/在线）*   **输入**：原始 RGB 图片文件夹。

*   **输出**：`<out_root>/<task>/<episode>.zarr` (包含 `per_frame_features` 等)。

**离线（推荐用于训练）**：*   **注意**：不再需要中间的 `.pt` 文件转换步骤。

```bash

# 提取 4 个模型的特征（croco/vggt/dinov3/da3）---

python tools/run_extract_features.py \

  --model croco --rgb_root rgb_dataset/RGB --out_root rgb_dataset --device cuda --all### 第二步：点云 (ULIP) 特征提取



python tools/run_extract_features.py \使用 ULIP 模型处理点云数据，作为 Teacher 信号。

  --model vggt --rgb_root rgb_dataset/RGB --out_root rgb_dataset --device cuda --all

```bash

python tools/run_extract_features.py \python tools/extract_ulip_features_to_zarr.py

  --model dinov3 --rgb_root rgb_dataset/RGB --out_root rgb_dataset --device cuda --all```



python tools/run_extract_features.py \*   **注意**：你需要修改脚本内的 `PC_SOURCE_DIR` 和 `OUTPUT_ZARR_DIR` 路径。

  --model da3 --rgb_root rgb_dataset/RGB --out_root rgb_dataset --device cuda --all*   **过程**：

```    1.  读取 `.ply` 点云文件。

    2.  输入 ULIP (PointBERT) 编码器。

**在线（推理时实时提特征）**：    3.  提取全局特征 (Global Feature, `[1280]`) 或逐点特征。

- 集成 4 个 backbone 到推理脚本（见 `RoBoTwin/policy/infer_dp_4models.py` 示例）    4.  保存为 `.zarr` 文件。

*   **输出**：`<OUTPUT_ZARR_DIR>/<task>/<episode>.zarr`。

---

---

### 2. RGB→PC 对齐训练（冻结后给 DP 用）

### 第三步：融合与对齐训练

```bash

python tools/train_rgb2pc_distill.py \这是核心步骤，训练 Student 逼近 Teacher。

  --config configs/train_rgb2pc_distill_default.yaml \

  --epochs 100 --batch_size 32 --num_workers 4 \**3.1 融合机制 (Fusion)**

  --save_dir outputs/train_rgb2pc_runs/run_01

```在 `features_common/fusion.py` 中实现：



产出：`outputs/train_rgb2pc_runs/.../ckpt_step_XXXXXX.pt`（包含 adapters/fusion/proj_student）*   **WeightedFusion (默认)**：

    *   为 4 个模型各学习一个全局权重 $w_i$。

---    *   公式：$F_{fused} = \sum (w_i \cdot F_i)$。

    *   特点：计算量小，稳定。

### 3. DP Head 单任务训练（4模型 + 冻结encoder）*   **MoEFusion (混合专家)**：

    *   根据输入特征动态计算权重 $w_i(x)$。

```bash    *   特点：能根据图像内容（如纹理丰富区用 DINO，几何丰富区用 DepthAnything）动态调整依赖。

python tools/train_dp_rgb_single_task_4models.py \

  --task beat_block_hammer-demo_randomized-20_head_camera \**3.2 维度对齐**

  --encoder_ckpt outputs/train_rgb2pc_runs/.../ckpt_step_XXXXXX.pt \

  --rgb_zarr_roots_4 \*   **问题**：4 个模型输出维度不同 (e.g., 768, 1024, 384)，Teacher 维度也不同 (1280)。

      rgb_dataset/features_croco_encoder_dict_unified_zarr \*   **解决**：

      rgb_dataset/features_vggt_encoder_dict_unified_zarr \    1.  **Adapter**：每个视觉模型后接一个 MLP，将维度统一映射到 `fuse_dim` (例如 1280)。

      rgb_dataset/features_dinov3_encoder_dict_unified_zarr \    2.  **Fusion**：在 `fuse_dim` 空间内进行加权求和。

      rgb_dataset/features_da3_encoder_dict_unified_zarr \    3.  **Projector**：融合后的特征再经过一层 MLP (Student Projector) 增强表达能力。

  --traj_root raw_data \

  --horizon 8 --n_obs_steps 2 \**3.3 训练命令**

  --epochs 50 --batch_size 32 \

  --save_dir outputs/dp_rgb_runs/beat_block_hammer_4models \```bash

  --use_left_arm --use_right_arm --fuse_arms \python tools/train_rgb2pc_distill.py \

  --include_gripper  # 若要训练 14 维 action（7+7）  --config configs/train_rgb2pc_distill_default.yaml \

```  --num_workers 4 \

  --batch_size 64

产出：`outputs/dp_rgb_runs/.../final_head.pt`（包含 head_state + normalizer + encoder_ckpt）```



------



### 4. 推理## 3. 数据流 (Data Flow)



**离线推理（从 zarr 读特征）**：整个过程的数据形状变化如下：

```bash

python tools/infer_dp_rgb_4models.py \1.  **输入 (RGB)**: `[Batch, 3, H, W]` (原始图像)

  --head_ckpt outputs/dp_rgb_runs/.../final_head.pt \2.  **视觉编码器**: 输出 `[Batch, Hf, Wf, C_i]` (特征图，C_i 不等)

  --rgb_zarr_roots_4 \3.  **采样 (Sampling)**: 随机采样 K 个 Token -> `[Batch, K, C_i]`

      rgb_dataset/features_croco_encoder_dict_unified_zarr \4.  **Adapter**: 映射到统一维度 -> `[Batch, K, 1280]`

      rgb_dataset/features_vggt_encoder_dict_unified_zarr \5.  **Fusion**: 4 个特征融合 -> `[Batch, K, 1280]`

      rgb_dataset/features_dinov3_encoder_dict_unified_zarr \6.  **Pooling**: 平均池化 -> `[Batch, 1280]` (Student Embedding)

      rgb_dataset/features_da3_encoder_dict_unified_zarr \7.  **Teacher (PC)**: 读取 ULIP 特征 -> `[Batch, 1280]` (Teacher Embedding)

  --task beat_block_hammer-demo_randomized-20_head_camera \8.  **Loss**: 计算 Student 和 Teacher 的 InfoNCE Loss。

  --episode episode_0 \

  --start_frame 0 \

  --exec_steps 1 \## 4. 优化建议 (Optimization)

  --device cuda

```如果你想进一步提升效果：



**在 RoBoTwin 仿真中推理**：1.  **数据增强 (Data Augmentation)**:

```bash    *   目前是直接读取预提取特征，无法做图像增强。

python RoBoTwin/policy/infer_dp_4models.py \    *   **优化**：可以在 Feature 层面做 Dropout 或加噪声，模拟数据增强。

  --head_ckpt outputs/dp_rgb_runs/.../final_head.pt \

  --rgb_zarr_roots_4 \2.  **MoE 融合**:

      rgb_dataset/features_croco_encoder_dict_unified_zarr \    *   尝试将 `configs/train_rgb2pc_distill_default.yaml` 中的 `fusion` 改为 `moe`。MoE 通常能更好地利用不同模型的优势。

      rgb_dataset/features_vggt_encoder_dict_unified_zarr \

      rgb_dataset/features_dinov3_encoder_dict_unified_zarr \3.  **扩大 Batch Size**:

      rgb_dataset/features_da3_encoder_dict_unified_zarr \    *   对比学习 (InfoNCE) 非常依赖大 Batch Size。如果显存允许，尽量调大 `batch_size`，或者增加 `grad_accum_steps`。

  --task beat_block_hammer-demo_randomized-20_head_camera \

  --episode episode_0 \4.  **更多数据**:

  --start_frame 0 \    *   目前只用了 20 个 Episode。对比学习需要海量数据才能学到通用的语义对齐。建议扩大到所有可用任务。

  --exec_steps 1 \

  --device cuda \5.  **时序建模**:

  --robotwin_env_class beat_block_hammer  # 可选：集成环境    *   目前是单帧对齐。可以尝试 `window` 模式（在 yaml 中设置 `sample_unit: window`），让模型学习一段视频与一段点云的对应关系。

```



---## 4模型 + 冻结对齐encoder + 单任务DP Head（当前推荐入口）



## 关键文件索引如果你要跑你现在最关心的闭环：



### Dataset> RGB/video → 4视觉模型特征 → RGB→PC对齐encoder(冻结) → Diffusion Policy head → 动作



- `features_common/dp_rgb_dataset_4models.py`：4模型 zarr + traj pkl → `obs [To,4,2048]` + `action [H,A]`请直接看这份说明：

  - 支持 `include_gripper=True` 输出 7/14 维 action

- `docs/DP_RGB_4MODELS_PIPELINE.md`

### Model

对应脚本入口：

- `features_common/rgb2pc_aligned_encoder_4models.py`：4模型对齐 encoder（冻结）

- `features_common/dp_rgb_policy_single.py`：单任务 DP head（`HeadSpec` + `DiffusionRGBHead`）- 训练：`tools/train_dp_rgb_single_task_4models.py`

- `features_common/dp_rgb_policy_multitask.py`：多任务包装（共享 encoder + 多个 head）- 离线推理（读zarr）：`tools/infer_dp_rgb_4models.py`

- 在线wrapper（缺特征时自动提取）：`tools/infer_dp_rgb_4models_online.py`

### Tools

- `tools/train_dp_rgb_single_task_4models.py`：单任务训练入口
- `tools/infer_dp_rgb_4models.py`：离线推理入口
- `tools/infer_dp_rgb_4models_online.py`：在线 wrapper（缺特征时先提取）
- `RoBoTwin/policy/infer_dp_4models.py`：RoBoTwin 仿真推理入口

---

## Action 维度说明

| 配置 | Action 维度 | 说明 |
|-----|------------|------|
| `--use_left_arm` | 6 或 7 | 单臂（6=关节，7=关节+夹爪） |
| `--use_right_arm` | 6 或 7 | 单臂（右） |
| `--use_left_arm --use_right_arm --fuse_arms` | 12 或 14 | 双臂（12=6+6，14=7+7） |
| `--include_gripper` | +1/+2 | 额外添加夹爪维度 |

**RoBoTwin 执行时需要 14 维**（7+7），如果训练时只用 12 维，推理时需要补 gripper（见 `RoBoTwin/policy/infer_dp_4models.py` 的 `predict_action_14d` 函数）。

---

## 测试

```bash
pytest -q tests/
```

---

## 常见问题

### Q1: To=2 与对齐训练 8 帧矛盾吗？

**不矛盾**。对齐训练用 8 帧是**数据增强策略**，DP 推理时 To 可以独立选择。详见 [To 取值建议](docs/TO_SELECTION_GUIDE.md)。

### Q2: 为什么有 `dp_rgb_policy_multitask.py` 又有 `train_dp_rgb_single_task_4models.py`？

**Policy 是模型库，Train 是训练入口**。详见 [职责划分说明](docs/POLICY_VS_TRAIN_SCRIPT.md)。

### Q3: "新数据不落盘直接串联"算在线还是离线？

**在线推理**。判断标准是"是否闭环"，不是"是否落盘"。详见 [在线 vs 离线定义](docs/ONLINE_VS_OFFLINE.md)。

### Q4: 用新数据训练 head，能在新数据集推理吗？

**同任务新 episode 可以，跨任务需重新训练**。详见 [对齐训练与泛化](docs/ALIGNMENT_TRAINING_AND_GENERALIZATION.md)。

### Q5: Action 维度到底是 6/7/12/14？

- **raw_data traj**：只记录 arm 关节（6 维/臂），不含 gripper
- **RoBoTwin 执行**：需要 arm+gripper（7 维/臂 = 14 双臂）
- **训练时**：可以只训 12（arm only），推理时补 gripper；或用 `--include_gripper` 训练 14

---

## 贡献者

（项目信息）

## 许可证

（许可证信息）
