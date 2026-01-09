# Features Model Pipeline

> **RGB-Only 蒸馏对齐训练全流程** —— 从多视角 RGB 到机器人动作的端到端流水线

## 🚀 快速开始

### 1. 训练 14 维 DP Head（双臂 + 夹爪）

```bash
# 使用快速脚本
bash scripts/quick_train_14d.sh

# 或手动运行
python tools/train_dp_rgb_single_task_4models.py \
  --task beat_block_hammer-demo_randomized-20_head_camera \
  --encoder_ckpt outputs/train_rgb2pc_runs/run_best_bs32/ckpt_step_0010000.pt \
  --use_left_arm --use_right_arm --fuse_arms --include_gripper \
  --epochs 50 --batch_size 16 \
  --save_dir outputs/dp_rgb_runs/beat_block_hammer_14d \
  --tqdm
```

### 2. 提取特征

```bash
# 提取 RGB 特征（4 个模型）
python tools/run_extract_features.py \
  --model croco \
  --rgb_root rgb_dataset/RGB \
  --out_root rgb_dataset \
  --device cuda --all

# 提取点云特征（ULIP）
python tools/extract_ulip_features_to_zarr.py
```

---

## 📖 完整文档

| 文档 | 说明 |
|-----|------|
| **[README_PIPELINE.md](README_PIPELINE.md)** | 📘 **主流程文档**（必读） |
| [DATA_PATHS.md](DATA_PATHS.md) | 数据路径和软链接说明 |
| [CHECKLIST.md](CHECKLIST.md) | 最近修改清单和验证结果 |

### 专题文档（docs/）

- [DP_RGB_4MODELS_PIPELINE.md](docs/DP_RGB_4MODELS_PIPELINE.md) - 4 模型流水线详解
- [TO_SELECTION_GUIDE.md](docs/TO_SELECTION_GUIDE.md) - To 参数取值建议
- [POLICY_VS_TRAIN_SCRIPT.md](docs/POLICY_VS_TRAIN_SCRIPT.md) - 代码职责划分
- [ONLINE_VS_OFFLINE.md](docs/ONLINE_VS_OFFLINE.md) - 在线/离线定义
- [ALIGNMENT_TRAINING_AND_GENERALIZATION.md](docs/ALIGNMENT_TRAINING_AND_GENERALIZATION.md) - 对齐训练泛化

---

## 🏗️ 项目结构

```
features_model/
├── features_common/     # 核心算法代码
├── tools/              # 训练和推理入口脚本
├── configs/            # 配置文件
├── docs/               # 详细文档
├── scripts/            # 快速开始脚本
├── tests/              # 单元测试
├── integration/        # 外部项目集成代码备份
├── rgb_dataset/        # RGB 特征（软链接）
├── pc_dataset/         # 点云特征（软链接）
└── raw_data/           # 轨迹数据（软链接）
```

---

## 🔧 环境依赖

### 外部项目（需单独克隆）

- **RoBoTwin**：机器人仿真环境
- **DP (Diffusion Policy)**：扩散策略库
- **CroCo**、**VGGT**、**DINOv3**、**Depth-Anything-3**：视觉特征提取模型
- **ULIP**：点云特征提取模型

### Python 依赖

```bash
pip install -r requirements.txt
pip install pytorch3d  # FPS 采样加速（可选）
```

---

## 📊 核心功能

### ✅ 已实现

1. **4 模型特征提取**：CroCo + VGGT + DINOv3 + DA3
2. **RGB→PC 对齐训练**：Student-Teacher 蒸馏
3. **Diffusion Policy Head 训练**：支持 6/7/12/14 维 action
4. **离线推理**：从 zarr 读取特征
5. **在线推理**：实时提取特征（RoBoTwin 集成）

### 🔄 训练流程

```
RGB 图片 → 4 视觉模型特征 → RGB→PC 对齐 encoder（冻结）→ DP head → 机器人动作
```

---

## 🎯 Action 维度说明

| 配置 | Action 维度 | 说明 |
|-----|------------|------|
| 默认（单臂） | 6 | left_arm 关节 |
| `--include_gripper` | 7 | left_arm + gripper |
| `--use_left_arm --use_right_arm --fuse_arms` | 12 | 双臂关节 |
| `--use_left_arm --use_right_arm --fuse_arms --include_gripper` | **14** | **双臂 + 夹爪**（RoBoTwin 推荐） |

---

## 🧪 测试

```bash
pytest -q tests/
```

---

## 📄 许可证

（待补充）

## 👥 贡献者

（待补充）

---

## ⚠️ 注意事项

1. **数据路径**：使用软链接管理，部署到新环境需重新创建（见 [DATA_PATHS.md](DATA_PATHS.md)）
2. **外部依赖**：需要单独克隆 RoBoTwin、DP 等项目（见 `.gitignore`）
3. **GPU 要求**：训练需要至少 16GB 显存（batch_size=16）

---

## 🆘 常见问题

**Q: To=2 与对齐训练 8 帧矛盾吗？**  
A: 不矛盾。对齐训练用 8 帧是数据增强，DP 推理时 To 可独立选择。详见 [TO_SELECTION_GUIDE.md](docs/TO_SELECTION_GUIDE.md)

**Q: 训练脚本缺少参数？**  
A: 所有特征路径都有默认值，只需指定 `--task` 和 `--encoder_ckpt`。详见 [CHECKLIST.md](CHECKLIST.md)

**Q: Action 维度应该用几维？**  
A: RoBoTwin 推荐 14 维（双臂+夹爪）。详见 [README_PIPELINE.md](README_PIPELINE.md)

---

## 权重（Weights）管理

模型权重通常体积很大，会显著增加仓库大小。我们采用 `third_party/` 作为 vendored 源码快照，并**不**默认将大权重推送到远端。下面说明如何在本地恢复权重，以及如何使用 Git LFS（可选）把权重上传到远端。

1. 使用本地权重（推荐，最简单）

- 如果你已经在本地下载了权重并保存在原始子仓库路径（例如 `vggt/weight/model.pt`），可以运行：

```bash
# 将本地原始权重复制到 third_party 的占位位置（覆盖占位文件）
bash scripts/fetch_weights.sh
```

- 脚本会把 `third_party_weights_manifest.json` 中列出的权重复制到 `third_party/` 对应位置（如果原始路径存在）。

2. 使用远程存储（推荐用于公开/团队共享）

- 把大权重上传到一个外部存储（S3/网盘/NAS），并在 `third_party_weights_manifest.json` 中记录下载 URL（或者在 `DATA_PATHS.md` 中写明获取方式）。我们建议不要直接把权重提交到主仓库，除非你使用 Git LFS 并且了解配额影响。

3. 可选：使用 Git LFS 将权重托管到仓库（需慎用）

- 如果你确实需要把权重包含在仓库历史里，可用 Git LFS：

```bash
# 启用 LFS
git lfs install
# 为 manifest 中的每个权重路径添加追踪（脚本自动完成，见下面）
bash scripts/prepare_lfs.sh
# 把权重文件添加到索引并提交（确认 .gitattributes 里已有条目）
git add <weight-files>
git commit -m "Add weight files via Git LFS"
git push origin master
```

- 注意：Git LFS 有存储/带宽配额（GitHub 限制），请确认你能承担这些配额或使用私有的大文件存储。

4. 说明文档与工具

- `third_party/third_party_weights_manifest.json`：列出所有被排除的大权重（path/size/sha256）。
- `scripts/fetch_weights.sh`：把本地原始权重复制到 `third_party/` 的占位位置。
- `scripts/prepare_lfs.sh`：帮助把 manifest 中的权重路径加入 Git LFS track（仅生成 .gitattributes，需你检查后提交）。

如果你要我为你启用 Git LFS 并把某些权重上传，请告诉我你要托管的权重清单（或确认使用 manifest 中的全部权重），我会一步步帮你配置并执行（我会先确认 quota/风险再继续）。

---

**开始你的训练之旅吧！** 🚀
