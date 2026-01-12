# 项目重组计划

## 当前问题

1. **文件混乱**: tools/ 目录下有 25+ 个文件，离线训练、在线训练、特征提取混在一起
2. **配置分散**: 参数全部通过命令行传递，难以管理和复用
3. **功能重复**: 多个版本的训练脚本（单模型、多模型、双臂等）
4. **缺少文档**: task 命名规则、相机配置、数据结构等缺少说明

## 重组方案

### 新目录结构

```
features_model/
├── configs/                          # 配置文件
│   ├── alignment/                    # 对齐训练配置
│   │   └── train_rgb2pc_default.yaml
│   ├── offline_training/             # 离线训练配置
│   │   ├── single_arm.yaml
│   │   ├── dual_arm.yaml
│   │   └── multitask.yaml
│   ├── online_training/              # 在线训练配置
│   │   ├── default.yaml
│   │   └── with_cache.yaml
│   └── extraction/                   # 特征提取配置
│       └── extract_all.yaml
│
├── scripts/                          # 主要训练/推理脚本
│   ├── alignment/
│   │   └── train_rgb2pc.py
│   ├── offline_training/
│   │   ├── train_single_task.py
│   │   └── train_multitask.py
│   ├── online_training/
│   │   ├── train.py
│   │   └── train_with_cache.py
│   └── extraction/
│       └── extract_features.py
│
├── features_common/                  # 核心模块
│   ├── datasets/
│   │   ├── dp_rgb_offline.py
│   │   ├── dp_rgb_online.py
│   │   └── alignment.py
│   ├── models/
│   │   ├── encoder.py
│   │   ├── head.py
│   │   └── extractors.py
│   └── utils/
│       ├── config.py
│       ├── cache.py
│       └── multiview.py
│
├── tools/                            # 辅助工具
│   ├── test_pipeline.py
│   ├── validate_data.py
│   ├── visualize_features.py
│   └── convert_format.py
│
├── docs/                             # 文档
│   ├── DATA_FORMAT.md                # 数据格式说明
│   ├── TASK_NAMING.md                # Task 命名规则
│   ├── CAMERA_CONFIG.md              # 相机配置
│   ├── TRAINING_GUIDE.md             # 训练指南
│   └── API.md                        # API 文档
│
└── archived/                         # 归档的旧文件
    └── old_scripts/
```

### 分阶段执行

#### 阶段 1: 创建新结构（不删除旧文件）
- 创建新目录
- 复制并重构关键文件
- 保持向后兼容

#### 阶段 2: 测试新结构
- 验证新脚本功能正常
- 文档完善

#### 阶段 3: 迁移和清理
- 标记旧文件为 deprecated
- 移动旧文件到 archived/
- 更新 README

## 配置文件标准

### 所有配置文件包含:
1. **data**: 数据路径、任务列表、相机配置
2. **model**: 模型架构、checkpoint 路径
3. **training**: 训练超参数
4. **device**: GPU 配置
5. **output**: 输出目录、日志配置

### 命令行使用:
```bash
# 使用配置文件
python scripts/online_training/train.py --config configs/online_training/default.yaml

# 覆盖特定参数
python scripts/online_training/train.py \\
  --config configs/online_training/default.yaml \\
  --data.tasks task1,task2 \\
  --training.batch_size 16
```

## Task 命名规则

### 格式
```
<task_name>-demo_randomized-<N>_<camera_spec>
```

### 组成部分
1. **task_name**: 任务名称（如 beat_block_hammer）
2. **demo_randomized**: 数据类型标识
3. **N**: 数据集大小或版本号
4. **camera_spec**: 相机配置标识

### 示例
```
beat_block_hammer-demo_randomized-20_sapien_head_camera
├── task_name: beat_block_hammer
├── data_type: demo_randomized
├── version: 20
└── camera: sapien_head_camera

beat_block_hammer-demo_randomized-20_head_camera
├── task_name: beat_block_hammer
├── data_type: demo_randomized
├── version: 20
└── camera: head_camera (不同的相机配置)
```

### 区别
- `sapien_head_camera`: Sapien 模拟器的头部相机
- `head_camera`: 真实机器人或其他系统的头部相机
- 数据格式可能不同（分辨率、视角等）

## 多相机支持

### 数据结构
```
raw_data/
  └── <task>/
      ├── sapien_head_camera/      ← 头部相机
      │   └── episode*/
      ├── sapien_wrist_camera/     ← 腕部相机
      │   └── episode*/
      ├── left_camera/             ← 左侧相机
      │   └── episode*/
      └── _traj_data/              ← 轨迹数据（所有相机共享）
          └── episode*.pkl
```

### 配置指定相机
```yaml
data:
  camera_name: sapien_head_camera  # 只提取这个相机的数据
  # 或者使用多相机
  cameras:
    - sapien_head_camera
    - sapien_wrist_camera
```

## 特征缓存策略

### 问题
在线训练每次都要跑 4 个 backbone，显存占用大（20GB），训练慢（5-10x）

### 解决方案 1: 特征缓存
```python
# 第一次训练：提取并缓存特征
python scripts/online_training/train_with_cache.py \\
  --config configs/online_training/with_cache.yaml \\
  --feature_cache.enabled true \\
  --feature_cache.cache_dir .cache/features

# 后续训练：直接读缓存（快速）
python scripts/online_training/train_with_cache.py \\
  --config configs/online_training/with_cache.yaml \\
  --feature_cache.enabled true \\
  --feature_cache.cache_dir .cache/features
```

缓存结构:
```
.cache/features/
  └── <task>/
      └── <episode>/
          ├── frame_0000.pt  # {croco: [1024], vggt: [2048], ...}
          ├── frame_0001.pt
          └── ...
```

### 解决方案 2: 多 GPU 并行
```python
# 使用 DataParallel 在多 GPU 上并行提特征
CUDA_VISIBLE_DEVICES=0,1 python scripts/online_training/train.py \\
  --config configs/online_training/default.yaml \\
  --device.cuda_visible_devices "0,1" \\
  --device.multi_gpu true
```

### 推荐策略
1. **小数据集（<100 episodes）**: 直接在线训练
2. **中等数据集（100-500 episodes）**: 第一次训练时缓存特征
3. **大数据集（>500 episodes）**: 先离线提特征，用 zarr 格式

## 执行计划

### 立即执行
1. ✅ 创建配置文件系统
2. ⏳ 创建新目录结构
3. ⏳ 移动核心文件到新位置
4. ⏳ 创建文档

### 后续执行
5. 实现特征缓存
6. 实现多 GPU 支持
7. 测试验证
8. 清理旧文件
