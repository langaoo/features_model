# Task 命名规则和数据结构说明

## Task 命名格式

### 标准格式
```
<task_name>-<data_type>-<version>_<camera_spec>
```

### 组成部分

1. **task_name** (必需)
   - 任务的名称标识
   - 例如: `beat_block_hammer`, `pick_and_place`, `door_opening`

2. **data_type** (必需)
   - 数据采集方式
   - 常见值:
     - `demo_randomized`: 随机化演示数据
     - `demo_fixed`: 固定场景演示数据
     - `teleoperation`: 遥操作数据
     - `autonomous`: 自主采集数据

3. **version** (必需)
   - 数据集版本号或大小标识
   - 通常是数字，表示 episode 数量或版本
   - 例如: `20`, `50`, `100`, `v1`, `v2`

4. **camera_spec** (可选)
   - 相机配置标识
   - 标识使用的相机系统或配置
   - 例如: `sapien_head_camera`, `head_camera`, `wrist_camera`

### 示例对比

```
beat_block_hammer-demo_randomized-20_sapien_head_camera
beat_block_hammer-demo_randomized-20_head_camera
```

**区别**:
- `sapien_head_camera`: 
  - Sapien 物理模拟器中的头部相机
  - 通常分辨率 640x480 或 512x512
  - 相机外参由 Sapien 定义
  - 数据路径: `raw_data/<task>/sapien_head_camera/`

- `head_camera`:
  - 真实机器人或其他系统的头部相机
  - 分辨率和外参可能不同
  - 数据路径: `raw_data/<task>/head_camera/`

**如何选择**:
- 如果只使用模拟器数据 → 使用 `sapien_head_camera`
- 如果使用真实机器人数据 → 使用 `head_camera` 或自定义名称
- 如果混合使用 → 分别创建两个 task，或使用多相机配置

### 完整示例

```
# Sapien 模拟器数据
beat_block_hammer-demo_randomized-20_sapien_head_camera

# 真实机器人数据
beat_block_hammer-teleoperation-50_head_camera

# 多相机数据（不在 task 名称中指定，在配置文件中指定）
beat_block_hammer-demo_randomized-20
```

## 数据目录结构

### 单相机数据
```
raw_data/
  └── beat_block_hammer-demo_randomized-20_sapien_head_camera/
      ├── sapien_head_camera/          ← RGB 图像
      │   ├── episode0/
      │   │   ├── 000000.jpg
      │   │   ├── 000001.jpg
      │   │   └── ...
      │   ├── episode1/
      │   └── ...
      └── _traj_data/                  ← 动作轨迹
          ├── episode0.pkl
          ├── episode1.pkl
          └── ...
```

### 多相机数据
```
raw_data/
  └── beat_block_hammer-demo_randomized-20/
      ├── sapien_head_camera/          ← 头部相机
      │   ├── episode0/
      │   │   ├── 000000.jpg
      │   │   └── ...
      │   └── ...
      ├── sapien_wrist_camera/         ← 腕部相机
      │   ├── episode0/
      │   │   ├── 000000.jpg
      │   │   └── ...
      │   └── ...
      ├── left_camera/                 ← 左侧相机
      │   ├── episode0/
      │   └── ...
      └── _traj_data/                  ← 轨迹数据（所有相机共享）
          ├── episode0.pkl
          └── ...
```

### 轨迹数据格式 (pkl)
```python
{
    'left_joint_path': {
        'position': np.ndarray,  # [T, 6] - 左臂关节位置
        'velocity': np.ndarray,  # [T, 6] - 左臂关节速度
        'gripper': np.ndarray,   # [T, 1] - 左夹爪状态 (可选)
    },
    'right_joint_path': {
        'position': np.ndarray,  # [T, 6] - 右臂关节位置
        'velocity': np.ndarray,  # [T, 6] - 右臂关节速度
        'gripper': np.ndarray,   # [T, 1] - 右夹爪状态 (可选)
    },
    'metadata': {
        'fps': 10,
        'task_name': 'beat_block_hammer',
        'success': True,
        ...
    }
}
```

## 在配置文件中指定相机

### 单相机配置
```yaml
# configs/train_online_default.yaml
data:
  raw_data_root: raw_data
  tasks:
    - beat_block_hammer-demo_randomized-20_sapien_head_camera
  camera_name: sapien_head_camera  # 要使用的相机
```

### 多相机配置
```yaml
# configs/train_online_multiview.yaml
data:
  raw_data_root: raw_data
  tasks:
    - beat_block_hammer-demo_randomized-20
  cameras:                          # 多相机列表
    - sapien_head_camera
    - sapien_wrist_camera
  camera_fusion: concat             # 融合方式: concat | attention | late_fusion
```

## Task 自动发现

Dataset 会自动尝试多种路径查找数据:

```python
# 给定 task = "beat_block_hammer-demo_randomized-20_sapien_head_camera"
# 会依次尝试:
1. raw_data/beat_block_hammer-demo_randomized-20_sapien_head_camera/sapien_head_camera/
2. raw_data/beat_block_hammer-demo_randomized-20_sapien_head_camera/head_camera/
3. raw_data/beat_block_hammer/demo_randomized/sapien_head_camera/

# 给定 task = "beat_block_hammer"
# 会依次尝试:
1. raw_data/beat_block_hammer/sapien_head_camera/
2. raw_data/beat_block_hammer/head_camera/
3. raw_data/beat_block_hammer/demo_randomized/sapien_head_camera/
```

## 最佳实践

### 1. 命名一致性
```bash
# 好的命名（一致、清晰）
beat_block_hammer-demo_randomized-20_sapien_head_camera
pick_cube-demo_randomized-50_sapien_head_camera
door_opening-teleoperation-100_head_camera

# 不推荐（不一致）
beat_block_hammer_demo_20
pick-cube-random-50-sapien
door_open_teleop_100
```

### 2. 版本管理
```bash
# 使用版本号追踪数据迭代
beat_block_hammer-demo_randomized-20_sapien_head_camera  # v1
beat_block_hammer-demo_randomized-50_sapien_head_camera  # v2 (更多数据)
beat_block_hammer-demo_randomized-50_v2_sapien_head_camera  # v2.1 (修复了错误)
```

### 3. 多相机数据组织
```bash
# 推荐：不在 task 名称中指定相机，在配置文件中指定
beat_block_hammer-demo_randomized-20/
  ├── sapien_head_camera/
  ├── sapien_wrist_camera/
  └── _traj_data/

# 配置文件:
data:
  tasks: [beat_block_hammer-demo_randomized-20]
  cameras: [sapien_head_camera, sapien_wrist_camera]
```

### 4. 处理遗留数据
```bash
# 如果已有旧格式数据
beat_block_hammer/
  ├── sapien_head_camera/
  └── _traj_data/

# 可以创建软链接
ln -s beat_block_hammer beat_block_hammer-demo_randomized-20

# 或在配置中使用短名称
data:
  tasks: [beat_block_hammer]
  camera_name: sapien_head_camera
```

## FAQ

### Q: 为什么需要 camera_spec?
A: 因为:
1. 不同相机的数据格式可能不同（分辨率、畸变等）
2. 方便区分模拟器数据和真实数据
3. 支持多视角训练

### Q: 如果相机名称不同怎么办?
A: Dataset 会自动尝试多种路径，或者你可以在配置文件中明确指定:
```yaml
data:
  camera_name: my_custom_camera
```

### Q: 可以混合使用不同 task 吗?
A: 可以，在配置文件中指定多个 task:
```yaml
data:
  tasks:
    - beat_block_hammer-demo_randomized-20_sapien_head_camera
    - pick_cube-demo_randomized-30_sapien_head_camera
    - door_opening-teleoperation-50_head_camera
```

### Q: 如何验证 task 名称正确?
A: 使用验证工具:
```bash
python tools/validate_data.py --task beat_block_hammer-demo_randomized-20_sapien_head_camera
```

输出示例:
```
✓ Task found: beat_block_hammer-demo_randomized-20_sapien_head_camera
✓ Camera dir: raw_data/.../sapien_head_camera (20 episodes)
✓ Trajectory dir: raw_data/.../_traj_data (20 episodes)
✓ All episodes have matching RGB + trajectory data
```
