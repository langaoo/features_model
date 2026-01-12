# Head训练完整流程指南

生成时间: 2026年1月10日

---

## 📊 问题1: Head阶段的数据输入

### ✅ 数据流程图

```
输入数据源:
├─ 4模型zarr特征 (rgb_dataset/features_*_zarr/)
│   ├─ features_croco_encoder_dict_unified_zarr/
│   ├─ features_vggt_encoder_dict_unified_zarr/
│   ├─ features_dinov3_encoder_dict_unified_zarr/
│   └─ features_da3_encoder_dict_unified_zarr/
│
└─ Trajectory数据 (raw_data/)
    └─ <task>/demo_randomized/_traj_data/episode*.pkl

处理流程:
1. DPRGB4ModelDataset读取zarr → obs[To, 4, 2048]
2. DPRGB4ModelDataset解析pkl → action[Ta, A]
3. RGB2PCAlignedEncoder(冻结) → obs_feat[To, 1280]
4. DiffusionRGBHead → 预测action
```

### 🎯 数据格式说明

#### Zarr特征格式
```
features_croco_encoder_dict_unified_zarr/
└── beat_block_hammer-demo_randomized-20_sapien_head_camera/
    ├── episode_0.zarr/
    │   ├── per_frame_features  # zarr array [W, T=8, Hf, Wf, C=1024]
    │   ├── frame_paths.json
    │   └── meta.json
    ├── episode_1.zarr/
    └── ...
```

#### Trajectory格式
```python
# episode0.pkl内容:
{
    'left_joint_path': [
        {
            'status': 'success',
            'position': np.array([N, 6], dtype=float32),  # 6D关节角度
            'velocity': np.array([N, 6], dtype=float32),
        },
        ...  # 5个waypoints
    ],
    'right_joint_path': [...]  # 同左臂
}
```

### 💻 使用你的数据训练

```bash
# 1. 测试数据pipeline
python tools/test_head_training_pipeline.py \
  --task beat_block_hammer-demo_randomized-20_sapien_head_camera \
  --traj_root /home/gl/features_model/raw_data \
  --encoder_ckpt outputs/train_rgb2pc_runs/run_best_bs32/ckpt_step_0002000.pt

# 2. 运行训练 (单臂6D)
python tools/train_dp_rgb_single_task_4models.py \
  --task beat_block_hammer-demo_randomized-20_sapien_head_camera \
  --traj_root /home/gl/features_model/raw_data \
  --encoder_ckpt outputs/train_rgb2pc_runs/run_best_bs32/ckpt_step_0002000.pt \
  --use_left_arm \
  --epochs 50 \
  --batch_size 16 \
  --save_dir outputs/dp_rgb_runs/beat_block_hammer_6d

# 3. 训练双臂12D
python tools/train_dp_rgb_single_task_4models.py \
  --task beat_block_hammer-demo_randomized-20_sapien_head_camera \
  --encoder_ckpt outputs/train_rgb2pc_runs/run_best_bs32/ckpt_step_0002000.pt \
  --use_left_arm --use_right_arm --fuse_arms \
  --epochs 50 \
  --save_dir outputs/dp_rgb_runs/beat_block_hammer_12d

# 4. 训练双臂+夹爪14D
python tools/train_dp_rgb_single_task_4models.py \
  --task beat_block_hammer-demo_randomized-20_sapien_head_camera \
  --encoder_ckpt outputs/train_rgb2pc_runs/run_best_bs32/ckpt_step_0002000.pt \
  --use_left_arm --use_right_arm --fuse_arms --include_gripper \
  --epochs 50 \
  --save_dir outputs/dp_rgb_runs/beat_block_hammer_14d
```

### 📁 数据处理逻辑 (已实现)

位置: `features_common/dp_rgb_dataset_4models.py`

```python
class DPRGB4ModelDataset:
    def __getitem__(self, idx):
        # 1. 从4个zarr读取特征
        packs = [load_zarr_pack(root/task/episode.zarr) for root in roots_4]
        
        # 2. 取[Wi, Ti]帧特征,每个模型平均池化到一个向量
        for pack in packs:
            f = pack.get_frame(wi, ti)  # [Hf, Wf, C]
            f = f.reshape(-1, C).mean(axis=0)  # [C]
        
        # 3. 堆叠4个模型 → obs[To, 4, 2048]
        obs = stack_frames_across_models()
        
        # 4. 从pkl解析action
        traj = pickle.load(traj_pkl)
        left_path = traj['left_joint_path']
        action = parse_joint_path(left_path)  # [T, 6]
        
        # 5. 切片action窗口 → action[Ta, A]
        action = action[start:start+horizon]
        
        return obs, action
```

---

## 📊 问题2: 离线推理阶段

### ✅ 是的!离线推理使用相同的数据处理流程

```
训练阶段:
  zarr特征 + pkl轨迹 → Dataset → Encoder+Head → 预测action → 计算loss

离线推理阶段:
  zarr特征 (no pkl) → Dataset → Encoder+Head → 预测action → 保存/可视化
```

### 💻 离线推理使用方法

```bash
# 方法1: 使用infer_dp_rgb_4models.py
python tools/infer_dp_rgb_4models.py \
  --ckpt outputs/dp_rgb_runs/beat_block_hammer_6d/final_head.pt \
  --task beat_block_hammer-demo_randomized-20_sapien_head_camera \
  --episode episode_0 \
  --exec_steps 50
```

### 🔍 判断推理是否成功

#### 方法1: 检查输出维度和数值
```python
# infer_dp_rgb_4models.py 会打印:
Predicted actions shape: [T, A]  # T是执行步数, A是动作维度
Action range: [min, max]
```

#### 方法2: 可视化action轨迹
```bash
# 保存预测的action并与ground truth比较
python - << 'PY'
import torch
import matplotlib.pyplot as plt

# 加载预测
pred = torch.load('outputs/predicted_actions.pt')
# 加载真实action
import pickle
with open('raw_data/.../episode_0.pkl', 'rb') as f:
    gt = pickle.load(f)['left_joint_path']

# 绘制对比图
plt.plot(pred[:, 0], label='pred joint 0')
plt.plot(gt[0]['position'][:, 0], label='gt joint 0')
plt.legend()
plt.savefig('action_comparison.png')
PY
```

#### 方法3: 在RoBoTwin中执行 (问题3会详细讲)

---

## 📊 问题3: 在线推理 - 集成到RoBoTwin

### ⚠️ 重要区别

```
离线推理: 使用预提取的zarr特征
在线推理: 从环境实时获取RGB → 提取特征 → 预测action
```

### 🔧 在RoBoTwin中运行的步骤

#### 步骤1: 修改RoBoTwin的policy接口

在 `RoBoTwin/policy/` 下创建你的policy类:

```python
# RoBoTwin/policy/DP_RGB_4Models/dp_rgb_4models_policy.py
import torch
import sys
from pathlib import Path

# 添加你的项目路径
FEATURES_MODEL_ROOT = Path('/home/gl/features_model')
sys.path.insert(0, str(FEATURES_MODEL_ROOT))

from features_common.rgb2pc_aligned_encoder_4models import RGB2PCAlignedEncoder4Models
from features_common.dp_rgb_policy_multitask import DiffusionRGBHead

class DPRGB4ModelsPolicy:
    def __init__(self, ckpt_path, device='cuda'):
        # 加载checkpoint
        ckpt = torch.load(ckpt_path, map_location='cpu')
        
        # 加载encoder
        encoder_ckpt = ckpt['encoder_ckpt']
        self.encoder = RGB2PCAlignedEncoder4Models.from_checkpoint(
            encoder_ckpt, freeze=True
        ).to(device).eval()
        
        # 加载head
        self.head = DiffusionRGBHead(...)
        self.head.load_state_dict(ckpt['head_state'])
        self.head.to(device).eval()
        
        # 加载normalizer
        self.normalizer = ckpt['normalizer']
        
        self.device = device
        self.obs_buffer = deque(maxlen=2)  # n_obs_steps=2
    
    def reset(self):
        self.obs_buffer.clear()
    
    def predict(self, obs_dict):
        """
        Args:
            obs_dict: {
                'head_camera': np.array([H, W, 3]),  # RGB图像
                # 或者直接提供特征:
                'features_4models': np.array([4, C]),  # 如果已经提取
            }
        Returns:
            action: np.array([A])
        """
        # 获取特征 (这里需要实现4模型的特征提取)
        if 'features_4models' in obs_dict:
            feat = obs_dict['features_4models']
        else:
            # 在线提取特征 (需要加载4个backbone,显存需求大)
            feat = self._extract_features_online(obs_dict['head_camera'])
        
        # 添加到buffer
        self.obs_buffer.append(feat)
        
        # 如果buffer不够,填充
        while len(self.obs_buffer) < 2:
            self.obs_buffer.append(feat)
        
        # 构建obs tensor [To, 4, C]
        obs_seq = np.stack(list(self.obs_buffer), axis=0)
        obs_t = torch.from_numpy(obs_seq).unsqueeze(0).float().to(self.device)
        
        # Encoder
        with torch.no_grad():
            obs_feat = self.encoder(obs_t)  # [1, To, 1280]
            
            # Head预测
            action_pred = self.head.predict_action({'obs': obs_feat})
            action = action_pred['action'][0, 0].cpu().numpy()  # 取第一步
        
        # Denormalize
        action = self.normalizer['action'].unnormalize(
            torch.from_numpy(action).unsqueeze(0)
        ).squeeze(0).numpy()
        
        return action

def get_model(args):
    """RoBoTwin调用的接口"""
    return DPRGB4ModelsPolicy(
        ckpt_path=args['ckpt_path'],
        device='cuda'
    )
```

#### 步骤2: 修改RoBoTwin的eval配置

```yaml
# RoBoTwin/task_config/beat_block_hammer_dp_rgb.yml
policy_name: "DP_RGB_4Models"
ckpt_path: "/home/gl/features_model/outputs/dp_rgb_runs/beat_block_hammer_6d/final_head.pt"
data_type:
  use_rgbd_pointcloud: false  # 不需要点云
  use_rgb_features: true      # 使用RGB特征
```

#### 步骤3: 运行评估

```bash
cd /home/gl/features_model/RoBoTwin

python script/eval_policy.py \
  --task_name beat_block_hammer \
  --task_config beat_block_hammer_dp_rgb \
  --policy_name DP_RGB_4Models \
  --ckpt_setting final_head \
  --seed 0 \
  --eval_test_num 10
```

### 🚀 简化方案: 先用离线特征验证

```python
# 在RoBoTwin中使用预提取的zarr特征(不需要在线提取)
class DPRGB4ModelsPolicyOffline:
    def __init__(self, ckpt_path, zarr_roots, task, episode):
        # 加载模型
        ...
        
        # 加载zarr特征
        self.features_cache = {}
        for i, root in enumerate(zarr_roots):
            pack = load_zarr_pack(root / task / f"{episode}.zarr")
            self.features_cache[i] = pack
        
        self.step_idx = 0
    
    def predict(self, obs_dict):
        # 直接从zarr读取特征
        wi = self.step_idx // 8
        ti = self.step_idx % 8
        
        feat_4 = []
        for i in range(4):
            f = self.features_cache[i].get_frame(wi, ti)
            f = f.reshape(-1, f.shape[-1]).mean(axis=0)
            feat_4.append(f)
        
        feat_4 = np.stack(feat_4, axis=0)  # [4, C]
        
        # 添加到buffer并预测
        ...
        
        self.step_idx += 1
        return action
```

---

## 📊 问题4: 冗余文件扫描

让我扫描tools/下的文件...