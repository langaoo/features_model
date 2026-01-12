#!/usr/bin/env python3
"""
RoBoTwin仿真环境完整推理脚本
支持已训练DP模型在RoBoTwin中的推理部署

依赖项:
- sapien: RoBoTwin仿真引擎核心依赖
  安装: pip install sapien
  或参考: https://sapien.ucsd.edu/docs/latest/tutorial/installation.html
"""
import sys
import os
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import argparse
import yaml
# 保存原始工作目录
ORIGINAL_CWD = Path.cwd()

# 添加路径
sys.path.insert(0, str(Path(__file__).parent.parent))
robotwin_path = Path(__file__).parent.parent / "RoBoTwin"
if robotwin_path.exists():
    sys.path.insert(0, str(robotwin_path))
else:
    print(f"[WARNING] RoBoTwin路径不存在: {robotwin_path}")

from features_common.rgb2pc_aligned_encoder_4models import RGB2PCAlignedEncoder4Models
from features_common.multi_gpu_extractors import MultiGPUFeatureExtractors

# 检查sapien依赖
try:
    import sapien
    HAS_SAPIEN = True
except ImportError:
    HAS_SAPIEN = False
    print("[WARNING] sapien未安装，RoBoTwin推理功能将不可用")
    print("         安装方法: pip install sapien")


def load_embodiment_config(embodiment_name: str = "aloha-agilex"):
    """加载机器人embodiment配置"""
    robotwin_dir = Path(__file__).parent.parent / "RoBoTwin"
    
    # 读取embodiment配置路径
    embodiment_config_path = robotwin_dir / "task_config" / "_embodiment_config.yml"
    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        embodiment_types = yaml.load(f, Loader=yaml.FullLoader)
    
    if embodiment_name not in embodiment_types:
        raise ValueError(f"Embodiment {embodiment_name} 不存在. 可用: {list(embodiment_types.keys())}")
    
    robot_file_path = embodiment_types[embodiment_name]["file_path"]
    robot_full_path = robotwin_dir / robot_file_path.lstrip("./")
    
    # 读取robot配置
    robot_config_file = robot_full_path / "config.yml"
    with open(robot_config_file, "r", encoding="utf-8") as f:
        robot_config = yaml.load(f, Loader=yaml.FullLoader)
    
    return {
        "robot_file": str(robot_full_path),
        "config": robot_config
    }


def load_camera_config(camera_type: str = "D435"):
    """加载相机配置"""
    robotwin_dir = Path(__file__).parent.parent / "RoBoTwin"
    camera_config_path = robotwin_dir / "task_config" / "_camera_config.yml"
    
    with open(camera_config_path, "r", encoding="utf-8") as f:
        camera_configs = yaml.load(f, Loader=yaml.FullLoader)
    
    if camera_type not in camera_configs:
        raise ValueError(f"Camera {camera_type} 不存在")
    
    return camera_configs[camera_type]


class RoBoTwinInferenceEngine:
    """RoBoTwin推理引擎"""
    
    def __init__(
        self,
        rgb2pc_ckpt: str,
        head_ckpt: str,
        gpu_ids: list = [0, 1, 2, 3],
        device: str = 'cuda:0',
    ):
        self.device = device
        print("="*60)
        print("初始化RoBoTwin推理引擎")
        print("="*60)
        
        # ★ 直接使用传入的绝对路径 (已在run_robotwin_inference中转换)
        # 不在这里调用Path.resolve()，避免相对于RoBoTwin目录解析
        
        # 1. 加载4个特征提取器
        print("\n[1/3] 加载特征提取器...")
        self.extractors = MultiGPUFeatureExtractors(gpu_ids=gpu_ids)
        print("✓ 特征提取器就绪")
        
        # 2. 加载RGB2PC对齐编码器
        print("\n[2/3] 加载RGB2PC对齐编码器...")
        self.encoder = RGB2PCAlignedEncoder4Models.from_checkpoint(
            rgb2pc_ckpt,
            freeze=True
        )
        self.encoder = self.encoder.to(device).eval()
        self.fuse_dim = self.encoder.spec.fuse_dim
        print(f"✓ 对齐编码器加载: {rgb2pc_ckpt}")
        print(f"  fuse_dim={self.fuse_dim}")
        
        # 3. 加载DP动作头
        print("\n[3/3] 加载DP动作头...")
        head_state = torch.load(head_ckpt, map_location='cpu')
        
        # 从checkpoint解析配置
        if 'config' in head_state:
            config = head_state['config']
            self.n_obs_steps = config['data']['n_obs_steps']
            self.horizon = config['data']['horizon']
            action_dim = self._compute_action_dim(config)
            obs_dim = self.n_obs_steps * self.fuse_dim
            hidden_dim = config['policy'].get('hidden_dim', 512)
        else:
            # 尝试从policy state推断
            print("[WARNING] checkpoint无config，尝试从state_dict推断...")
            policy_state = head_state.get('policy', head_state)
            
            # 从第一层推断obs_dim
            first_weight = policy_state['net.0.weight']
            obs_dim = first_weight.shape[1]
            self.n_obs_steps = obs_dim // self.fuse_dim
            
            # 从最后一层推断action_dim和horizon
            last_weight = policy_state['net.4.weight']
            output_dim = last_weight.shape[0]
            self.horizon = 8  # 默认
            action_dim = output_dim // self.horizon
            hidden_dim = 512
            
            print(f"  推断: obs_dim={obs_dim}, action_dim={action_dim}")
        
        # 构建SimpleDPHead
        import torch.nn as nn
        class SimpleDPHead(nn.Module):
            def __init__(self, obs_dim, action_dim, horizon, hidden_dim=512):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(obs_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim // 2, action_dim * horizon),
                )
                self.action_dim = action_dim
                self.horizon = horizon
            
            def forward(self, obs):
                B = obs.shape[0]
                obs_flat = obs.reshape(B, -1)
                out = self.net(obs_flat)
                return out.reshape(B, self.horizon, self.action_dim)
        
        self.policy = SimpleDPHead(obs_dim, action_dim, self.horizon, hidden_dim)
        
        # 加载权重
        policy_state = head_state.get('policy', head_state)
        self.policy.load_state_dict(policy_state)
        self.policy = self.policy.to(device).eval()
        
        self.action_dim = action_dim
        
        print(f"✓ DP动作头加载: {head_ckpt}")
        print(f"  n_obs_steps={self.n_obs_steps}, horizon={self.horizon}, action_dim={action_dim}")
        print("\n" + "="*60)
        print("推理引擎初始化完成！")
        print("="*60 + "\n")
        
        # 观测缓冲区
        self.obs_buffer = []
    
    def _compute_action_dim(self, config):
        """从配置计算动作维度"""
        action_dim = 0
        if config['data']['use_left_arm']:
            action_dim += 6
            if config['data']['include_gripper']:
                action_dim += 1
        if config['data']['use_right_arm']:
            action_dim += 6
            if config['data']['include_gripper']:
                action_dim += 1
        return action_dim
    
    def reset(self):
        """重置推理引擎（新episode开始时调用）"""
        self.obs_buffer = []
    
    def predict(self, image) -> np.ndarray:
        """
        单步推理
        
        Args:
            image: RGB图像 (PIL.Image 或 numpy array [H,W,3])
        
        Returns:
            action: [action_dim] 当前步动作
        """
        # 转换图像格式
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype(np.uint8))
        
        # 添加到观测缓冲
        self.obs_buffer.append(image)
        
        # 维持n_obs_steps窗口
        if len(self.obs_buffer) > self.n_obs_steps:
            self.obs_buffer = self.obs_buffer[-self.n_obs_steps:]
        
        # 填充不足的帧
        while len(self.obs_buffer) < self.n_obs_steps:
            self.obs_buffer.insert(0, self.obs_buffer[0] if self.obs_buffer else image)
        
        # 提取特征
        obs_features = []
        for img in self.obs_buffer:
            feat_4 = self.extractors(img)  # [4, 2048]
            obs_features.append(feat_4)
        
        # [n_obs_steps, 4, 2048] -> [1, n_obs_steps, 4, 2048]
        obs = np.stack(obs_features, axis=0)
        obs = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        
        # 对齐编码
        with torch.no_grad():
            obs_encoded = self.encoder(obs)  # [1, n_obs_steps, fuse_dim]
        
        # 预测动作序列
        with torch.no_grad():
            actions = self.policy(obs_encoded)  # [1, horizon, action_dim]
        
        # 返回第一步动作
        action = actions[0, 0].cpu().numpy()  # [action_dim]
        return action


def run_robotwin_inference(
    task_name: str = 'beat_block_hammer',
    rgb2pc_ckpt: str = None,
    head_ckpt: str = None,
    max_steps: int = 1000,
    num_episodes: int = 10,
    gpu_ids: list = [0, 1, 2, 3],
    render: bool = False,
):
    """在RoBoTwin中运行推理"""
    
    # ★ 关键: 在切换工作目录前转换为绝对路径
    rgb2pc_ckpt = str(Path(rgb2pc_ckpt).resolve())
    head_ckpt = str(Path(head_ckpt).resolve())
    
    # 检查sapien依赖
    if not HAS_SAPIEN:
        print("="*60)
        print("错误: sapien未安装")
        print("="*60)
        print("RoBoTwin依赖sapien仿真引擎")
        print("\n安装方法:")
        print("  pip install sapien")
        print("  或参考: https://sapien.ucsd.edu/docs/latest/tutorial/installation.html")
        print("="*60)
        return
    
    # 切换到RoBoTwin目录（环境需要从该目录加载资源）
    robotwin_dir = Path(__file__).parent.parent / "RoBoTwin"
    if not robotwin_dir.exists():
        print(f"错误: RoBoTwin目录不存在: {robotwin_dir}")
        return
    
    os.chdir(robotwin_dir)
    print(f"✓ 切换工作目录到: {robotwin_dir}")
    
    # 导入RoBoTwin环境
    try:
        from envs import beat_block_hammer
        print("✓ RoBoTwin环境导入成功")
    except ImportError as e:
        print("="*60)
        print(f"错误: 无法导入RoBoTwin环境")
        print("="*60)
        print(f"详细信息: {e}")
        print("\n请确保:")
        print("  1. RoBoTwin目录存在于项目根目录")
        print("  2. envs模块路径正确")
        print(f"  3. 当前sys.path包含: {robotwin_path}")
        print("="*60)
        os.chdir(ORIGINAL_CWD)  # 恢复工作目录
        return
    
    # 初始化推理引擎 (checkpoint路径已经是绝对路径)
    inference = RoBoTwinInferenceEngine(
        rgb2pc_ckpt=rgb2pc_ckpt,
        head_ckpt=head_ckpt,
        gpu_ids=gpu_ids,
    )
    
    print("\n" + "="*60)
    print(f"开始推理: 任务={task_name}, episodes={num_episodes}")
    print("="*60 + "\n")
    
    success_count = 0
    
    # 加载embodiment配置
    embodiment_name = "aloha-agilex"  # 默认机器人
    left_embodiment = load_embodiment_config(embodiment_name)
    right_embodiment = load_embodiment_config(embodiment_name)
    
    # 加载相机配置
    camera_config = load_camera_config("D435")
    
    for ep in range(num_episodes):
        print(f"\n[Episode {ep+1}/{num_episodes}]")
        
        # 创建环境
        env = beat_block_hammer.beat_block_hammer()
        
        # 设置环境参数
        setup_kwargs = {
            'headless': (not render),
            'use_ray_tracing': False,
            'task_name': task_name,
            'now_ep_num': ep,
            'seed': ep,
            'domain_randomization': {
                'random_background': False,
                'cluttered_table': False,
                'clean_background_rate': 1.0,
                'random_head_camera_dis': 0,
                'random_table_height': 0,
                'random_light': False,
                'crazy_random_light_rate': 0,
                'random_embodiment': False,
            },
            'left_robot_file': left_embodiment['robot_file'],
            'right_robot_file': right_embodiment['robot_file'],
            'left_embodiment_config': left_embodiment['config'],
            'right_embodiment_config': right_embodiment['config'],
            'dual_arm_embodied': True,
            'head_camera_h': camera_config['h'],
            'head_camera_w': camera_config['w'],
        }
        env.setup_demo(**setup_kwargs)
        
        # 重置
        obs = env.reset()
        inference.reset()
        
        # 获取初始图像
        if 'head_camera' in obs['observation']:
            rgb = obs['observation']['head_camera']['rgb']
        else:
            # 尝试其他相机
            cam_keys = [k for k in obs['observation'].keys() if 'camera' in k]
            if cam_keys:
                rgb = obs['observation'][cam_keys[0]]['rgb']
            else:
                print("✗ 无法找到相机图像")
                continue
        
        # Rollout
        done = False
        step = 0
        rewards = []
        
        while not done and step < max_steps:
            # 推理动作
            action = inference.predict(rgb)
            
            # 执行动作
            # RoBoTwin的action格式需要确认
            # 通常是字典格式: {'left_arm': [...], 'right_arm': [...]}
            # 需要根据实际情况转换
            
            # 假设action是展平的 [left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)]
            if inference.action_dim == 12:
                # 双臂无夹爪
                action_dict = {
                    'left_arm': action[:6],
                    'right_arm': action[6:12],
                }
            elif inference.action_dim == 14:
                # 双臂有夹爪
                action_dict = {
                    'left_arm': action[:6],
                    'left_gripper': action[6:7],
                    'right_arm': action[7:13],
                    'right_gripper': action[13:14],
                }
            else:
                action_dict = {'arm': action}
            
            # 执行
            obs, reward, done, info = env.step(action_dict)
            
            # 更新图像
            if 'head_camera' in obs['observation']:
                rgb = obs['observation']['head_camera']['rgb']
            
            rewards.append(reward)
            step += 1
            
            if step % 50 == 0:
                print(f"  Step {step}: avg_reward={np.mean(rewards):.3f}")
        
        # Episode结束
        success = info.get('success', False) if isinstance(info, dict) else done
        if success:
            success_count += 1
        
        print(f"  完成: steps={step}, total_reward={sum(rewards):.2f}, success={success}")
        
        # 关闭环境
        env.close()
    
    print("\n" + "="*60)
    print(f"推理完成: 成功率={success_count}/{num_episodes} ({100*success_count/num_episodes:.1f}%)")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="RoBoTwin推理部署")
    parser.add_argument('--config', type=str, default='configs/robotwin_inference.yaml',
                       help='配置文件路径')
    # 保留命令行参数以便快速覆盖
    parser.add_argument('--task', type=str, default=None,
                       help='任务名称 (覆盖配置文件)')
    parser.add_argument('--rgb2pc_ckpt', type=str, default=None,
                       help='RGB2PC checkpoint (覆盖配置文件)')
    parser.add_argument('--head_ckpt', type=str, default=None,
                       help='DP头checkpoint (覆盖配置文件)')
    parser.add_argument('--max_steps', type=int, default=None,
                       help='最大步数 (覆盖配置文件)')
    parser.add_argument('--num_episodes', type=int, default=None,
                       help='Episode数量 (覆盖配置文件)')
    parser.add_argument('--render', action='store_true',
                       help='是否渲染 (覆盖配置文件)')
    args = parser.parse_args()
    
    # 加载配置文件
    import yaml
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"配置文件不存在: {config_path}")
        print("使用命令行参数...")
        config = {
            'checkpoints': {
                'rgb2pc': args.rgb2pc_ckpt or 'outputs/train_rgb2pc_runs/run_best_bs32/ckpt_step_0010000.pt',
                'head': args.head_ckpt or '_runs/online_training/beat_block_hammer_run/ckpt_epoch_0050.pt',
            },
            'task': {
                'name': args.task or 'beat_block_hammer',
                'num_episodes': args.num_episodes or 10,
                'max_steps': args.max_steps or 1000,
            },
            'device': {
                'gpu_ids': [0, 1, 2, 3],
            },
            'render': {
                'enabled': args.render,
            }
        }
    else:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 命令行参数覆盖配置文件
        if args.task is not None:
            config['task']['name'] = args.task
        if args.rgb2pc_ckpt is not None:
            config['checkpoints']['rgb2pc'] = args.rgb2pc_ckpt
        if args.head_ckpt is not None:
            config['checkpoints']['head'] = args.head_ckpt
        if args.max_steps is not None:
            config['task']['max_steps'] = args.max_steps
        if args.num_episodes is not None:
            config['task']['num_episodes'] = args.num_episodes
        if args.render:
            config['render']['enabled'] = True
    
    # 打印配置
    print("="*60)
    print("推理配置")
    print("="*60)
    print(f"任务: {config['task']['name']}")
    print(f"Episodes: {config['task']['num_episodes']}")
    print(f"最大步数: {config['task']['max_steps']}")
    print(f"RGB2PC: {config['checkpoints']['rgb2pc']}")
    print(f"Head: {config['checkpoints']['head']}")
    print(f"GPU: {config['device']['gpu_ids']}")
    print(f"渲染: {config['render']['enabled']}")
    print("="*60 + "\n")
    
    run_robotwin_inference(
        task_name=config['task']['name'],
        rgb2pc_ckpt=config['checkpoints']['rgb2pc'],
        head_ckpt=config['checkpoints']['head'],
        max_steps=config['task']['max_steps'],
        num_episodes=config['task']['num_episodes'],
        gpu_ids=config['device']['gpu_ids'],
        render=config['render']['enabled'],
    )


if __name__ == '__main__':
    main()
