#!/usr/bin/env python3
"""
RoBoTwin原生Policy适配器
将训练好的模型(4特征模型 + RGB2PC + DP头)封装为RoBoTwin标准Policy接口
"""
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from PIL import Image
import sys

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from features_common.multi_gpu_extractors import MultiGPUFeatureExtractors
from features_common.rgb2pc_aligned_encoder_4models import RGB2PCAlignedEncoder4Models

# 导入正版DP
try:
    DP_OUTER = Path(__file__).parent / "DP" / "diffusion_policy"
    if DP_OUTER.exists():
        sys.path.insert(0, str(DP_OUTER))
        from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
        from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
        HAS_OFFICIAL_DP = True
    else:
        HAS_OFFICIAL_DP = False
except:
    HAS_OFFICIAL_DP = False


class RoBoTwinDPPolicy(nn.Module):
    """
    RoBoTwin原生Policy - 端到端推理
    
    输入: RoBoTwin env.reset()/env.step()返回的RGB观测
    输出: 机械臂动作 [left_arm_6dof + right_arm_6dof]
    """
    
    def __init__(
        self,
        rgb2pc_ckpt: str,
        head_ckpt: str,
        gpu_ids: list = [0, 1],
        device: str = 'cuda:0',
    ):
        super().__init__()
        
        self.device = device
        self.gpu_ids = gpu_ids
        
        print("="*60)
        print("初始化RoBoTwin Policy")
        print("="*60)
        
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
        print(f"✓ 对齐编码器: {rgb2pc_ckpt}")
        print(f"  fuse_dim={self.fuse_dim}")
        
        # 3. 加载DP动作头
        print("\n[3/3] 加载DP动作头...")
        head_state = torch.load(head_ckpt, map_location='cpu')
        
        # 推断模型类型和配置
        if 'policy' in head_state:
            policy_state = head_state['policy']
        else:
            policy_state = head_state
        
        # 从checkpoint推断参数
        obs_dim = None
        action_dim = None
        horizon = None
        
        # 尝试从config获取
        if 'config' in head_state:
            cfg = head_state['config']
            if 'data' in cfg:
                n_obs_steps = cfg['data'].get('n_obs_steps', 2)
                horizon = cfg['data'].get('horizon', 8)
                obs_dim = n_obs_steps * self.fuse_dim
                
                # 计算action_dim
                action_dim = 0
                if cfg['data'].get('use_left_arm', True):
                    action_dim += 6
                if cfg['data'].get('use_right_arm', True):
                    action_dim += 6
                if cfg['data'].get('include_gripper', False):
                    if cfg['data'].get('use_left_arm', True):
                        action_dim += 1
                    if cfg['data'].get('use_right_arm', True):
                        action_dim += 1
        
        # Fallback: 从state_dict推断
        if obs_dim is None:
            # 查找obs_encoder.0.weight的输入维度
            if 'obs_encoder.0.weight' in policy_state:
                obs_dim = policy_state['obs_encoder.0.weight'].shape[1]
                n_obs_steps = obs_dim // self.fuse_dim
            else:
                # SimpleDPHead
                if 'net.0.weight' in policy_state:
                    obs_dim = policy_state['net.0.weight'].shape[1]
                    n_obs_steps = obs_dim // self.fuse_dim
        
        if action_dim is None or horizon is None:
            # 从noise_pred_net或net的输出推断
            if 'net.4.weight' in policy_state:
                # SimpleDPHead
                out_dim = policy_state['net.4.weight'].shape[0]
                # 假设horizon=8, action_dim=12
                horizon = 8
                action_dim = out_dim // horizon
            else:
                # 默认值
                action_dim = 12  # 双臂各6自由度
                horizon = 8
        
        # 检测是否为正版DP
        is_official_dp = 'obs_encoder.0.weight' in policy_state and 'noise_pred_net.time_embed.0.weight' in policy_state
        
        if is_official_dp and HAS_OFFICIAL_DP:
            print("  模型类型: 正版Diffusion Policy")
            
            # 重建正版DP
            class DPRGBPolicy(nn.Module):
                def __init__(self, obs_dim, action_dim, horizon, n_obs_steps, num_inference_steps=100):
                    super().__init__()
                    self.obs_dim = obs_dim
                    self.action_dim = action_dim
                    self.horizon = horizon
                    self.n_obs_steps = n_obs_steps
                    self.num_inference_steps = num_inference_steps
                    
                    self.obs_encoder = nn.Sequential(
                        nn.Linear(obs_dim, 512),
                        nn.ReLU(),
                        nn.Linear(512, 256),
                        nn.ReLU(),
                    )
                    
                    self.noise_pred_net = ConditionalUnet1D(
                        input_dim=action_dim,
                        global_cond_dim=256,
                        diffusion_step_embed_dim=128,
                        down_dims=[256, 512, 1024],
                        kernel_size=5,
                        n_groups=8,
                        cond_predict_scale=True,
                    )
                    
                    self.noise_scheduler = DDPMScheduler(
                        num_train_timesteps=100,
                        beta_schedule='squaredcos_cap_v2',
                        clip_sample=True,
                        prediction_type='epsilon'
                    )
                
                def forward(self, obs):
                    B = obs.shape[0]
                    device = obs.device
                    
                    obs_flat = obs.reshape(B, -1)
                    obs_cond = self.obs_encoder(obs_flat)
                    
                    action = torch.randn((B, self.n_obs_steps, self.action_dim), device=device)
                    self.noise_scheduler.set_timesteps(self.num_inference_steps)
                    
                    for t in self.noise_scheduler.timesteps:
                        noise_pred = self.noise_pred_net(
                            action,
                            t.unsqueeze(0).expand(B).to(device),
                            global_cond=obs_cond
                        )
                        action = self.noise_scheduler.step(noise_pred, t, action).prev_sample
                    
                    return action
            
            self.head = DPRGBPolicy(obs_dim, action_dim, horizon, n_obs_steps)
        else:
            print("  模型类型: SimpleDPHead")
            
            # SimpleDPHead
            class SimpleDPHead(nn.Module):
                def __init__(self, obs_dim, action_dim, horizon):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(obs_dim, 512),
                        nn.ReLU(),
                        nn.Linear(512, 256),
                        nn.ReLU(),
                        nn.Linear(256, action_dim * horizon),
                    )
                    self.action_dim = action_dim
                    self.horizon = horizon
                
                def forward(self, obs):
                    B = obs.shape[0]
                    obs_flat = obs.reshape(B, -1)
                    out = self.net(obs_flat)
                    return out.reshape(B, self.horizon, self.action_dim)
            
            self.head = SimpleDPHead(obs_dim, action_dim, horizon)
        
        # 加载权重
        self.head.load_state_dict(policy_state)
        self.head = self.head.to(device).eval()
        
        self.n_obs_steps = n_obs_steps
        self.horizon = horizon
        self.action_dim = action_dim
        
        print(f"✓ DP动作头: {head_ckpt}")
        print(f"  n_obs_steps={n_obs_steps}, horizon={horizon}, action_dim={action_dim}")
        
        # 观测历史缓冲
        self.obs_history = []
        
        print("\n" + "="*60)
        print("Policy初始化完成！")
        print("="*60 + "\n")
    
    def reset(self):
        """重置观测历史"""
        self.obs_history = []
    
    @torch.no_grad()
    def predict(self, rgb_obs: np.ndarray) -> np.ndarray:
        """
        预测动作 - RoBoTwin标准接口
        
        Args:
            rgb_obs: RGB图像 [H, W, 3] numpy array (uint8, 0-255)
        
        Returns:
            action: 机械臂动作 [12] numpy array (left_6dof + right_6dof)
        """
        # 1. RGB转PIL
        if isinstance(rgb_obs, np.ndarray):
            if rgb_obs.dtype != np.uint8:
                rgb_obs = (rgb_obs * 255).astype(np.uint8)
            img = Image.fromarray(rgb_obs)
        else:
            img = rgb_obs
        
        # 2. 提取特征
        features = self.extractors.extract(img)  # [4, 2048]
        features = torch.from_numpy(features).float().to(self.device)
        
        # 3. 添加到历史
        self.obs_history.append(features)
        if len(self.obs_history) < self.n_obs_steps:
            # 不足n_obs_steps，重复第一帧
            while len(self.obs_history) < self.n_obs_steps:
                self.obs_history.insert(0, features)
        elif len(self.obs_history) > self.n_obs_steps:
            # 超过n_obs_steps，保留最新的
            self.obs_history = self.obs_history[-self.n_obs_steps:]
        
        # 4. 构建观测 [1, n_obs_steps, 4, 2048]
        obs = torch.stack(self.obs_history, dim=0).unsqueeze(0)  # [1, To, 4, 2048]
        
        # 5. 编码
        obs_encoded = self.encoder(obs)  # [1, To, 1280]
        
        # 6. 预测动作
        action_seq = self.head(obs_encoded)  # [1, Ta, A]
        
        # 7. 返回第一步动作
        action = action_seq[0, 0].cpu().numpy()  # [A]
        
        return action
    
    def __call__(self, rgb_obs: np.ndarray) -> np.ndarray:
        """兼容接口"""
        return self.predict(rgb_obs)


def load_robotwin_policy(
    rgb2pc_ckpt: str,
    head_ckpt: str,
    gpu_ids: list = [0, 1],
    device: str = 'cuda:0',
) -> RoBoTwinDPPolicy:
    """
    快速加载RoBoTwin Policy
    
    Args:
        rgb2pc_ckpt: RGB2PC对齐编码器checkpoint路径
        head_ckpt: DP头checkpoint路径
        gpu_ids: GPU列表 (用于4个特征提取器)
        device: Policy主设备
    
    Returns:
        policy: RoBoTwinDPPolicy实例
    """
    return RoBoTwinDPPolicy(
        rgb2pc_ckpt=rgb2pc_ckpt,
        head_ckpt=head_ckpt,
        gpu_ids=gpu_ids,
        device=device,
    )
