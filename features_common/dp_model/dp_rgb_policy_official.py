#!/usr/bin/env python3
"""
正版DP接入方案 - 使用diffusion_policy官方实现
"""
import torch
import torch.nn as nn
from pathlib import Path
import sys

# 添加正版DP路径
DP_ROOT = Path("/home/gl/features_model/DP/diffusion_policy")
sys.path.insert(0, str(DP_ROOT))

from diffusion_policy.policy.diffusion_unet_lowdim_policy import DiffusionUnetLowdimPolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D


class DPRGBPolicy(nn.Module):
    """
    正版Diffusion Policy for RGB特征
    
    输入: RGB特征序列 [B, To, D]
    输出: 动作序列 [B, Ta, A]
    """
    
    def __init__(
        self,
        obs_dim: int,           # 观测维度: n_obs_steps * fuse_dim (e.g., 2*1280=2560)
        action_dim: int,        # 动作维度: 7 for single arm, 14 for dual
        horizon: int = 8,       # 预测horizon
        n_obs_steps: int = 2,   # 观测步数
        n_action_steps: int = 8,  # 执行动作步数
        num_inference_steps: int = 100,  # 扩散步数
        device: str = 'cuda',
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.num_inference_steps = num_inference_steps
        
        # 观测编码器（将扁平化的obs映射到latent）
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        
        # Diffusion UNet (正版DP核心)
        self.noise_pred_net = ConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=256,  # 来自obs_encoder
            diffusion_step_embed_dim=128,
            down_dims=[256, 512, 1024],
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
        )
        
        # 噪声调度器
        from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=100,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon'
        )
        
    def forward(self, obs: torch.Tensor, train: bool = True) -> torch.Tensor:
        """
        训练/推理forward
        
        Args:
            obs: [B, To, D] - RGB特征序列
            train: 是否训练模式
            
        Returns:
            action: [B, Ta, A] - 预测动作
        """
        B = obs.shape[0]
        device = obs.device
        
        # 1. 编码观测
        obs_flat = obs.reshape(B, -1)  # [B, To*D]
        obs_cond = self.obs_encoder(obs_flat)  # [B, 256]
        
        if train:
            # 训练模式：预测噪声
            # 需要GT动作来训练，从外部传入
            raise NotImplementedError("Use compute_loss() for training")
        else:
            # 推理模式：从噪声逐步去噪
            return self._inference(obs_cond)
    
    def compute_loss(
        self, 
        obs: torch.Tensor, 
        action_gt: torch.Tensor
    ) -> torch.Tensor:
        """
        计算扩散损失
        
        Args:
            obs: [B, To, D]
            action_gt: [B, Ta, A]
            
        Returns:
            loss: scalar
        """
        B = obs.shape[0]
        device = obs.device
        
        # 编码观测
        obs_flat = obs.reshape(B, -1)
        obs_cond = self.obs_encoder(obs_flat)  # [B, 256]
        
        # 随机采样时间步
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (B,), device=device
        ).long()
        
        # 添加噪声到GT动作
        noise = torch.randn(action_gt.shape, device=device)
        noisy_actions = self.noise_scheduler.add_noise(
            action_gt, noise, timesteps
        )
        
        # 预测噪声
        noise_pred = self.noise_pred_net(
            noisy_actions,
            timesteps,
            global_cond=obs_cond
        )
        
        # MSE损失
        loss = nn.functional.mse_loss(noise_pred, noise)
        return loss
    
    def _inference(self, obs_cond: torch.Tensor) -> torch.Tensor:
        """
        推理：从随机噪声逐步去噪得到动作
        
        Args:
            obs_cond: [B, 256]
            
        Returns:
            action: [B, Ta, A]
        """
        B = obs_cond.shape[0]
        device = obs_cond.device
        
        # 初始化随机噪声
        action = torch.randn(
            (B, self.n_action_steps, self.action_dim),
            device=device
        )
        
        # 设置推理步数
        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        
        # 逐步去噪
        for t in self.noise_scheduler.timesteps:
            # 预测噪声
            noise_pred = self.noise_pred_net(
                action,
                t.unsqueeze(0).expand(B).to(device),
                global_cond=obs_cond
            )
            
            # 去噪一步
            action = self.noise_scheduler.step(
                noise_pred, t, action
            ).prev_sample
        
        return action


# 使用示例
if __name__ == "__main__":
    # 初始化
    policy = DPRGBPolicy(
        obs_dim=2 * 1280,  # n_obs_steps=2, fuse_dim=1280
        action_dim=7,      # 单臂6自由度+1夹爪
        horizon=8,
        n_obs_steps=2,
        n_action_steps=8,
    ).cuda()
    
    # 模拟输入
    obs = torch.randn(4, 2, 1280).cuda()  # [B=4, To=2, D=1280]
    action_gt = torch.randn(4, 8, 7).cuda()  # [B=4, Ta=8, A=7]
    
    # 训练
    loss = policy.compute_loss(obs, action_gt)
    print(f"Training loss: {loss.item():.4f}")
    
    # 推理
    policy.eval()
    with torch.no_grad():
        obs_cond = policy.obs_encoder(obs.reshape(4, -1))
        action_pred = policy._inference(obs_cond)
        print(f"Predicted action shape: {action_pred.shape}")
