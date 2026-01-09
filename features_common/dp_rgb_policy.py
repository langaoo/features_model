"""features_common/dp_rgb_policy.py

基于 RGB 蒸馏特征的 Diffusion Policy。

使用你训练好的 RGB2PC 蒸馏模型作为 observation encoder，
替代原始的图像编码器。
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add DP to path
DP_ROOT = Path(__file__).resolve().parents[1] / "DP" / "diffusion_policy"
if str(DP_ROOT) not in sys.path:
    sys.path.insert(0, str(DP_ROOT))

from typing import Any, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
try:  # pragma: no cover
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler  # type: ignore
except Exception:  # pragma: no cover
    DDPMScheduler = None  # type: ignore


class _SimpleDDPMScheduler:
    """Minimal DDPM scheduler fallback.

    This is intentionally tiny: only what's needed by our policy loss.
    It matches the parts of diffusers' DDPMScheduler API we use:
    - .config.num_train_timesteps
    - .add_noise(sample, noise, timesteps)
    """

    class _Config:
        def __init__(self, num_train_timesteps: int):
            self.num_train_timesteps = num_train_timesteps

    def __init__(self, num_train_timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 2e-2):
        self.config = self._Config(num_train_timesteps=num_train_timesteps)
        betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer = lambda name, t: setattr(self, name, t)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)

    def add_noise(self, original_samples: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        # timesteps: [B]
        a = self.alphas_cumprod[timesteps].view(-1, *([1] * (original_samples.ndim - 1)))
        return torch.sqrt(a) * original_samples + torch.sqrt(1.0 - a) * noise

from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.model.common.normalizer import LinearNormalizer


class RGBFeatureEncoder(nn.Module):
    """
    从预训练的 RGB2PC 蒸馏模型中加载特征提取器。
    
    输入: RGB features [B, To, C_rgb]  (来自 zarr pack)
    输出: Encoded features [B, To, D]
    """
    
    def __init__(
        self,
        ckpt_path: str,
        input_dim: int,
        output_dim: int = 256,
        freeze: bool = False,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # 加载预训练的 adapter + fusion
        print(f"Loading RGB2PC checkpoint from: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location='cpu')
        
        # 提取模型参数
        # 假设 checkpoint 结构: {'model_state': {...}, ...}
        if 'model_state' in ckpt:
            state_dict = ckpt['model_state']
        elif 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt
        
        # 构建一个简单的投影层 (你可以根据实际 ckpt 结构调整)
        # 这里假设你的模型输出是 fuse_dim 维度
        fuse_dim = ckpt.get('config', {}).get('fuse_dim', 1280)
        
        self.projector = nn.Sequential(
            nn.Linear(input_dim, fuse_dim),
            nn.GELU(),
            nn.Linear(fuse_dim, output_dim),
            nn.LayerNorm(output_dim),
        )
        
        # 尝试加载 projector 权重
        projector_keys = [k for k in state_dict.keys() if 'projector' in k or 'proj' in k]
        if projector_keys:
            print(f"Found projector keys in checkpoint: {len(projector_keys)}")
            # 加载部分权重
            new_state = {}
            for k, v in state_dict.items():
                if 'projector' in k or 'proj' in k:
                    new_key = k.replace('module.', '').replace('projector.', '')
                    new_state[new_key] = v
            
            try:
                self.projector.load_state_dict(new_state, strict=False)
                print("Loaded projector weights from checkpoint")
            except Exception as e:
                print(f"Warning: failed to load projector weights: {e}")
        
        if freeze:
            for param in self.parameters():
                param.requires_grad = False
            print("RGBFeatureEncoder frozen")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, To, C_rgb]
        return: [B, To, D]
        """
        return self.projector(x)


class DiffusionRGBPolicy(BaseLowdimPolicy):
    """
    Diffusion Policy 使用 RGB 蒸馏特征作为 observation。
    
    参数:
        obs_dim: RGB 特征维度
        action_dim: 动作维度
        horizon: 预测时域
        n_obs_steps: 观测步数
        n_action_steps: 执行步数
        rgb_ckpt_path: RGB2PC 蒸馏模型 checkpoint 路径
        freeze_encoder: 是否冻结特征编码器
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        horizon: int,
        n_obs_steps: int,
        n_action_steps: int,
        rgb_ckpt_path: Optional[str] = None,
        freeze_encoder: bool = False,
        obs_encoder_dim: int = 256,
        num_inference_steps: Optional[int] = None,
        # Diffusion model params
        diffusion_step_embed_dim: int = 256,
        down_dims: tuple = (256, 512, 1024),
        kernel_size: int = 5,
        n_groups: int = 8,
        cond_predict_scale: bool = True,
        obs_as_global_cond: bool = True,
        # Noise scheduler
        noise_scheduler: Optional[Any] = None,
        **kwargs,
    ):
        super().__init__()
        
        # Feature encoder
        if rgb_ckpt_path is not None:
            self.obs_encoder = RGBFeatureEncoder(
                ckpt_path=rgb_ckpt_path,
                input_dim=obs_dim,
                output_dim=obs_encoder_dim,
                freeze=freeze_encoder,
            )
        else:
            # 简单的 MLP encoder
            self.obs_encoder = nn.Sequential(
                nn.Linear(obs_dim, obs_encoder_dim),
                nn.GELU(),
                nn.LayerNorm(obs_encoder_dim),
            )
        
        # Diffusion model
        obs_feature_dim = obs_encoder_dim
        
        if obs_as_global_cond:
            input_dim = action_dim
            global_cond_dim = obs_feature_dim * n_obs_steps
        else:
            input_dim = action_dim + obs_feature_dim
            global_cond_dim = None
        
        self.model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
        )
        
        if noise_scheduler is None:
            if DDPMScheduler is None:
                noise_scheduler = _SimpleDDPMScheduler(num_train_timesteps=100)
            else:
                noise_scheduler = DDPMScheduler(
                    num_train_timesteps=100,
                    beta_start=0.0001,
                    beta_end=0.02,
                    beta_schedule='squaredcos_cap_v2',
                    variance_type='fixed_small',
                    clip_sample=True,
                    prediction_type='epsilon',
                )
        
        self.noise_scheduler = noise_scheduler
        
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False,
        )
        
        self.normalizer = LinearNormalizer()
        
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs
        
        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps
        
        print(f"DiffusionRGBPolicy:")
        print(f"  Encoder params: {sum(p.numel() for p in self.obs_encoder.parameters()):,}")
        print(f"  Diffusion params: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        推理接口。
        
        obs_dict: {
            'obs': [B, To, C_rgb]
        }
        
        返回: {
            'action': [B, Ta, A],
            'action_pred': [B, horizon, A]
        }
        """
        # Encode observations
        obs = obs_dict['obs']  # [B, To, C_rgb]
        
        # Normalize obs
        nbatch = self.normalizer.normalize({'obs': obs})
        nobs = nbatch['obs']
        
        B = obs.shape[0]
        
        # Forward encoder
        obs_features = self.obs_encoder(nobs)  # [B, To, D]
        
        # Prepare condition
        if self.obs_as_global_cond:
            # Flatten temporal dim
            global_cond = obs_features.reshape(B, -1)  # [B, To*D]
        else:
            global_cond = None
        
        # Sample noise
        shape = (B, self.horizon, self.action_dim)
        cond_data = torch.zeros(shape, device=obs.device, dtype=obs.dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        
        # Run diffusion
        nsample = self.conditional_sample(
            cond_data,
            cond_mask,
            local_cond=None,
            global_cond=global_cond,
        )
        
        # Denormalize
        naction_pred = nsample[..., :self.action_dim]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)
        
        # Extract action steps
        start = self.n_obs_steps - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]
        
        return {
            'action': action,
            'action_pred': action_pred,
        }
    
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        训练损失。
        
        batch: {
            'obs': [B, To, C_rgb],
            'action': [B, Ta, A]
        }
        """
        # Normalize batch
        nbatch = self.normalizer.normalize(batch)
        nobs = nbatch['obs']
        naction = nbatch['action']
        
        B = nobs.shape[0]
        
        # Encode observations
        obs_features = self.obs_encoder(nobs)  # [B, To, D]
        
        # Prepare trajectory
        trajectory = naction
        
        # Pad to horizon
        if trajectory.shape[1] < self.horizon:
            pad = torch.zeros(
                (B, self.horizon - trajectory.shape[1], self.action_dim),
                device=trajectory.device,
                dtype=trajectory.dtype,
            )
            trajectory = torch.cat([trajectory, pad], dim=1)
        
        # Prepare condition
        if self.obs_as_global_cond:
            global_cond = obs_features.reshape(B, -1)  # [B, To*D]
            local_cond = None
            trajectory_cond = trajectory
        else:
            global_cond = None
            # Repeat obs for each time step
            local_cond = obs_features  # [B, To, D]
            # Pad local_cond to horizon
            if local_cond.shape[1] < self.horizon:
                pad = torch.zeros(
                    (B, self.horizon - local_cond.shape[1], self.obs_feature_dim),
                    device=local_cond.device,
                    dtype=local_cond.dtype,
                )
                local_cond = torch.cat([local_cond, pad], dim=1)
            trajectory_cond = torch.cat([trajectory, local_cond], dim=-1)
        
        # Sample noise
        noise = torch.randn_like(trajectory)
        
        # Sample timestep
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (B,),
            device=trajectory.device,
        ).long()
        
        # Add noise
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)
        
        # Predict
        noise_pred = self.model(
            noisy_trajectory,
            timesteps,
            local_cond=local_cond,
            global_cond=global_cond,
        )
        
        # Loss
        loss = F.mse_loss(noise_pred, noise)
        
        return loss
    
    def conditional_sample(
        self,
        condition_data,
        condition_mask,
        local_cond=None,
        global_cond=None,
        generator=None,
        **kwargs,
    ):
        """Conditional sampling (去噪过程)"""
        model = self.model
        scheduler = self.noise_scheduler
        
        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator,
        )
        
        # Set timesteps
        scheduler.set_timesteps(self.num_inference_steps)
        
        for t in scheduler.timesteps:
            # Apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]
            
            # Predict
            model_output = model(
                trajectory,
                t,
                local_cond=local_cond,
                global_cond=global_cond,
            )
            
            # Step
            trajectory = scheduler.step(
                model_output,
                t,
                trajectory,
                generator=generator,
                **kwargs,
            ).prev_sample
        
        # Final conditioning
        trajectory[condition_mask] = condition_data[condition_mask]
        
        return trajectory
    
    def set_normalizer(self, normalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())
