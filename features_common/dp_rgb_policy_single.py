"""features_common/dp_rgb_policy_single.py

单任务 Diffusion Policy Head 实现（从 dp_rgb_policy_multitask.py 拆分出来）。

为什么拆分：
- 原本 dp_rgb_policy_multitask.py 既有单 head 实现（DiffusionRGBHead），又有多任务包装（MultiTaskDiffusionRGBPolicy）。
- 文件名"multitask"容易误导：以为只能多任务用，实际上单任务也在这里。
- 拆分后：
  - 本文件 dp_rgb_policy_single.py：纯粹的单 head 实现（HeadSpec + DiffusionRGBHead）
  - dp_rgb_policy_multitask.py：多任务包装（共享 encoder + 多个 head）

契约（contract）
- 输入：
  - obs_features: FloatTensor[B, To, fuse_dim]（来自冻结的对齐 encoder）
  - action: FloatTensor[B, horizon, action_dim]
- 输出：
  - compute_loss(...) -> scalar（训练时）
  - predict_action(...) -> dict{'action_pred': FloatTensor[B, horizon, action_dim]}（推理时）

设计说明
- obs normalizer：identity（因为 obs_features 是 encoder 输出，已归一化）
- action normalizer：从训练数据拟合 SingleFieldLinearNormalizer

使用场景
- 单任务训练（tools/train_dp_rgb_single_task_4models.py）
- 单任务推理（tools/infer_dp_rgb_4models.py）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any
import os
import sys
from pathlib import Path

# Add DP to path
DP_ROOT = Path(__file__).resolve().parents[1] / "DP" / "diffusion_policy"
if str(DP_ROOT) not in sys.path:
    sys.path.insert(0, str(DP_ROOT))

import torch
import torch.nn as nn

# diffusers fallback
try:
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
except Exception:
    print("[dp_rgb_policy_single] diffusers not available, using minimal DDPM scheduler")
    # minimal scheduler (same as dp_rgb_policy_multitask.py)
    class DDPMScheduler:
        def __init__(self, num_train_timesteps: int = 1000, beta_schedule: str = "squaredcos_cap_v2", clip_sample: bool = True, prediction_type: str = "epsilon"):
            self.num_train_timesteps = num_train_timesteps
            self.timesteps = torch.arange(0, num_train_timesteps)
            betas = torch.linspace(1e-4, 0.02, num_train_timesteps)
            alphas = 1.0 - betas
            self.alphas_cumprod = torch.cumprod(alphas, dim=0)
            self.prediction_type = prediction_type
            self.clip_sample = clip_sample

        def add_noise(self, x: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
            sqrt_alpha_prod = self.alphas_cumprod[timesteps] ** 0.5
            sqrt_alpha_prod = sqrt_alpha_prod.flatten()
            while len(sqrt_alpha_prod.shape) < len(x.shape):
                sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_prod = (1 - self.alphas_cumprod[timesteps]) ** 0.5
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
            while len(sqrt_one_minus_alpha_prod.shape) < len(x.shape):
                sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
            return sqrt_alpha_prod * x + sqrt_one_minus_alpha_prod * noise

        def step(self, model_output: torch.Tensor, timestep: int, sample: torch.Tensor) -> Any:
            class Out:
                def __init__(self, prev):
                    self.prev_sample = prev
            t = int(timestep)
            alpha_prod_t = self.alphas_cumprod[t]
            alpha_prod_t_prev = self.alphas_cumprod[t - 1] if t > 0 else torch.tensor(1.0)
            beta_prod_t = 1 - alpha_prod_t
            pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
            if self.clip_sample:
                pred_original_sample = torch.clamp(pred_original_sample, -1, 1)
            pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
            prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
            return Out(prev_sample)

        def set_timesteps(self, num_inference_steps: int, device=None):
            self.timesteps = torch.linspace(self.num_train_timesteps - 1, 0, num_inference_steps, dtype=torch.long, device=device)


@dataclass
class HeadSpec:
    """单 head 的所有超参（action/obs/horizon/...）"""
    action_dim: int
    horizon: int
    n_obs_steps: int
    obs_feature_dim: int
    num_inference_steps: int = 16
    down_dims: tuple[int, ...] = (256, 512, 1024)
    diffusion_step_embed_dim: int = 128
    kernel_size: int = 5


class DiffusionRGBHead(nn.Module):
    """单任务 Diffusion Policy Head（ConditionalUnet1D + DDPM）"""

    def __init__(self, spec: HeadSpec):
        super().__init__()
        self.spec = spec
        # import ConditionalUnet1D
        try:
            from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
        except Exception:
            # fallback: minimal placeholder (you'd need full DP code for real training)
            class ConditionalUnet1D(nn.Module):
                def __init__(self, input_dim, global_cond_dim, diffusion_step_embed_dim, down_dims, kernel_size, n_groups=8):
                    super().__init__()
                    self.input_dim = input_dim
                    self.global_cond_dim = global_cond_dim
                    self.dummy = nn.Linear(input_dim + global_cond_dim, input_dim)

                def forward(self, x, t, global_cond):
                    # minimal placeholder: just return x
                    return x

        self.model = ConditionalUnet1D(
            input_dim=spec.action_dim,
            global_cond_dim=spec.obs_feature_dim * spec.n_obs_steps,
            diffusion_step_embed_dim=spec.diffusion_step_embed_dim,
            down_dims=spec.down_dims,
            kernel_size=spec.kernel_size,
            n_groups=8,
        )
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=100,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon',
        )

    def compute_loss(
        self,
        obs_features: torch.Tensor,
        action: torch.Tensor,
        normalizer_obs: Any,
        normalizer_action: Any,
    ) -> torch.Tensor:
        """训练时计算 noise-pred MSE loss"""
        B = obs_features.shape[0]
        obs_norm = normalizer_obs.normalize(obs_features)
        obs_cond = obs_norm.reshape(B, -1)
        action_norm = normalizer_action.normalize(action)

        noise = torch.randn_like(action_norm, device=action_norm.device)
        timesteps = torch.randint(0, self.noise_scheduler.num_train_timesteps, (B,), device=action_norm.device, dtype=torch.long)
        noisy_action = self.noise_scheduler.add_noise(action_norm, noise, timesteps)

        noise_pred = self.model(noisy_action.permute(0, 2, 1), timesteps, global_cond=obs_cond).permute(0, 2, 1)
        loss = torch.nn.functional.mse_loss(noise_pred, noise)
        return loss

    def predict_action(
        self,
        obs_features: torch.Tensor,
        normalizer_obs: Any,
        normalizer_action: Any,
    ) -> Dict[str, torch.Tensor]:
        """推理时采样动作"""
        device = obs_features.device
        B = obs_features.shape[0]
        obs_norm = normalizer_obs.normalize(obs_features)
        obs_cond = obs_norm.reshape(B, -1)

        # CRITICAL FIX: 采样张量必须在 obs_features.device 上（避免 GPU device mismatch）
        x = torch.randn((B, self.spec.horizon, self.spec.action_dim), device=device, dtype=torch.float32)
        self.noise_scheduler.set_timesteps(self.spec.num_inference_steps, device=device)

        for t in self.noise_scheduler.timesteps:
            tt = t.unsqueeze(0).expand(B).to(device)
            noise_pred = self.model(x.permute(0, 2, 1), tt, global_cond=obs_cond).permute(0, 2, 1)
            x = self.noise_scheduler.step(noise_pred, t, x).prev_sample

        action_pred = normalizer_action.unnormalize(x)
        return {'action_pred': action_pred}
