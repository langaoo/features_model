"""features_common/dp_rgb_policy_multitask.py

多任务版本：共享 RGB2PCStudentEncoder + 任务专属 Diffusion Policy Head。

契约（contract）
- 输入 batch dict：
  - obs: FloatTensor[B, To, C_in]
  - action: FloatTensor[B, Ta, A_task]  (注意：不同 task 的 A 不同，因此一个 batch 内可混 task，但 action tensor 需要按 task 分组后再使用)
  - task: list[str]  (来自 dp_rgb_dataset.collate_fn)

- 输出：
  - compute_loss(batch) -> scalar loss（对 batch 内各 task 分别算loss再加权平均）

设计说明
- obs normalizer：共享一个（key='obs'）
- action normalizer：每个 task 一个（key=f'action/{task}'）

注意：当前 dp_rgb_dataset 的 action 已经是固定维度（由 use_left_arm/use_right_arm 决定）。
若要在一个 DataLoader 内混合不同 action_dim 的 task，需要让 Dataset 也按 task 返回不同 shape。
本文件假设你会：
- 训练时可选择「每次只训一组相同 action_dim 的任务」
- 或者未来扩展 dataset，让 action 在 collate 时 padding + mask（我们不推荐，但可做）。

这里我们实现推荐方案：同 action_dim 的任务放一起训练；不同 action_dim 分开跑，
但 encoder checkpoint 共用，head 不共用。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Any
import os

import sys
from pathlib import Path

# Add DP to path (keep consistent with features_common/dp_rgb_policy.py)
DP_ROOT = Path(__file__).resolve().parents[1] / "DP" / "diffusion_policy"
# the python package is at DP_ROOT/diffusion_policy
if str(DP_ROOT) not in sys.path:
    sys.path.insert(0, str(DP_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F

# diffusers 在一些环境里可能缺失或版本不兼容；提供一个最小 DDPM scheduler 作为 fallback。
try:  # pragma: no cover
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler  # type: ignore
except Exception:  # pragma: no cover
    DDPMScheduler = None  # type: ignore


class _SimpleDDPMScheduler:
    """最小可用 DDPM scheduler（仅支持训练所需 add_noise + 反推 step）。

    目标：让本仓库 DP RGB 训练在没有 diffusers 的环境也能跑通。
    这不是完整实现，但足够用于训练/调试。
    """

    def __init__(
        self,
        *,
        num_train_timesteps: int = 100,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
    ):
        self.num_train_timesteps = int(num_train_timesteps)
        betas = torch.linspace(float(beta_start), float(beta_end), self.num_train_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register = {
            'betas': betas,
            'alphas_cumprod': alphas_cumprod,
            'sqrt_alphas_cumprod': torch.sqrt(alphas_cumprod),
            'sqrt_one_minus_alphas_cumprod': torch.sqrt(1.0 - alphas_cumprod),
        }

        class _Cfg:
            def __init__(self, n):
                self.num_train_timesteps = n

        self.config = _Cfg(self.num_train_timesteps)

    def add_noise(self, x0: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        # x_t = sqrt(a_bar)*x0 + sqrt(1-a_bar)*eps
        device = x0.device
        t = timesteps.to(device=device)
        s1 = self.register['sqrt_alphas_cumprod'].to(device=device, dtype=x0.dtype)[t].view(-1, 1, 1)
        s2 = self.register['sqrt_one_minus_alphas_cumprod'].to(device=device, dtype=x0.dtype)[t].view(-1, 1, 1)
        return s1 * x0 + s2 * noise


# DP core
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.model.common.normalizer import LinearNormalizer

from features_common.rgb2pc_student_encoder import RGB2PCStudentEncoder


@dataclass
class HeadSpec:
    action_dim: int
    horizon: int
    n_obs_steps: int
    n_action_steps: int
    obs_feature_dim: int
    obs_as_global_cond: bool = True


class DiffusionRGBHead(nn.Module):
    def __init__(
        self,
        *,
        spec: HeadSpec,
        noise_scheduler: Optional[Any] = None,
        diffusion_step_embed_dim: int = 256,
        down_dims: tuple = (256, 512, 1024),
        kernel_size: int = 5,
        n_groups: int = 8,
        cond_predict_scale: bool = True,
    ):
        super().__init__()
        self.spec = spec

        action_dim = int(spec.action_dim)
        obs_feature_dim = int(spec.obs_feature_dim)

        if spec.obs_as_global_cond:
            input_dim = action_dim
            global_cond_dim = obs_feature_dim * int(spec.n_obs_steps)
            local_cond_dim = None
        else:
            input_dim = action_dim + obs_feature_dim
            global_cond_dim = None
            local_cond_dim = None

        self.model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=local_cond_dim,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
        )

        if noise_scheduler is None:
            if DDPMScheduler is not None:
                noise_scheduler = DDPMScheduler(
                    num_train_timesteps=100,
                    beta_start=0.0001,
                    beta_end=0.02,
                    beta_schedule='squaredcos_cap_v2',
                    variance_type='fixed_small',
                    clip_sample=True,
                    prediction_type='epsilon',
                )
            else:
                noise_scheduler = _SimpleDDPMScheduler(num_train_timesteps=100, beta_start=0.0001, beta_end=0.02)
        self.noise_scheduler = noise_scheduler

        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if spec.obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=int(spec.n_obs_steps),
            fix_obs_steps=True,
            action_visible=False,
        )

        # per-head inference steps
        self.num_inference_steps = int(noise_scheduler.config.num_train_timesteps)

    def compute_loss(
        self,
        *,
        obs_features: torch.Tensor,
        action: torch.Tensor,
        normalizer_obs,
        normalizer_action,
    ) -> torch.Tensor:
        """obs_features: [B,To,D], action: [B,Ta,A]"""

        # normalize
        nobs = normalizer_obs.normalize(obs_features)
        naction = normalizer_action.normalize(action)

        B = nobs.shape[0]
        horizon = int(self.spec.horizon)
        action_dim = int(self.spec.action_dim)

        # pad action to horizon
        trajectory = naction
        if trajectory.shape[1] < horizon:
            pad = torch.zeros(
                (B, horizon - trajectory.shape[1], action_dim),
                device=trajectory.device,
                dtype=trajectory.dtype,
            )
            trajectory = torch.cat([trajectory, pad], dim=1)

        if self.spec.obs_as_global_cond:
            global_cond = nobs.reshape(B, -1)
            local_cond = None
        else:
            global_cond = None
            local_cond = nobs

        noise = torch.randn_like(trajectory)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (B,),
            device=trajectory.device,
        ).long()

        noisy = self.noise_scheduler.add_noise(trajectory, noise, timesteps)
        noise_pred = self.model(noisy, timesteps, local_cond=local_cond, global_cond=global_cond)
        loss = F.mse_loss(noise_pred, noise)
        return loss


    @torch.no_grad()
    def predict_action(
        self,
        *,
        obs_features: torch.Tensor,
        normalizer_obs,
        normalizer_action,
        num_inference_steps: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """采样得到动作序列。

        Args:
            obs_features: [B,To,D]
            normalizer_obs: SingleFieldLinearNormalizer for obs_features
            normalizer_action: SingleFieldLinearNormalizer for action
            num_inference_steps: optional override, default self.num_inference_steps

        Returns:
            dict with key 'action_pred': FloatTensor[B,horizon,action_dim]
        """

        # Make sure internal model is on the same device as inputs.
        dev = obs_features.device
        self.model.to(dev)
        if os.environ.get('DP_DEBUG_DEVICES', '0') == '1':
            try:
                print('[DP_DEBUG_DEVICES] obs_features', obs_features.device, obs_features.dtype, tuple(obs_features.shape))
                p0 = next(iter(self.model.parameters()))
                print('[DP_DEBUG_DEVICES] unet.param0', p0.device, p0.dtype)
                # diffusion_step_encoder is the usual mismatch hot spot
                p1 = next(iter(self.model.diffusion_step_encoder.parameters()))
                print('[DP_DEBUG_DEVICES] timestep_enc.param0', p1.device, p1.dtype)
            except Exception as _e:
                print('[DP_DEBUG_DEVICES] unable to dump devices:', _e)

        nobs = normalizer_obs.normalize(obs_features).to(device=dev)
        B = nobs.shape[0]
        horizon = int(self.spec.horizon)
        action_dim = int(self.spec.action_dim)

        if self.spec.obs_as_global_cond:
            global_cond = nobs.reshape(B, -1)
            local_cond = None
        else:
            global_cond = None
            local_cond = nobs

        # start from pure noise in normalized action space
        x = torch.randn((B, horizon, action_dim), device=dev, dtype=nobs.dtype)

        # Sampling note:
        # Some diffusers scheduler versions have device quirks (CPU timesteps) that can break CUDA sampling.
        # For robustness in this repo, we use a simple deterministic Euler-style loop that stays fully on-device.
        n_steps = int(num_inference_steps or self.num_inference_steps)
        T = int(getattr(getattr(self.noise_scheduler, 'config', object()), 'num_train_timesteps', n_steps))
        for t in reversed(range(min(T, n_steps))):
            # Keep timesteps as int64 on CUDA to match ConditionalUnet1D's embedding path.
            tt = torch.full((B,), int(t), device=dev, dtype=torch.long)
            if os.environ.get('DP_DEBUG_DEVICES', '0') == '1' and t == min(T, n_steps) - 1:
                print('[DP_DEBUG_DEVICES] x', x.device, x.dtype, tuple(x.shape))
                print('[DP_DEBUG_DEVICES] tt', tt.device, tt.dtype, tuple(tt.shape))
            noise_pred = self.model(x, tt, local_cond=local_cond, global_cond=global_cond)
            x = x - noise_pred / float(n_steps)

        action = normalizer_action.unnormalize(x)
        return {'action_pred': action}


class MultiTaskDiffusionRGBPolicy(nn.Module):
    """共享 encoder + 任务专属 head 的包装器。"""

    def __init__(
        self,
        *,
        encoder: RGB2PCStudentEncoder,
        head_specs: Dict[str, HeadSpec],
    ):
        super().__init__()
        self.encoder = encoder
        self.heads = nn.ModuleDict({k: DiffusionRGBHead(spec=v) for k, v in head_specs.items()})

        # normalizers
        self.normalizer = LinearNormalizer()
        # you must set these via set_normalizers

    def set_normalizers(self, *, obs_normalizer, action_normalizers: Dict[str, object]):
        # LinearNormalizer is dict-like; we store in a single container for checkpoint friendliness
        self.normalizer['obs'] = obs_normalizer
        for task, an in action_normalizers.items():
            self.normalizer[f'action/{task}'] = an

    def compute_loss(self, batch: Dict) -> torch.Tensor:
        obs = batch['obs']
        action = batch['action']
        tasks = batch['task']

        if not isinstance(tasks, list):
            raise TypeError("batch['task'] must be a list[str]")

        # encode obs once
        obs_feat = self.encoder(obs)  # [B,To,D]

        # group indices by task
        task_to_idx: Dict[str, list[int]] = {}
        for i, t in enumerate(tasks):
            task_to_idx.setdefault(str(t), []).append(i)

        losses = []
        total = 0
        for t, idxs in task_to_idx.items():
            if t not in self.heads:
                raise KeyError(f"Task head not found: {t}. Available: {list(self.heads.keys())[:10]}")
            head = self.heads[t]
            ix = torch.tensor(idxs, device=obs.device, dtype=torch.long)
            loss_t = head.compute_loss(
                obs_features=obs_feat.index_select(0, ix),
                action=action.index_select(0, ix),
                normalizer_obs=self.normalizer['obs'],
                normalizer_action=self.normalizer[f'action/{t}'],
            )
            losses.append(loss_t * len(idxs))
            total += len(idxs)

        return sum(losses) / max(1, total)
