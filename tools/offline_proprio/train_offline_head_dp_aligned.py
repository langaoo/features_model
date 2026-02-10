#!/usr/bin/env python3
"""
离线Head训练脚本 (DP-Aligned Proprio)

关键改动（与 train_offline_head_proprio.py 的区别）：
- agent_pos 不再通过单独的 MLP 编码器
- agent_pos 归一化后直接拼接到 obs_aligned 特征上
- obs_encoder 的输入维度自动适配 (To * (feat_dim + proprio_dim))
- 完全对齐原版 DP 的 MultiImageObsEncoder 处理方式
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import argparse
import sys
import os
from tqdm import tqdm
import logging
import yaml
import numpy as np
import time
import zarr

logging.basicConfig(level=logging.ERROR)
os.environ['PYTHONWARNINGS'] = 'ignore'

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from path_config import DP_ROOT

HAS_OFFICIAL_DP = False
try:
    DP_OUTER = DP_ROOT
    if DP_OUTER.exists() and (DP_OUTER / "diffusion_policy").exists():
        sys.path.insert(0, str(DP_OUTER))
        from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
        from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
        from diffusion_policy.model.common.normalizer import LinearNormalizer
        HAS_OFFICIAL_DP = True
    else:
        print(f"[WARNING] DP not found at {DP_OUTER}")
except ImportError as e:
    print(f"[WARNING] DP import failed: {e}")


class DPAlignedPolicy(nn.Module):
    """
    DP-Aligned Diffusion Policy

    关键设计：agent_pos 的处理方式与原版 DP 完全一致
    - agent_pos 通过 LinearNormalizer 归一化到 [-1, 1]
    - 归一化后直接拼接到视觉特征后面（不经过任何 MLP）
    - obs_encoder 接收拼接后的向量

    原版 DP 的 MultiImageObsEncoder:
        features = [image_feat, agent_pos]  # 直接 concat
        result = torch.cat(features, dim=-1)

    本 Policy:
        obs_input = [vis_feat, normalized_agent_pos]  # 直接 concat
        obs_cond = obs_encoder(obs_input.reshape(B, -1))
    """

    def __init__(
        self,
        vis_dim: int,
        action_dim: int,
        horizon: int,
        n_obs_steps: int,
        n_action_steps: int,
        num_inference_steps: int = 100,
        use_proprio: bool = False,
        proprio_dim: int = 14,
    ):
        super().__init__()
        if not HAS_OFFICIAL_DP:
            raise RuntimeError("DP not loaded")

        self.vis_dim = vis_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.num_inference_steps = num_inference_steps
        self.use_proprio = bool(use_proprio)
        self.proprio_dim = int(proprio_dim) if use_proprio else 0

        # LinearNormalizer: fit on {"action": ..., "agent_pos": ...}
        self.normalizer = LinearNormalizer()

        # obs_encoder 输入维度：n_obs_steps * (vis_dim + proprio_dim)
        # 这与原版 DP 的 global_cond_dim = obs_feature_dim * n_obs_steps 一致
        per_step_dim = vis_dim + (self.proprio_dim if use_proprio else 0)
        obs_input_dim = n_obs_steps * per_step_dim

        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_input_dim, 512),
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

    def _build_obs_input(self, obs: torch.Tensor, agent_pos: torch.Tensor = None) -> torch.Tensor:
        """
        构建观测输入向量，模仿原版 DP 的 MultiImageObsEncoder

        Args:
            obs: [B, To, vis_dim] 视觉特征（已对齐）
            agent_pos: [B, To, proprio_dim] 本体感知（可选）

        Returns:
            obs_flat: [B, To * (vis_dim + proprio_dim)]
        """
        B = obs.shape[0]
        if self.use_proprio and agent_pos is not None:
            # 归一化 agent_pos（模仿 DP 的 self.normalizer.normalize(obs_dict)）
            npos = self.normalizer["agent_pos"].normalize(agent_pos).to(obs.device)
            # 直接拼接：[B, To, vis_dim] || [B, To, proprio_dim] → [B, To, vis_dim+proprio_dim]
            obs_combined = torch.cat([obs, npos], dim=-1)
        else:
            obs_combined = obs
        return obs_combined.reshape(B, -1)

    def compute_loss(self, obs: torch.Tensor, action_gt: torch.Tensor,
                     agent_pos: torch.Tensor = None) -> torch.Tensor:
        """
        训练 loss

        Args:
            obs: [B, To, vis_dim]
            action_gt: [B, Ta, action_dim] (未归一化)
            agent_pos: [B, To, proprio_dim] (未归一化)
        """
        B = obs.shape[0]
        device = obs.device

        # 归一化 action
        nactions = self.normalizer['action'].normalize(action_gt).to(device)

        # 构建观测条件
        obs_flat = self._build_obs_input(obs, agent_pos)
        obs_cond = self.obs_encoder(obs_flat)

        # Diffusion training
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (B,), device=device
        ).long()
        noise = torch.randn(nactions.shape, device=device)
        noisy_actions = self.noise_scheduler.add_noise(nactions, noise, timesteps)
        noise_pred = self.noise_pred_net(noisy_actions, timesteps, global_cond=obs_cond)
        return nn.functional.mse_loss(noise_pred, noise)

    def forward(self, obs: torch.Tensor, agent_pos: torch.Tensor = None) -> torch.Tensor:
        """
        推理：去噪采样

        Args:
            obs: [B, To, vis_dim]
            agent_pos: [B, To, proprio_dim] (未归一化, normalizer 会处理)

        Returns:
            action: [B, Ta, action_dim] (已反归一化)
        """
        B = obs.shape[0]
        device = obs.device

        obs_flat = self._build_obs_input(obs, agent_pos)
        obs_cond = self.obs_encoder(obs_flat)

        action = torch.randn((B, self.n_action_steps, self.action_dim), device=device)
        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        for t in self.noise_scheduler.timesteps:
            noise_pred = self.noise_pred_net(
                action,
                t.unsqueeze(0).expand(B).to(device),
                global_cond=obs_cond,
            )
            action = self.noise_scheduler.step(noise_pred, t, action).prev_sample

        # 反归一化 action
        action = self.normalizer['action'].unnormalize(action)
        return action


class DPAlignedDualStreamPolicy(nn.Module):
    """
    DP-Aligned Dual-Stream Policy

    与 DPAlignedPolicy 相同的 proprio 处理方式 + token cross-attention
    """

    def __init__(
        self,
        vis_dim: int,
        token_dim: int,
        action_dim: int,
        horizon: int,
        n_obs_steps: int,
        n_action_steps: int,
        num_inference_steps: int = 100,
        token_dropout: float = 0.0,
        ctx_dropout: float = 0.0,
        token_gate_init: float = -4.0,
        use_proprio: bool = False,
        proprio_dim: int = 14,
    ):
        super().__init__()
        if not HAS_OFFICIAL_DP:
            raise RuntimeError("DP not loaded")

        self.vis_dim = vis_dim
        self.token_dim = token_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.num_inference_steps = num_inference_steps
        self.use_proprio = bool(use_proprio)
        self.proprio_dim = int(proprio_dim) if use_proprio else 0

        self.normalizer = LinearNormalizer()
        self.use_normalizer = True  # 始终使用

        per_step_dim = vis_dim + (self.proprio_dim if use_proprio else 0)
        obs_input_dim = n_obs_steps * per_step_dim

        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )

        self.token_proj = nn.Linear(token_dim, 256)
        self.query_proj = nn.Linear(256, 256)
        self.cross_attn = nn.MultiheadAttention(256, num_heads=8, batch_first=True)
        self.token_dropout = nn.Dropout(float(token_dropout)) if float(token_dropout) > 0 else nn.Identity()
        self.ctx_dropout = nn.Dropout(float(ctx_dropout)) if float(ctx_dropout) > 0 else nn.Identity()
        self.token_gate = nn.Parameter(torch.tensor(float(token_gate_init)))

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

    def _build_obs_input(self, obs: torch.Tensor, agent_pos: torch.Tensor = None) -> torch.Tensor:
        B = obs.shape[0]
        if self.use_proprio and agent_pos is not None:
            npos = self.normalizer["agent_pos"].normalize(agent_pos).to(obs.device)
            obs_combined = torch.cat([obs, npos], dim=-1)
        else:
            obs_combined = obs
        return obs_combined.reshape(B, -1)

    def _build_cond(self, obs_global: torch.Tensor, obs_tokens: torch.Tensor,
                    agent_pos: torch.Tensor = None,
                    force_gate: float = None) -> torch.Tensor:
        B = obs_global.shape[0]

        obs_flat = self._build_obs_input(obs_global, agent_pos)
        global_cond = self.obs_encoder(obs_flat)

        tokens = obs_tokens.reshape(B, -1, obs_tokens.shape[-1])
        tokens = self.token_proj(tokens)
        tokens = self.token_dropout(tokens)
        query = self.query_proj(global_cond).unsqueeze(1)
        ctx, _ = self.cross_attn(query, tokens, tokens)
        ctx = self.ctx_dropout(ctx.squeeze(1))

        if force_gate is not None:
            gate = force_gate
        else:
            gate = torch.sigmoid(self.token_gate)
        return global_cond + gate * ctx

    def compute_loss(self, obs_global: torch.Tensor, obs_tokens: torch.Tensor,
                     action_gt: torch.Tensor, agent_pos: torch.Tensor = None,
                     force_gate: float = None,
                     gate_regularization: float = 0.0) -> torch.Tensor:
        B = obs_global.shape[0]
        device = obs_global.device

        nactions = self.normalizer['action'].normalize(action_gt).to(device)
        obs_cond = self._build_cond(obs_global, obs_tokens, agent_pos, force_gate=force_gate)

        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (B,), device=device
        ).long()
        noise = torch.randn(nactions.shape, device=device)
        noisy_actions = self.noise_scheduler.add_noise(nactions, noise, timesteps)
        noise_pred = self.noise_pred_net(noisy_actions, timesteps, global_cond=obs_cond)

        mse_loss = nn.functional.mse_loss(noise_pred, noise)
        if gate_regularization > 0:
            gate_sigmoid = torch.sigmoid(self.token_gate)
            gate_reg_loss = gate_regularization * (gate_sigmoid - 0.5).pow(2)
            return mse_loss + gate_reg_loss
        return mse_loss

    def forward(self, obs_global: torch.Tensor, obs_tokens: torch.Tensor,
                agent_pos: torch.Tensor = None) -> torch.Tensor:
        B = obs_global.shape[0]
        device = obs_global.device
        obs_cond = self._build_cond(obs_global, obs_tokens, agent_pos)

        action = torch.randn((B, self.n_action_steps, self.action_dim), device=device)
        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        for t in self.noise_scheduler.timesteps:
            noise_pred = self.noise_pred_net(
                action,
                t.unsqueeze(0).expand(B).to(device),
                global_cond=obs_cond,
            )
            action = self.noise_scheduler.step(noise_pred, t, action).prev_sample

        action = self.normalizer['action'].unnormalize(action)
        return action


# ──────────────────────────────── Dataset ────────────────────────────────

class OfflineFeatureDataset(Dataset):
    """与 train_offline_head_proprio.py 中相同的数据集"""

    def __init__(self, dataset_dir, tasks, horizon=8, n_obs_steps=2,
                 use_proprio=False, action_offset=0):
        self.dataset_dir = Path(dataset_dir)
        self.tasks = tasks
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.use_proprio = bool(use_proprio)
        self.action_offset = int(action_offset)
        self.samples = []

        for task in tasks:
            task_dir = self.dataset_dir / task
            if not task_dir.exists():
                print(f"[Dataset] Warning: {task_dir} not found")
                continue
            zarr_files = sorted(list(task_dir.glob("*.zarr")))
            for zf in zarr_files:
                try:
                    root = zarr.open(str(zf), mode='r')
                    n_frames = root.attrs['num_frames']
                    effective_frames = max(0, int(n_frames) - self.action_offset)
                    valid_starts = max(0, effective_frames - (horizon + n_obs_steps) + 1)
                    for i in range(valid_starts):
                        self.samples.append((task, zf.stem, i, str(zf)))
                except Exception as e:
                    print(f"Error reading {zf}: {e}")

        print(f"[Dataset] Loaded {len(self.samples)} samples from {len(tasks)} tasks")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        task, ep_name, start_idx, zpath = self.samples[idx]
        root = zarr.open(zpath, mode='r')
        n_frames = root.attrs['num_frames']

        obs_end = min(start_idx + 1, n_frames)
        obs_start = max(0, obs_end - self.n_obs_steps)

        obs_data = root['obs_aligned'][obs_start:obs_end]
        obs_tokens = root['obs_tokens'][obs_start:obs_end] if 'obs_tokens' in root else None
        agent_pos = None
        if self.use_proprio and 'agent_pos' in root:
            agent_pos = root['agent_pos'][obs_start:obs_end]

        # Pad if needed
        if obs_data.shape[0] < self.n_obs_steps:
            pad_len = self.n_obs_steps - obs_data.shape[0]
            obs_data = np.concatenate([np.tile(obs_data[:1], (pad_len, 1)), obs_data], axis=0)
            if obs_tokens is not None:
                obs_tokens = np.concatenate([np.tile(obs_tokens[:1], (pad_len, 1, 1)), obs_tokens], axis=0)
            if agent_pos is not None:
                agent_pos = np.concatenate([np.tile(agent_pos[:1], (pad_len, 1)), agent_pos], axis=0)

        # Action
        act_start = start_idx + self.action_offset
        act_end = min(act_start + self.horizon, n_frames)
        action_data = root['action'][act_start:act_end]
        if action_data.shape[0] < self.horizon:
            pad_len = self.horizon - action_data.shape[0]
            action_data = np.concatenate([action_data, np.tile(action_data[-1:], (pad_len, 1))], axis=0)

        payload = {
            'obs': torch.from_numpy(obs_data).float(),
            'action': torch.from_numpy(action_data).float(),
        }
        if obs_tokens is not None:
            payload['obs_tokens'] = torch.from_numpy(obs_tokens).float()
        if agent_pos is not None:
            payload['agent_pos'] = torch.from_numpy(agent_pos).float()
        return payload


# ──────────────────────────────── Main ────────────────────────────────

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    gpu_ids = config['device']['gpu_ids']
    device = torch.device(f"cuda:{gpu_ids[0]}")

    use_proprio = bool(config.get('policy', {}).get('use_proprio', False))
    proprio_dim = int(config.get('policy', {}).get('proprio_dim', 14))
    action_offset = int(config.get('data', {}).get('action_offset', 0))

    # 1. Dataset
    print("Creating Datasets...")
    dataset = OfflineFeatureDataset(
        dataset_dir=config['data']['features_dataset_dir'],
        tasks=config['data']['tasks'],
        horizon=config['data']['horizon'],
        n_obs_steps=config['data']['n_obs_steps'],
        use_proprio=use_proprio,
        action_offset=action_offset,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config['train']['batch_size'],
        shuffle=True,
        num_workers=config['train']['num_workers'],
        pin_memory=True,
        prefetch_factor=2 if config['train']['num_workers'] > 0 else None,
    )
    print(f"Dataset size: {len(dataset)}")

    # 2. Policy
    sample = dataset[0]
    action_dim = sample['action'].shape[-1]
    vis_dim = 1280  # aligned feature dim
    use_tokens = bool(config.get('policy', {}).get('use_tokens', False)) and 'obs_tokens' in sample

    if use_proprio and 'agent_pos' not in sample:
        print("[Warning] use_proprio=True but agent_pos not found. Fallback to pure visual.")
        use_proprio = False

    print(f"Vis Dim: {vis_dim}, Action Dim: {action_dim}, Proprio: {use_proprio} (dim={proprio_dim})")

    if use_tokens:
        token_dim = sample['obs_tokens'].shape[-1]
        token_dropout = float(config.get('policy', {}).get('token_dropout', 0.0))
        ctx_dropout = float(config.get('policy', {}).get('ctx_dropout', 0.0))
        token_gate_init = float(config.get('policy', {}).get('token_gate_init', -4.0))
        policy = DPAlignedDualStreamPolicy(
            vis_dim=vis_dim,
            token_dim=token_dim,
            action_dim=action_dim,
            horizon=config['data']['horizon'],
            n_obs_steps=config['data']['n_obs_steps'],
            n_action_steps=config['data']['horizon'],
            num_inference_steps=config['policy']['num_inference_steps'],
            token_dropout=token_dropout,
            ctx_dropout=ctx_dropout,
            token_gate_init=token_gate_init,
            use_proprio=use_proprio,
            proprio_dim=proprio_dim,
        ).to(device)
    else:
        policy = DPAlignedPolicy(
            vis_dim=vis_dim,
            action_dim=action_dim,
            horizon=config['data']['horizon'],
            n_obs_steps=config['data']['n_obs_steps'],
            n_action_steps=config['data']['horizon'],
            num_inference_steps=config['policy']['num_inference_steps'],
            use_proprio=use_proprio,
            proprio_dim=proprio_dim,
        ).to(device)

    # 3. Fit normalizer
    print("Fitting normalizer...")
    all_actions = []
    all_agent_pos = [] if use_proprio else None
    for i in range(min(len(dataset), 1000)):
        s = dataset[i]
        all_actions.append(s['action'])
        if use_proprio and 'agent_pos' in s:
            all_agent_pos.append(s['agent_pos'])
    all_actions = torch.stack(all_actions)
    norm_data = {'action': all_actions}
    if use_proprio and all_agent_pos:
        all_agent_pos = torch.stack(all_agent_pos)
        norm_data['agent_pos'] = all_agent_pos
    policy.normalizer.fit(
        norm_data,
        last_n_dims=1,
        mode='limits',
        output_min=-1.0,
        output_max=1.0,
    )
    print(f"  Action range: [{policy.normalizer['action'].params_dict['input_stats'].min[:3]}..., "
          f"{policy.normalizer['action'].params_dict['input_stats'].max[:3]}...]")
    if use_proprio and "agent_pos" in policy.normalizer.params_dict:
        ap_stats = policy.normalizer['agent_pos'].params_dict['input_stats']
        print(f"  Agent_pos range: [{ap_stats.min[:3]}..., {ap_stats.max[:3]}...]")
        # 验证 agent_pos 与 action 的差异性
        try:
            diff = (all_agent_pos[:, -1, :] - all_actions[:, 0, :]).pow(2).mean().item()
            same_ratio = (all_agent_pos[:, -1, :] == all_actions[:, 0, :]).all(dim=-1).float().mean().item()
            print(f"  agent_pos[-1] vs action[0] MSE: {diff:.6f}, same_ratio: {same_ratio:.2%}")
        except Exception:
            pass

    # 4. Optimizer
    lr = float(config['train']['lr'])
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr,
                                   weight_decay=float(config['train'].get('weight_decay', 1e-6)))

    # 4.1 Resume
    resume_ckpt = config.get('train', {}).get('resume_ckpt')
    start_epoch = 0
    if resume_ckpt:
        rp = Path(str(resume_ckpt))
        if rp.exists():
            pl = torch.load(rp, map_location=device)
            missing, unexpected = policy.load_state_dict(pl['policy'], strict=False)
            if missing:
                print(f"[Resume] Missing: {missing}")
            if unexpected:
                print(f"[Resume] Unexpected: {unexpected}")
            if 'normalizer' in pl:
                policy.normalizer.load_state_dict(pl['normalizer'])
            if 'optimizer' in pl:
                try:
                    optimizer.load_state_dict(pl['optimizer'])
                except Exception:
                    print("[WARN] Optimizer state incompatible.")
            start_epoch = int(pl.get('epoch', 0))
            print(f"[Resume] Loaded {rp}, epoch={start_epoch}")

    # 5. Train
    print("Starting Offline Training (DP-Aligned Proprio)...")
    total_epochs = int(config['train']['epochs'])
    gate_warmup_epochs = int(config.get('train', {}).get('gate_warmup_epochs', total_epochs // 3))
    gate_regularization = float(config.get('train', {}).get('gate_regularization', 0.01))

    for epoch in range(start_epoch, total_epochs):
        policy.train()
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}", ncols=100)

        force_gate = 0.5 if (use_tokens and epoch < gate_warmup_epochs) else None
        current_gate_reg = 0.0 if force_gate is not None else gate_regularization

        for batch in pbar:
            obs = batch['obs'].to(device)
            action = batch['action'].to(device)
            agent_pos = batch.get('agent_pos', None)
            if agent_pos is not None:
                agent_pos = agent_pos.to(device)

            if use_tokens:
                obs_tokens = batch['obs_tokens'].to(device)
                loss = policy.compute_loss(
                    obs, obs_tokens, action,
                    agent_pos=agent_pos,
                    force_gate=force_gate,
                    gate_regularization=current_gate_reg,
                )
            else:
                loss = policy.compute_loss(obs, action, agent_pos=agent_pos)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_loss = epoch_loss / len(dataloader)

        if use_tokens and hasattr(policy, 'token_gate'):
            gs = torch.sigmoid(policy.token_gate).item()
            status = "FORCED=0.5" if force_gate is not None else "learnable"
            print(f"Epoch {epoch+1}: loss={avg_loss:.4f}, gate_sig={gs:.4f} ({status})")
        else:
            print(f"Epoch {epoch+1}: loss={avg_loss:.4f}")

        # Save
        if (epoch + 1) % config['output']['save_every_n_epochs'] == 0:
            task_name = config['data']['tasks'][0]
            ckpt_setting = config['checkpoint']['ckpt_setting']
            num = config['checkpoint']['expert_data_num']
            seed = config['checkpoint']['seed']
            save_dir_name = f"{task_name}-{ckpt_setting}-{num}-{seed}"
            output_root = config.get('output', {}).get('dir')
            if output_root:
                save_path = Path(output_root) / save_dir_name
            else:
                save_path = Path(__file__).parent.parent / "checkpoints" / save_dir_name
            save_path.mkdir(parents=True, exist_ok=True)
            ckpt_file = save_path / f"{epoch+1}.ckpt"
            torch.save({
                'policy': policy.state_dict(),
                'normalizer': policy.normalizer.state_dict(),
                'policy_type': 'dp_aligned_dual_stream' if use_tokens else 'dp_aligned',
                'policy_class': 'DPAlignedDualStreamPolicy' if use_tokens else 'DPAlignedPolicy',
                'token_dim': int(token_dim) if use_tokens else None,
                'optimizer': optimizer.state_dict(),
                'epoch': epoch + 1,
                'config': config,
                'loss': avg_loss,
            }, ckpt_file)
            print(f"Saved: {ckpt_file}")


if __name__ == "__main__":
    main()
