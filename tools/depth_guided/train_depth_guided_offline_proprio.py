#!/usr/bin/env python3
"""
tools/depth_guided/train_depth_guided_offline_proprio.py

Depth-Guided Cross-Attention Fusion + proprio 训练【离线版】
============================================================

方案说明
--------
  - 语义组 [CroCo, VGGT, DINOv3] 提供 Query（内容理解）
  - 几何组 [DA3]                  提供 Key/Value（空间结构）
  - Cross-Attention 让语义 tokens 按深度感知空间重新聚合
  - 训练信号：仅 Action Loss（无需点云 teacher，无需对齐模块）
  - 特征格式：token-level（保留空间拓扑），不做 mean-pool

与 train_direct_fusion_offline_proprio.py 的对比
------------------------------------------------
  - 相同：数据加载逻辑、normalizer、DP Head、训练循环、超参数
  - 不同：SimpleFusionEncoder → DepthGuidedFusionEncoder
          token-level 特征（而非 mean-pooled 向量）
"""
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import argparse
import sys
import os
from tqdm import tqdm
import warnings
import yaml
from typing import Dict, Any, List, Tuple
import numpy as np
import zarr
import h5py

warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

# 添加路径
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

# 导入正版 DP
HAS_OFFICIAL_DP = False
try:
    DP_OUTER = REPO_ROOT / "third_party" / "DP" / "diffusion_policy"
    if DP_OUTER.exists():
        sys.path.insert(0, str(DP_OUTER))
        from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
        from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
        from diffusion_policy.model.common.normalizer import LinearNormalizer
        HAS_OFFICIAL_DP = True
        print("[INFO] 正版DP已加载")
except ImportError as e:
    print(f"[WARNING] 正版DP导入失败: {e}")
    sys.exit(1)

# 导入 Depth-Guided Encoder
from features_common.depth_guided import DepthGuidedFusionEncoder


# ============================================================
# Policy
# ============================================================

class DepthGuidedPolicy(nn.Module):
    """Depth-Guided Cross-Attention Fusion + proprio + Diffusion Policy。

    与 DirectFusionProprioPolicy 完全相同的 DP Head 结构，
    区别只在 fusion_encoder 换成了 DepthGuidedFusionEncoder。

    输入（训练时）：
        tokens_list: List[4 Tensor]，每个 [B, To, K_i, C_i]
        agent_pos:   [B, To, proprio_dim]
        actions:     [B, Ta, action_dim]

    输入（推理时）：
        tokens_list + agent_pos
    """

    def __init__(
        self,
        fusion_encoder: DepthGuidedFusionEncoder,
        proprio_dim: int = 14,
        action_dim: int = 14,
        horizon: int = 8,
        n_obs_steps: int = 3,
        num_inference_steps: int = 100,
    ):
        super().__init__()

        if not HAS_OFFICIAL_DP:
            raise RuntimeError("正版 DP 未加载")

        self.fusion_encoder = fusion_encoder
        self.proprio_dim = proprio_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.num_inference_steps = num_inference_steps
        self.use_proprio = True

        self.normalizer = LinearNormalizer()

        # obs_encoder 输入维度 = (out_dim + proprio_dim) * n_obs_steps
        per_step_dim = fusion_encoder.out_dim + proprio_dim
        obs_encoder_dim = per_step_dim * n_obs_steps

        # 与 Direct Fusion / tokens_full 实验保持完全一致的 obs_encoder 和 UNet
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_encoder_dim, 512),
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
            prediction_type='epsilon',
        )

    def compute_loss(
        self,
        tokens_list: List[torch.Tensor],
        agent_pos: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        tokens_list: List[4 Tensor]，每个 [B, To, K_i, C_i]
        agent_pos:   [B, To, 14]（未归一化）
        actions:     [B, Ta, 14]（未归一化）
        """
        B = actions.shape[0]
        device = actions.device

        # 归一化
        nactions = self.normalizer.normalize({'action': actions})['action'].to(device)
        nagent_pos = self.normalizer.normalize({'agent_pos': agent_pos})['agent_pos'].to(device)

        # 融合：4 组 tokens → [B, To, out_dim]
        fused = self.fusion_encoder(tokens_list)

        # 拼接 proprio
        obs_combined = torch.cat([fused, nagent_pos], dim=-1)  # [B, To, out_dim+14]
        obs_flat = obs_combined.reshape(B, -1)
        obs_cond = self.obs_encoder(obs_flat)                  # [B, 256]

        # Diffusion noise
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (B,), device=device,
        ).long()

        noise = torch.randn_like(nactions)
        noisy_actions = self.noise_scheduler.add_noise(nactions, noise, timesteps)
        noise_pred = self.noise_pred_net(noisy_actions, timesteps, global_cond=obs_cond)

        return nn.functional.mse_loss(noise_pred, noise)

    @torch.no_grad()
    def predict_action(
        self,
        tokens_list: List[torch.Tensor],
        agent_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        tokens_list: List[4 Tensor]，每个 [B, To, K_i, C_i]
        agent_pos:   [B, To, 14]（未归一化，可选）
        Returns: [B, horizon, 14]（未归一化）
        """
        B = tokens_list[0].shape[0]
        device = tokens_list[0].device

        fused = self.fusion_encoder(tokens_list)

        if agent_pos is not None:
            nagent_pos = self.normalizer.normalize({'agent_pos': agent_pos})['agent_pos'].to(device)
            obs_combined = torch.cat([fused, nagent_pos], dim=-1)
        else:
            zeros = torch.zeros(B, fused.shape[1], self.proprio_dim, device=device)
            obs_combined = torch.cat([fused, zeros], dim=-1)

        obs_flat = obs_combined.reshape(B, -1)
        obs_cond = self.obs_encoder(obs_flat)

        action = torch.randn((B, self.horizon, self.action_dim), device=device)
        self.noise_scheduler.set_timesteps(self.num_inference_steps)

        for t in self.noise_scheduler.timesteps:
            noise_pred = self.noise_pred_net(
                action,
                t.unsqueeze(0).expand(B).to(device),
                global_cond=obs_cond,
            )
            action = self.noise_scheduler.step(noise_pred, t, action).prev_sample

        return self.normalizer.unnormalize({'action': action})['action']


# ============================================================
# Dataset
# ============================================================

# 模型列表（顺序固定！）
MODEL_NAMES = ['croco', 'vggt', 'dinov3', 'da3']
MODEL_IN_DIMS = [1024, 2048, 768, 2048]


class DepthGuidedDataset(Dataset):
    """Token-level 特征数据集，保留空间结构供 Cross-Attention 使用。

    关键点：
      - 直接加载 per_frame_features 的空间维度（不做 mean-pool）
      - 每帧形状: zarr [W, T, Hf, Wf, C] → 取 window[0] → [Hf*Wf, C] tokens

    时间对齐（与其他实验完全一致）：
      state[t]  = vector[t]       frames 0..T-2
      action[t] = vector[t+1]     frames 1..T-1
    """

    def __init__(
        self,
        vis_zarr_roots: List[str],     # 4 个模型的 zarr 根目录（顺序：croco/vggt/dinov3/da3）
        robotwin_data_root: str,
        task_name: str,
        task_config: str,
        horizon: int = 8,
        n_obs_steps: int = 3,
        expert_data_num: int = 50,
        camera_name: str = 'head_camera',
        max_tokens: int | None = None,    # 每个模型最多保留多少 tokens（None=全部）
        zarr_expert_num: int | None = None,  # zarr 目录名中的 expert 数（默认=expert_data_num）
    ):
        super().__init__()
        assert len(vis_zarr_roots) == 4, "必须提供 4 个模型的 zarr 根目录"

        self.vis_zarr_roots = vis_zarr_roots
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.task_name = task_name
        self.task_config = task_config
        self.camera_name = camera_name
        self.expert_data_num = expert_data_num
        self.max_tokens = max_tokens

        _zarr_num = zarr_expert_num if zarr_expert_num is not None else expert_data_num
        self.raw_data_root = os.path.join(robotwin_data_root, task_name, task_config, 'data')
        self.episodes = [f'episode_{i}' for i in range(expert_data_num)]
        self.zarr_subdir = f"{task_name}-{task_config}-{_zarr_num}_sapien_{camera_name}"

        print(f"[DepthGuided] 收集数据样本...")
        self.samples: List[Tuple] = self._collect_samples()
        print(f"[DepthGuided] 共 {len(self.samples)} 个样本")

    def _load_vector(self, ep_num: int) -> torch.Tensor:
        """加载 joint_action/vector，返回 [T, 14]。"""
        hdf5_path = os.path.join(self.raw_data_root, f'episode{ep_num}.hdf5')
        with h5py.File(hdf5_path, 'r') as f:
            if 'joint_action/vector' in f:
                vec = f['joint_action/vector'][:]
            else:
                left = f['joint_action/left_arm'][:]
                right = f['joint_action/right_arm'][:]
                lg = f['joint_action/left_gripper'][:] if 'joint_action/left_gripper' in f else np.zeros((left.shape[0], 1))
                rg = f['joint_action/right_gripper'][:] if 'joint_action/right_gripper' in f else np.zeros((right.shape[0], 1))
                if lg.ndim == 1:
                    lg = lg[:, None]
                if rg.ndim == 1:
                    rg = rg[:, None]
                vec = np.concatenate([left, lg, right, rg], axis=-1)
        return torch.from_numpy(vec).float()

    def _load_episode_tokens(self, episode: str) -> List[torch.Tensor]:
        """加载 4 个模型的 per-frame tokens，返回 List[4 × Tensor[T, K_i, C_i]]。

        ws1 zarr 格式：per_frame_features [T, 1, Hf, Wf, C]
          → squeeze dim=1 → [T, Hf, Wf, C]
          → reshape → [T, Hf*Wf, C]

        兼容其他格式（ws>1）：[W, ws, Hf, Wf, C]
          → 取 ws 维的第 0 帧（最新帧）→ [W, Hf, Wf, C]
          → reshape → [W, Hf*Wf, C]
        """
        tokens_by_model = []
        for mi, zarr_root in enumerate(self.vis_zarr_roots):
            zarr_path = os.path.join(zarr_root, self.zarr_subdir, f"{episode}.zarr")
            z = zarr.open(zarr_path, mode='r')
            pf = z['per_frame_features']  # [T, ws, Hf, Wf, C]

            if pf.ndim == 5:
                # [T, ws, Hf, Wf, C]：取 ws 维第 0 帧
                T_frames, ws, Hf, Wf, C = pf.shape
                arr = pf[:, 0, :, :, :]          # [T, Hf, Wf, C]
                arr = arr.reshape(T_frames, Hf * Wf, C)
            elif pf.ndim == 4:
                # [T, Hf, Wf, C]
                T_frames, Hf, Wf, C = pf.shape
                arr = pf[:].reshape(T_frames, Hf * Wf, C)
            elif pf.ndim == 3:
                # [T, K, C]（已经是 token 形式）
                arr = pf[:]
                T_frames = arr.shape[0]
            else:
                raise ValueError(f"未知 zarr 维度: {pf.ndim}")

            t = torch.from_numpy(arr.astype(np.float32))  # [T, K, C]

            # 可选：限制 token 数量（节省内存，加速训练）
            if self.max_tokens is not None and t.shape[1] > self.max_tokens:
                # 等间隔采样，保持空间均匀性
                idx = torch.linspace(0, t.shape[1] - 1, self.max_tokens).long()
                t = t[:, idx, :]

            tokens_by_model.append(t)  # [T, K_i, C_i]

        return tokens_by_model

    def _collect_samples(self) -> List[Tuple]:
        """构建所有 (obs_tokens_list, obs_state, action) 样本。"""
        samples = []

        for ep_idx, ep in enumerate(self.episodes):
            try:
                if ep_idx % 10 == 0:
                    print(f"  {ep_idx}/{len(self.episodes)}: {ep}")

                ep_num = int(ep.split('_')[1])
                tokens_list = self._load_episode_tokens(ep)   # List[4 × [T, K_i, C_i]]
                vector = self._load_vector(ep_num)             # [T_raw, 14]

                # 对齐帧数
                T = min(tokens_list[0].shape[0], len(vector))

                # 时间偏移
                states  = vector[:T-1]                          # [T-1, 14]
                actions = vector[1:T]                           # [T-1, 14]
                toks = [t[:T-1] for t in tokens_list]           # [T-1, K_i, C_i] per model

                T_eff = T - 1
                if T_eff < self.n_obs_steps + self.horizon:
                    print(f"  跳过 {ep}: T_eff={T_eff} < {self.n_obs_steps + self.horizon}")
                    continue

                for t_idx in range(T_eff - self.n_obs_steps - self.horizon + 1):
                    obs_toks = [
                        tok[t_idx:t_idx + self.n_obs_steps]  # [To, K_i, C_i]
                        for tok in toks
                    ]
                    obs_state  = states[t_idx:t_idx + self.n_obs_steps]   # [To, 14]
                    act_window = actions[t_idx + self.n_obs_steps:
                                         t_idx + self.n_obs_steps + self.horizon]  # [Ta, 14]
                    samples.append((obs_toks, obs_state, act_window))

            except Exception as e:
                print(f"  跳过 {ep}: {e}")
                import traceback
                traceback.print_exc()
                continue

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        obs_toks, agent_pos, action = self.samples[idx]
        # obs_toks: List[4 × Tensor[To, K_i, C_i]]
        return {
            'obs_tokens': obs_toks,  # 列表，不做 collate（由 collate_fn 处理）
            'agent_pos': agent_pos,  # [To, 14]
            'action': action,        # [Ta, 14]
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """自定义 collate，处理 obs_tokens 列表。"""
    B = len(batch)
    # 对每个模型分别 stack
    n_models = len(batch[0]['obs_tokens'])
    obs_tokens_batched = []
    for mi in range(n_models):
        stacked = torch.stack([batch[b]['obs_tokens'][mi] for b in range(B)], dim=0)
        obs_tokens_batched.append(stacked)   # [B, To, K_i, C_i]

    agent_pos = torch.stack([batch[b]['agent_pos'] for b in range(B)], dim=0)
    action    = torch.stack([batch[b]['action']    for b in range(B)], dim=0)

    return {
        'obs_tokens': obs_tokens_batched,
        'agent_pos': agent_pos,
        'action': action,
    }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Depth-Guided Cross-Attention Fusion 训练")
    parser.add_argument('--config', type=str, required=True, help='YAML 配置文件路径')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    print("=" * 65)
    print("Depth-Guided Cross-Attention Fusion + proprio 训练 (离线版)")
    print("=" * 65)

    # 设备
    gpu_ids = config['device']['gpu_ids']
    if isinstance(gpu_ids, list):
        gpu_id = gpu_ids[0]
    else:
        gpu_id = int(gpu_ids)
    device = f'cuda:{gpu_id}'
    print(f"Device: {device}")

    task_name = config['data']['tasks']
    if isinstance(task_name, list):
        task_name = task_name[0]
    task_config   = config['data'].get('task_config', 'demo_clean')
    expert_num    = config.get('checkpoint', {}).get('expert_data_num', 50)
    data_root     = config['data'].get('robotwin_data_root', '/home/gl/RoboTwin/data')
    horizon       = int(config['data']['horizon'])
    n_obs_steps   = int(config['data']['n_obs_steps'])
    max_tokens    = config['encoder'].get('max_tokens', None)  # 可选截断

    # ──────────────── 1. 数据集 ────────────────
    print("\n1. 创建数据集...")
    dataset = DepthGuidedDataset(
        vis_zarr_roots=config['data']['vis_zarr_roots'],
        robotwin_data_root=data_root,
        task_name=task_name,
        task_config=task_config,
        horizon=horizon,
        n_obs_steps=n_obs_steps,
        expert_data_num=expert_num,
        camera_name=config['data'].get('camera_name', 'head_camera'),
        max_tokens=max_tokens,
        zarr_expert_num=config['checkpoint'].get('zarr_expert_num', expert_num),
    )
    print(f"✓ Dataset: {len(dataset)} 样本")

    dataloader = DataLoader(
        dataset,
        batch_size=config['train']['batch_size'],
        shuffle=True,
        num_workers=config['train'].get('num_workers', 4),
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )

    # ──────────────── 2. 模型 ────────────────
    print("\n2. 创建 Depth-Guided Encoder + Policy...")
    enc_cfg = config['encoder']
    fusion_encoder = DepthGuidedFusionEncoder(
        semantic_in_dims=tuple(enc_cfg.get('semantic_in_dims', [1024, 2048, 768])),
        geometric_in_dim=int(enc_cfg.get('geometric_in_dim', 2048)),
        proj_dim=int(enc_cfg.get('proj_dim', 512)),
        n_heads=int(enc_cfg.get('n_heads', 8)),
        n_layers=int(enc_cfg.get('n_layers', 2)),
        out_dim=int(enc_cfg.get('out_dim', 1280)),
        semantic_fusion=str(enc_cfg.get('semantic_fusion', 'concat_proj')),
        pool=str(enc_cfg.get('pool', 'mean')),
        dropout=float(enc_cfg.get('dropout', 0.1)),
    ).to(device)

    policy = DepthGuidedPolicy(
        fusion_encoder=fusion_encoder,
        proprio_dim=14,
        action_dim=14,
        horizon=horizon,
        n_obs_steps=n_obs_steps,
        num_inference_steps=int(config['policy'].get('num_inference_steps', 100)),
    ).to(device)

    total_params = sum(p.numel() for p in policy.parameters()) / 1e6
    encoder_params = sum(p.numel() for p in fusion_encoder.parameters()) / 1e6
    print(f"✓ 模型总参数: {total_params:.2f}M  (encoder: {encoder_params:.2f}M)")
    print(f"  obs_encoder 输入维度: {(enc_cfg.get('out_dim', 1280) + 14) * n_obs_steps}")

    # ──────────────── 3. Fit normalizer ────────────────
    print("\n3. Fit normalizer...")
    all_actions   = torch.stack([s[2] for s in dataset.samples])  # [N, Ta, 14]
    all_agent_pos = torch.stack([s[1] for s in dataset.samples])  # [N, To, 14]

    policy.normalizer.fit(
        {'action': all_actions, 'agent_pos': all_agent_pos},
        last_n_dims=1,
        mode='limits',
        output_min=-1.0,
        output_max=1.0,
    )
    try:
        policy.normalizer.to(device)
    except Exception:
        pass

    # 时间偏移验证
    s0_state  = dataset.samples[0][1]
    s0_action = dataset.samples[0][2]
    if torch.allclose(s0_state[0], s0_action[0], atol=1e-5):
        print("  ⚠ 警告: state[0] ≈ action[0]，时间偏移可能有问题！")
    else:
        diff = (s0_state[0] - s0_action[0]).abs().mean().item()
        print(f"  ✓ state[0] ≠ action[0]，平均差异={diff:.6f}（时间偏移正确）")

    # ──────────────── 4. 优化器 ────────────────
    train_cfg = config['train']
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=float(train_cfg['lr']),
        weight_decay=float(train_cfg.get('weight_decay', 1e-6)),
    )

    # 余弦退火 LR（与其他实验一致）
    total_epochs = int(train_cfg['epochs'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_epochs, eta_min=float(train_cfg['lr']) * 0.01
    )

    # ──────────────── 5. 训练循环 ────────────────
    print("\n4. 开始训练...")
    save_dir = Path(config['checkpoint']['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)

    save_every  = int(config['checkpoint'].get('save_every', 100))
    best_loss   = float('inf')
    avg_loss    = float('inf')

    for epoch in range(total_epochs):
        policy.train()
        epoch_loss = 0.0
        n_batches  = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{total_epochs}", leave=False)
        for batch in pbar:
            # 移动 tokens 到 device
            tokens_list = [t.to(device, non_blocking=True) for t in batch['obs_tokens']]
            agent_pos   = batch['agent_pos'].to(device, non_blocking=True)
            action      = batch['action'].to(device, non_blocking=True)

            loss = policy.compute_loss(tokens_list, agent_pos, action)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches  += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{scheduler.get_last_lr()[0]:.2e}'})

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        print(f"Epoch {epoch+1:4d}/{total_epochs}: Loss={avg_loss:.6f}  LR={scheduler.get_last_lr()[0]:.2e}")

        # 保存 checkpoint
        if (epoch + 1) % save_every == 0:
            ckpt = {
                'policy': policy.state_dict(),
                'normalizer': policy.normalizer.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'config': config,
                'loss': avg_loss,
                'policy_class': 'DepthGuidedPolicy',
                'policy_type': 'depth_guided',
                # 保存 encoder 超参以便 deploy 时重建
                'encoder_cfg': {
                    'semantic_in_dims': list(enc_cfg.get('semantic_in_dims', [1024, 2048, 768])),
                    'geometric_in_dim': int(enc_cfg.get('geometric_in_dim', 2048)),
                    'proj_dim': int(enc_cfg.get('proj_dim', 512)),
                    'n_heads': int(enc_cfg.get('n_heads', 8)),
                    'n_layers': int(enc_cfg.get('n_layers', 2)),
                    'out_dim': int(enc_cfg.get('out_dim', 1280)),
                    'semantic_fusion': str(enc_cfg.get('semantic_fusion', 'concat_proj')),
                    'pool': str(enc_cfg.get('pool', 'mean')),
                    'dropout': float(enc_cfg.get('dropout', 0.1)),
                },
            }
            ckpt_path = save_dir / f"{epoch+1}.ckpt"
            torch.save(ckpt, ckpt_path)
            print(f"  ✓ Saved: {ckpt_path}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_path = save_dir / "best.ckpt"
                torch.save(ckpt, best_path)
                print(f"  ✓ Best model: {best_path}")

    # 最终保存
    ckpt = {
        'policy': policy.state_dict(),
        'normalizer': policy.normalizer.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': total_epochs - 1,
        'config': config,
        'loss': avg_loss,
        'policy_class': 'DepthGuidedPolicy',
        'policy_type': 'depth_guided',
        'encoder_cfg': {
            'semantic_in_dims': list(enc_cfg.get('semantic_in_dims', [1024, 2048, 768])),
            'geometric_in_dim': int(enc_cfg.get('geometric_in_dim', 2048)),
            'proj_dim': int(enc_cfg.get('proj_dim', 512)),
            'n_heads': int(enc_cfg.get('n_heads', 8)),
            'n_layers': int(enc_cfg.get('n_layers', 2)),
            'out_dim': int(enc_cfg.get('out_dim', 1280)),
            'semantic_fusion': str(enc_cfg.get('semantic_fusion', 'concat_proj')),
            'pool': str(enc_cfg.get('pool', 'mean')),
            'dropout': float(enc_cfg.get('dropout', 0.1)),
        },
    }
    final_path = save_dir / f"{total_epochs}.ckpt"
    torch.save(ckpt, final_path)
    print(f"\n✓ 最终 ckpt: {final_path}")
    print(f"训练完成！最佳 Loss: {best_loss:.6f}")


if __name__ == '__main__':
    main()
