#!/usr/bin/env python3
"""
3模型直接融合 + 本体感知(proprio) 训练 【离线版】
与 train_direct_fusion_offline_proprio.py 的区别:
- 只用3个RGB模型: CroCo(1024d) + VGGT(2048d) + DINOv3(768d)，去掉 DA3
- in_dims = (1024, 2048, 768)，M = 3
- 作为 depth_guided (3RGB + DA3深度信息) 的对比基线
- policy_class 仍保存为 'DirectFusionProprioPolicy'（兼容 deploy_direct_fusion_policy.py）
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
from typing import Dict, Any, List
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
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# 导入正版DP
HAS_OFFICIAL_DP = False
try:
    DP_OUTER = Path(__file__).parent.parent.parent / "third_party" / "DP" / "diffusion_policy"
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


class SimpleFusionEncoder(nn.Module):
    """简单融合编码器: 3模型特征 → 融合 → 输出"""
    
    def __init__(
        self,
        in_dims=(1024, 2048, 768),   # 3模型: CroCo, VGGT, DINOv3
        fusion_type='weighted',
        out_dim=1280,
    ):
        super().__init__()
        self.in_dims = in_dims
        self.fusion_type = fusion_type
        self.out_dim = out_dim
        
        if fusion_type == 'weighted':
            self.projectors = nn.ModuleList([
                nn.Linear(dim, out_dim) for dim in in_dims
            ])
            self.fusion_weights = nn.Parameter(torch.ones(len(in_dims)) / len(in_dims))
        elif fusion_type == 'concat':
            total_dim = sum(in_dims)
            self.projector = nn.Sequential(
                nn.Linear(total_dim, out_dim * 2),
                nn.ReLU(),
                nn.Linear(out_dim * 2, out_dim),
                nn.LayerNorm(out_dim),
            )
        elif fusion_type == 'mean':
            self.projectors = nn.ModuleList([
                nn.Linear(dim, out_dim) for dim in in_dims
            ])
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")
    
    def forward(self, x):
        """
        x: [B, To, M, C_max]  M=3
        Returns: [B, To, out_dim]
        """
        B, To, M, _ = x.shape
        
        if self.fusion_type == 'weighted':
            feats = []
            for i, proj in enumerate(self.projectors):
                feat_i = x[:, :, i, :self.in_dims[i]].reshape(B * To, self.in_dims[i])
                feat_i = proj(feat_i).reshape(B, To, self.out_dim)
                feats.append(feat_i)
            feats = torch.stack(feats, dim=2)  # [B, To, M, out_dim]
            weights = torch.softmax(self.fusion_weights, dim=0)
            out = (feats * weights.view(1, 1, M, 1)).sum(dim=2)
        elif self.fusion_type == 'concat':
            feats = [x[:, :, i, :self.in_dims[i]] for i in range(M)]
            feats_concat = torch.cat(feats, dim=-1).reshape(B * To, -1)
            out = self.projector(feats_concat).reshape(B, To, self.out_dim)
        elif self.fusion_type == 'mean':
            feats = []
            for i, proj in enumerate(self.projectors):
                feat_i = x[:, :, i, :self.in_dims[i]].reshape(B * To, self.in_dims[i])
                feat_i = proj(feat_i).reshape(B, To, self.out_dim)
                feats.append(feat_i)
            out = torch.stack(feats, dim=0).mean(dim=0)
        
        return out


class DirectFusionProprioPolicy(nn.Module):
    """
    3模型直接融合 + proprio + Diffusion Policy

    与4模型版完全一致，只是 fusion_encoder 使用3个模型。
    policy_class 保存为 'DirectFusionProprioPolicy'，与 deploy_direct_fusion_policy.py 兼容。
    """
    
    def __init__(
        self,
        fusion_encoder: SimpleFusionEncoder,
        proprio_dim: int = 14,
        action_dim: int = 14,
        horizon: int = 8,
        n_obs_steps: int = 3,
        num_inference_steps: int = 100,
    ):
        super().__init__()
        
        if not HAS_OFFICIAL_DP:
            raise RuntimeError("正版DP未加载")
        
        self.fusion_encoder = fusion_encoder
        self.proprio_dim = proprio_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.num_inference_steps = num_inference_steps
        self.use_proprio = True
        
        self.normalizer = LinearNormalizer()
        
        # 每步维度 = 融合后视觉 + proprio
        per_step_dim = fusion_encoder.out_dim + proprio_dim
        obs_encoder_dim = per_step_dim * n_obs_steps
        
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
            prediction_type='epsilon'
        )
    
    def compute_loss(self, rgb_feats, agent_pos, actions):
        """
        rgb_feats: [B, To, M, C_max]   M=3
        agent_pos: [B, To, proprio_dim]
        actions: [B, Ta, A]
        """
        B = rgb_feats.shape[0]
        device = rgb_feats.device
        
        nactions = self.normalizer.normalize({'action': actions})['action'].to(device)
        nagent_pos = self.normalizer.normalize({'agent_pos': agent_pos})['agent_pos'].to(device)
        
        fused = self.fusion_encoder(rgb_feats)  # [B, To, fuse_dim]
        
        obs_combined = torch.cat([fused, nagent_pos], dim=-1)
        obs_flat = obs_combined.reshape(B, -1)
        obs_cond = self.obs_encoder(obs_flat)
        
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (B,), device=device
        ).long()
        
        noise = torch.randn_like(nactions)
        noisy_actions = self.noise_scheduler.add_noise(nactions, noise, timesteps)
        
        noise_pred = self.noise_pred_net(
            noisy_actions, timesteps, global_cond=obs_cond
        )
        
        loss = nn.functional.mse_loss(noise_pred, noise)
        return loss
    
    def forward(self, rgb_feats, agent_pos=None):
        """推理"""
        B = rgb_feats.shape[0]
        device = rgb_feats.device
        
        fused = self.fusion_encoder(rgb_feats)
        
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
                action, t.unsqueeze(0).expand(B).to(device), global_cond=obs_cond
            )
            action = self.noise_scheduler.step(noise_pred, t, action).prev_sample
        
        action = self.normalizer.unnormalize({'action': action})['action']
        return action


class DirectFusion3ModelDataset(Dataset):
    """
    3模型直融 + proprio 离线数据集 (去掉DA3)
    """
    
    def __init__(
        self,
        vis_zarr_roots: List[str],
        robotwin_data_root: str,
        task_name: str,
        task_config: str,
        horizon: int = 8,
        n_obs_steps: int = 3,
        expert_data_num: int = 50,
        camera_name: str = 'head_camera',
    ):
        super().__init__()
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        # 3模型维度: CroCo, VGGT, DINOv3
        self.in_dims = [1024, 2048, 768]
        self.C_max = 2048
        self.M = 3
        
        assert len(vis_zarr_roots) == self.M, \
            f"期望3个zarr路径，得到{len(vis_zarr_roots)}个"
        
        self.vis_zarr_roots = vis_zarr_roots
        self.task_name = task_name
        self.task_config = task_config
        self.camera_name = camera_name
        self.expert_data_num = expert_data_num
        
        self.raw_data_root = os.path.join(robotwin_data_root, task_name, task_config, 'data')
        self.episodes = [f'episode_{i}' for i in range(expert_data_num)]
        
        print(f"[DirectFusion3Model+proprio] 收集数据样本...")
        self.samples = self._collect_samples()
        print(f"[DirectFusion3Model+proprio] 共 {len(self.samples)} 个样本")
    
    def _load_episode_vector(self, episode: str):
        """加载完整 vector"""
        ep_num = episode.split('_')[1]
        hdf5_path = os.path.join(self.raw_data_root, f'episode{ep_num}.hdf5')
        with h5py.File(hdf5_path, 'r') as f:
            if 'joint_action/vector' in f:
                vector = f['joint_action/vector'][:]
            elif 'joint_action/left_arm' in f:
                left = f['joint_action/left_arm'][:]
                right = f['joint_action/right_arm'][:]
                left_g = f['joint_action/left_gripper'][:] if 'joint_action/left_gripper' in f else np.zeros((left.shape[0], 1))
                right_g = f['joint_action/right_gripper'][:] if 'joint_action/right_gripper' in f else np.zeros((right.shape[0], 1))
                if left_g.ndim == 1:
                    left_g = left_g[:, None]
                if right_g.ndim == 1:
                    right_g = right_g[:, None]
                vector = np.concatenate([left, left_g, right, right_g], axis=-1)
            else:
                raise KeyError("HDF5中缺少joint_action数据")
        return torch.from_numpy(vector).float()
    
    def _load_episode_feat(self, episode: str):
        """加载3模型特征 [T, M, C_max]"""
        feat_list = []
        zarr_subdir = f"{self.task_name}-{self.task_config}-{self.expert_data_num}_sapien_{self.camera_name}"
        
        for mi in range(self.M):
            zarr_root = self.vis_zarr_roots[mi]
            zarr_path = os.path.join(zarr_root, zarr_subdir, f"{episode}.zarr")
            
            feat_zarr = zarr.open(zarr_path, mode='r')
            feat = feat_zarr['per_frame_features'][:]
            
            # 空间平均池化
            if feat.ndim == 5:  # [T, 1, Hf, Wf, C]
                feat = feat[:, 0, :, :, :]
                T, Hf, Wf, C = feat.shape
                feat = feat.reshape(T, Hf * Wf, C).mean(axis=1)
            elif feat.ndim == 4:
                feat = feat.mean(axis=(1, 2))
            elif feat.ndim == 3:
                feat = feat.mean(axis=1)
            
            feat = torch.from_numpy(feat).float()
            
            # padding 到 C_max
            pad_len = self.C_max - feat.shape[-1]
            if pad_len > 0:
                feat = torch.nn.functional.pad(feat, (0, pad_len))
            feat_list.append(feat)
        
        feats = torch.stack(feat_list, dim=0).transpose(0, 1)  # [T, M, C_max]
        return feats
    
    def _collect_samples(self):
        """收集带时间偏移的样本"""
        samples = []
        
        for ep_idx, ep in enumerate(self.episodes):
            try:
                if ep_idx % 10 == 0:
                    print(f"  处理: {ep_idx}/{len(self.episodes)}")
                
                feats = self._load_episode_feat(ep)    # [T_feat, M, C_max]
                vector = self._load_episode_vector(ep)  # [T_raw, 14]
                
                T = min(feats.shape[0], len(vector))
                
                # 时间对齐: state[t]=vector[t], action[t]=vector[t+1]
                states = vector[:T-1]      # [T-1, 14]
                actions = vector[1:T]      # [T-1, 14]
                vis_feats = feats[:T-1]    # [T-1, M, C_max]
                
                T_eff = T - 1
                if T_eff < self.n_obs_steps + self.horizon:
                    print(f"  跳过 {ep}: T_eff={T_eff} < {self.n_obs_steps + self.horizon}")
                    continue
                
                for t in range(T_eff - self.n_obs_steps - self.horizon + 1):
                    obs_vis = vis_feats[t:t+self.n_obs_steps]        # [To, M, C_max]
                    obs_state = states[t:t+self.n_obs_steps]         # [To, 14]
                    act_window = actions[t+self.n_obs_steps:t+self.n_obs_steps+self.horizon]  # [Ta, 14]
                    samples.append((obs_vis, obs_state, act_window))
                    
            except Exception as e:
                print(f"  跳过 {ep}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        vis_feat, agent_pos, action = self.samples[idx]
        return {'obs': vis_feat, 'agent_pos': agent_pos, 'action': action}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print("=" * 60)
    print("3模型直接融合 + proprio 训练 (离线版)")
    print("模型: CroCo(1024d) + VGGT(2048d) + DINOv3(768d)")
    print("=" * 60)
    
    gpu_id = config['device']['gpu_ids'][0] if isinstance(config['device']['gpu_ids'], list) else config['device']['gpu_ids']
    device = f'cuda:{gpu_id}'
    
    task_name = config['data']['tasks'][0] if isinstance(config['data']['tasks'], list) else config['data']['tasks']
    task_config = config['data'].get('task_config', 'demo_clean')
    expert_data_num = config.get('checkpoint', {}).get('expert_data_num', 50)
    robotwin_data_root = config['data'].get('robotwin_data_root', '/home/gl/RoboTwin/data')
    
    # 1. 创建数据集
    print("\n1. 创建数据集...")
    dataset = DirectFusion3ModelDataset(
        vis_zarr_roots=config['data']['vis_zarr_roots'],
        robotwin_data_root=robotwin_data_root,
        task_name=task_name,
        task_config=task_config,
        horizon=config['data']['horizon'],
        n_obs_steps=config['data']['n_obs_steps'],
        expert_data_num=expert_data_num,
        camera_name=config['data']['camera_name'],
    )
    print(f"✓ Dataset: {len(dataset)} samples")
    
    dataloader = DataLoader(
        dataset,
        batch_size=config['train']['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    
    # 2. 创建模型
    print("\n2. 创建模型 (3模型)...")
    fusion_encoder = SimpleFusionEncoder(
        in_dims=(1024, 2048, 768),   # CroCo, VGGT, DINOv3
        fusion_type=config['fusion']['type'],
        out_dim=config['fusion']['out_dim'],
    ).to(device)
    
    policy = DirectFusionProprioPolicy(
        fusion_encoder=fusion_encoder,
        proprio_dim=14,
        action_dim=14,
        horizon=config['data']['horizon'],
        n_obs_steps=config['data']['n_obs_steps'],
        num_inference_steps=config['policy']['num_inference_steps'],
    ).to(device)
    
    print(f"✓ 模型参数: {sum(p.numel() for p in policy.parameters()) / 1e6:.2f}M")
    print(f"  obs_encoder 输入维度: {(config['fusion']['out_dim'] + 14) * config['data']['n_obs_steps']}")
    
    # 3. Fit normalizer
    print("\n3. Fit normalizer...")
    all_actions = torch.stack([s[2] for s in dataset.samples])
    all_agent_pos = torch.stack([s[1] for s in dataset.samples])
    
    policy.normalizer.fit(
        {'action': all_actions, 'agent_pos': all_agent_pos},
        last_n_dims=1,
        mode='limits',
        output_min=-1.0,
        output_max=1.0
    )
    try:
        policy.normalizer.to(device)
    except Exception:
        pass
    
    print(f"  Action: min={policy.normalizer['action'].params_dict['input_stats'].min[:3]}...")
    print(f"  Action: max={policy.normalizer['action'].params_dict['input_stats'].max[:3]}...")
    
    # 验证时间偏移
    sample_state = dataset.samples[0][1]
    sample_action = dataset.samples[0][2]
    if torch.allclose(sample_state[0], sample_action[0], atol=1e-5):
        print("  ⚠ 警告: state[0] ≈ action[0]，时间偏移可能有问题!")
    else:
        diff = (sample_state[0] - sample_action[0]).abs().mean().item()
        print(f"  ✓ state[0] ≠ action[0]，平均差异 = {diff:.6f}（时间偏移正确）")
    
    # 4. 优化器
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=config['train']['lr'],
        weight_decay=config['train']['weight_decay'],
    )
    
    # 5. 训练
    print("\n4. 开始训练...")
    save_dir = Path(config['checkpoint']['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    
    best_loss = float('inf')
    avg_loss = float('inf')
    
    for epoch in range(config['train']['epochs']):
        policy.train()
        epoch_loss = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['train']['epochs']}")
        
        for batch in pbar:
            obs = batch['obs'].to(device, non_blocking=True)
            agent_pos = batch['agent_pos'].to(device, non_blocking=True)
            action = batch['action'].to(device, non_blocking=True)
            
            loss = policy.compute_loss(obs, agent_pos, action)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.6f}")
        
        if (epoch + 1) % config['checkpoint']['save_every'] == 0:
            ckpt_data = {
                'policy': policy.state_dict(),
                'normalizer': policy.normalizer.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'config': config,
                'loss': avg_loss,
                'policy_class': 'DirectFusionProprioPolicy',
                # 3模型标记，deploy 侧读取 in_dims
                'fusion_in_dims': [1024, 2048, 768],
                'fusion_n_models': 3,
            }
            
            ckpt_path = save_dir / f"{epoch+1}.ckpt"
            torch.save(ckpt_data, ckpt_path)
            print(f"✓ Saved: {ckpt_path}")
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_path = save_dir / "best.ckpt"
                torch.save(ckpt_data, best_path)
                print(f"✓ Best model saved: {best_path}")
    
    print("\n训练完成!")
    print(f"  最终 Loss: {avg_loss:.6f}")
    print(f"  最佳 Loss: {best_loss:.6f}")
    print(f"  Checkpoint 目录: {save_dir}")


if __name__ == '__main__':
    main()
