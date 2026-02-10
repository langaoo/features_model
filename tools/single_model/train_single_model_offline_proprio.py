#!/usr/bin/env python3
"""
单模型 + 本体感知(proprio) 训练
直接使用单个视觉模型的特征 + agent_pos，训练 DP Head
agent_pos 采用 DP-aligned 方式：归一化后直接拼接到视觉特征，不使用独立 MLP

与 train_single_model_offline.py 的区别：
- 新增 agent_pos 加载（来自 HDF5 的 joint_action/vector）
- obs_dim = vis_dim + proprio_dim (如 768 + 14 = 782)
- 时间对齐：state[t]=vector[t], action[t]=vector[t+1]，共 T-1 有效帧
- normalizer 同时 fit action 和 agent_pos（与原版 DP 一致）
"""
import torch
import torch.nn as nn
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


class SingleModelProprioPolicy(nn.Module):
    """
    单模型 + proprio + Diffusion Policy
    
    与原版 DP 对齐的设计：
    - agent_pos 归一化后直接拼接到视觉特征（不使用独立 MLP）
    - normalizer 同时管理 action 和 agent_pos 的归一化
    - obs_encoder 输入 = n_obs_steps * (vis_dim + proprio_dim)
    """
    
    def __init__(
        self,
        vis_dim: int,
        proprio_dim: int = 14,
        action_dim: int = 14,
        horizon: int = 8,
        n_obs_steps: int = 3,
        num_inference_steps: int = 100,
    ):
        super().__init__()
        
        self.vis_dim = vis_dim
        self.proprio_dim = proprio_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.num_inference_steps = num_inference_steps
        self.use_proprio = True
        
        # LinearNormalizer：同时管理 action 和 agent_pos
        self.normalizer = LinearNormalizer()
        
        # 每步观测维度 = 视觉特征 + 本体感知
        per_step_dim = vis_dim + proprio_dim
        obs_encoder_dim = per_step_dim * n_obs_steps
        
        # 观测编码器（与原 SingleModelDPPolicy 保持相同结构）
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_encoder_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        
        # Diffusion UNet
        self.noise_pred_net = ConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=256,
            diffusion_step_embed_dim=128,
            down_dims=[256, 512, 1024],
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
        )
        
        # 噪声调度器
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=100,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon'
        )
    
    def compute_loss(self, vis_feat, agent_pos, actions):
        """
        训练时的 loss 计算
        vis_feat: [B, To, vis_dim] 视觉特征
        agent_pos: [B, To, proprio_dim] 本体感知（已归一化）
        actions: [B, Ta, A] 原始动作（未归一化）
        """
        B = vis_feat.shape[0]
        device = vis_feat.device
        
        # 归一化 action
        nactions = self.normalizer.normalize({'action': actions})['action'].to(device)
        
        # 归一化 agent_pos 并拼接到视觉特征
        nagent_pos = self.normalizer.normalize({'agent_pos': agent_pos})['agent_pos'].to(device)
        obs_combined = torch.cat([vis_feat, nagent_pos], dim=-1)  # [B, To, vis_dim + proprio_dim]
        
        # 编码观测
        obs_flat = obs_combined.reshape(B, -1)
        obs_cond = self.obs_encoder(obs_flat)
        
        # 随机采样时间步
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (B,), device=device
        ).long()
        
        # 添加噪声
        noise = torch.randn_like(nactions)
        noisy_actions = self.noise_scheduler.add_noise(nactions, noise, timesteps)
        
        # 预测噪声
        noise_pred = self.noise_pred_net(
            noisy_actions, timesteps, global_cond=obs_cond
        )
        
        # MSE loss
        loss = nn.functional.mse_loss(noise_pred, noise)
        return loss
    
    def forward(self, vis_feat, agent_pos=None):
        """
        推理时的去噪采样
        vis_feat: [B, To, vis_dim]
        agent_pos: [B, To, proprio_dim]（可选，推理时提供）
        返回: actions [B, Ta, A] - 原始尺度
        """
        B = vis_feat.shape[0]
        device = vis_feat.device
        
        # 归一化 agent_pos 并拼接
        if agent_pos is not None:
            nagent_pos = self.normalizer.normalize({'agent_pos': agent_pos})['agent_pos'].to(device)
            obs_combined = torch.cat([vis_feat, nagent_pos], dim=-1)
        else:
            # 无 proprio 时用零填充
            zeros = torch.zeros(B, vis_feat.shape[1], self.proprio_dim, device=device)
            obs_combined = torch.cat([vis_feat, zeros], dim=-1)
        
        # 编码观测
        obs_flat = obs_combined.reshape(B, -1)
        obs_cond = self.obs_encoder(obs_flat)
        
        # 初始化随机噪声
        action = torch.randn((B, self.horizon, self.action_dim), device=device)
        
        # 去噪采样
        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        
        for t in self.noise_scheduler.timesteps:
            noise_pred = self.noise_pred_net(
                action,
                t.unsqueeze(0).expand(B).to(device),
                global_cond=obs_cond
            )
            action = self.noise_scheduler.step(noise_pred, t, action).prev_sample
        
        # 反归一化
        action = self.normalizer.unnormalize({'action': action})['action']
        
        return action


class SingleModelProprioDataset(Dataset):
    """
    单模型 + proprio 离线数据集
    
    数据来源：
    - 视觉特征：zarr 文件（单模型的 per_frame_features）
    - 动作 + agent_pos：HDF5 文件（joint_action/vector）
    
    时间对齐（与原版 DP 一致）：
    - state[t] = vector[t]      (frames 0 .. T-2)
    - action[t] = vector[t+1]   (frames 1 .. T-1)
    - 共 T-1 有效帧
    """
    
    def __init__(
        self,
        vis_zarr_root: str,
        robotwin_data_root: str,
        task_name: str,
        task_config: str,
        horizon: int = 8,
        n_obs_steps: int = 3,
        expert_data_num: int = 50,
        camera_name: str = 'head_camera',
        model_name: str = 'dinov3',
    ):
        super().__init__()
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.model_name = model_name
        
        self.vis_zarr_root = vis_zarr_root
        self.task_name = task_name
        self.task_config = task_config
        self.camera_name = camera_name
        self.expert_data_num = expert_data_num
        
        self.raw_data_root = os.path.join(robotwin_data_root, task_name, task_config, 'data')
        self.episodes = [f'episode_{i}' for i in range(expert_data_num)]
        
        print(f"[{model_name}+proprio] 收集数据样本...")
        self.samples = self._collect_samples()
        print(f"[{model_name}+proprio] 共 {len(self.samples)} 个样本")
    
    def _load_episode_vector(self, episode: str):
        """加载单 episode 的完整 vector（用于 state 和 action）"""
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
        return torch.from_numpy(vector).float()  # [T_raw, 14]
    
    def _load_episode_feat(self, episode: str):
        """加载单 episode 的视觉特征 [T, C]"""
        zarr_subdir = f"{self.task_name}-{self.task_config}-{self.expert_data_num}_sapien_{self.camera_name}"
        zarr_path = os.path.join(self.vis_zarr_root, zarr_subdir, f"{episode}.zarr")
        
        feat_zarr = zarr.open(zarr_path, mode='r')
        feat = feat_zarr['per_frame_features'][:]
        
        # 处理 5D 特征：空间平均池化
        if feat.ndim == 5:  # [T, 1, Hf, Wf, C]
            feat = feat[:, 0, :, :, :]  # [T, Hf, Wf, C]
            T, Hf, Wf, C = feat.shape
            feat = feat.reshape(T, Hf * Wf, C).mean(axis=1)  # [T, C]
        elif feat.ndim == 4:  # [T, Hf, Wf, C]
            T, Hf, Wf, C = feat.shape
            feat = feat.reshape(T, Hf * Wf, C).mean(axis=1)
        elif feat.ndim == 3:  # [T, tokens, C]
            feat = feat.mean(axis=1)
        
        return torch.from_numpy(feat).float()
    
    def _collect_samples(self):
        """收集所有时序对齐的样本"""
        samples = []
        all_states = []
        all_actions = []
        
        for ep_idx, ep in enumerate(self.episodes):
            try:
                feats = self._load_episode_feat(ep)     # [T_feat, vis_dim]
                vector = self._load_episode_vector(ep)   # [T_raw, 14]
                
                T_feat = len(feats)
                T_raw = len(vector)
                T = min(T_feat, T_raw)
                
                # ✅ 时间对齐（与原版 DP 一致）：
                # state[t] = vector[t]      (t = 0 .. T-2)
                # action[t] = vector[t+1]   (t = 0 .. T-2)
                # vis_feat[t] 对应 state[t]
                states = vector[:T-1]    # [T-1, 14]  — agent_pos
                actions = vector[1:T]    # [T-1, 14]  — 下一帧的关节状态作为动作
                vis_feats = feats[:T-1]  # [T-1, vis_dim]
                
                T_eff = T - 1  # 有效帧数
                if T_eff < self.n_obs_steps + self.horizon:
                    print(f"  跳过 {ep}: 有效帧数 {T_eff} < {self.n_obs_steps + self.horizon}")
                    continue
                
                # 收集用于 normalizer 的数据
                all_states.append(states)
                all_actions.append(actions)
                
                # 滑动窗口生成样本
                for t in range(T_eff - self.n_obs_steps - self.horizon + 1):
                    obs_vis = vis_feats[t:t+self.n_obs_steps]           # [To, vis_dim]
                    obs_state = states[t:t+self.n_obs_steps]            # [To, 14]
                    act_window = actions[t+self.n_obs_steps:t+self.n_obs_steps+self.horizon]  # [Ta, 14]
                    samples.append((obs_vis, obs_state, act_window))
                    
            except Exception as e:
                print(f"  跳过 {ep}: {e}")
                continue
        
        # 保存全局统计信息用于 normalizer
        if all_states:
            self.all_states = torch.cat(all_states, dim=0)   # [N, 14]
            self.all_actions = torch.cat(all_actions, dim=0)  # [N, 14]
        else:
            self.all_states = torch.zeros(1, 14)
            self.all_actions = torch.zeros(1, 14)
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        vis_feat, agent_pos, action = self.samples[idx]
        return {'vis_feat': vis_feat, 'agent_pos': agent_pos, 'action': action}


def train_single_model_proprio(config):
    """训练单模型 + proprio"""
    device = torch.device(f"cuda:{config['device']['gpu_id']}" if torch.cuda.is_available() else "cpu")
    
    # 1. 创建数据集
    print("\n1. 加载数据集...")
    dataset = SingleModelProprioDataset(
        vis_zarr_root=config['data']['vis_zarr_root'],
        robotwin_data_root=config['data']['robotwin_data_root'],
        task_name=config['data']['task_name'],
        task_config=config['data']['task_config'],
        horizon=config['data']['horizon'],
        n_obs_steps=config['data']['n_obs_steps'],
        expert_data_num=config['data']['expert_data_num'],
        camera_name=config['data']['camera_name'],
        model_name=config['model']['name'],
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config['train']['batch_size'],
        shuffle=True,
        num_workers=config['train']['num_workers'],
        pin_memory=True,
    )
    
    # 2. 创建 policy
    print("\n2. 创建 Policy...")
    vis_dim = dataset.samples[0][0].shape[-1]  # 视觉特征维度
    proprio_dim = dataset.samples[0][1].shape[-1]  # 本体感知维度
    print(f"  vis_dim = {vis_dim}, proprio_dim = {proprio_dim}")
    print(f"  per_step_dim = {vis_dim + proprio_dim}")
    print(f"  obs_encoder_input = {(vis_dim + proprio_dim) * config['data']['n_obs_steps']}")
    
    policy = SingleModelProprioPolicy(
        vis_dim=vis_dim,
        proprio_dim=proprio_dim,
        action_dim=14,
        horizon=config['data']['horizon'],
        n_obs_steps=config['data']['n_obs_steps'],
        num_inference_steps=config['policy']['num_inference_steps'],
    ).to(device)
    
    # 3. Fit normalizer（与原版 DP 一致：同时 fit action 和 agent_pos）
    print("\n3. Fit normalizer...")
    
    # 使用 dataset 收集的全局统计
    # action 的维度是 [N_total, 14]，需要包装成 [N, Ta, A] 来 fit
    # agent_pos 的维度是 [N_total, 14]，需要包装成 [N, To, D] 来 fit
    all_actions = torch.stack([s[2] for s in dataset.samples])       # [N, Ta, 14]
    all_agent_pos = torch.stack([s[1] for s in dataset.samples])     # [N, To, 14]
    
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
    
    print(f"  Action normalizer: min={policy.normalizer['action'].params_dict['input_stats'].min[:3]}...")
    print(f"  Action normalizer: max={policy.normalizer['action'].params_dict['input_stats'].max[:3]}...")
    print(f"  AgentPos normalizer: min={policy.normalizer['agent_pos'].params_dict['input_stats'].min[:3]}...")
    print(f"  AgentPos normalizer: max={policy.normalizer['agent_pos'].params_dict['input_stats'].max[:3]}...")
    
    # 4. 验证 state ≠ action（时间偏移是否生效）
    print("\n  验证时间偏移...")
    sample_state = dataset.samples[0][1]  # [To, 14]
    sample_action = dataset.samples[0][2]  # [Ta, 14]
    if torch.allclose(sample_state[0], sample_action[0], atol=1e-5):
        print("  ⚠ 警告: state[0] ≈ action[0]，时间偏移可能有问题!")
    else:
        diff = (sample_state[0] - sample_action[0]).abs().mean().item()
        print(f"  ✓ state[0] ≠ action[0]，平均差异 = {diff:.6f}（时间偏移正确）")
    
    # 5. 优化器
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=config['train']['lr'],
        weight_decay=config['train']['weight_decay'],
    )
    
    # 6. 训练
    print("\n4. 开始训练...")
    save_dir = Path(config['checkpoint']['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    
    best_loss = float('inf')
    
    for epoch in range(config['train']['epochs']):
        policy.train()
        epoch_loss = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['train']['epochs']}")
        
        for batch in pbar:
            vis_feat = batch['vis_feat'].to(device)     # [B, To, vis_dim]
            agent_pos = batch['agent_pos'].to(device)   # [B, To, 14]
            action = batch['action'].to(device)         # [B, Ta, 14]
            
            loss = policy.compute_loss(vis_feat, agent_pos, action)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.6f}")
        
        # 保存 checkpoint
        if (epoch + 1) % config['checkpoint']['save_every'] == 0:
            ckpt_data = {
                'policy': policy.state_dict(),
                'normalizer': policy.normalizer.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'config': config,
                'loss': avg_loss,
                'policy_class': 'SingleModelProprioPolicy',
                'vis_dim': vis_dim,
                'proprio_dim': proprio_dim,
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    train_single_model_proprio(config)


if __name__ == '__main__':
    main()
