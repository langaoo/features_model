#!/usr/bin/env python3
"""
单模型训练 - 测试是否是融合破坏了语义
直接使用单个视觉模型的特征训练DP Head
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


class SingleModelDPPolicy(nn.Module):
    """单模型 + Diffusion Policy + LinearNormalizer"""
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int = 14,
        horizon: int = 8,
        n_obs_steps: int = 2,
        num_inference_steps: int = 100,
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.num_inference_steps = num_inference_steps
        
        # ✅ 使用LinearNormalizer自动管理归一化
        self.normalizer = LinearNormalizer()
        
        # 观测编码器
        obs_encoder_dim = obs_dim * n_obs_steps
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
    
    def compute_loss(self, obs, actions):
        """
        训练时的loss计算
        obs: [B, To, D] 观测特征
        actions: [B, Ta, A] 原始动作 (未归一化)
        """
        B = obs.shape[0]
        device = obs.device
        
        # ✅ 使用normalizer归一化action
        nactions = self.normalizer.normalize({'action': actions})['action'].to(device)
        
        # 编码观测
        obs_flat = obs.reshape(B, -1)
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
    
    def forward(self, obs):
        """
        推理时的去噪采样
        obs: [B, To, D]
        返回: actions [B, Ta, A] - 原始尺度
        """
        B = obs.shape[0]
        device = obs.device
        
        # 编码观测
        obs_flat = obs.reshape(B, -1)
        obs_cond = self.obs_encoder(obs_flat)
        
        # 初始化随机噪声
        action = torch.randn((B, self.horizon, self.action_dim), device=device)
        
        # 设置推理步数
        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        
        # 逐步去噪
        for t in self.noise_scheduler.timesteps:
            noise_pred = self.noise_pred_net(
                action,
                t.unsqueeze(0).expand(B).to(device),
                global_cond=obs_cond
            )
            action = self.noise_scheduler.step(noise_pred, t, action).prev_sample
        
        # ✅ 使用normalizer反归一化
        action = self.normalizer.unnormalize({'action': action})['action']
        
        return action


class SingleModelOfflineDataset(Dataset):
    """单模型离线数据集"""
    
    def __init__(
        self,
        vis_zarr_root: str,  # 单个模型的zarr根路径
        robotwin_data_root: str,
        task_name: str,
        task_config: str,
        horizon: int = 8,
        n_obs_steps: int = 2,
        expert_data_num: int = 50,
        camera_name: str = 'head_camera',
        model_name: str = 'dinov3',  # 模型名称用于日志
    ):
        super().__init__()
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.model_name = model_name
        
        # 加载离线zarr特征
        self.vis_zarr_root = vis_zarr_root
        self.task_name = task_name
        self.task_config = task_config
        self.camera_name = camera_name
        self.expert_data_num = expert_data_num
        
        # 加载动作标签
        self.raw_data_root = os.path.join(robotwin_data_root, task_name, task_config, 'data')
        self.episodes = [f'episode_{i}' for i in range(expert_data_num)]
        
        # 收集样本
        print(f"[{model_name}] 收集数据样本...")
        self.samples = self._collect_samples()
        print(f"[{model_name}] 共 {len(self.samples)} 个样本")
    
    def _load_episode_action(self, episode: str):
        """加载单episode的动作"""
        ep_num = episode.split('_')[1]
        hdf5_path = os.path.join(self.raw_data_root, f'episode{ep_num}.hdf5')
        with h5py.File(hdf5_path, 'r') as f:
            if 'joint_action/left_arm' in f:
                left = f['joint_action/left_arm'][:]
                right = f['joint_action/right_arm'][:]
                left_g = f['joint_action/left_gripper'][:] if 'joint_action/left_gripper' in f else np.zeros((left.shape[0], 1))
                right_g = f['joint_action/right_gripper'][:] if 'joint_action/right_gripper' in f else np.zeros((right.shape[0], 1))
                if left_g.ndim == 1:
                    left_g = left_g[:, None]
                if right_g.ndim == 1:
                    right_g = right_g[:, None]
                actions = np.concatenate([left, left_g, right, right_g], axis=-1)
            elif 'joint_action/vector' in f:
                actions = f['joint_action/vector'][:]
            else:
                raise KeyError("HDF5中缺少joint_action数据")
        return torch.from_numpy(actions).float()
    
    def _load_episode_feat(self, episode: str):
        """加载单episode的特征 [T, C]"""
        zarr_subdir = f"{self.task_name}-{self.task_config}-{self.expert_data_num}_sapien_{self.camera_name}"
        zarr_path = os.path.join(self.vis_zarr_root, zarr_subdir, f"{episode}.zarr")
        
        feat_zarr = zarr.open(zarr_path, mode='r')
        feat = feat_zarr['per_frame_features'][:]
        
        # ✅ 修复：正确处理5D特征，保留时序信息
        if feat.ndim == 5:  # [W, T, Hf, Wf, C]
            # 取每个窗口的第一帧（stride=1时，这等同于连续帧序列）
            feat = feat[:, 0, :, :, :]  # [W, Hf, Wf, C]
            # 对空间维度做全局平均池化
            W, Hf, Wf, C = feat.shape
            feat = feat.reshape(W, Hf * Wf, C).mean(axis=1)  # [W, C]
        elif feat.ndim == 4:  # [T, Hf, Wf, C]
            T, Hf, Wf, C = feat.shape
            feat = feat.reshape(T, Hf * Wf, C).mean(axis=1)  # [T, C]
        elif feat.ndim == 3:  # [T, tokens, C]
            feat = feat.mean(axis=1)  # [T, C]
        
        return torch.from_numpy(feat).float()
    
    def _collect_samples(self):
        """收集所有时序对齐的样本"""
        samples = []
        for ep_idx, ep in enumerate(self.episodes):
            try:
                feats = self._load_episode_feat(ep)  # [T, C]
                actions = self._load_episode_action(ep)  # [T, A]
                
                T = min(len(feats), len(actions))
                if T < self.n_obs_steps + self.horizon:
                    continue
                
                # 滑动窗口
                for t in range(T - self.n_obs_steps - self.horizon + 1):
                    obs_window = feats[t:t+self.n_obs_steps]  # [To, C]
                    act_window = actions[t+self.n_obs_steps:t+self.n_obs_steps+self.horizon]  # [Ta, A]
                    samples.append((obs_window, act_window))
                    
            except Exception as e:
                print(f"  跳过 {ep}: {e}")
                continue
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        obs, action = self.samples[idx]
        return {'obs': obs, 'action': action}


def train_single_model(config):
    """训练单模型"""
    device = torch.device(f"cuda:{config['device']['gpu_id']}" if torch.cuda.is_available() else "cpu")
    
    # 1. 创建数据集
    print("\n1. 加载数据集...")
    dataset = SingleModelOfflineDataset(
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
    
    # 2. 创建policy
    print("\n2. 创建Policy...")
    obs_dim = dataset.samples[0][0].shape[-1]  # 特征维度
    policy = SingleModelDPPolicy(
        obs_dim=obs_dim,
        action_dim=14,
        horizon=config['data']['horizon'],
        n_obs_steps=config['data']['n_obs_steps'],
        num_inference_steps=config['policy']['num_inference_steps'],
    ).to(device)
    
    # ✅ 使用normalizer自动fit统计值
    print("\n3. Fit normalizer...")
    all_actions = torch.stack([s[1] for s in dataset.samples])  # [N, Ta, A]
    policy.normalizer.fit(
        {'action': all_actions},
        last_n_dims=1,
        mode='limits',
        output_min=-1.0,
        output_max=1.0
    )
    try:
        policy.normalizer.to(device)
    except Exception:
        pass
    print(f"  Action stats: min={policy.normalizer['action'].params_dict['input_stats'].min[:3]}...")
    print(f"  Action stats: max={policy.normalizer['action'].params_dict['input_stats'].max[:3]}...")
    
    # 4. 创建优化器
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
    
    for epoch in range(config['train']['epochs']):
        policy.train()
        epoch_loss = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['train']['epochs']}")
        
        for batch in pbar:
            obs = batch['obs'].to(device)  # [B, To, C]
            action = batch['action'].to(device)  # [B, Ta, A]
            
            # 前向
            loss = policy.compute_loss(obs, action)
            
            # 反向
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.6f}")
        
        # 保存
        if (epoch + 1) % config['checkpoint']['save_every'] == 0:
            ckpt_path = save_dir / f"{epoch+1}.ckpt"
            torch.save({
                'policy': policy.state_dict(),
                'normalizer': policy.normalizer.state_dict(),  # ✅ 保存normalizer
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'config': config,
                'loss': avg_loss,
            }, ckpt_path)
            print(f"✓ Saved: {ckpt_path}")
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_path = save_dir / "best.ckpt"
                torch.save({
                    'policy': policy.state_dict(),
                    'normalizer': policy.normalizer.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'config': config,
                    'loss': avg_loss,
                }, best_path)
                print(f"✓ Best model saved: {best_path}")
    
    print("\n训练完成!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    train_single_model(config)


if __name__ == '__main__':
    main()
