#!/usr/bin/env python3
"""
直接融合训练: 跳过对齐模块,直接训练 4模型融合特征 → Head → 动作

目的: 
1. 测试是否是对齐模块破坏了特征
2. 验证融合本身的有效性
3. 提供更简单快速的baseline

流程:
  RGB → 4个特征提取器 → 简单融合(weighted/concat) → Head → 动作
  
与完整流程的区别:
- 无对齐模块 (no context_encoder, no projection)
- 只训练fusion权重和Head
- 4个backbone可选冻结或微调
"""
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
import sys
import os
from tqdm import tqdm
import logging
import warnings
import yaml
from typing import Dict, Any
import numpy as np
import time

# 关闭不必要的日志
logging.basicConfig(level=logging.ERROR)
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

# 设置multiprocessing
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

# 添加路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from features_common.dp_rgb_dataset_from_hdf5 import DPRGBOnlineDataset, make_batch_collate_fn
from features_common.multi_gpu_extractors import MultiGPUFeatureExtractors

# 导入正版DP
HAS_OFFICIAL_DP = False
try:
    DP_OUTER = Path(__file__).parent.parent.parent / "DP" / "diffusion_policy"
    if DP_OUTER.exists() and (DP_OUTER / "diffusion_policy").exists():
        sys.path.insert(0, str(DP_OUTER))
        from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
        from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
        HAS_OFFICIAL_DP = True
        print("[INFO] 正版DP已加载")
except ImportError as e:
    print(f"[WARNING] 正版DP导入失败: {e}")


class SimpleFusionEncoder(nn.Module):
    """简单融合编码器: 4模型特征 → 融合 → 输出"""
    
    def __init__(
        self,
        in_dims=(1024, 2048, 768, 2048),
        fusion_type='weighted',  # 'weighted', 'concat', 'mean'
        out_dim=1280,
    ):
        super().__init__()
        self.in_dims = in_dims
        self.fusion_type = fusion_type
        self.out_dim = out_dim
        
        if fusion_type == 'weighted':
            # 可学习的加权融合
            # 先投影到统一维度,再加权
            self.projectors = nn.ModuleList([
                nn.Linear(dim, out_dim) for dim in in_dims
            ])
            self.fusion_weights = nn.Parameter(torch.ones(len(in_dims)) / len(in_dims))
            
        elif fusion_type == 'concat':
            # 拼接后降维
            total_dim = sum(in_dims)
            self.projector = nn.Sequential(
                nn.Linear(total_dim, out_dim * 2),
                nn.ReLU(),
                nn.Linear(out_dim * 2, out_dim),
                nn.LayerNorm(out_dim),
            )
            
        elif fusion_type == 'mean':
            # 简单平均(先投影)
            self.projectors = nn.ModuleList([
                nn.Linear(dim, out_dim) for dim in in_dims
            ])
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")
    
    def forward(self, x):
        """
        Args:
            x: [B, To, M, C_max] 其中M=4
        Returns:
            out: [B, To, out_dim]
        """
        B, To, M, _ = x.shape
        
        if self.fusion_type == 'weighted':
            # 投影 + 加权
            feats = []
            for i, proj in enumerate(self.projectors):
                feat_i = x[:, :, i, :self.in_dims[i]]  # [B, To, C_i]
                feat_i = feat_i.reshape(B * To, self.in_dims[i])
                feat_i = proj(feat_i)  # [B*To, out_dim]
                feat_i = feat_i.reshape(B, To, self.out_dim)
                feats.append(feat_i)
            
            feats = torch.stack(feats, dim=2)  # [B, To, M, out_dim]
            weights = torch.softmax(self.fusion_weights, dim=0)  # [M]
            out = (feats * weights.view(1, 1, M, 1)).sum(dim=2)  # [B, To, out_dim]
            
        elif self.fusion_type == 'concat':
            # 拼接
            feats = []
            for i in range(M):
                feat_i = x[:, :, i, :self.in_dims[i]]
                feats.append(feat_i)
            feats_concat = torch.cat(feats, dim=-1)  # [B, To, sum(dims)]
            feats_concat = feats_concat.reshape(B * To, -1)
            out = self.projector(feats_concat)
            out = out.reshape(B, To, self.out_dim)
            
        elif self.fusion_type == 'mean':
            # 简单平均
            feats = []
            for i, proj in enumerate(self.projectors):
                feat_i = x[:, :, i, :self.in_dims[i]]
                feat_i = feat_i.reshape(B * To, self.in_dims[i])
                feat_i = proj(feat_i)
                feat_i = feat_i.reshape(B, To, self.out_dim)
                feats.append(feat_i)
            out = torch.stack(feats, dim=0).mean(dim=0)  # [B, To, out_dim]
        
        return out


class DirectFusionDPPolicy(nn.Module):
    """直接融合 + Diffusion Policy"""
    
    def __init__(
        self,
        fusion_encoder: SimpleFusionEncoder,
        action_dim: int = 14,
        horizon: int = 4,
        n_obs_steps: int = 4,
        num_inference_steps: int = 100,
    ):
        super().__init__()
        
        if not HAS_OFFICIAL_DP:
            raise RuntimeError("正版DP未加载")
        
        self.fusion_encoder = fusion_encoder
        self.action_dim = action_dim
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.num_inference_steps = num_inference_steps
        
        obs_dim = n_obs_steps * fusion_encoder.out_dim
        
        # 观测编码器
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 512),
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
    
    def compute_loss(self, rgb_feats, actions):
        """
        训练时的loss计算
        Args:
            rgb_feats: [B, To, M, C] 4模型特征
            actions: [B, Ta, A] ground truth动作
        """
        B = rgb_feats.shape[0]
        device = rgb_feats.device
        
        # 融合特征
        obs_fused = self.fusion_encoder(rgb_feats)  # [B, To, D]
        
        # 编码观测
        obs_flat = obs_fused.reshape(B, -1)
        obs_cond = self.obs_encoder(obs_flat)  # [B, 256]
        
        # 随机采样时间步
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (B,), device=device
        ).long()
        
        # 添加噪声
        noise = torch.randn_like(actions)
        noisy_actions = self.noise_scheduler.add_noise(actions, noise, timesteps)
        
        # 预测噪声
        noise_pred = self.noise_pred_net(
            noisy_actions, timesteps, global_cond=obs_cond
        )
        
        # MSE loss
        loss = nn.functional.mse_loss(noise_pred, noise)
        return loss
    
    def forward(self, rgb_feats):
        """
        推理时的去噪采样
        Args:
            rgb_feats: [B, To, M, C]
        Returns:
            actions: [B, Ta, A]
        """
        B = rgb_feats.shape[0]
        device = rgb_feats.device
        
        # 融合特征
        obs_fused = self.fusion_encoder(rgb_feats)
        
        # 编码观测
        obs_flat = obs_fused.reshape(B, -1)
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
        
        return action


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                       help='配置文件路径')
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    print("="*60)
    print("直接融合训练 (无对齐模块)")
    print("="*60)
    print(f"配置文件: {args.config}")
    print(f"任务: {config['data']['tasks']}")
    print(f"融合方式: {config['fusion']['type']}")
    print(f"Horizon: {config['data']['horizon']}, N_obs: {config['data']['n_obs_steps']}")
    print("="*60)
    
    gpu_ids = config['device']['gpu_ids']
    
    # 1. 加载特征提取器
    print("\n1. 加载4个特征提取器...")
    extractors = MultiGPUFeatureExtractors(gpu_ids=gpu_ids)
    
    # 2. 创建Dataset
    print("\n2. 创建Dataset...")
    robotwin_data_root = config['data'].get('robotwin_data_root', '/home/gl/RoboTwin/data')
    task_config = config['data'].get('task_config', 'demo_clean')
    expert_data_num = config.get('checkpoint', {}).get('expert_data_num', 50)
    
    task_name = config['data']['tasks'][0]
    raw_data_root = os.path.join(robotwin_data_root, task_name, task_config, 'data')
    
    dataset = DPRGBOnlineDataset(
        raw_data_root=raw_data_root,
        tasks=config['data']['tasks'],
        horizon=config['data']['horizon'],
        n_obs_steps=config['data']['n_obs_steps'],
        feature_extractors=extractors,
        camera_name=config['data']['camera_name'],
        use_left_arm=config['data']['use_left_arm'],
        use_right_arm=config['data']['use_right_arm'],
        fuse_arms=config['data']['fuse_arms'],
        include_gripper=config['data']['include_gripper'],
        batch_extract=True,
    )
    print(f"✓ Dataset: {len(dataset)} samples")
    
    # 3. DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=config['train']['batch_size'],
        shuffle=True,
        num_workers=0,
        collate_fn=make_batch_collate_fn(extractors),
        pin_memory=True,
    )
    
    # 4. 创建模型
    print("\n3. 创建直接融合模型...")
    fusion_encoder = SimpleFusionEncoder(
        in_dims=(1024, 2048, 768, 2048),
        fusion_type=config['fusion']['type'],
        out_dim=config['fusion']['out_dim'],
    ).to(f'cuda:{gpu_ids[0]}')
    
    policy = DirectFusionDPPolicy(
        fusion_encoder=fusion_encoder,
        action_dim=14,
        horizon=config['data']['horizon'],
        n_obs_steps=config['data']['n_obs_steps'],
        num_inference_steps=config['policy']['num_inference_steps'],
    ).to(f'cuda:{gpu_ids[0]}')
    
    print(f"✓ 模型参数: {sum(p.numel() for p in policy.parameters()) / 1e6:.2f}M")
    
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
    
    global_step = 0
    best_loss = float('inf')
    
    for epoch in range(config['train']['epochs']):
        policy.train()
        epoch_loss = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['train']['epochs']}")
        
        for batch in pbar:
            obs = batch['obs'].to(f'cuda:{gpu_ids[0]}')  # [B, To, M, C]
            action_gt = batch['action'].to(f'cuda:{gpu_ids[0]}')  # [B, Ta, A]
            
            # 前向
            loss = policy.compute_loss(obs, action_gt)
            
            # 反向
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            global_step += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.6f}")
        
        # 保存
        if (epoch + 1) % config['checkpoint']['save_every'] == 0:
            ckpt_path = save_dir / f"{epoch+1}.ckpt"
            torch.save({
                'policy': policy.state_dict(),
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
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'config': config,
                    'loss': avg_loss,
                }, best_path)
                print(f"✓ Best model saved: {best_path}")
    
    print("\n训练完成!")


if __name__ == '__main__':
    main()
