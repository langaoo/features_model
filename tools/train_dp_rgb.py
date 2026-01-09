"""tools/train_dp_rgb.py

训练 Diffusion Policy with RGB 蒸馏特征。

用法:
    python tools/train_dp_rgb.py --config configs/train_dp_rgb_default.yaml
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

try:
    import wandb
except ImportError:
    wandb = None

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ensure vendored diffusion_policy package is importable
DP_ROOT = REPO_ROOT / "DP" / "diffusion_policy"
if str(DP_ROOT) not in sys.path:
    sys.path.insert(0, str(DP_ROOT))

from features_common.dp_rgb_dataset import DPRGBDataset, collate_fn
from features_common.dp_rgb_policy import DiffusionRGBPolicy
from diffusion_policy.model.common.normalizer import LinearNormalizer


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_normalizer(dataset: DPRGBDataset, num_samples: int = 1000) -> LinearNormalizer:
    """
    创建数据归一化器。
    
    从数据集中采样 num_samples 个样本，计算 obs 和 action 的统计量。
    """
    print(f"Computing normalizer from {num_samples} samples...")
    
    obs_list = []
    action_list = []
    
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    for idx in indices:
        sample = dataset[idx]
        obs_list.append(sample.obs)
        action_list.append(sample.action)
    
    obs_all = torch.stack(obs_list, dim=0)  # [N, To, C]
    action_all = torch.stack(action_list, dim=0)  # [N, Ta, A]
    
    from diffusion_policy.model.common.normalizer import SingleFieldLinearNormalizer
    
    normalizer = LinearNormalizer()
    
    # Compute stats
    normalizer['obs'] = SingleFieldLinearNormalizer.create_fit(obs_all)
    normalizer['action'] = SingleFieldLinearNormalizer.create_fit(action_all)
    
    # Print stats
    obs_stats = normalizer['obs'].get_input_stats()
    action_stats = normalizer['action'].get_input_stats()
    print(f"  Obs stats: min={obs_stats['min'].mean():.3f}, max={obs_stats['max'].mean():.3f}, "
          f"mean={obs_stats['mean'].mean():.3f}, std={obs_stats['std'].mean():.3f}")
    print(f"  Action stats: min={action_stats['min'].mean():.3f}, max={action_stats['max'].mean():.3f}, "
          f"mean={action_stats['mean'].mean():.3f}, std={action_stats['std'].mean():.3f}")
    
    return normalizer


def train_one_epoch(
    policy: DiffusionRGBPolicy,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    grad_accum_steps: int = 1,
    use_amp: bool = False,
    log_every: int = 50,
    use_tqdm: bool = True,
    wandb_run = None,
) -> dict:
    """训练一个 epoch"""
    policy.train()
    
    total_loss = 0.0
    num_batches = 0
    
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    iterator = enumerate(dataloader)
    if use_tqdm and tqdm is not None:
        iterator = tqdm(iterator, total=len(dataloader), desc=f"Epoch {epoch}")
    
    optimizer.zero_grad()
    
    for batch_idx, batch in iterator:
        # Move tensors to device (collate may include non-tensor metadata like task/episode)
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        
        # Forward
        # Only pass tensor fields expected by policy/normalizer
        batch_policy = {
            'obs': batch['obs'],
            'action': batch['action'],
        }

        if use_amp:
            with torch.cuda.amp.autocast():
                loss = policy.compute_loss(batch_policy)
        else:
            loss = policy.compute_loss(batch_policy)
        
        # Backward
        if scaler is not None:
            scaler.scale(loss / grad_accum_steps).backward()
        else:
            (loss / grad_accum_steps).backward()
        
        # Step
        if (batch_idx + 1) % grad_accum_steps == 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            
            optimizer.zero_grad()
        
        # Log
        total_loss += loss.item()
        num_batches += 1
        
        if (batch_idx + 1) % log_every == 0:
            avg_loss = total_loss / num_batches
            print(f"  Batch {batch_idx+1}/{len(dataloader)}: loss={loss.item():.4f}, avg={avg_loss:.4f}")
            
            if wandb_run is not None:
                wandb_run.log({
                    'train/loss': loss.item(),
                    'train/loss_avg': avg_loss,
                    'train/batch': batch_idx + epoch * len(dataloader),
                })
    
    avg_loss = total_loss / max(1, num_batches)
    return {'loss': avg_loss}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="配置文件路径")
    
    # 数据
    ap.add_argument("--rgb_zarr_roots", nargs="+", help="RGB zarr 根目录列表")
    ap.add_argument("--traj_root", type=str, help="轨迹数据根目录")
    ap.add_argument("--tasks", nargs="+", help="任务列表")
    
    # 训练
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--grad_accum_steps", type=int, default=1)
    ap.add_argument("--amp", action="store_true", help="使用混合精度训练")
    ap.add_argument("--seed", type=int, default=0)
    
    # Policy
    ap.add_argument("--rgb_ckpt_path", type=str, help="RGB2PC 蒸馏模型路径")
    ap.add_argument("--freeze_encoder", action="store_true", help="冻结特征编码器")
    ap.add_argument("--horizon", type=int, help="预测时域")
    ap.add_argument("--n_obs_steps", type=int, help="观测步数")
    ap.add_argument("--n_action_steps", type=int, help="执行步数")
    
    # 日志
    ap.add_argument("--save_dir", type=str, default="outputs/dp_rgb_runs")
    ap.add_argument("--save_every", type=int, default=10)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--tqdm", action="store_true")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb_project", type=str, default="dp_rgb")
    ap.add_argument("--wandb_entity", type=str, default="")
    ap.add_argument("--wandb_run_name", type=str, default="")
    
    args = ap.parse_args()
    
    # 加载配置文件
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        # 命令行参数覆盖配置文件
        for k, v in vars(args).items():
            if v is not None and k != 'config':
                config[k] = v
    else:
        config = vars(args)
    
    # 设置随机种子
    set_seed(config['seed'])
    
    # 设备
    device = torch.device(config['device'])
    print(f"Using device: {device}")
    
    # 创建保存目录
    save_dir = Path(config['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存配置
    with open(save_dir / "config.yaml", 'w') as f:
        yaml.dump(config, f)
    
    # WandB
    wandb_run = None
    if config.get('wandb', False) and wandb is not None:
        wandb_run = wandb.init(
            project=config.get('wandb_project', 'dp_rgb'),
            entity=config.get('wandb_entity', None),
            name=config.get('wandb_run_name', None),
            config=config,
        )
    
    # 创建数据集
    print("Creating dataset...")
    print(f"  RGB zarr roots: {config['rgb_zarr_roots']}")
    print(f"  Traj root: {config['traj_root']}")
    print(f"  Tasks: {config['tasks']}")
    print(f"  Horizon: {config['horizon']}")
    
    dataset = DPRGBDataset(
        rgb_zarr_roots=config['rgb_zarr_roots'],
        traj_root=config['traj_root'],
        tasks=config['tasks'],
        horizon=config['horizon'],
        n_obs_steps=config['n_obs_steps'],
        pad_before=config.get('pad_before', 0),
        pad_after=config.get('pad_after', 0),
        use_left_arm=config.get('use_left_arm', True),
        use_right_arm=config.get('use_right_arm', False),
        fuse_arms=config.get('fuse_arms', False),
        seed=config['seed'],
    )
    
    print(f"Dataset: {len(dataset)} samples")
    
    # 创建 DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=config['num_workers'] > 0,
    )
    
    # 获取 obs_dim 和 action_dim
    sample = dataset[0]
    obs_dim = sample.obs.shape[-1]
    action_dim = sample.action.shape[-1]
    
    print(f"Obs dim: {obs_dim}, Action dim: {action_dim}")
    
    # 创建 normalizer
    normalizer = create_normalizer(dataset)
    
    # 创建 policy
    print("Creating policy...")
    policy = DiffusionRGBPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        horizon=config['horizon'],
        n_obs_steps=config['n_obs_steps'],
        n_action_steps=config['n_action_steps'],
        rgb_ckpt_path=config.get('rgb_ckpt_path'),
        freeze_encoder=config.get('freeze_encoder', False),
        obs_encoder_dim=config.get('obs_encoder_dim', 256),
        obs_as_global_cond=config.get('obs_as_global_cond', True),
    )
    
    # Set normalizer
    policy.set_normalizer(normalizer)
    
    policy.set_normalizer(normalizer)
    policy = policy.to(device)
    
    # 优化器
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=config['lr'],
        betas=config.get('betas', [0.95, 0.999]),
        eps=config.get('eps', 1e-8),
        weight_decay=config.get('weight_decay', 1e-6),
    )
    
    # 训练循环
    print("Starting training...")
    for epoch in range(config['epochs']):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{config['epochs']}")
        print(f"{'='*60}")
        
        metrics = train_one_epoch(
            policy=policy,
            dataloader=dataloader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            grad_accum_steps=config.get('grad_accum_steps', 1),
            use_amp=config.get('amp', False),
            log_every=config.get('log_every', 50),
            use_tqdm=config.get('tqdm', False),
            wandb_run=wandb_run,
        )
        
        print(f"Epoch {epoch+1} metrics: {metrics}")
        
        if wandb_run is not None:
            wandb_run.log({
                'epoch': epoch + 1,
                'train/epoch_loss': metrics['loss'],
            })
        
        # 保存 checkpoint
        if (epoch + 1) % config.get('save_every', 10) == 0:
            ckpt_path = save_dir / f"ckpt_epoch_{epoch+1:04d}.pt"
            torch.save({
                'epoch': epoch + 1,
                'policy_state': policy.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'normalizer': normalizer.state_dict(),
                'config': config,
                'metrics': metrics,
            }, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")
    
    # 保存最终模型
    final_path = save_dir / "final_policy.pt"
    torch.save({
        'policy_state': policy.state_dict(),
        'normalizer': normalizer.state_dict(),
        'config': config,
    }, final_path)
    print(f"Saved final model: {final_path}")
    
    if wandb_run is not None:
        wandb_run.finish()
    
    print("\nTraining completed!")


if __name__ == "__main__":
    main()
