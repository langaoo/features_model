"""tools/test_dp_rgb_pipeline.py

测试 DP RGB 训练管道的各个组件。

用法:
    python tools/test_dp_rgb_pipeline.py --config configs/train_dp_rgb_default.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from features_common.dp_rgb_dataset import DPRGBDataset, collate_fn
from features_common.dp_rgb_policy import DiffusionRGBPolicy
from torch.utils.data import DataLoader


def test_dataset(config: dict):
    """测试数据集"""
    print("\n" + "="*80)
    print("测试 1: 数据集")
    print("="*80)
    
    try:
        dataset = DPRGBDataset(
            rgb_zarr_roots=config['rgb_zarr_roots'],
            traj_root=config['traj_root'],
            tasks=config['tasks'],
            horizon=config['horizon'],
            n_obs_steps=config['n_obs_steps'],
            use_left_arm=config.get('use_left_arm', True),
            use_right_arm=config.get('use_right_arm', False),
            fuse_arms=config.get('fuse_arms', False),
        )
        
        print(f"✅ 数据集创建成功: {len(dataset)} 个样本")
        
        # 测试采样
        print("\n测试采样...")
        sample = dataset[0]
        print(f"  Task: {sample.task}")
        print(f"  Episode: {sample.episode}")
        print(f"  Start idx: {sample.start_idx}")
        print(f"  Obs shape: {sample.obs.shape}")  # [To, C]
        print(f"  Action shape: {sample.action.shape}")  # [Ta, A]
        
        # 测试 DataLoader
        print("\n测试 DataLoader...")
        loader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn,
        )
        
        batch = next(iter(loader))
        print(f"  Batch obs shape: {batch['obs'].shape}")  # [B, To, C]
        print(f"  Batch action shape: {batch['action'].shape}")  # [B, Ta, A]
        
        print("\n✅ 数据集测试通过!")
        return True, dataset, sample
    
    except Exception as e:
        print(f"\n❌ 数据集测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None


def test_policy(config: dict, sample):
    """测试 policy"""
    print("\n" + "="*80)
    print("测试 2: Policy")
    print("="*80)
    
    try:
        obs_dim = sample.obs.shape[-1]
        action_dim = sample.action.shape[-1]
        
        print(f"  Obs dim: {obs_dim}")
        print(f"  Action dim: {action_dim}")
        
        # 创建 policy
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
        
        print(f"\n✅ Policy 创建成功")
        
        # 测试前向传播
        print("\n测试前向传播...")
        device = torch.device(config.get('device', 'cpu'))
        policy = policy.to(device)
        policy.eval()
        
        # 准备输入
        obs = sample.obs.unsqueeze(0).to(device)  # [1, To, C]
        action = sample.action.unsqueeze(0).to(device)  # [1, Ta, A]
        
        batch = {
            'obs': obs,
            'action': action,
        }
        
        # 测试 compute_loss
        print("  测试 compute_loss...")
        with torch.no_grad():
            loss = policy.compute_loss(batch)
        print(f"    Loss: {loss.item():.4f}")
        
        # 测试 predict_action
        print("  测试 predict_action...")
        with torch.no_grad():
            result = policy.predict_action({'obs': obs})
        
        print(f"    Action shape: {result['action'].shape}")  # [1, Ta, A]
        print(f"    Action pred shape: {result['action_pred'].shape}")  # [1, horizon, A]
        
        print("\n✅ Policy 测试通过!")
        return True, policy
    
    except Exception as e:
        print(f"\n❌ Policy 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_training_step(config: dict, dataset, policy):
    """测试训练步骤"""
    print("\n" + "="*80)
    print("测试 3: 训练步骤")
    print("="*80)
    
    try:
        device = torch.device(config.get('device', 'cpu'))
        policy = policy.to(device)
        policy.train()
        
        # 创建 optimizer
        optimizer = torch.optim.AdamW(
            policy.parameters(),
            lr=config.get('lr', 1e-4),
        )
        
        # 创建 DataLoader
        loader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn,
        )
        
        # 训练一个 batch
        batch = next(iter(loader))
        batch = {k: v.to(device) for k, v in batch.items()}
        
        print("  训练一个 batch...")
        optimizer.zero_grad()
        loss = policy.compute_loss(batch)
        loss.backward()
        optimizer.step()
        
        print(f"    Loss: {loss.item():.4f}")
        print(f"    梯度范数: {sum(p.grad.norm().item() for p in policy.parameters() if p.grad is not None):.4f}")
        
        print("\n✅ 训练步骤测试通过!")
        return True
    
    except Exception as e:
        print(f"\n❌ 训练步骤测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="配置文件路径")
    args = ap.parse_args()
    
    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print("="*80)
    print("DP RGB 管道测试")
    print("="*80)
    print(f"配置文件: {args.config}")
    print(f"任务: {config.get('tasks', [])}")
    print(f"RGB ckpt: {config.get('rgb_ckpt_path', 'None')}")
    
    # 测试数据集
    success, dataset, sample = test_dataset(config)
    if not success:
        print("\n❌ 测试失败: 数据集")
        return
    
    # 测试 policy
    success, policy = test_policy(config, sample)
    if not success:
        print("\n❌ 测试失败: Policy")
        return
    
    # 测试训练步骤
    success = test_training_step(config, dataset, policy)
    if not success:
        print("\n❌ 测试失败: 训练步骤")
        return
    
    print("\n" + "="*80)
    print("🎉 所有测试通过!")
    print("="*80)
    print("\n你可以开始训练了:")
    print(f"  python tools/train_dp_rgb.py --config {args.config}")


if __name__ == "__main__":
    main()
