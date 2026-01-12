#!/usr/bin/env python3
"""
在线训练脚本 - 多GPU并行 + 从HDF5读取数据
支持16GB显存的2卡训练
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
import sys
import os

# 添加路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from features_common.dp_rgb_dataset_from_hdf5 import DPRGBOnlineDataset, collate_fn_online_4, make_batch_collate_fn
from features_common.multi_gpu_extractors import MultiGPUFeatureExtractors
from features_common.rgb2pc_aligned_encoder_4models import RGB2PCAlignedEncoder4Models


def parse_args():
    parser = argparse.ArgumentParser()
    
    # 数据
    parser.add_argument('--raw_data_root', type=str, default='raw_data',
                       help='原始数据根目录（包含HDF5文件）')
    parser.add_argument('--task', type=str, required=True,
                       help='任务名（支持简写如beat_block_hammer或完整名）')
    parser.add_argument('--camera_name', type=str, default='head_camera',
                       help='相机名称')
    
    # 采样
    parser.add_argument('--horizon', type=int, default=8)
    parser.add_argument('--n_obs_steps', type=int, default=2)
    
    # 手臂配置
    parser.add_argument('--use_left_arm', action='store_true', help='使用左臂')
    parser.add_argument('--use_right_arm', action='store_true', help='使用右臂')
    parser.add_argument('--fuse_arms', action='store_true', help='融合双臂')
    parser.add_argument('--include_gripper', action='store_true', help='包含夹爪')
    
    # 模型
    parser.add_argument('--encoder_ckpt', type=str, required=True,
                       help='对齐编码器checkpoint路径')
    
    # 训练
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=4,
                       help='在线训练建议小batch (2-4)')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=2)
    
    # GPU
    parser.add_argument('--gpu_ids', type=str, default='0,1',
                       help='使用的GPU ID，逗号分隔，如 0,1')
    
    # 输出
    parser.add_argument('--output_dir', type=str, default='_runs/online_training_hdf5')
    parser.add_argument('--run_name', type=str, default=None)
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 解析GPU IDs
    gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(',')]
    print(f"\n{'='*60}")
    print(f"在线训练 - 多GPU并行 + HDF5数据源")
    print(f"{'='*60}")
    print(f"GPU配置: {gpu_ids}")
    print(f"任务: {args.task}")
    print(f"相机: {args.camera_name}")
    print(f"批大小: {args.batch_size}")
    print(f"{'='*60}\n")
    
    # 1. 加载特征提取器（分布到多GPU）
    print("1. 加载4个特征提取器到多GPU...")
    extractors = MultiGPUFeatureExtractors(gpu_ids=gpu_ids)
    print()
    
    # 2. 创建Dataset
    print("2. 创建Dataset（从HDF5读取）...")
    dataset = DPRGBOnlineDataset(
        raw_data_root=args.raw_data_root,
        tasks=[args.task],
        horizon=args.horizon,
        n_obs_steps=args.n_obs_steps,
        feature_extractors=extractors,
        camera_name=args.camera_name,
        use_left_arm=args.use_left_arm,
        use_right_arm=args.use_right_arm,
        fuse_arms=args.fuse_arms,
        include_gripper=args.include_gripper,
        batch_extract=True,  # 启用批量提取模式
    )
    print(f"✓ Dataset: {len(dataset)} samples\n")
    
    # 3. DataLoader - 使用批量提取collate
    print("3. 创建DataLoader（批量特征提取模式）...")
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=make_batch_collate_fn(extractors),  # 使用批量提取
        pin_memory=True,
    )
    print(f"✓ Batchsize={args.batch_size} 将真正生效（批量提取）\n")
    
    # 4. 加载对齐编码器
    print("3. 加载对齐编码器...")
    encoder = RGB2PCAlignedEncoder4Models.from_checkpoint(
        args.encoder_ckpt,
        freeze=True
    )
    encoder = encoder.to(f'cuda:{gpu_ids[0]}')  # 编码器放第一张卡
    encoder.eval()
    fuse_dim = encoder.spec.fuse_dim  # 获取实际输出维度（1280）
    print(f"✓ Encoder loaded from {args.encoder_ckpt}")
    print(f"  fuse_dim = {fuse_dim}\n")
    
    # 5. DP Policy Head（示例）
    print("4. 创建DP Policy...")
    action_dim = 6  # 根据实际配置调整
    if args.use_left_arm and args.use_right_arm and args.fuse_arms:
        action_dim = 12
    if args.include_gripper:
        action_dim += 1 if not args.fuse_arms else 2
    
    class SimpleDPHead(nn.Module):
        def __init__(self, obs_dim, action_dim, horizon):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(obs_dim, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, action_dim * horizon),
            )
            self.action_dim = action_dim
            self.horizon = horizon
        
        def forward(self, obs):
            # obs: [B, To, D]
            B = obs.shape[0]
            obs_flat = obs.reshape(B, -1)
            out = self.net(obs_flat)
            return out.reshape(B, self.horizon, self.action_dim)
    
    obs_dim = args.n_obs_steps * fuse_dim  # 对齐后的融合特征（1280维）
    policy = SimpleDPHead(obs_dim, action_dim, args.horizon)
    policy = policy.to(f'cuda:{gpu_ids[0]}')
    print(f"✓ Policy: obs_dim={obs_dim}, action_dim={action_dim}\n")
    
    # 6. 优化器
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    
    # 7. 训练循环
    print("5. 开始训练...")
    print(f"{'='*60}\n")
    
    for epoch in range(args.epochs):
        policy.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(dataloader):
            obs = batch['obs'].to(f'cuda:{gpu_ids[0]}')  # [B, To, 4, 2048]
            action_gt = batch['action'].to(f'cuda:{gpu_ids[0]}')  # [B, Ta, A]
            
            # 前向传播 - 使用正确的encoder输入格式
            # encoder期望输入: [B, To, M, C]，即obs已经是正确格式
            with torch.no_grad():
                obs_encoded = encoder(obs)  # [B, To, fuse_dim]
            
            action_pred = policy(obs_encoded)  # [B, Ta, A]
            
            # 计算损失
            loss = criterion(action_pred, action_gt)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}] "
                      f"Batch [{batch_idx+1}/{len(dataloader)}] "
                      f"Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        print(f"\n==> Epoch {epoch+1} Average Loss: {avg_loss:.4f}\n")
        
        # 保存checkpoint
        if (epoch + 1) % 5 == 0:
            output_dir = Path(args.output_dir) / (args.run_name or f"{args.task}_run")
            output_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = output_dir / f"ckpt_epoch_{epoch+1:04d}.pt"
            torch.save({
                'epoch': epoch + 1,
                'policy': policy.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': avg_loss,
            }, ckpt_path)
            print(f"✓ Saved checkpoint: {ckpt_path}\n")
    
    print(f"\n{'='*60}")
    print("训练完成！")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
