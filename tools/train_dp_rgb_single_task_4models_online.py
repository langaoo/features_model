"""tools/train_dp_rgb_single_task_4models_online.py

从 raw_data 实时提取特征并训练 DP head（在线训练）

与 train_dp_rgb_single_task_4models.py 的区别：
- 旧版：读离线 zarr 特征（快速，适合大数据集）
- 本版：实时从 RGB 提取特征（灵活，适合新数据/小数据集）

用法:
  python tools/train_dp_rgb_single_task_4models_online.py \\
    --raw_data_root raw_data \\
    --task beat_block_hammer-demo_randomized-20_sapien_head_camera \\
    --encoder_ckpt outputs/train_rgb2pc_runs/run_best_bs32/ckpt_step_0002000.pt \\
    --use_left_arm --epochs 50 --batch_size 8

注意:
- 需要加载 4 个 backbone，显存占用大（约 20GB）
- 训练速度慢于离线版本（约 5-10 倍）
- 适合快速验证新数据，大规模训练建议先提特征
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from features_common.dp_rgb_dataset_online_4models import (
    DPRGBOnline4ModelDataset,
    collate_fn_online_4,
)
from features_common.online_extractors import load_all_extractors
from features_common.rgb2pc_aligned_encoder_4models import RGB2PCAlignedEncoder4Models

# Import DP Head (假设你有 DiffusionRGBHead 实现)
try:
    from DP.diffusion_policy.model.diffusion.diffusion_rgb_head import DiffusionRGBHead
    HAS_DP_HEAD = True
except ImportError:
    print("Warning: DiffusionRGBHead not found, will use dummy head for testing")
    HAS_DP_HEAD = False


class DummyDPHead(nn.Module):
    """Dummy DP head for testing"""
    def __init__(self, obs_dim, action_dim, horizon):
        super().__init__()
        self.net = nn.Linear(obs_dim, action_dim * horizon)
        self.action_dim = action_dim
        self.horizon = horizon
    
    def forward(self, obs_feat):
        # obs_feat: [B, To, D]
        B, To, D = obs_feat.shape
        # 简单 MLP
        out = self.net(obs_feat[:, -1, :])  # [B, action_dim * horizon]
        return out.view(B, self.horizon, self.action_dim)
    
    def compute_loss(self, obs_feat, action):
        pred = self.forward(obs_feat)
        return nn.functional.mse_loss(pred, action)


def main():
    parser = argparse.ArgumentParser(description="在线训练 DP RGB Head (4 models)")
    
    # Data paths
    parser.add_argument('--rgb_root', type=str, default='rgb_dataset/RGB_ORI',
                       help='RGB 图像根目录')
    parser.add_argument('--traj_root', type=str, default='raw_data',
                       help='轨迹数据根目录')
    parser.add_argument('--task', type=str, required=True,
                       help='任务名称（例如 beat_block_hammer-demo_randomized-20_sapien_head_camera）')
    parser.add_argument('--camera_name', type=str, default=None,
                       help='相机名称（None=自动检测，或指定 head_camera, sapien_head_camera 等）')
    
    # Encoder checkpoint
    parser.add_argument('--encoder_ckpt', type=str, required=True,
                       help='对齐 Encoder checkpoint 路径')
    
    # Model config
    parser.add_argument('--n_obs_steps', type=int, default=2,
                       help='观测步数')
    parser.add_argument('--horizon', type=int, default=8,
                       help='Action horizon')
    
    # Action config
    parser.add_argument('--use_left_arm', action='store_true',
                       help='使用左臂')
    parser.add_argument('--use_right_arm', action='store_true',
                       help='使用右臂')
    parser.add_argument('--fuse_arms', action='store_true',
                       help='融合双臂（仅当两个 arm 都启用时）')
    parser.add_argument('--include_gripper', action='store_true',
                       help='包含夹爪')
    
    # Training config
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size（在线训练建议用小 batch）')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=0,
                       help='DataLoader workers（在线提特征建议用 0）')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='outputs/dp_rgb_online_runs',
                       help='输出目录')
    
    args = parser.parse_args()
    
    # Validate
    if not args.use_left_arm and not args.use_right_arm:
        print("Error: At least one arm must be enabled")
        sys.exit(1)
    
    device = torch.device(args.device)
    
    # ========== 1. 加载 4 个 backbone extractors ==========
    print("\n" + "="*60)
    print("加载 4 个视觉 backbone...")
    print("="*60)
    extractors = load_all_extractors(device=args.device)
    print("✓ Extractors loaded\n")
    
    # ========== 2. 创建 Dataset ==========
    print("="*60)
    print("创建 Online Dataset...")
    print("="*60)
    print(f"  rgb_root: {args.rgb_root}")
    print(f"  traj_root: {args.traj_root}")
    print(f"  task: {args.task}")
    print(f"  camera: {args.camera_name or 'auto-detect'}")
    print(f"  n_obs_steps: {args.n_obs_steps}, horizon: {args.horizon}")
    
    dataset = DPRGBOnline4ModelDataset(
        rgb_root=args.rgb_root,
        traj_root=args.traj_root,
        tasks=[args.task],
        horizon=args.horizon,
        n_obs_steps=args.n_obs_steps,
        feature_extractors=extractors,
        device=args.device,
        camera_name=args.camera_name,
        use_left_arm=args.use_left_arm,
        use_right_arm=args.use_right_arm,
        fuse_arms=args.fuse_arms,
        include_gripper=args.include_gripper,
    )
    
    print(f"✓ Dataset created: {len(dataset)} samples")
    
    # Test sample
    print("\n测试采样...")
    sample = dataset[0]
    print(f"  task: {sample.task}")
    print(f"  episode: {sample.episode}")
    print(f"  obs shape: {sample.obs.shape}")
    print(f"  action shape: {sample.action.shape}")
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn_online_4,
        pin_memory=True if args.device == 'cuda' else False,
    )
    
    # ========== 3. 加载 Encoder（冻结）==========
    print("\n" + "="*60)
    print("加载对齐 Encoder（冻结）...")
    print("="*60)
    encoder = RGB2PCAlignedEncoder4Models.from_checkpoint(
        str(args.encoder_ckpt),
        freeze=True
    ).to(device)
    encoder.eval()
    print(f"✓ Encoder loaded: {args.encoder_ckpt}")
    print(f"  Spec: in_dims={encoder.spec.in_dims}, fuse_dim={encoder.spec.fuse_dim}\n")
    
    # ========== 4. 创建 DP Head ==========
    print("="*60)
    print("创建 DP Head...")
    print("="*60)
    
    action_dim = sample.action.shape[1]
    obs_feat_dim = encoder.spec.fuse_dim  # 1280
    
    if HAS_DP_HEAD:
        head = DiffusionRGBHead(
            obs_dim=obs_feat_dim,
            action_dim=action_dim,
            horizon=args.horizon,
            n_obs_steps=args.n_obs_steps,
        ).to(device)
        print(f"✓ Using DiffusionRGBHead")
    else:
        head = DummyDPHead(
            obs_dim=obs_feat_dim,
            action_dim=action_dim,
            horizon=args.horizon,
        ).to(device)
        print(f"⚠ Using DummyDPHead (for testing)")
    
    print(f"  obs_dim: {obs_feat_dim}, action_dim: {action_dim}, horizon: {args.horizon}\n")
    
    # ========== 5. Optimizer ==========
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr)
    
    # ========== 6. Training Loop ==========
    print("="*60)
    print("开始训练...")
    print("="*60)
    
    output_dir = Path(args.output_dir) / f"{args.task}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_dir}\n")
    
    for epoch in range(args.epochs):
        head.train()
        total_loss = 0.0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            obs = batch['obs'].to(device)  # [B, To, 4, 2048]
            action = batch['action'].to(device)  # [B, Ta, A]
            
            # Encoder forward（冻结，no_grad）
            with torch.no_grad():
                obs_feat = encoder(obs)  # [B, To, 1280]
            
            # Head forward + loss
            if HAS_DP_HEAD:
                loss = head.compute_loss(obs_feat, action)
            else:
                loss = head.compute_loss(obs_feat, action)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{args.epochs}, Avg Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            ckpt_path = output_dir / f"head_epoch_{epoch+1}.pt"
            torch.save({
                'epoch': epoch + 1,
                'head_state_dict': head.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'encoder_ckpt': args.encoder_ckpt,
                'config': vars(args),
            }, ckpt_path)
            print(f"  Saved: {ckpt_path}")
    
    # ========== 7. Save Final ==========
    final_path = output_dir / "final_head.pt"
    torch.save({
        'head_state_dict': head.state_dict(),
        'encoder_ckpt': args.encoder_ckpt,
        'config': vars(args),
    }, final_path)
    print(f"\n✓ Training complete! Final model: {final_path}")


if __name__ == '__main__':
    main()
