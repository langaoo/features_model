"""tools/train_dp_rgb_single_task_4models.py

单任务训练：4 模型特征（离线 zarr） + 4 模型对齐 encoder（冻结） + DP head。

这实现你要的 ②③④（① 在线跑 4 backbone 不在训练阶段做，而是在推理/部署阶段做；训练阶段用离线特征更稳定）。

用法（示例）：
  conda run -n DP3_ULIP python tools/train_dp_rgb_single_task_4models.py \
    --task beat_block_hammer-demo_randomized-20_head_camera \
    --croco_root /home/gl/features_model/rgb_dataset/features_croco_encoder_dict_unified_zarr \
    --vggt_root  /home/gl/features_model/rgb_dataset/features_vggt_encoder_dict_unified_zarr \
    --dino_root  /home/gl/features_model/rgb_dataset/features_dinov3_encoder_dict_unified_zarr \
    --da3_root   /home/gl/features_model/rgb_dataset/features_da3_encoder_dict_unified_zarr \
    --traj_root /home/gl/features_model/raw_data \
    --rgb2pc_ckpt /home/gl/features_model/outputs/train_rgb2pc_runs/run_best_bs32/ckpt_step_0010000.pt \
    --save_dir outputs/dp_rgb_runs/beat_block_hammer_4models

产物：
- final_policy.pt（包含 DP head + normalizer + config；encoder 冻结不必保存也可保存）
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]

import sys
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DP_ROOT = REPO_ROOT / 'DP' / 'diffusion_policy'
if str(DP_ROOT) not in sys.path:
    sys.path.insert(0, str(DP_ROOT))

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.common.normalizer import SingleFieldLinearNormalizer

from features_common.dp_rgb_dataset_4models import DPRGB4ModelDataset, collate_fn_4
from features_common.rgb2pc_aligned_encoder_4models import RGB2PCAlignedEncoder4Models
from features_common.dp_rgb_policy_multitask import DiffusionRGBHead, HeadSpec  # reuse head implementation


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_action_normalizer(dataset: DPRGB4ModelDataset, num_samples: int = 1000) -> SingleFieldLinearNormalizer:
    action_list = []
    idxs = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    for i in idxs:
        s = dataset[int(i)]
        action_list.append(s.action)
    act_all = torch.stack(action_list, dim=0)   # [N,H,A]
    return SingleFieldLinearNormalizer.create_fit(act_all)


def create_identity_obs_normalizer(obs_feature_dim: int) -> SingleFieldLinearNormalizer:
    # Fit on zeros -> scale=1, offset=0 (i.e., identity) for the required dim.
    dummy = torch.zeros(32, 1, int(obs_feature_dim), dtype=torch.float32)
    return SingleFieldLinearNormalizer.create_fit(dummy)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', type=str, required=True)
    
    # 特征路径（提供默认值）
    ap.add_argument('--croco_root', type=str, 
                    default='/home/gl/features_model/rgb_dataset/features_croco_encoder_dict_unified_zarr')
    ap.add_argument('--vggt_root', type=str,
                    default='/home/gl/features_model/rgb_dataset/features_vggt_encoder_dict_unified_zarr')
    ap.add_argument('--dino_root', type=str,
                    default='/home/gl/features_model/rgb_dataset/features_dinov3_encoder_dict_unified_zarr')
    ap.add_argument('--da3_root', type=str,
                    default='/home/gl/features_model/rgb_dataset/features_da3_encoder_dict_unified_zarr')
    ap.add_argument('--traj_root', type=str,
                    default='/home/gl/features_model/raw_data')
    
    # encoder checkpoint（支持两个参数名）
    ap.add_argument('--rgb2pc_ckpt', '--encoder_ckpt', type=str, required=True,
                    dest='rgb2pc_ckpt', help='RGB2PC aligned encoder checkpoint')
    ap.add_argument('--save_dir', type=str, required=True)

    # arm/action config
    ap.add_argument('--use_left_arm', action='store_true', help='use left arm actions')
    ap.add_argument('--use_right_arm', action='store_true', help='use right arm actions')
    ap.add_argument('--fuse_arms', action='store_true', help='concat left+right actions when both enabled')
    ap.add_argument('--include_gripper', action='store_true', help='include gripper in action (7 or 14 dim)')

    ap.add_argument('--device', type=str, default='cuda')
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--seed', type=int, default=42)

    # DP params
    ap.add_argument('--horizon', type=int, default=8)
    ap.add_argument('--n_obs_steps', type=int, default=2)
    ap.add_argument('--n_action_steps', type=int, default=4)
    ap.add_argument('--obs_as_global_cond', action='store_true')

    ap.add_argument('--save_every', type=int, default=10)
    ap.add_argument('--tqdm', action='store_true')

    args = ap.parse_args()
    set_seed(args.seed)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # defaults: keep backward compatible behavior (left-only)
    use_left = bool(args.use_left_arm or (not args.use_right_arm))
    use_right = bool(args.use_right_arm)

    dataset = DPRGB4ModelDataset(
        rgb_zarr_roots_4=[args.croco_root, args.vggt_root, args.dino_root, args.da3_root],
        traj_root=args.traj_root,
        tasks=[args.task],
        horizon=args.horizon,
        n_obs_steps=args.n_obs_steps,
        use_left_arm=use_left,
        use_right_arm=use_right,
        fuse_arms=bool(args.fuse_arms),
        include_gripper=bool(args.include_gripper),
    )

    # infer dims
    sample0 = dataset[0]
    c_in = int(sample0.obs.shape[-1])
    action_dim = int(sample0.action.shape[-1])

    print(f"Dataset size={len(dataset)} obs_c={c_in} action_dim={action_dim}")

    dl = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn_4)

    device = torch.device(args.device)

    # encoder: load full 4-model ckpt and freeze
    encoder = RGB2PCAlignedEncoder4Models.from_checkpoint(args.rgb2pc_ckpt, freeze=True).to(device)

    # DP head: reuse DiffusionRGBHead (expects obs_features: [B,To,D])
    head_spec = HeadSpec(
        action_dim=action_dim,
        horizon=args.horizon,
        n_obs_steps=args.n_obs_steps,
        n_action_steps=args.n_action_steps,
        obs_feature_dim=int(encoder.spec.fuse_dim),
        obs_as_global_cond=True,
    )
    head = DiffusionRGBHead(spec=head_spec).to(device)

    # normalizer:
    # - obs normalizer is identity at the encoder output dimension (we don't normalize raw 4-model features here)
    # - action normalizer is fitted from dataset actions
    normalizer = LinearNormalizer()
    normalizer['obs'] = create_identity_obs_normalizer(int(encoder.spec.fuse_dim))
    normalizer['action'] = create_action_normalizer(dataset)
    normalizer = normalizer.to(device)

    opt = torch.optim.AdamW(head.parameters(), lr=args.lr)

    def forward_loss(batch: dict) -> torch.Tensor:
        obs = batch['obs'].to(device)       # [B,To,4,C]
        action = batch['action'].to(device) # [B,H,A]

        with torch.no_grad():
            z = encoder(obs)  # [B,To,fuse_dim]

        # head expects obs_features/action + per-field normalizers
        return head.compute_loss(
            obs_features=z,
            action=action,
            normalizer_obs=normalizer['obs'],
            normalizer_action=normalizer['action'],
        )

    for ep in range(1, args.epochs + 1):
        head.train()
        total = 0.0
        n = 0
        for batch in dl:
            opt.zero_grad()
            loss = forward_loss(batch)
            loss.backward()
            opt.step()
            total += float(loss.item())
            n += 1
        print(f"Epoch {ep}/{args.epochs} loss={total/max(n,1):.6f}")

        if (ep % args.save_every) == 0:
            torch.save({
                'head_state': head.state_dict(),
                'normalizer': normalizer.state_dict(),
                'config': vars(args),
                'encoder_ckpt': args.rgb2pc_ckpt,
                'encoder_spec': {
                    'in_dims': tuple(int(x) for x in encoder.spec.in_dims),
                    'fuse_dim': int(encoder.spec.fuse_dim),
                    'fusion': str(encoder.spec.fusion),
                },
                'action_dim': action_dim,
                'obs_c': c_in,
            }, save_dir / f'ckpt_ep_{ep:04d}.pt')

    # final save
    torch.save({
        'head_state': head.state_dict(),
        'normalizer': normalizer.state_dict(),
        'config': vars(args),
        'encoder_ckpt': args.rgb2pc_ckpt,
        'encoder_spec': {
            'in_dims': tuple(int(x) for x in encoder.spec.in_dims),
            'fuse_dim': int(encoder.spec.fuse_dim),
            'fusion': str(encoder.spec.fusion),
        },
        'action_dim': action_dim,
        'obs_c': c_in,
    }, save_dir / 'final_head.pt')

    print('Done. Saved:', save_dir / 'final_head.pt')


if __name__ == '__main__':
    main()
