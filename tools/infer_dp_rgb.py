"""tools/infer_dp_rgb.py

最小推理脚本：加载训练好的 DP RGB policy，给定“已提取好的 RGB 特征序列”，输出动作。

为什么是“特征”而不是直接视频？
- 你的仓库里视频->特征抽取已经有独立脚本（例如 tools/run_extract_features.py）。
- DP 训练/推理侧目前使用的是 zarr 里保存的 per-frame 特征（例如 croco/vggt/dino 等）。

用法示例：
    conda run -n DP3_ULIP python tools/infer_dp_rgb.py \
      --ckpt outputs/dp_rgb_runs/smoke_run/final_policy.pt \
      --rgb_zarr_root /home/gl/features_model/rgb_dataset/features_croco_encoder_dict_unified_zarr \
      --task shake_bottle-demo_randomized-20_head_camera \
      --episode episode_0 \
      --start 0

输出：
- action: [n_action_steps, action_dim]
- action_pred: [horizon, action_dim]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DP_ROOT = REPO_ROOT / "DP" / "diffusion_policy"
if str(DP_ROOT) not in sys.path:
    sys.path.insert(0, str(DP_ROOT))

from features_common.zarr_pack import load_zarr_pack


def _load_obs_feature(rgb_zarr_root: Path, task: str, episode: str, start: int, n_obs_steps: int) -> torch.Tensor:
    """从 zarr pack 读取特征并做简单 mean-pool，返回 [1, To, C]"""
    zp = rgb_zarr_root / task / f"{episode}.zarr"
    pack = load_zarr_pack(zp)

    # ZarrPack 存的是 per_frame_features: [W, T, Hf, Wf, C]
    W, T = pack.shape[0], pack.shape[1]
    total_steps = W * T
    if start < 0 or start + n_obs_steps > total_steps:
        raise ValueError(f"start out of range: start={start}, n_obs_steps={n_obs_steps}, total_steps={total_steps}")

    frames = []
    for s in range(start, start + n_obs_steps):
        wi = s // T
        ti = s % T
        f = pack.get_frame(wi, ti)  # [Hf, Wf, C]
        f = f.reshape(-1, f.shape[-1]).mean(axis=0)  # [C]
        frames.append(f)
    x = torch.from_numpy(__import__('numpy').stack(frames, axis=0))  # [To, C]
    x = x.unsqueeze(0)  # [1, To, C]
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', type=str, required=True, help='训练保存的 policy ckpt (final_policy.pt)')
    ap.add_argument('--rgb_zarr_root', type=str, required=True)
    ap.add_argument('--task', type=str, required=True)
    ap.add_argument('--episode', type=str, required=True)
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--device', type=str, default='cuda')
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt)
    rgb_zarr_root = Path(args.rgb_zarr_root)

    payload = torch.load(ckpt_path, map_location='cpu')
    if not (isinstance(payload, dict) and 'policy_state' in payload and 'config' in payload):
        raise ValueError(f"Unexpected checkpoint format keys={list(payload.keys()) if hasattr(payload,'keys') else type(payload)}")

    from features_common.dp_rgb_policy import DiffusionRGBPolicy
    cfg = payload['config']

    # Infer dims (train script doesn't persist them explicitly in config)
    obs_dim = cfg.get('obs_dim', None)
    action_dim = cfg.get('action_dim', None)
    if obs_dim is None:
        obs_dim = int(cfg.get('obs_encoder_dim', 256))
        # Better: infer from first Linear layer in encoder projector
        for k, v in payload['policy_state'].items():
            if k.endswith('obs_encoder.projector.0.weight') and v.ndim == 2:
                # [out, in]
                obs_dim = int(v.shape[1])
                break
    if action_dim is None:
        for k, v in payload['policy_state'].items():
            if k.endswith('model.final_conv.1.weight') and v.ndim == 3:
                # [out_channels, in_channels, 1]; out_channels == action_dim for our setup
                action_dim = int(v.shape[0])
                break
    if action_dim is None:
        raise ValueError('Failed to infer action_dim from checkpoint. Please re-save ckpt with action_dim in config.')
    policy = DiffusionRGBPolicy(
        obs_dim=int(obs_dim),
        action_dim=int(action_dim),
        horizon=cfg['horizon'],
        n_obs_steps=cfg['n_obs_steps'],
        n_action_steps=cfg['n_action_steps'],
        rgb_ckpt_path=cfg.get('rgb_ckpt_path', None),
        freeze_encoder=cfg.get('freeze_encoder', False),
        obs_encoder_dim=cfg.get('obs_encoder_dim', 256),
        obs_as_global_cond=cfg.get('obs_as_global_cond', True),
    )
    policy.load_state_dict(payload['policy_state'], strict=True)
    # train_dp_rgb.py uses key 'normalizer'
    policy.normalizer.load_state_dict(payload['normalizer'], strict=True)

    device = torch.device(args.device)
    policy.to(device)
    policy.eval()

    # 从 ckpt config 取 n_obs_steps
    n_obs_steps = cfg['n_obs_steps']

    obs = _load_obs_feature(
        rgb_zarr_root=rgb_zarr_root,
        task=args.task,
        episode=args.episode,
        start=args.start,
        n_obs_steps=n_obs_steps,
    ).to(device).float()

    with torch.no_grad():
        out = policy.predict_action({'obs': obs})

    action = out['action'].detach().cpu()
    action_pred = out['action_pred'].detach().cpu()

    print(f"obs: {tuple(obs.shape)}")
    print(f"action: {tuple(action.shape)}")
    print(action)
    print(f"action_pred: {tuple(action_pred.shape)}")


if __name__ == '__main__':
    main()
