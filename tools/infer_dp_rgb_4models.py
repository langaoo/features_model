"""tools/infer_dp_rgb_4models.py

推理：4 模型 zarr 特征 -> 对齐 encoder(冻结) -> 单任务 DP head -> 输出动作

这条脚本能把“4模型+对齐+DP head”的闭环串起来，且与机器人端执行逻辑兼容。

用法示例：
  conda run -n DP3_ULIP python tools/infer_dp_rgb_4models.py \
    --ckpt outputs/dp_rgb_runs/beat_block_hammer_4models/final_head.pt \
    --croco_root /home/gl/features_model/rgb_dataset/features_croco_encoder_dict_unified_zarr \
    --vggt_root  /home/gl/features_model/rgb_dataset/features_vggt_encoder_dict_unified_zarr \
    --dino_root  /home/gl/features_model/rgb_dataset/features_dinov3_encoder_dict_unified_zarr \
    --da3_root   /home/gl/features_model/rgb_dataset/features_da3_encoder_dict_unified_zarr \
    --task beat_block_hammer-demo_randomized-20_head_camera \
    --episode episode_0 \
    --start 0 \
    --exec_steps 1

输出：
- action_exec: [exec_steps, action_dim]  (机器人直接执行)
- action_pred: [horizon, action_dim]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DP_ROOT = REPO_ROOT / 'DP' / 'diffusion_policy'
if str(DP_ROOT) not in sys.path:
    sys.path.insert(0, str(DP_ROOT))

from features_common.zarr_pack import load_zarr_pack
from features_common.rgb2pc_aligned_encoder_4models import RGB2PCAlignedEncoder4Models
from features_common.dp_rgb_policy_multitask import DiffusionRGBHead, HeadSpec
from diffusion_policy.model.common.normalizer import LinearNormalizer


def _pad_or_trim_1d(x: np.ndarray, target: int) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim != 1:
        x = x.reshape(-1)
    if x.shape[0] >= target:
        return x[:target]
    out = np.zeros((target,), dtype=x.dtype)
    out[: x.shape[0]] = x
    return out


def load_obs_4(
    roots: list[Path],
    task: str,
    episode: str,
    start: int,
    n_obs_steps: int,
    *,
    in_dims: tuple[int, int, int, int] = (1024, 2048, 768, 2048),
    max_dim: int = 2048,
) -> torch.Tensor:
    packs = [load_zarr_pack(r / task / f"{episode}.zarr") for r in roots]
    W, T = packs[0].shape[0], packs[0].shape[1]
    total_steps = W * T
    if start < 0 or start + n_obs_steps > total_steps:
        raise ValueError(f"start out of range: start={start}, n_obs_steps={n_obs_steps}, total_steps={total_steps}")

    frames = []
    for s in range(start, start + n_obs_steps):
        wi = s // T
        ti = s % T
        per_model = []
        for mi, pack in enumerate(packs):
            f = pack.get_frame(wi, ti)  # [Hf,Wf,C]
            f = f.reshape(-1, f.shape[-1]).mean(axis=0)
            f = _pad_or_trim_1d(f, int(in_dims[mi]))
            f = _pad_or_trim_1d(f, int(max_dim))
            per_model.append(f)
        frames.append(np.stack(per_model, axis=0))  # [4,max_dim]

    obs = torch.from_numpy(np.stack(frames, axis=0)).to(torch.float32)  # [To,4,max_dim]
    return obs.unsqueeze(0)  # [1,To,4,C]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', type=str, required=True)
    ap.add_argument('--croco_root', type=str, required=True)
    ap.add_argument('--vggt_root', type=str, required=True)
    ap.add_argument('--dino_root', type=str, required=True)
    ap.add_argument('--da3_root', type=str, required=True)

    ap.add_argument('--task', type=str, required=True)
    ap.add_argument('--episode', type=str, required=True)
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--device', type=str, default='cuda')
    ap.add_argument('--exec_steps', type=int, default=1)
    # purely informational for now (action_dim comes from ckpt), but keeps CLI aligned with training.
    ap.add_argument('--use_left_arm', action='store_true')
    ap.add_argument('--use_right_arm', action='store_true')
    ap.add_argument('--fuse_arms', action='store_true')
    args = ap.parse_args()

    device = torch.device(args.device)

    payload = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    cfg = payload['config']

    encoder = RGB2PCAlignedEncoder4Models.from_checkpoint(payload['encoder_ckpt'], freeze=True).to(device)

    head_spec = HeadSpec(
        action_dim=int(payload['action_dim']),
        horizon=int(cfg['horizon']),
        n_obs_steps=int(cfg['n_obs_steps']),
        n_action_steps=int(cfg['n_action_steps']),
        obs_feature_dim=int(encoder.spec.fuse_dim),
        obs_as_global_cond=True,
    )
    head = DiffusionRGBHead(spec=head_spec).to(device)
    head.load_state_dict(payload['head_state'], strict=True)

    normalizer = LinearNormalizer().to(device)
    normalizer.load_state_dict(payload['normalizer'], strict=True)

    roots = [Path(args.croco_root), Path(args.vggt_root), Path(args.dino_root), Path(args.da3_root)]
    obs = load_obs_4(roots, args.task, args.episode, args.start, int(cfg['n_obs_steps'])).to(device)

    with torch.no_grad():
        z = encoder(obs)
        out = head.predict_action(
            obs_features=z,
            normalizer_obs=normalizer['obs'],
            normalizer_action=normalizer['action'],
        )

    action_pred = out['action_pred'][0].detach().cpu()  # [H,A]
    exec_steps = max(1, min(int(args.exec_steps), action_pred.shape[0]))
    action_exec = action_pred[:exec_steps]

    print('obs:', tuple(obs.shape))
    print('z:', tuple(z.shape))
    print('action_pred:', tuple(action_pred.shape))
    print('action_exec:', tuple(action_exec.shape))
    print(action_exec)


if __name__ == '__main__':
    main()
