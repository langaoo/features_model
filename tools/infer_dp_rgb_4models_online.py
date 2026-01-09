"""tools/infer_dp_rgb_4models_online.py

在线推理入口：RGB(root) -> 在线提四模型zarr特征(必要时) -> 对齐encoder(冻结) -> DP head -> 动作

说明
- 训练仍推荐使用离线特征（更快更稳）。
- 在线推理时，如果指定的 4 个特征 zarr 不存在，本脚本会调用 `tools/run_extract_features.py`
  自动把这一个 task 的所有 episode 导出到 out_root 下的四个 `features_*_encoder_dict_unified_zarr`。
- 导出完成后复用 `tools/infer_dp_rgb_4models.py` 做真正推理。

用法示例（只保证闭环；后续可把 video/相机流接进来）:
  conda run -n DP3_ULIP python tools/infer_dp_rgb_4models_online.py \
    --ckpt outputs/dp_rgb_runs/beat_block_hammer_4models_smoke/final_head.pt \
    --rgb_root /home/gl/features_model/rgb_dataset/RGB \
    --out_root /home/gl/features_model/rgb_dataset \
    --task beat_block_hammer-demo_randomized-20_head_camera \
    --episode episode_0 \
    --exec_steps 2 \
    --device cuda

"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def _run(cmd: list[str]) -> None:
    print('[RUN]', ' '.join(cmd))
    subprocess.run(cmd, check=True)


def _ensure_features_exist(out_root: Path, task: str, episode: str) -> bool:
    roots = {
        'croco': out_root / 'features_croco_encoder_dict_unified_zarr',
        'vggt': out_root / 'features_vggt_encoder_dict_unified_zarr',
        'dinov3': out_root / 'features_dinov3_encoder_dict_unified_zarr',
        'da3': out_root / 'features_da3_encoder_dict_unified_zarr',
    }
    for r in roots.values():
        if not (r / task / f'{episode}.zarr').exists():
            return False
    return True


def _extract_for_task(rgb_root: Path, out_root: Path, device: str, task: str) -> None:
    # 注意：当前 run_extract_features.py 不支持只抽一个 task；这里用 smoke/全量都能跑。
    # 为了“一轮对话完成闭环”，我们优先跑 --all（确保输出齐全），后续可再做 task filter 优化。
    for m in ['croco', 'vggt', 'dinov3', 'da3']:
        _run([
            'python',
            str(Path(__file__).resolve().parents[1] / 'tools' / 'run_extract_features.py'),
            '--model', m,
            '--rgb_root', str(rgb_root),
            '--out_root', str(out_root),
            '--device', device,
            '--all',
        ])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', type=str, required=True)
    ap.add_argument('--rgb_root', type=str, required=True, help='RGB dataset root (contains task/episode_*/frames...)')
    ap.add_argument('--out_root', type=str, required=True, help='where features_*_encoder_dict_unified_zarr lives/will be written')
    ap.add_argument('--task', type=str, required=True)
    ap.add_argument('--episode', type=str, required=True)
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--exec_steps', type=int, default=1)
    ap.add_argument('--device', type=str, default='cuda')
    ap.add_argument('--force_extract', action='store_true')
    args = ap.parse_args()

    rgb_root = Path(args.rgb_root)
    out_root = Path(args.out_root)

    if args.force_extract or (not _ensure_features_exist(out_root, args.task, args.episode)):
        print('[INFO] features not found (or force_extract). extracting 4-model features...')
        _extract_for_task(rgb_root, out_root, args.device, args.task)

    # now call the offline infer script with the out_root-produced feature roots
    croco = out_root / 'features_croco_encoder_dict_unified_zarr'
    vggt = out_root / 'features_vggt_encoder_dict_unified_zarr'
    dino = out_root / 'features_dinov3_encoder_dict_unified_zarr'
    da3 = out_root / 'features_da3_encoder_dict_unified_zarr'

    _run([
        'python',
        str(Path(__file__).resolve().parents[1] / 'tools' / 'infer_dp_rgb_4models.py'),
        '--ckpt', args.ckpt,
        '--croco_root', str(croco),
        '--vggt_root', str(vggt),
        '--dino_root', str(dino),
        '--da3_root', str(da3),
        '--task', args.task,
        '--episode', args.episode,
        '--start', str(args.start),
        '--exec_steps', str(args.exec_steps),
        '--device', args.device,
    ])


if __name__ == '__main__':
    main()
