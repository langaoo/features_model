"""CroCo 统一 CLI wrapper（与 VGGT / DINOv3 / DA3 命令格式完全一致）

目的
- 不改原 CroCo 提取实现（复用 `croco/extract_multi_frame_croco_features.py` 的 process_episode 等逻辑）
- 提供一致的参数语义：
  --rgb_root / --out_root
  --window_size / --stride
  --task / --episode / --all / --smoke
  --device / --save_dtype
  --keep_time_dim（默认 True）
  --also_save_fused（默认 False）

输出
- 仍由原脚本保存 dict：per_frame_features/frame_paths/meta（可选 features）

说明
- CroCo 原脚本的 `list_task_episode_dirs()` 会一次性枚举所有 task/episode。
  为了支持精确选择，本 wrapper 直接复用其工具函数来枚举，再做过滤后逐个调用 `process_episode()`。

"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch  # type: ignore

# 允许直接 `python croco/xxx.py` 运行：把仓库根目录加入 sys.path，避免要求 croco 必须是可安装包
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# 直接复用原脚本中的函数（不改逻辑）
from croco.extract_multi_frame_croco_features import (  # noqa: E402
    build_preprocess,
    get_autocast_dtype,
    get_device,
    list_task_episode_dirs,
    load_croco_encoder,
    process_episode,
)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CroCo v2 统一入口：多帧滑窗特征导出（对齐 VGGT/DINOv3/DA3）")

    p.add_argument("--rgb_root", type=str, default="/home/gl/features_model/rgb_dataset/RGB")
    p.add_argument(
        "--out_root",
        type=str,
        default="/home/gl/features_model/rgb_dataset/features_croco_v2_encoder_dict",
        help="输出根目录（将按 task/episode 保存 .pt）",
    )

    p.add_argument(
        "--ckpt_path",
        type=str,
        default="/home/gl/features_model/croco/pretrained_models/CroCo_V2_ViTLarge_BaseDecoder.pth",
    )

    p.add_argument("--window_size", type=int, default=8)
    p.add_argument("--stride", type=int, default=1)

    p.add_argument("--task", type=str, default=None, help="只处理某一个 task 目录名")
    p.add_argument("--episode", type=str, default=None, help="只处理某一个 episode 目录名")
    p.add_argument("--all", action="store_true", help="全量导出（默认就是全量；加这个只是为了命令风格统一）")
    p.add_argument("--smoke", action="store_true", help="快速冒烟：只跑1个task/episode/window（CroCo 用 1 个 episode 冒烟）")

    p.add_argument("--device", type=str, default="cuda")

    # 统一成 save_dtype（croco 原脚本只支持 fp16/fp32）
    p.add_argument("--save_dtype", type=str, default="fp16", choices=["fp16", "fp32"])

    # 你现在的默认需求：保留时间维、不做融合
    p.add_argument("--keep_time_dim", action="store_true", default=True)
    p.add_argument("--also_save_fused", action="store_true", default=False)

    # 仍保留 fuse 选项（但如果你不打开 also_save_fused，融合结果不会保存）
    p.add_argument("--fusion", type=str, default="mean", choices=["mean", "max", "concat"])

    return p


def main() -> None:
    args = build_argparser().parse_args()

    device = get_device(args.device)
    autocast_dtype = get_autocast_dtype("auto")

    if not os.path.exists(args.ckpt_path):
        raise FileNotFoundError(f"找不到 ckpt: {args.ckpt_path}")

    model = load_croco_encoder(args.ckpt_path, device=device)
    model._ckpt_path = args.ckpt_path

    img_size = int(model.patch_embed.img_size[0])
    preprocess = build_preprocess(img_size=img_size)

    pairs = list_task_episode_dirs(args.rgb_root)

    # smoke：只跑 1 个 episode
    if args.smoke:
        pairs = pairs[:1]

    # task/episode 精确过滤
    filtered = []
    for task_dir, ep_dir in pairs:
        task_name = Path(task_dir).name
        ep_name = Path(ep_dir).name
        if args.task is not None and task_name != args.task:
            continue
        if args.episode is not None and ep_name != args.episode:
            continue
        filtered.append((task_dir, ep_dir))

    if not filtered:
        raise FileNotFoundError(
            f"没有匹配到任何 episode：rgb_root={args.rgb_root}, task={args.task}, episode={args.episode}"
        )

    # 逐 episode 导出（复用原逻辑，保证输出范式一致）
    for _task_dir, episode_dir in filtered:
        process_episode(
            model=model,
            episode_dir=episode_dir,
            out_root=args.out_root,
            preprocess=preprocess,
            window_size=int(args.window_size),
            stride=int(args.stride),
            fuse=args.fusion,
            keep_time_dim=bool(args.keep_time_dim),
            also_save_fused=bool(args.also_save_fused),
            save_dtype=str(args.save_dtype),
            autocast_dtype=autocast_dtype,
            device=device,
            overwrite=False,
        )


if __name__ == "__main__":
    # 让 torch 在导入时就初始化（某些环境下更稳定）
    _ = torch.__version__
    main()
