"""tools/run_extract_first_n_episodes.py

统一批处理入口：在某个 task 下选前 N 个 episode，并循环调用“某个模型的提特征脚本”。

你当前的四个模型脚本都支持统一参数：
- --rgb_root --out_root --task --episode
并且默认只保存 per_frame_features，不融合。

本脚本不依赖 99999 hack，也不要求各脚本必须实现 --max_episodes。

注意
- 该脚本是“调度器”，会启动子进程逐个 episode 调用。
- 你可以用同一套调度器分别跑 croco/vggt/dinov3/da3，只需要换 --script。

示例
- 跑 CroCo：
  python tools/run_extract_first_n_episodes.py \
    --script croco/extract_multi_frame_croco_features_unified.py \
    --task beat_block_hammer-demo_randomized-50_head_camera \
    --n 20 \
    --out_root /abs/features_croco_v2_encoder_dict_unified

- 跑 VGGT：
  python tools/run_extract_first_n_episodes.py \
    --script vggt/extract_multi_frame_vggt_features_wrapper.py \
    --task beat_block_hammer-demo_randomized-50_head_camera \
    --n 20 \
    --out_root /abs/features_vggt_encoder_dict_unified

"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
from pathlib import Path


_EP_RE = re.compile(r"^episode_(\d+)$")


def list_episodes(rgb_root: str, task: str, *, mode: str) -> list[str]:
    task_dir = Path(rgb_root) / task
    eps = [p.name for p in task_dir.iterdir() if p.is_dir()]
    if mode == "lexicographic":
        return sorted(eps)

    def key(x: str):
        m = _EP_RE.match(x)
        return int(m.group(1)) if m else 10**9

    return sorted(eps, key=key)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", type=str, required=True, help="要调用的提特征脚本相对路径")
    ap.add_argument("--rgb_root", type=str, default="/home/gl/features_model/rgb_dataset/RGB")
    ap.add_argument("--out_root", type=str, required=True)
    ap.add_argument("--task", type=str, required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--mode", type=str, default="numeric", choices=["numeric", "lexicographic"])

    # 透传常用参数（所有脚本都有，不同脚本多余参数会被忽略吗？不会——argparse 会报错）
    # 所以我们这里仅透传“所有脚本都确认存在”的公共参数。
    ap.add_argument("--window_size", type=int, default=8)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--device", type=str, default=None, help="不传则由子脚本自己决定")
    ap.add_argument("--save_dtype", type=str, default=None, help="不传则由子脚本自己决定")
    ap.add_argument("--keep_time_dim", action="store_true", default=True)

    # 允许额外参数：用 --extra "--model_dir ..." 这种方式传给子脚本
    ap.add_argument("--extra", type=str, default="", help="额外参数字符串（会按 shell 方式拆分）")

    args = ap.parse_args()

    script_path = Path("/home/gl/features_model") / args.script
    if not script_path.exists():
        raise FileNotFoundError(f"script 不存在: {script_path}")

    eps = list_episodes(args.rgb_root, args.task, mode=args.mode)[: max(args.n, 0)]
    if not eps:
        raise RuntimeError("没有找到任何 episode")

    # 兼容两类脚本接口：
    # 1) unified:  --rgb_root --out_root --task --episode
    # 2) legacy(CroCo): --dataset_root --output_root --task --episode
    #
    # 不能“多传参数让子脚本忽略”，因为 argparse 会直接报错。
    help_out = subprocess.run(
        ["python", str(script_path), "-h"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=os.environ.copy(),
    ).stdout

    supports_unified = ("--rgb_root" in help_out) and ("--out_root" in help_out)
    supports_legacy = ("--dataset_root" in help_out) and ("--output_root" in help_out)
    if supports_unified:
        base = [
            "python",
            str(script_path),
            "--rgb_root",
            args.rgb_root,
            "--out_root",
            args.out_root,
            "--task",
            args.task,
        ]
    elif supports_legacy:
        base = [
            "python",
            str(script_path),
            "--dataset_root",
            args.rgb_root,
            "--output_root",
            args.out_root,
            "--task",
            args.task,
        ]
    else:
        raise RuntimeError(
            "无法识别脚本参数接口：未检测到 (--rgb_root,--out_root) 或 (--dataset_root,--output_root)。"
        )

    # window/stride/device/save_dtype/keep_time_dim：尽量传；若子脚本不支持则报错，用户可用 --extra 自己控制。
    base += ["--window_size", str(args.window_size), "--stride", str(args.stride)]
    if args.keep_time_dim:
        base += ["--keep_time_dim"]
    if args.device:
        base += ["--device", str(args.device)]
    if args.save_dtype:
        base += ["--save_dtype", str(args.save_dtype)]

    extra = shlex.split(args.extra) if args.extra.strip() else []

    for e in eps:
        cmd = base + ["--episode", e] + extra
        print("\n[RUN]", " ".join(cmd))
        subprocess.run(cmd, check=True, env=os.environ.copy())


if __name__ == "__main__":
    main()
