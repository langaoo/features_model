"""tools/select_episodes_subset.py

在不改动任何特征提取脚本的前提下，帮你选出某个 task 的前 N 个 episode 名称。

为什么需要它？
- 你希望先在 beat_block_hammer 上取 20 个 episode。
- 目录排序是字符串排序：episode_10 会排在 episode_2 前面。
- 这里提供两种选择策略：
  1) --mode numeric: 按 episode_\d+ 的数字排序（推荐）
  2) --mode lexicographic: 按文件名字符串排序（不推荐）

输出
- 默认打印 episode 名称（每行一个），可重定向到文件。
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


_EP_RE = re.compile(r"^episode_(\d+)$")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rgb_root", type=str, default="/home/gl/features_model/rgb_dataset/RGB")
    ap.add_argument("--task", type=str, required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--mode", type=str, default="numeric", choices=["numeric", "lexicographic"])
    args = ap.parse_args()

    task_dir = Path(args.rgb_root) / args.task
    if not task_dir.is_dir():
        raise FileNotFoundError(f"task 不存在: {task_dir}")

    eps = [p.name for p in task_dir.iterdir() if p.is_dir()]

    if args.mode == "lexicographic":
        eps = sorted(eps)
    else:
        def key(x: str):
            m = _EP_RE.match(x)
            return int(m.group(1)) if m else 10**9

        eps = sorted(eps, key=key)

    for e in eps[: max(args.n, 0)]:
        print(e)


if __name__ == "__main__":
    main()
