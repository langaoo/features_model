"""tools/convert_all_vis_roots_to_zarr.py

把多个视觉特征 root 下的所有 <task>/episode_*.pt 批量转换为 <out_root>/<task>/episode_*.zarr。

动机
- 路线2训练需要随机访问某个 step/frame 的 token。
- 直接 torch.load 整个 episode_*.pt（可能 100~500MB）会导致 CPU RSS 暴涨。
- Zarr 可以按 chunk（window）读取，避免读放大。

用法示例（四模型）：

python tools/convert_all_vis_roots_to_zarr.py \
  --in_roots \
    /home/gl/features_model/rgb_dataset/features_croco_v2_encoder_dict_unified \
    /home/gl/features_model/rgb_dataset/features_da3_encoder_dict_unified \
    /home/gl/features_model/rgb_dataset/features_dinov3_encoder_dict_unified_safe \
    /home/gl/features_model/rgb_dataset/features_vggt_encoder_dict_unified \
  --out_suffix _zarr \
  --skip_existing

可选参数
- --tasks：只转换指定任务
- --episodes：只转换 episode_0 episode_1 ...（可多次传入）
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.convert_episode_pt_to_zarr import convert_one  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch convert multiple vis_roots pt -> zarr")
    ap.add_argument("--in_roots", type=str, nargs="+", required=True)
    ap.add_argument("--out_roots", type=str, nargs="*", default=[])
    ap.add_argument("--out_suffix", type=str, default="_zarr", help="当未提供 --out_roots 时，输出 root = in_root + out_suffix")
    ap.add_argument("--tasks", type=str, nargs="*", default=[], help="只转换这些 task（默认转换全部）")
    ap.add_argument("--episodes", type=str, nargs="*", default=[], help="只转换这些 episode（例如 episode_0 episode_1；默认转换全部）")
    ap.add_argument("--skip_existing", action="store_true")
    ap.add_argument("--remove_pt", action="store_true", help="转换成功后删除源 .pt（节省磁盘，谨慎使用）")
    ap.add_argument("--compressor", type=str, default="zstd", choices=["zstd", "lz4"])
    ap.add_argument("--clevel", type=int, default=3)

    args = ap.parse_args()

    in_roots = [Path(p) for p in args.in_roots]
    if args.out_roots:
        if len(args.out_roots) != len(in_roots):
            raise SystemExit("--out_roots 数量必须与 --in_roots 一致")
        out_roots = [Path(p) for p in args.out_roots]
    else:
        out_roots = [Path(str(p) + str(args.out_suffix)) for p in in_roots]

    tasks_filter = set(args.tasks) if args.tasks else None
    episodes_filter = set(args.episodes) if args.episodes else None

    ok_cnt = 0
    skip_cnt = 0
    fail_cnt = 0

    for in_root, out_root in zip(in_roots, out_roots):
        if not in_root.exists():
            print(f"[skip] missing root: {in_root}")
            continue
        out_root.mkdir(parents=True, exist_ok=True)

        tasks = [p for p in sorted(in_root.iterdir()) if p.is_dir()]
        for task_dir in tasks:
            task = task_dir.name
            if tasks_filter is not None and task not in tasks_filter:
                continue

            for pt in sorted(task_dir.glob("episode_*.pt")):
                if episodes_filter is not None and pt.stem not in episodes_filter:
                    continue

                out = out_root / task / f"{pt.stem}.zarr"
                if args.skip_existing and out.exists():
                    print(f"[skip] exists: {out}")
                    skip_cnt += 1
                    continue

                try:
                    convert_one(pt, out, compressor=str(args.compressor), clevel=int(args.clevel))
                    print(f"[ok] {pt} -> {out}")
                    ok_cnt += 1
                    if bool(getattr(args, "remove_pt", False)):
                        pt.unlink(missing_ok=True)
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    fail_cnt += 1
                    print(f"[fail] {pt} -> {out} err={type(e).__name__}: {e}")

    print(f"[done] ok={ok_cnt} skip={skip_cnt} fail={fail_cnt}")


if __name__ == "__main__":
    main()
