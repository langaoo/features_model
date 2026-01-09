"""tools/convert_episode_pt_to_zarr.py

把某个视觉模型导出的 `<task>/<episode>.pt` 转换为可随机访问的 Zarr 格式。

动机
- 目前 episode 的 per_frame_features 是一个巨大 tensor： [W,T,Hf,Wf,C]
  例如 CroCo 的 episode_0.pt 就有 515MB。
- 在训练中如果每次只需要某个 window/frame 的少量 token，整体 torch.load 会造成：
  - CPU RSS 上升/碎片化
  - IO 读放大
  - 多任务时很难稳定训练 5000 步

Zarr 方案
- 将 per_frame_features 存为 chunked array，按 window 维度切块（chunk=(1,T,Hf,Wf,C)）
- 训练时可以只读取某个 window 的数据：zarr_array[wi] -> [T,Hf,Wf,C]
- 同时保存 frame_paths（JSON）与 meta（JSON）

用法（单文件）
    PYTHONPATH=/home/gl/features_model python tools/convert_episode_pt_to_zarr.py \
      --pt rgb_dataset/features_croco_v2_encoder_dict_unified/<task>/episode_0.pt \
      --out rgb_dataset/features_croco_v2_encoder_dict_unified_zarr/<task>/episode_0.zarr

也支持 root 批量转换：
    --in_root <features_root> --out_root <features_root>_zarr --tasks ... --episodes 0..19
"""

from __future__ import annotations

import argparse
import json
import sys
import gc
from pathlib import Path
from typing import Iterable

# 允许直接用 `python tools/convert_episode_pt_to_zarr.py ...` 运行
# 注意：必须在导入 features_common 之前注入 repo root，否则会 ModuleNotFoundError。
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import zarr
from numcodecs import Blosc

from features_common.feature_pack import load_feature_pack


def _iter_episodes(episodes: int) -> Iterable[str]:
    for i in range(int(episodes)):
        yield f"episode_{i}"


def _discover_tasks(in_root: Path) -> list[str]:
    return sorted([p.name for p in in_root.iterdir() if p.is_dir()])


def _discover_episode_pts(task_dir: Path) -> list[Path]:
    return sorted(task_dir.glob("episode_*.pt"))


def convert_one(pt_path: Path, out_path: Path, *, compressor: str = "zstd", clevel: int = 3) -> None:
    pack = load_feature_pack(pt_path)
    if pack.per_frame_features is None:
        raise RuntimeError(f"per_frame_features is None: {pt_path}")
    if pack.frame_paths is None:
        raise RuntimeError(f"frame_paths is None: {pt_path}")

    pf = pack.per_frame_features
    # to numpy (keep dtype)
    # 注意：这里会把整个 episode tensor materialize 成 numpy（峰值内存=pf大小*2左右）。
    # 转换完成后会显式释放并 gc，配合 remove_pt 可以避免磁盘与内存双压力。
    pf_np = pf.detach().cpu().numpy()
    if pf_np.ndim != 5:
        raise ValueError(f"expect [W,T,Hf,Wf,C] but got {pf_np.shape}: {pt_path}")
    W, T, Hf, Wf, C = pf_np.shape

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        # overwrite
        import shutil

        shutil.rmtree(out_path)

    store = zarr.DirectoryStore(str(out_path))
    root = zarr.group(store=store)

    # compressor
    if compressor.lower() == "zstd":
        comp = Blosc(cname="zstd", clevel=int(clevel), shuffle=Blosc.SHUFFLE)
    elif compressor.lower() == "lz4":
        comp = Blosc(cname="lz4", clevel=int(clevel), shuffle=Blosc.SHUFFLE)
    else:
        comp = Blosc(cname="zstd", clevel=int(clevel), shuffle=Blosc.SHUFFLE)

    # chunk by window
    chunks = (1, T, Hf, Wf, C)
    arr = root.create_dataset(
        "per_frame_features",
        shape=(W, T, Hf, Wf, C),
        chunks=chunks,
        dtype=pf_np.dtype,
        compressor=comp,
        overwrite=True,
    )
    # write window by window to keep memory stable
    for wi in range(W):
        arr[wi] = pf_np[wi]

    # store small metadata as JSON
    (out_path / "frame_paths.json").write_text(json.dumps(pack.frame_paths), encoding="utf-8")
    (out_path / "meta.json").write_text(json.dumps(pack.meta), encoding="utf-8")
    (out_path / "shape.json").write_text(
        json.dumps({"W": W, "T": T, "Hf": Hf, "Wf": Wf, "C": C, "dtype": str(pf_np.dtype)}),
        encoding="utf-8",
    )

    # 显式释放大对象，尽量降低 RSS 峰值
    del pf_np, pf, pack
    gc.collect()


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert episode .pt FeaturePack to zarr")

    ap.add_argument("--pt", type=str, default="", help="单个 episode .pt 路径")
    ap.add_argument("--out", type=str, default="", help="输出 .zarr 路径")

    ap.add_argument("--in_root", type=str, default="", help="批量转换：features root")
    ap.add_argument("--out_root", type=str, default="", help="批量转换：输出 root")
    ap.add_argument("--tasks", type=str, nargs="*", default=[], help="批量转换：任务名列表")
    ap.add_argument("--episodes", type=int, default=0, help="批量转换：每个task的episode数（0表示不批量）")

    ap.add_argument("--auto", action="store_true", help="自动扫描 in_root 下所有 <task>/episode_*.pt 进行转换")
    ap.add_argument("--skip_existing", action="store_true", help="若目标 .zarr 已存在则跳过")
    ap.add_argument("--remove_pt", action="store_true", help="转换成功后删除源 .pt（节省磁盘，谨慎使用）")

    ap.add_argument("--compressor", type=str, default="zstd", choices=["zstd", "lz4"])
    ap.add_argument("--clevel", type=int, default=3)

    args = ap.parse_args()

    if str(args.pt).strip():
        pt = Path(str(args.pt))
        out = Path(str(args.out)) if str(args.out).strip() else pt.with_suffix(".zarr")
        convert_one(pt, out, compressor=str(args.compressor), clevel=int(args.clevel))
        print(f"[ok] {pt} -> {out}")
        return

    if not (str(args.in_root).strip() and str(args.out_root).strip()):
        raise SystemExit("Provide either --pt/--out OR --in_root/--out_root (plus --auto or --tasks/--episodes)")

    in_root = Path(str(args.in_root))
    out_root = Path(str(args.out_root))

    if bool(getattr(args, "auto", False)):
        tasks = list(args.tasks) if args.tasks else _discover_tasks(in_root)
        for task in tasks:
            task_dir = in_root / str(task)
            if not task_dir.exists():
                continue
            for pt in _discover_episode_pts(task_dir):
                out = out_root / str(task) / f"{pt.stem}.zarr"
                if bool(getattr(args, "skip_existing", False)) and out.exists():
                    print(f"[skip] exists: {out}")
                    continue
                convert_one(pt, out, compressor=str(args.compressor), clevel=int(args.clevel))
                print(f"[ok] {pt} -> {out}")
                if bool(getattr(args, "remove_pt", False)):
                    pt.unlink(missing_ok=True)
        return

    # legacy: explicit tasks + episodes count
    if not (args.tasks and int(args.episodes) > 0):
        raise SystemExit("Provide --auto OR --tasks/--episodes")

    for task in args.tasks:
        for ep in _iter_episodes(int(args.episodes)):
            pt = in_root / str(task) / f"{ep}.pt"
            if not pt.exists():
                print(f"[skip] missing: {pt}")
                continue
            out = out_root / str(task) / f"{ep}.zarr"
            if bool(getattr(args, "skip_existing", False)) and out.exists():
                print(f"[skip] exists: {out}")
                continue
            convert_one(pt, out, compressor=str(args.compressor), clevel=int(args.clevel))
            print(f"[ok] {pt} -> {out}")
            if bool(getattr(args, "remove_pt", False)):
                pt.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
