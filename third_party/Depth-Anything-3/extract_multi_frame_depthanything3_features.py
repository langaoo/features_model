"""Depth Anything 3 多帧滑动窗口特征导出脚本（对齐 CroCo / VGGT / DINOv3 范式）

目标
- 输入：rgb_dataset/RGB 下的 task/episode 连续帧（.png/.jpg）
- 切分：window_size=8, stride=1（可配）
- 提取：使用 Depth Anything 3 的 backbone (DINOv2) 中间层 patch tokens
- 输出：每个 episode 保存一个 .pt，内容为 dict：
  - per_frame_features: [NumWindows, T, Hf, Wf, C]（默认）
  - frame_paths: List[List[str]]  # 每个窗口的帧路径
  - meta: dict  # 记录配置、shape、模型等
  - （可选）features: 融合后的 [NumWindows, Hf, Wf, C]

实现选择
- Depth Anything 3 的 backbone 直接支持多视图输入 x: (B, N, 3, H, W)。我们把“多帧视频”当作 N 张 view 输入。
- 通过 DepthAnything3.forward(export_feat_layers=[layer]) 取回 output["aux"][layer]。
  该 aux feature 的形状为 (B, N, num_patches, C)，其中 num_patches = (H/14)*(W/14)。
- 由于 DA3 预处理会把每张图 resize 到 process_res 附近并保证 H/W 都能被 14 整除，因此 Hf=Wf=H/14,W/14。

注意
- 该脚本不依赖 HuggingFace 下载；默认从本地 `Depth-Anything-3/weight` 加载 config + safetensors。
- 你需要在 conda 环境 `depth3` 中运行，并确保 `pip install -e Depth-Anything-3`（或把该 repo 加到 PYTHONPATH）。

"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import zarr
import numpy as np
import json


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _is_image(p: str) -> bool:
    return Path(p).suffix.lower() in IMG_EXTS


def list_episode_frames(episode_dir: str) -> list[str]:
    """列出一个 episode 下的所有帧路径（按文件名排序）。"""
    paths = [str(p) for p in sorted(Path(episode_dir).iterdir()) if p.is_file() and _is_image(str(p))]
    return paths


def iter_sliding_windows(frame_paths: list[str], window_size: int, stride: int):
    n = len(frame_paths)
    if n < window_size:
        return
    for start in range(0, n - window_size + 1, stride):
        yield start, frame_paths[start : start + window_size]


def fuse_time_mean(per_frame: torch.Tensor) -> torch.Tensor:
    """[T,Hf,Wf,C] -> [Hf,Wf,C]"""
    return per_frame.mean(dim=0)


def tokens_to_map(tokens: torch.Tensor, hf: int, wf: int) -> torch.Tensor:
    """把 patch tokens reshape 回 feature map。

    tokens: [num_patches, C]
    return: [Hf, Wf, C]
    """
    return tokens.reshape(hf, wf, tokens.shape[-1])


@dataclass
class EpisodePack:
    per_frame_features: torch.Tensor  # [W,T,Hf,Wf,C]
    frame_paths: list[list[str]]
    meta: dict[str, Any]
    features: torch.Tensor | None = None  # [W,Hf,Wf,C]


def build_da3(model_dir: str, device: torch.device):
    """从本地目录加载 DA3 模型（config.json + model.safetensors）。"""
    from depth_anything_3.api import DepthAnything3

    model_dir = str(model_dir)
    model = DepthAnything3.from_pretrained(model_dir)
    model = model.to(device=device)
    model.eval()
    return model


@torch.inference_mode()
def extract_window_features(
    model,
    image_paths: list[str],
    *,
    export_layer: int,
    process_res: int,
    process_res_method: str,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """提取一个窗口 (T 张图) 的 per-frame patch feature map。

    Returns:
      per_frame_map: [T,Hf,Wf,C] (dtype)
      meta_extra: 记录处理分辨率、patch grid 等
    """
    # 用 DepthAnything3 API 的 preprocess，保证跟官方一致。
    imgs_cpu, _, _ = model._preprocess_inputs(
        image_paths,
        extrinsics=None,
        intrinsics=None,
        process_res=process_res,
        process_res_method=process_res_method,
    )
    imgs, _, _ = model._prepare_model_inputs(imgs_cpu, None, None)
    imgs = imgs.to(device)

    # 走一条更稳的路径：直接从 backbone 抽中间层 tokens。
    # 原因：DepthAnything3Net 会尝试把 backbone 返回的 aux_feats 进一步 reshape，且有 assert 校验。
    # 在部分配置/版本下，aux_feats 可能为空导致断言失败。
    # backbone.get_intermediate_layers 的返回更稳定。
    backbone = model.model.backbone  # depth_anything_3.model.dinov2.dinov2.DinoV2
    # 该调用返回：
    #   feats: tuple(zip(outputs, camera_tokens)) 其中 outputs 是 [B,N,num_patches,C]
    # 我们只需要最后一个（或者指定 n=[export_layer] 的逻辑在本实现里等价用 out_layers 控制）
    feats, _aux = backbone(x=imgs, export_feat_layers=[export_layer])
    if not feats:
        raise RuntimeError("backbone 没有返回任何中间层输出（feats 为空）。")
    # 取第一个返回（当前配置下只请求了一个 layer）
    tokens, _cam_tokens = feats[0]  # tokens: [B,N,num_patches,C]
    if tokens.dim() != 4:
        raise RuntimeError(f"backbone tokens 形状不符合预期: {tuple(tokens.shape)}")
    assert tokens.shape[0] == 1, "该脚本按 episode 逐窗口处理，batch 固定为 1"
    t = int(tokens.shape[1])

    # 根据 preprocess 后的 H/W 推回 patch grid（DA3 预处理保证可整除 14）
    h, w = int(imgs.shape[-2]), int(imgs.shape[-1])
    patch = 14
    if h % patch != 0 or w % patch != 0:
        raise RuntimeError(f"预处理后 H/W 不能被 14 整除: H={h}, W={w}")
    hf, wf = h // patch, w // patch

    b, t_, n_patches, c = tokens.shape
    if hf * wf != n_patches:
        raise RuntimeError(f"num_patches 不匹配: got {n_patches}, expected {hf*wf} (hf={hf}, wf={wf})")
    tokens = tokens[0]  # [T,HW,C]
    maps = torch.stack([tokens_to_map(tokens[i], hf, wf) for i in range(t_)], dim=0)

    maps = maps.to(dtype=dtype).contiguous()
    meta_extra = {
        "processed_hw": (int(h), int(w)),
        "patch_size": patch,
        "map_hw": (int(hf), int(wf)),
        "channels": int(c),
    }
    return maps, meta_extra


def process_episode(
    *,
    model,
    episode_dir: str,
    out_path: str,
    window_size: int,
    stride: int,
    export_layer: int,
    process_res: int,
    process_res_method: str,
    keep_time_dim: bool,
    also_save_fused: bool,
    device: torch.device,
    save_dtype: str,
    max_windows: int | None,
):
    frame_paths = list_episode_frames(episode_dir)
    windows = list(iter_sliding_windows(frame_paths, window_size, stride))
    # IMPORTANT: 正式导出不允许截断 windows（会丢数据段）。
    # max_windows 仅作调试用途；为了防止误传导致 silent truncation，这里选择直接报错。
    if max_windows is not None:
        raise ValueError(
            f"不允许在正式导出中使用 --max_windows（会丢数据段）。"
            f"请去掉该参数；如需冒烟请用 --smoke。收到 max_windows={max_windows}"
        )

    if len(windows) == 0:
        return False

    expected_windows = (len(frame_paths) - window_size) // stride + 1 if len(frame_paths) >= window_size else 0
    if expected_windows != len(windows):
        raise RuntimeError(
            f"DA3 windows 数不一致：num_frames={len(frame_paths)} window_size={window_size} stride={stride} "
            f"expected_windows={expected_windows} actual_windows={len(windows)} (episode_dir={episode_dir})"
        )

    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    if save_dtype not in dtype_map:
        raise ValueError(f"save_dtype 必须是 {list(dtype_map.keys())}，但收到 {save_dtype}")
    dtype = dtype_map[save_dtype]

    # 输出路径：严格对齐 CroCo
    task_name = os.path.basename(os.path.dirname(episode_dir))
    episode_name = os.path.basename(episode_dir)
    save_dir = Path(out_path) / task_name
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{episode_name}.zarr"
    if save_path.exists():
        import shutil
        shutil.rmtree(save_path)

    # zarr dtype
    np_dtype = "float32"
    if save_dtype == "fp16":
        np_dtype = "float16"
    elif save_dtype == "bf16":
        np_dtype = "float16"

    # 先抽一个窗口确定 shape
    first_win_paths = windows[0][1]
    feat_0, meta_0 = extract_window_features(
        model,
        first_win_paths,
        export_layer=export_layer,
        process_res=process_res,
        process_res_method=process_res_method,
        device=device,
        dtype=dtype,
    )
    T, Hf, Wf, C = feat_0.shape
    num_windows = len(windows)

    store = zarr.DirectoryStore(str(save_path))
    root = zarr.group(store=store, overwrite=True)
    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=zarr.Blosc.SHUFFLE)
    ds_per_frame = root.create_dataset(
        "per_frame_features",
        shape=(num_windows, T, Hf, Wf, C),
        chunks=(1, T, Hf, Wf, C),
        dtype=np_dtype,
        compressor=compressor,
    )

    all_window_frame_paths: list[list[str]] = []
    for wi, (_start, w_paths) in enumerate(windows):
        feat, _ = extract_window_features(
            model,
            w_paths,
            export_layer=export_layer,
            process_res=process_res,
            process_res_method=process_res_method,
            device=device,
            dtype=dtype,
        )
        ds_per_frame[wi] = feat.to(torch.float32).cpu().numpy().astype(np_dtype)
        all_window_frame_paths.append(w_paths)

    meta_dict = {
        "model_name": "da3",
        "episode_dir": os.path.abspath(episode_dir),
        "task_name": task_name,
        "episode_name": episode_name,
        "num_frames": len(frame_paths),
        "window_size": window_size,
        "stride": stride,
        "num_windows": num_windows,
        "fuse": "mean",
        "keep_time_dim": bool(keep_time_dim),
        "also_save_fused": bool(also_save_fused),
        "img_size": int(meta_0.get("processed_hw", (0, 0))[0]) if isinstance(meta_0, dict) else None,
        "patch_size": int(meta_0.get("patch_size", 14)) if isinstance(meta_0, dict) else 14,
        "enc_embed_dim": int(C),
        "save_dtype": save_dtype,
    }
    root.attrs["meta"] = meta_dict
    root.attrs["frame_paths"] = all_window_frame_paths

    # 三个 json
    with open(os.path.join(str(save_path), "meta.json"), "w") as f:
        json.dump(meta_dict, f, indent=2)
    with open(os.path.join(str(save_path), "frame_paths.json"), "w") as f:
        json.dump(all_window_frame_paths, f)
    with open(os.path.join(str(save_path), "shape.json"), "w") as f:
        json.dump({"W": num_windows, "T": T, "Hf": Hf, "Wf": Wf, "C": C, "dtype": save_dtype}, f)

    return True


def extract_episode(
    model,
    episode_dir: Path,
    out_root: Path,
    window_size: int,
    stride: int,
    export_layer: int,
    process_res: int,
    process_res_method: str,
    device: torch.device,
    dtype: torch.dtype,
    overwrite: bool,
):
    # NOTE: legacy helper removed; main 导出逻辑使用 process_episode 直接写 CroCo 风格的 episode_x.zarr。
    raise RuntimeError("extract_episode 已弃用，请使用 process_episode")


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--rgb_root", type=str, default="/home/gl/features_model/rgb_dataset/RGB")
    ap.add_argument(
        "--out_root",
        type=str,
        default="/home/gl/features_model/rgb_dataset/features_depthanything3_encoder_dict",
    )

    ap.add_argument(
        "--model_dir",
        type=str,
        default="/home/gl/features_model/Depth-Anything-3/weight",
        help="本地权重目录（包含 config.json & model.safetensors）。",
    )

    ap.add_argument("--window_size", type=int, default=8)
    ap.add_argument("--stride", type=int, default=1)

    ap.add_argument(
        "--export_layer",
        type=int,
        default=23,
        help=(
            "导出 backbone 的中间层特征（对应 transformer block index）。"
            "对 DA3-Large（vitl）来说，官方多尺度 out_layers 常用 [11,15,19,23]；"
            "如果你想要更偏空间几何/深度 head 的表征，默认推荐 23。"
        ),
    )

    ap.add_argument("--process_res", type=int, default=504)
    ap.add_argument(
        "--process_res_method",
        type=str,
        default="upper_bound_resize",
        choices=["upper_bound_resize", "lower_bound_resize", "upper_bound_crop", "lower_bound_crop"],
    )

    ap.add_argument("--keep_time_dim", action="store_true", default=True)
    ap.add_argument("--also_save_fused", action="store_true", default=False)
    ap.add_argument("--save_dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])

    ap.add_argument(
        "--task",
        type=str,
        default=None,
        help="只处理某一个 task（对应 rgb_root 下的子目录名）。",
    )
    ap.add_argument(
        "--episode",
        type=str,
        default=None,
        help="只处理某一个 episode（对应 task 下的子目录名，例如 episode_0）。",
    )
    ap.add_argument(
        "--all",
        action="store_true",
        help="全量导出：遍历 rgb_root 下所有 task/episode。默认就是全量；加这个只是为了让命令风格更统一。",
    )
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="快速冒烟：等价于 --max_tasks 1 --max_episodes 1（用于验证环境/shape）。",
    )

    # 仍保留 max_* 作为调试选项（不建议在正式全量导出时使用）
    ap.add_argument("--max_tasks", type=int, default=None, help="[调试] 限制 task 数")
    ap.add_argument("--max_episodes", type=int, default=None, help="[调试] 限制 episode 数")
    ap.add_argument("--max_windows", type=int, default=None, help="[调试] 限制窗口数")

    ap.add_argument("--device", type=str, default="cuda")

    args = ap.parse_args()

    if args.smoke:
        args.max_tasks = 1
        args.max_episodes = 1
        # IMPORTANT: process_episode 明确禁止 max_windows（避免 silent truncation）。
        # smoke 只缩小 task/episode 范围，不截断窗口序列。

    device = torch.device(args.device)

    # 尽量保证能 import 到 depth_anything_3（建议已 pip install -e Depth-Anything-3）
    model = build_da3(args.model_dir, device)

    rgb_root = Path(args.rgb_root)
    out_root = Path(args.out_root)

    if args.task is not None:
        tasks = [rgb_root / args.task]
    else:
        tasks = [p for p in sorted(rgb_root.iterdir()) if p.is_dir()]
    if args.max_tasks is not None:
        tasks = tasks[: max(args.max_tasks, 0)]

    num_saved = 0
    for task_dir in tasks:
        if args.episode is not None:
            episodes = [task_dir / args.episode]
        else:
            episodes = [p for p in sorted(task_dir.iterdir()) if p.is_dir()]
        if args.max_episodes is not None:
            episodes = episodes[: max(args.max_episodes, 0)]

        for ep_dir in episodes:
            ok = process_episode(
                model=model,
                episode_dir=str(ep_dir),
                out_path=str(out_root),
                window_size=args.window_size,
                stride=args.stride,
                export_layer=args.export_layer,
                process_res=args.process_res,
                process_res_method=args.process_res_method,
                keep_time_dim=args.keep_time_dim,
                also_save_fused=args.also_save_fused,
                device=device,
                save_dtype=args.save_dtype,
                max_windows=args.max_windows,
            )
            if ok:
                num_saved += 1

    print(f"Done. Saved {num_saved} episodes to {out_root}.")


if __name__ == "__main__":
    main()
