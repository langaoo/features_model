"""extract_multi_frame_vggt_features_wrapper.py

核心用途
- 在机器人多任务/多 episode/连续 RGB 帧的数据集上，使用 VGGT 的 aggregator 提取“多帧窗口”的 encoder 特征。
- 按滑动窗口切分（window_size=8, stride=1），每个窗口输出每帧的 patch feature map。
- 以与 CroCo 脚本一致的 dict 形式保存：
  - per_frame_features: [NumWindows, S, Hf, Wf, C]
  - (可选) features:      [NumWindows, Hf, Wf, C] 或其它融合形式
  - frame_paths:          List[List[str]]，每个窗口对应 S 帧路径
  - meta:                 dict（task/episode信息 + 预处理参数等）

重要约束
- 不修改任何 vggt 原文件（multi_frame_vggt_features.py 等）。
- 仅作为 wrapper 调用 vggt 包内部 API。

为什么你看到的 VGGT PCA 图很“平滑”？
- VGGT 输出是高维语义特征；PCA 压到 3 维本来就会丢很多边界信息。
- 另外 VGGT 的默认预处理会 resize 到 518x518，patch_size=14 => Hf/Wf 不是 14x14，而是 37x37（或 37x28 之类）。
  这种分辨率下做 PCA 伪彩色，人眼更容易觉得“色块连续”。

建议：用更可靠的 sanity check
- 同一窗口内相邻帧的 cosine 应该很高但 < 1。
- 不同窗口的 cosine 应该略低。
- per_frame_features 的方差/能量应明显非零且无 NaN/Inf。

"""

from __future__ import annotations

import argparse
from pathlib import Path
import glob
import os
from typing import List, Optional

import torch
from tqdm import tqdm
import zarr
import numpy as np
import json


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_amp_dtype() -> torch.dtype:
    # Ampere(8.0+) 用 bfloat16 更稳
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        return torch.bfloat16
    return torch.float16


def list_task_dirs(dataset_root: str) -> List[str]:
    return sorted([p for p in glob.glob(os.path.join(dataset_root, "*")) if os.path.isdir(p)])


def list_episode_dirs(task_dir: str) -> List[str]:
    return sorted([p for p in glob.glob(os.path.join(task_dir, "*")) if os.path.isdir(p)])


def get_image_paths(episode_dir: str) -> List[str]:
    exts = ("*.png", "*.jpg", "*.jpeg")
    paths: List[str] = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(episode_dir, ext)))
    paths.sort()
    return paths


def sliding_windows(paths: List[str], window_size: int, stride: int) -> List[List[str]]:
    n = len(paths)
    if n < window_size:
        return []
    return [paths[i : i + window_size] for i in range(0, n - window_size + 1, stride)]


def _import_vggt():
    # 让脚本可以从 workspace 根目录直接运行
    # /home/gl/features_model 是 repo 根，里面有 vggt/ (package root)
    from vggt.models.vggt import VGGT  # type: ignore
    from vggt.utils.load_fn import load_and_preprocess_images  # type: ignore

    return VGGT, load_and_preprocess_images


def load_model(model_path: str, device: str) -> torch.nn.Module:
    VGGT, _ = _import_vggt()

    print(f"[VGGT] 加载模型: {model_path}")
    model = VGGT()
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return model


def encode_window(
    model: torch.nn.Module,
    window_paths: List[str],
    device: str,
    amp_dtype: torch.dtype,
) -> torch.Tensor:
    """返回 feature_map: [S, Hf, Wf, C] (CPU tensor)."""

    _, load_and_preprocess_images = _import_vggt()

    # load_and_preprocess_images: [S, 3, H, W]，默认 H=W=518
    images = load_and_preprocess_images(window_paths).to(device)
    images = images.unsqueeze(0)  # [1, S, 3, H, W]

    with torch.no_grad():
        if device == "cuda":
            with torch.cuda.amp.autocast(dtype=amp_dtype):
                aggregated_tokens_list, patch_start_idx = model.aggregator(images)
        else:
            aggregated_tokens_list, patch_start_idx = model.aggregator(images)

        last_layer_tokens = aggregated_tokens_list[-1]  # [B, S, TotalTokens, C]
        patch_tokens = last_layer_tokens[:, :, patch_start_idx:, :]  # [B, S, N, C]
        b, s, n, c = patch_tokens.shape
        if b != 1:
            raise RuntimeError(f"仅支持 B=1 的推理，当前 B={b}")

        # patch_size 在原脚本里假定 14
        _, _, _, H, W = images.shape
        patch_size = 14
        Hf = H // patch_size
        Wf = W // patch_size
        if Hf * Wf != n:
            # 安全起见：如果模型 token 排列与简单 grid 不一致，直接报错提示
            raise RuntimeError(f"patch tokens 数量不匹配网格: n={n}, Hf*Wf={Hf*Wf}, H={H}, W={W}")

        feat_map = patch_tokens.view(b, s, Hf, Wf, c)[0]  # [S, Hf, Wf, C]

    return feat_map.detach().cpu()


def fuse_per_frame(
    per_frame: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    """per_frame: [S, Hf, Wf, C] -> fused: [Hf, Wf, C] or [Hf,Wf,S*C]."""

    if mode == "mean":
        return per_frame.mean(dim=0)
    if mode == "max":
        return per_frame.max(dim=0).values
    if mode == "concat":
        s, hf, wf, c = per_frame.shape
        return per_frame.permute(1, 2, 0, 3).reshape(hf, wf, s * c)
    raise ValueError(f"不支持的 fusion: {mode}")


def process_episode(
    model: torch.nn.Module,
    episode_dir: str,
    out_root: str,
    window_size: int,
    stride: int,
    device: str,
    amp_dtype: torch.dtype,
    overwrite: bool,
    max_windows: int | None = None,
) -> None:
    task_name = os.path.basename(os.path.dirname(episode_dir))
    episode_name = os.path.basename(episode_dir)
    
    save_dir = os.path.join(out_root, task_name)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{episode_name}.zarr")

    if os.path.exists(save_path):
        if not overwrite:
            print(f"[Skip] {save_path}")
            return
        import shutil
        shutil.rmtree(save_path)

    image_paths = get_image_paths(episode_dir)
    if not image_paths:
        return

    windows = sliding_windows(image_paths, window_size, stride)
    if not windows:
        return
    if max_windows is not None:
        windows = windows[:max_windows]

    # Infer shape
    # VGGT usually 518x518 -> 37x37 patches.
    # encode_window returns [T, Hf, Wf, C]
    with torch.no_grad():
        feat_0 = encode_window(model, windows[0], device, amp_dtype)
        T, Hf, Wf, C = feat_0.shape

    num_windows = len(windows)
    
    store = zarr.DirectoryStore(save_path)
    root = zarr.group(store=store, overwrite=True)
    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=zarr.Blosc.SHUFFLE)

    ds_per_frame = root.create_dataset(
        "per_frame_features",
        shape=(num_windows, T, Hf, Wf, C),
        chunks=(1, T, Hf, Wf, C),
        dtype="float32",
        compressor=compressor
    )

    all_window_frame_paths = []
    print(f"[Process] {task_name}/{episode_name} -> {save_path}")

    for wi, w_paths in enumerate(windows):
        feat = encode_window(model, w_paths, device, amp_dtype)
        ds_per_frame[wi] = feat.to(torch.float32).cpu().numpy()
        all_window_frame_paths.append(w_paths)

    meta_dict = {
        "model_name": "vggt",
        "episode_dir": episode_dir,
        "task_name": task_name,
        "episode_name": episode_name,
        "num_frames": len(image_paths),
        "window_size": window_size,
        "stride": stride,
        "num_windows": num_windows,
        "fuse": "mean",
        "keep_time_dim": True,
        "also_save_fused": False,
        "img_size": int(518),
        "patch_size": 14,
        "enc_embed_dim": int(C),
        "save_dtype": "fp32",
    }

    root.attrs["meta"] = meta_dict
    root.attrs["frame_paths"] = all_window_frame_paths

    # 严格对齐 CroCo：落盘三个 json 文件
    with open(os.path.join(save_path, "meta.json"), "w") as f:
        json.dump(meta_dict, f, indent=2)
    with open(os.path.join(save_path, "frame_paths.json"), "w") as f:
        json.dump(all_window_frame_paths, f)
    shape_dict = {"W": num_windows, "T": T, "Hf": Hf, "Wf": Wf, "C": C, "dtype": "fp32"}
    with open(os.path.join(save_path, "shape.json"), "w") as f:
        json.dump(shape_dict, f)
    print(f"[Done] {save_path}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="VGGT wrapper：多帧滑窗提特征并按 CroCo 风格保存 dict（含 frame_paths/meta）。"
    )

    # 统一命名（推荐使用）
    p.add_argument("--rgb_root", type=str, default="/home/gl/features_model/rgb_dataset/RGB")
    p.add_argument("--out_root", type=str, default="/home/gl/features_model/rgb_dataset/features_vggt_encoder_dict")
    # 兼容旧命名
    p.add_argument("--dataset_root", type=str, default=None, help="[兼容] 同 --rgb_root")
    p.add_argument("--output_root", type=str, default=None, help="[兼容] 同 --out_root")
    p.add_argument("--model_path", type=str, default="/home/gl/features_model/vggt/weight/model.pt")

    p.add_argument("--window_size", type=int, default=8)
    p.add_argument("--stride", type=int, default=1)

    p.add_argument("--keep_time_dim", action="store_true", help="保存 per_frame_features（保留时间维）")
    p.add_argument("--also_save_fused", action="store_true", help="同时保存融合后的 features")
    p.add_argument("--fusion", type=str, default="mean", choices=["mean", "max", "concat"], help="窗口内融合方式")

    p.add_argument("--save_dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"], help="保存到磁盘的 dtype")
    p.add_argument("--overwrite", action="store_true")

    # 统一控制：默认全量；--smoke 用于冒烟。
    p.add_argument("--task", type=str, default=None, help="只处理某一个 task 目录名")
    p.add_argument("--episode", type=str, default=None, help="只处理某一个 episode 目录名")
    p.add_argument("--all", action="store_true", help="全量导出（默认就是全量；加这个只是为了命令风格统一）")
    p.add_argument("--smoke", action="store_true", help="快速冒烟：只跑1个task/episode/window")

    # 仍保留 max_* 作为调试参数
    p.add_argument("--max_tasks", type=int, default=None)
    p.add_argument("--max_episodes", type=int, default=None)
    p.add_argument("--max_windows", type=int, default=None)

    p.add_argument("--cuda_visible_devices", type=str, default=None)

    return p


def main() -> None:
    args = build_argparser().parse_args()

    # 兼容旧命名
    if args.dataset_root is not None:
        args.rgb_root = args.dataset_root
    if args.output_root is not None:
        args.out_root = args.output_root

    if args.smoke:
        args.max_tasks = 1
        args.max_episodes = 1
        args.max_windows = 1

    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)

    device = get_device()
    amp_dtype = get_amp_dtype()
    print(f"[设备] {device} | amp_dtype={amp_dtype if device == 'cuda' else 'fp32'}")

    model = load_model(args.model_path, device=device)

    if args.task is not None:
        task_dirs = [str(Path(args.rgb_root) / args.task)]
    else:
        task_dirs = list_task_dirs(args.rgb_root)
    task_dirs = task_dirs[: args.max_tasks] if args.max_tasks is not None else task_dirs

    for task_dir in task_dirs:
        episode_dirs = list_episode_dirs(task_dir)
        episode_dirs = episode_dirs[: args.max_episodes] if args.max_episodes is not None else episode_dirs

        for episode_dir in episode_dirs:
            # episode 精确选择
            if args.episode is not None and Path(episode_dir).name != args.episode:
                continue

            process_episode(
                model=model,
                episode_dir=episode_dir,
                out_root=args.out_root,
                window_size=args.window_size,
                stride=args.stride,
                device=device,
                amp_dtype=amp_dtype,
                overwrite=args.overwrite,
                max_windows=args.max_windows,
            )


if __name__ == "__main__":
    main()
