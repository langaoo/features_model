# Copyright (C) 2025
# 说明：本文件为“独立新增脚本”，不修改仓库中任何既有代码。
#
# 目标：
#  - 基于机器人操作数据集（多任务/多episode/连续RGB帧），用 CroCo v2 (ViT-L encoder, Base decoder)
#    提取“编码器输出特征”，并按滑动窗口保存到磁盘。
#
# 数据集约定（用户提供）：
#   dataset_root/
#     task_name/
#       episode_name/
#         000000.png, 000001.png, ... (或jpg/jpeg)
#
# 核心设计（务必读）：
# 1) CroCo v2 原生是“跨视角补全”（两张图像作为输入：img1/img2），但编码器本质是标准 ViT 编码器。
#    我们本脚本只需要编码器特征，因此直接调用 CroCoNet._encode_image(img, do_mask=False)。
#    这会返回：
#      feat: [B, N, C]  (patch token 序列；N = (H/patch)*(W/patch))
#      pos : [B, N, 2]  (patch 位置)
#      mask: [B, N]     (这里 do_mask=False 时全False)
#    其中 C=enc_embed_dim（本checkpoint为1024），patch_size=16，默认img_size=224。
#
# 2) “多帧输入”在 CroCo v2 中没有像 VGGT 那样的原生时序聚合模块。
#    一般可行方案：
#      A. 逐帧编码器提特征，然后在“特征层”融合（推荐、简单稳定）
#      B. 直接把多帧当作 batch 维（等价于逐帧，不含时序交互）
#      C. token级拼接（concat tokens）+ 额外Transformer做时序建模（需要训练，不适合纯特征提取）
#      D. 以光流/位姿对齐后再融合（复杂、依赖额外信息）
#    对“从0到1可落地且无需训练”的目标，本脚本默认使用 A：
#      - 對窗口內每一幀提取 feat_i: [1, N, C]
#      - 融合得到 window_feat: [N, C] 或 [Hf, Wf, C]
#    融合方式可选：
#      - mean: 對時間維做均值（默認，最穩健）
#      - max : 對時間維做最大池化（突出出現過的強響應）
#      - concat: 在通道維拼接成 [N, C*T]（信息最多、文件更大）
#
# 3) 窗口大小 8 是否合理？
#    - 在“無訓練、僅做特徵匯聚”前提下，8 幀通常是合理的折中：
#      * 足夠覆蓋短動作（約0.25~1s，取決於FPS）
#      * 計算/存儲開銷線性隨幀數增長（逐幀提特徵）
#    - 如果你的 episode 變化很慢/動作很長，可以嘗試 12~16 幀獲得更平滑時序上下文；
#      如果顯存/速度受限或動作很快，可以嘗試 4 幀。
#    - 最推薦的“工程上可控”策略：保持 stride=1，調 window_size ∈ {4,8,16}，
#      先做小規模下游驗證（對齊你的任務指標）再定。
#
# 保存格式：
#  - 默認每個 episode 保存一個 .pt（torch.save），內容是一个 dict：
#      {
#        'features': Tensor[NumWindows, Hf, Wf, C_or_CxT] (float16 默認)
#        'frame_paths': List[List[str]]  # 每個窗口對應的幀路徑
#        'meta': {...}                  # 配置與shape信息
#      }
#
# 重要注意：CroCo 的 PatchEmbed 會強制輸入尺寸等於模型 img_size（默認224x224）。
#          因此必須 resize+crop/直接resize 到 (224,224)。
#          本腳本使用 torchvision.transforms.Resize((img_size,img_size))。

import os
import sys
import glob
import json
import argparse
from typing import List, Tuple, Dict, Any

import torch
from PIL import Image
from tqdm import tqdm
import zarr
import numpy as np

# 直接复用仓库内 CroCo 定义（不修改它）
# 说明：croco 目录本身不是一个 Python package（没有 __init__.py）。
# 为了让 `from models.xxx import ...` 这种仓库内写法在“从仓库根目录运行脚本”时也能工作，
# 我们把本文件所在目录（.../croco）加入 sys.path。
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from models.croco import CroCoNet
from models.croco_downstream import croco_args_from_ckpt

import torchvision.transforms as T


def get_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def get_autocast_dtype(dtype: str) -> torch.dtype:
    """推理精度：
    - fp16：普遍可用
    - bf16：Ampere(>=8.0) 更稳，数值范围更大
    """
    if dtype == "fp16":
        return torch.float16
    if dtype == "bf16":
        return torch.bfloat16
    if dtype == "auto":
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
            return torch.bfloat16
        return torch.float16
    raise ValueError(f"未知dtype: {dtype}")


def list_task_episode_dirs(dataset_root: str) -> List[Tuple[str, str]]:
    """返回 [(task_dir, episode_dir), ...]"""
    pairs: List[Tuple[str, str]] = []
    task_dirs = sorted([p for p in glob.glob(os.path.join(dataset_root, "*")) if os.path.isdir(p)])
    for task_dir in task_dirs:
        episode_dirs = sorted([p for p in glob.glob(os.path.join(task_dir, "*")) if os.path.isdir(p)])
        for ep in episode_dirs:
            pairs.append((task_dir, ep))
    return pairs


def get_image_paths(episode_dir: str) -> List[str]:
    exts = ("*.png", "*.jpg", "*.jpeg")
    paths: List[str] = []
    for e in exts:
        paths.extend(glob.glob(os.path.join(episode_dir, e)))
    return sorted(paths)


def sliding_windows(paths: List[str], window_size: int, stride: int) -> List[List[str]]:
    if window_size <= 0:
        raise ValueError("window_size 必须>0")
    if stride <= 0:
        raise ValueError("stride 必须>0")
    if len(paths) < window_size:
        return []
    return [paths[i : i + window_size] for i in range(0, len(paths) - window_size + 1, stride)]


def build_preprocess(img_size: int) -> T.Compose:
    """CroCo demo 使用 ImageNet mean/std，本脚本保持一致。"""
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    return T.Compose(
        [
            T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )


def load_croco_encoder(ckpt_path: str, device: torch.device) -> CroCoNet:
    """加载 CroCo v2 checkpoint，并返回 CroCoNet（包含encoder/decoder，但我们只用encoder）。"""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    croco_kwargs = croco_args_from_ckpt(ckpt)
    model = CroCoNet(**croco_kwargs).to(device)
    model.eval()
    msg = model.load_state_dict(ckpt["model"], strict=True)
    # msg 一般是 _IncompatibleKeys(missing_keys=[], unexpected_keys=[])
    _ = msg
    return model


def encode_frame(
    model: CroCoNet,
    image_tensor: torch.Tensor,
    autocast_dtype: torch.dtype,
) -> torch.Tensor:
    """单帧编码，返回 encoder token 特征：[1, N, C]"""
    with torch.inference_mode():
        if image_tensor.is_cuda:
            with torch.cuda.amp.autocast(dtype=autocast_dtype):
                feat, _pos, _mask = model._encode_image(image_tensor, do_mask=False, return_all_blocks=False)
        else:
            feat, _pos, _mask = model._encode_image(image_tensor, do_mask=False, return_all_blocks=False)
    return feat


def fuse_window_features(feats_tnc: torch.Tensor, fuse: str) -> torch.Tensor:
    """输入 feats_tnc: [T, N, C]，输出：
    - mean/max: [N, C]
    - concat:   [N, C*T]
    """
    if fuse == "mean":
        return feats_tnc.mean(dim=0)
    if fuse == "max":
        return feats_tnc.max(dim=0).values
    if fuse == "concat":
        # [T, N, C] -> [N, T, C] -> [N, T*C]
        return feats_tnc.permute(1, 0, 2).reshape(feats_tnc.size(1), -1)
    raise ValueError(f"未知融合方式: {fuse}")


def tokens_to_map(tokens_nc: torch.Tensor, img_size: int, patch_size: int) -> torch.Tensor:
    """把 [N, C] reshape 成 [Hf, Wf, C] 方便后续可视化/下游使用。"""
    hf = img_size // patch_size
    wf = img_size // patch_size
    n, c = tokens_nc.shape
    if n != hf * wf:
        raise RuntimeError(f"token数不匹配：N={n}, 但期望 Hf*Wf={hf*wf} (img_size={img_size}, patch={patch_size})")
    return tokens_nc.view(hf, wf, c)


def process_episode(
    model: CroCoNet,
    episode_dir: str,
    out_root: str,
    preprocess: T.Compose,
    window_size: int,
    stride: int,
    fuse: str,
    keep_time_dim: bool,
    also_save_fused: bool,
    save_dtype: str,
    autocast_dtype: torch.dtype,
    device: torch.device,
    overwrite: bool,
) -> None:
    task_name = os.path.basename(os.path.dirname(episode_dir))
    episode_name = os.path.basename(episode_dir)
    
    # 强制使用 .zarr 后缀
    save_dir = os.path.join(out_root, task_name)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{episode_name}.zarr")

    if os.path.exists(save_path):
        if not overwrite:
            print(f"[Skip] {save_path} 已存在")
            return
        else:
            import shutil
            shutil.rmtree(save_path)

    frame_paths = sorted(glob.glob(os.path.join(episode_dir, "*.png")) + glob.glob(os.path.join(episode_dir, "*.jpg")))
    if not frame_paths:
        print(f"[Warn] {episode_dir} 无图片")
        return

    # 滑动窗口
    windows = []
    if len(frame_paths) >= window_size:
        for i in range(0, len(frame_paths) - window_size + 1, stride):
            windows.append(frame_paths[i : i + window_size])
    
    if not windows:
        print(f"[Warn] {episode_dir} 图片数 {len(frame_paths)} < window_size {window_size}")
        return

    # 预计算 shape
    # 先跑一个 dummy 或者第一帧来确定 shape
    with torch.no_grad():
        dummy_img = Image.open(frame_paths[0]).convert("RGB")
        dummy_x = preprocess(dummy_img).unsqueeze(0).to(device)
        # CroCo output: [1, N, C]
        # We need Hf, Wf
        # CroCo patch_size=16, img_size=224 => 14x14
        img_size = int(model.patch_embed.img_size[0])
        patch_size = int(model.patch_embed.patch_size[0])
        Hf = Wf = img_size // patch_size
        C = int(model.enc_embed_dim)

    num_windows = len(windows)
    
    # 初始化 Zarr
    store = zarr.DirectoryStore(save_path)
    root = zarr.group(store=store, overwrite=True)
    
    # 压缩器
    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=zarr.Blosc.SHUFFLE)
    
    # 创建数组
    # per_frame_features: [W, T, Hf, Wf, C]
    
    # Map save_dtype to numpy dtype
    np_dtype = "float32"
    if save_dtype == "fp16":
        np_dtype = "float16"
    elif save_dtype == "fp32":
        np_dtype = "float32"
        
    if keep_time_dim:
        ds_per_frame = root.create_dataset(
            "per_frame_features",
            shape=(num_windows, window_size, Hf, Wf, C),
            chunks=(1, window_size, Hf, Wf, C), # chunk by window
            dtype=np_dtype,
            compressor=compressor
        )
    
    # features: [W, Hf, Wf, C']
    if (not keep_time_dim) or also_save_fused:
        C_prime = C * window_size if fuse == "concat" else C
        ds_features = root.create_dataset(
            "features",
            shape=(num_windows, Hf, Wf, C_prime),
            chunks=(1, Hf, Wf, C_prime),
            dtype=np_dtype,
            compressor=compressor
        )

    all_window_frame_paths = []

    print(f"[Process] {task_name}/{episode_name} -> {save_path} (Windows: {num_windows})")

    for wi, w_paths in enumerate(windows):
        # 提取一个窗口
        feats = []
        with torch.no_grad(), torch.amp.autocast(device_type=device.type, dtype=autocast_dtype):
            for p in w_paths:
                img = Image.open(p).convert("RGB")
                x = preprocess(img).unsqueeze(0).to(device, non_blocking=True)
                feat_1nc = encode_frame(model, x, autocast_dtype=autocast_dtype)
                feats.append(feat_1nc.squeeze(0))  # [N, C]

        feats_tnc = torch.stack(feats, dim=0)  # [T, N, C]

        # 写入 per_frame_features
        if keep_time_dim:
            # [T, N, C] -> [T, Hf, Wf, C]
            per_frame_maps = torch.stack(
                [tokens_to_map(feats_tnc[t], img_size=img_size, patch_size=patch_size) for t in range(feats_tnc.size(0))],
                dim=0,
            )
            # 转 numpy 写入 zarr
            ds_per_frame[wi] = per_frame_maps.to(torch.float32).cpu().numpy()

        # 写入 features
        if (not keep_time_dim) or also_save_fused:
            fused_nc = fuse_window_features(feats_tnc, fuse=fuse)
            fmap = tokens_to_map(fused_nc, img_size=img_size, patch_size=patch_size)
            ds_features[wi] = fmap.to(torch.float32).cpu().numpy()

        all_window_frame_paths.append(list(w_paths))

    # 保存 meta 和 frame_paths
    meta_dict = {
        "model_name": "croco_v2",
        "episode_dir": episode_dir,
        "task_name": task_name,
        "episode_name": episode_name,
        "num_frames": len(frame_paths),
        "window_size": window_size,
        "stride": stride,
        "num_windows": num_windows,
        "fuse": fuse,
        "keep_time_dim": keep_time_dim,
        "also_save_fused": also_save_fused,
        "img_size": img_size,
        "patch_size": patch_size,
        "enc_embed_dim": C,
        "save_dtype": save_dtype,
    }
    root.attrs["meta"] = meta_dict
    root.attrs["frame_paths"] = all_window_frame_paths

    # 额外保存 json 文件以保持与其他模型输出一致
    with open(os.path.join(save_path, "meta.json"), "w") as f:
        json.dump(meta_dict, f, indent=2)
    
    with open(os.path.join(save_path, "frame_paths.json"), "w") as f:
        json.dump(all_window_frame_paths, f)

    shape_dict = {
        "W": num_windows,
        "T": window_size,
        "Hf": Hf,
        "Wf": Wf,
        "C": C,
        "dtype": save_dtype
    }
    with open(os.path.join(save_path, "shape.json"), "w") as f:
        json.dump(shape_dict, f)

    print(f"[Done] Saved to {save_path}")

def main():
    parser = argparse.ArgumentParser(description="使用 CroCo v2 提取多帧滑窗的 encoder 特征并保存")
    parser.add_argument("--dataset_root", type=str, default="/home/gl/features_model/rgb_dataset/RGB", help="数据集根目录")
    parser.add_argument(
        "--output_root",
        type=str,
        default="/home/gl/features_model/rgb_dataset/features_croco_v2_encoder",
        help="输出根目录（将按task分子目录保存）",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="/home/gl/features_model/croco/pretrained_models/CroCo_V2_ViTLarge_BaseDecoder.pth",
        help="CroCo v2 checkpoint路径",
    )
    parser.add_argument("--window_size", type=int, default=8, help="滑动窗口大小（帧数）")
    parser.add_argument("--stride", type=int, default=1, help="滑动窗口步长")
    parser.add_argument(
        "--fuse",
        type=str,
        default="mean",
        choices=["mean", "max", "concat"],
        help="窗口内多帧特征融合方式（默认mean最稳健）",
    )
    parser.add_argument(
        "--keep_time_dim",
        action="store_true",
        help=(
            "是否保留时间维并保存 per_frame_features。"
            "开启后每个窗口会保存 [T,Hf,Wf,C]，更像真正的多帧输入；"
            "默认不保留（只保存融合后的features）。"
        ),
    )
    parser.add_argument(
        "--also_save_fused",
        action="store_true",
        help=(
            "当 --keep_time_dim 开启时，是否同时也保存融合后的 features（便于快速使用）。"
            "默认关闭时仅保存 per_frame_features。"
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="推理设备：auto/cuda/cuda:0/cpu 等",
    )
    parser.add_argument(
        "--amp_dtype",
        type=str,
        default="auto",
        choices=["auto", "fp16", "bf16"],
        help="CUDA autocast dtype（auto会在Ampere上选bf16，否则fp16）",
    )
    parser.add_argument(
        "--save_dtype",
        type=str,
        default="fp16",
        choices=["fp16", "fp32"],
        help="保存到磁盘的features dtype（fp16节省空间，fp32更稳）",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="若输出已存在，是否覆盖（默认跳过已存在文件）",
    )
    parser.add_argument(
        "--limit_episodes",
        type=int,
        default=0,
        help="调试用：限制处理的episode数量（0表示不限制）",
    )

    args = parser.parse_args()

    device = get_device(args.device)
    autocast_dtype = get_autocast_dtype(args.amp_dtype)

    if not os.path.exists(args.ckpt_path):
        raise FileNotFoundError(f"找不到ckpt: {args.ckpt_path}")

    # 加载model
    model = load_croco_encoder(args.ckpt_path, device=device)
    # 仅用于meta记录
    model._ckpt_path = args.ckpt_path

    img_size = int(model.patch_embed.img_size[0])
    preprocess = build_preprocess(img_size=img_size)

    pairs = list_task_episode_dirs(args.dataset_root)
    if args.limit_episodes and args.limit_episodes > 0:
        pairs = pairs[: args.limit_episodes]

    print(
        json.dumps(
            {
                "dataset_root": args.dataset_root,
                "output_root": args.output_root,
                "ckpt_path": args.ckpt_path,
                "window_size": args.window_size,
                "stride": args.stride,
                "fuse": args.fuse,
                "device": str(device),
                "amp_dtype": str(autocast_dtype),
                "save_dtype": args.save_dtype,
                "num_episodes": len(pairs),
                "model_img_size": img_size,
                "model_patch_size": int(model.patch_embed.patch_size[0]),
                "enc_embed_dim": int(model.enc_embed_dim),
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    for _task_dir, episode_dir in tqdm(pairs, desc="遍历episodes"):
        process_episode(
            model=model,
            episode_dir=episode_dir,
            out_root=args.output_root,
            preprocess=preprocess,
            window_size=args.window_size,
            stride=args.stride,
            fuse=args.fuse,
            keep_time_dim=args.keep_time_dim,
            also_save_fused=args.also_save_fused,
            save_dtype=args.save_dtype,
            autocast_dtype=autocast_dtype,
            device=device,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
