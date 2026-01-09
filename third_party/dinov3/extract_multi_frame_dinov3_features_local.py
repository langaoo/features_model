"""extract_multi_frame_dinov3_features_local.py

核心用途
- 在“机器人多任务/多 episode/连续 RGB 帧”的数据集上，使用本地 DINOv3 权重提取图像编码器特征。
- 将每个 episode 按滑动窗口切分为多个“多帧样本”，并对窗口内帧特征做融合。
- 特征以 .pt 形式保存到 output_root/task_name/episode_name.pt。

关键约束（符合你的要求）
- 只使用本地权重：默认从 /home/gl/features_model/dinov3/weight 加载（该目录包含 config.json、preprocessor_config.json、model.safetensors 等）。
- 不会访问互联网：显式设置 TRANSFORMERS_OFFLINE=1；并要求 model_dir 为本地目录。
- 不修改任何已有文件：这是一个独立新文件。
- 注释与说明全部中文。

关于“DINOv3 多帧输入”的落地方案（本脚本采用）
- DINOv3（ViT）本身是单张图像编码器，不像 VGGT 那样原生有 sequence 维度的时空注意力。
- 因此最稳妥的“多帧输入”方式是：
  1) 将窗口内的多帧当作 batch（形状 [S, C, H, W]）喂给 DINOv3，得到每帧的 encoder 特征。
  2) 再在窗口维度 S 上做融合（mean / max / attention / concat 等）。

本脚本提供两类输出：
- global: 每帧一个全局向量（优先用 pooler_output；如果模型没提供则用 CLS token 代替）
- patch: 每帧一个 patch token 序列（可选保存为 [S, N, C]，如果你后续要做 dense matching/分割等更适合）

注意：concat 会让维度变为 window_size * C，文件会变大。
"""

from __future__ import annotations

# 重要：很多机器上用户目录 ~/.local 会残留一些 pip 包，可能覆盖 conda 环境里的依赖，
# 导致出现“明明装了，但仍然导入到 ~/.local 的旧版本”的问题。
# 这里强制禁用用户站点包，保证依赖来自当前环境。
import os
os.environ.setdefault("PYTHONNOUSERSITE", "1")

import argparse
import glob
from dataclasses import dataclass
from typing import Iterable, List, Literal, Optional, Tuple

import torch
from PIL import Image
from tqdm import tqdm

# transformers 只用于加载本地 HuggingFace 格式权重（你当前 weight 文件夹就是这种格式）
import json
from pathlib import Path

import zarr
import numpy as np
import torchvision.transforms as T


# -------------------------
# 运行环境与数值类型
# -------------------------

def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_amp_dtype() -> torch.dtype:
    """根据显卡能力选择 autocast dtype。

    - Ampere(8.0+) 通常 bfloat16 更稳。
    - 其它 GPU 用 float16。
    - CPU 时不启用 autocast。
    """

    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        return torch.bfloat16
    return torch.float16


# -------------------------
# 数据集扫描与滑动窗口
# -------------------------

def list_task_dirs(dataset_root: str) -> List[str]:
    task_dirs = sorted([p for p in glob.glob(os.path.join(dataset_root, "*")) if os.path.isdir(p)])
    return task_dirs


def list_episode_dirs(task_dir: str) -> List[str]:
    episode_dirs = sorted([p for p in glob.glob(os.path.join(task_dir, "*")) if os.path.isdir(p)])
    return episode_dirs


def get_image_paths(episode_dir: str) -> List[str]:
    """获取 episode 下的所有图片并按文件名排序。

    这里假设 episode_dir 直接包含 png/jpg/jpeg。
    如果你的数据在 episode_dir/images 里，把 glob 的路径改一下即可。
    """

    exts = ("*.png", "*.jpg", "*.jpeg")
    paths: List[str] = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(episode_dir, ext)))
    paths.sort()
    return paths


def sliding_windows(paths: List[str], window_size: int, stride: int) -> List[List[str]]:
    """滑动窗口切分。

    例：40 帧，window_size=8，stride=1 -> 33 个窗口。
    """

    n = len(paths)
    if n < window_size:
        return []
    return [paths[i : i + window_size] for i in range(0, n - window_size + 1, stride)]


# -------------------------
# 多帧融合（窗口内）
# -------------------------

FusionMode = Literal["mean", "max", "attention", "concat"]
FeatureType = Literal["global", "patch"]


def fuse_window_features(
    per_frame_feat: torch.Tensor,
    mode: FusionMode,
) -> torch.Tensor:
    """对窗口内的每帧特征做融合。

    输入
    - per_frame_feat:
      - global 模式下：形状 [S, C]
      - patch 模式下：形状 [S, N, C]

    输出
    - mean/max/attention: 去掉 S，输出 [C] 或 [N, C]
    - concat: 展平拼接，输出 [S*C] 或 [N, S*C]

    说明
    - attention 这里用“窗口内自注意力”做一个轻量加权：
      - global: 先算 frame-frame 的相似度矩阵得到权重，再加权求和。
      - patch: 对每个 patch 位置，独立做 frame attention（更慢但更细）。
    """

    if mode == "mean":
        return per_frame_feat.mean(dim=0)

    if mode == "max":
        return per_frame_feat.max(dim=0).values

    if mode == "concat":
        # global: [S, C] -> [S*C]
        # patch:  [S, N, C] -> [N, S*C]
        if per_frame_feat.dim() == 2:
            return per_frame_feat.reshape(-1)
        if per_frame_feat.dim() == 3:
            s, n, c = per_frame_feat.shape
            return per_frame_feat.permute(1, 0, 2).reshape(n, s * c)
        raise ValueError(f"不支持的维度: {per_frame_feat.shape}")

    if mode == "attention":
        # 为了数值稳定，attention 全程用 float32 计算权重
        if per_frame_feat.dim() == 2:
            # [S, C]
            x = per_frame_feat.float()
            sim = x @ x.t()  # [S, S]
            w = torch.softmax(sim, dim=1)  # [S, S]
            # 用 w 的列平均做一个整体权重（你也可以改成取 CLS 与其它帧相似度）
            # 这里取每帧作为“query”时对所有帧的注意力分布，再平均成一个全局权重
            w_global = w.mean(dim=0)  # [S]
            w_global = w_global / (w_global.sum() + 1e-6)
            out = (per_frame_feat * w_global[:, None]).sum(dim=0)
            return out

        if per_frame_feat.dim() == 3:
            # [S, N, C]：对每个 patch 位置 n，做一次 [S, C] attention
            s, n, c = per_frame_feat.shape
            x = per_frame_feat.float()  # [S, N, C]
            # 把 (N) 当 batch，做 N 次 attention
            x_nsc = x.permute(1, 0, 2).contiguous()  # [N, S, C]
            sim = torch.matmul(x_nsc, x_nsc.transpose(1, 2))  # [N, S, S]
            w = torch.softmax(sim, dim=2)  # [N, S, S]
            w_global = w.mean(dim=1)  # [N, S]
            w_global = w_global / (w_global.sum(dim=1, keepdim=True) + 1e-6)
            out = (per_frame_feat.permute(1, 0, 2) * w_global[:, :, None]).sum(dim=1)  # [N, C]
            return out

        raise ValueError(f"不支持的维度: {per_frame_feat.shape}")

    raise ValueError(f"不支持的融合模式: {mode}")


# -------------------------
# 模型加载（本地，禁止联网）
# -------------------------


def load_local_hf_dinov3(model_dir: str, device: str) -> Tuple[torch.nn.Module, dict]:
    """从本地目录加载 DINOv3（禁止联网）。

    你提供的路径：/home/gl/features_model/dinov3/weight
    该目录包含：
    - config.json
    - preprocessor_config.json
    - model.safetensors（以及分片与 index）

    这正是 transformers 的 from_pretrained 本地加载格式。

    重要：如果你把 model_dir 指到单个 .safetensors 文件，transformers 无法知道 config。
    所以这里要求传入“目录”。
    """

    if not os.path.isdir(model_dir):
        raise FileNotFoundError(
            f"model_dir 必须是包含 config.json 和 model.safetensors 的目录，而不是单个文件: {model_dir}"
        )

    # 强制离线，避免 transformers 尝试联网
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    # 关键点：你的权重 config.json 里 model_type=dinov3_vit，transformers(稳定版) 并不识别该架构。
    # 因此不能用 AutoModel/AuoConfig 加载。
    # 我们改为使用本仓库自带的 dinov3.hub.backbones 来构建模型结构，并从“本地文件 URI”加载权重。

    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"缺少 {config_path}，无法确定网络结构")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # 从 config.json 推断 ViT 结构参数（尽量覆盖关键字段）
    patch_size = int(cfg.get("patch_size", 16))
    image_size = int(cfg.get("image_size", 224))
    embed_dim = int(cfg.get("hidden_size", 768))
    depth = int(cfg.get("num_hidden_layers", 12))
    num_heads = int(cfg.get("num_attention_heads", 12))
    intermediate_size = int(cfg.get("intermediate_size", embed_dim * 4))
    ffn_ratio = float(intermediate_size) / float(embed_dim)
    n_storage_tokens = int(cfg.get("num_register_tokens", 0))

    # 通过 num_heads 推断 compact_arch_name（仅用于选择“相近实现”的默认超参，结构参数我们会显式传入）
    # vits: 6 heads, vitb: 12 heads, vitl: 16 heads, vitg: 24 heads (常见)
    if num_heads <= 6:
        compact_arch_name = "vits"
    elif num_heads <= 12:
        compact_arch_name = "vitb"
    elif num_heads <= 16:
        compact_arch_name = "vitl"
    else:
        compact_arch_name = "vitg"

    # 1) 构建 dinov3 原生 ViT 模型结构
    from dinov3.models.vision_transformer import DinoVisionTransformer  # type: ignore

    model = DinoVisionTransformer(
        img_size=image_size,
        patch_size=patch_size,
        in_chans=int(cfg.get("num_channels", 3)),
        pos_embed_rope_base=float(cfg.get("rope_theta", 100.0)),
        pos_embed_rope_dtype="fp32",
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        ffn_ratio=ffn_ratio,
        qkv_bias=True,
        drop_path_rate=float(cfg.get("drop_path_rate", 0.0)),
        layerscale_init=float(cfg.get("layerscale_value", 1.0e-5)) if cfg.get("layerscale_value") is not None else None,
        norm_layer="layernormbf16" if (device == "cuda" and get_amp_dtype() == torch.bfloat16) else "layernorm",
        ffn_layer="mlp",
        ffn_bias=bool(cfg.get("mlp_bias", True)),
        proj_bias=bool(cfg.get("proj_bias", True)),
        n_storage_tokens=n_storage_tokens,
        mask_k_bias=bool(cfg.get("key_bias", False)),
    )

    def _convert_hf_keys_to_dinov3(sd: dict) -> dict:
        """把 HF 风格 key(embeddings.*, layer.*) 转成 dinov3 原生 key。

        说明：
        - 你的 safetensors 看起来来自 transformers 侧的实现（key 为 embeddings/layer）。
        - 本仓库 DinoVisionTransformer 的命名为 cls_token/storage_tokens/mask_token/patch_embed/blocks.*。
        - 这里做一个“最小必要”的映射，足以用于 backbone 特征提取。
        """

        out = {}
        for k, v in sd.items():
            nk = k
            # embeddings
            nk = nk.replace("embeddings.cls_token", "cls_token")
            nk = nk.replace("embeddings.mask_token", "mask_token")
            nk = nk.replace("embeddings.register_tokens", "storage_tokens")
            nk = nk.replace("embeddings.patch_embeddings.weight", "patch_embed.proj.weight")
            nk = nk.replace("embeddings.patch_embeddings.bias", "patch_embed.proj.bias")

            # layers
            if nk.startswith("layer."):
                # layer.{i}.xxxx -> blocks.{i}.xxxx
                nk = nk.replace("layer.", "blocks.")

                # attention projections
                nk = nk.replace(".attention.q_proj.", ".attn.q_proj.")
                nk = nk.replace(".attention.k_proj.", ".attn.k_proj.")
                nk = nk.replace(".attention.v_proj.", ".attn.v_proj.")
                nk = nk.replace(".attention.o_proj.", ".attn.proj.")

                # norms
                nk = nk.replace(".norm1.", ".norm1.")
                nk = nk.replace(".norm2.", ".norm2.")

                # gated MLP (HF: gate_proj/up_proj/down_proj)
                nk = nk.replace(".mlp.gate_proj.", ".mlp.fc1_gated.")
                nk = nk.replace(".mlp.up_proj.", ".mlp.fc1.")
                nk = nk.replace(".mlp.down_proj.", ".mlp.fc2.")

                # layer scales
                nk = nk.replace(".layer_scale1.lambda1", ".ls1.gamma")
                nk = nk.replace(".layer_scale2.lambda1", ".ls2.gamma")

            # token 参数形状对齐：
            # - DinoVisionTransformer.cls_token 期望 [1, 1, C]
            # - DinoVisionTransformer.mask_token 期望 [1, C]
            if nk == "mask_token" and v.ndim == 3 and v.shape[0] == 1 and v.shape[1] == 1:
                v = v.squeeze(1)  # [1, C]
            if nk == "cls_token" and v.ndim == 2 and v.shape[0] == 1:
                v = v.unsqueeze(1)  # [1, 1, C]
            if nk == "storage_tokens" and v.ndim == 3 and v.shape[0] == 1:
                # storage_tokens 期望 [1, n_storage, C]，这里保持不变
                pass

            out[nk] = v

        return out

    # 2) 加载 safetensors 权重（支持分片；不会联网，也不会复制到 torch hub cache）
    # 你的目录里同时存在：
    # - model.safetensors（看起来像是“占位/索引聚合文件”，某些环境下可能无法直接读取）
    # - model-00001-of-00006.safetensors ... model-00006-of-00006.safetensors
    # - model.safetensors.index.json（权重到分片文件的映射）
    # 因此我们优先按 index.json 逐片加载并合并。

    from safetensors.torch import load_file  # type: ignore

    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    state_dict = {}
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)
        shard_files = sorted(set(index.get("weight_map", {}).values()))
        if not shard_files:
            raise RuntimeError(f"{index_path} 中 weight_map 为空")

        missing_shards = []
        for shard in shard_files:
            shard_path = os.path.join(model_dir, shard)
            if not os.path.exists(shard_path):
                missing_shards.append(shard_path)
                continue
            part = load_file(shard_path)
            state_dict.update(part)

        # 有些目录只提供了单体 model.safetensors，但误带了 index.json。
        # 如果分片不存在，则回退到直接加载 model.safetensors。
        if missing_shards:
            weights_path = os.path.join(model_dir, "model.safetensors")
            if not os.path.exists(weights_path):
                # 保留原始报错信息，方便用户定位
                raise FileNotFoundError(
                    "index.json 指向的分片文件不存在，且目录中也没有 model.safetensors 可回退加载。\n"
                    f"缺失分片示例: {missing_shards[0]}"
                )
            print(f"[提示] 发现 index.json 但缺少分片文件，将回退到单体权重: {weights_path}")
            state_dict = load_file(weights_path)
    else:
        weights_path = os.path.join(model_dir, "model.safetensors")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"缺少权重文件 {weights_path}")
        state_dict = load_file(weights_path)
    state_dict = _convert_hf_keys_to_dinov3(state_dict)

    msg = model.load_state_dict(state_dict, strict=False)
    # 这里不 strict=True：因为不同导出方式可能多一些无关 key。
    # 但如果缺失非常多，后续 forward 大概率会报错；我们打印出来方便你定位。
    if len(msg.missing_keys) > 0 or len(msg.unexpected_keys) > 0:
        print(f"[提示] load_state_dict non-strict: missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)}")

    model.eval()
    model.to(device)

    preprocess_cfg_path = os.path.join(model_dir, "preprocessor_config.json")
    if not os.path.exists(preprocess_cfg_path):
        raise FileNotFoundError(f"缺少 {preprocess_cfg_path}，无法确定 resize/normalize 参数")

    with open(preprocess_cfg_path, "r", encoding="utf-8") as f:
        pre_cfg = json.load(f)

    # 仅取我们需要的字段
    processor_cfg = {
        "size": pre_cfg.get("size", {"height": 224, "width": 224}),
        "image_mean": pre_cfg.get("image_mean", [0.485, 0.456, 0.406]),
        "image_std": pre_cfg.get("image_std", [0.229, 0.224, 0.225]),
        "do_resize": bool(pre_cfg.get("do_resize", True)),
        "do_rescale": bool(pre_cfg.get("do_rescale", True)),
        "rescale_factor": float(pre_cfg.get("rescale_factor", 1.0 / 255.0)),
    }
    return model, processor_cfg


def load_small_dinov3_from_pth(
    *,
    backbone: str,
    weights_pth: str,
    device: str,
) -> Tuple[torch.nn.Module, dict]:
    """加载“小一点的 dinov3 vision backbone”（结构来自 dinov3.hub.backbones，权重来自本地 .pth）。

    设计动机
    - 你的 7B safetensors 在 16GB 显存上不可行。
    - 你允许换小模型先把整个 pipeline 跑通。
    - 但你要求不要联网下载，所以这里必须用你本地已经存在的 .pth。

    你需要做什么
    - 把一个小 backbone 的权重文件（例如 dinov3_vits16 或 dinov3_vitb16 的 .pth）放到本地。
    - 运行时传：--weights_pth /abs/path/to/xxx.pth --backbone dinov3_vits16

    预处理
    - 小模型这里用通用 224/imagenet mean/std（与你的 preprocessor_config 也一致）。
    """

    if weights_pth is None or weights_pth == "":
        raise ValueError("使用小 backbone 需要提供 --weights_pth（本地 .pth 文件路径）。")
    if not os.path.exists(weights_pth):
        raise FileNotFoundError(f"weights_pth 不存在: {weights_pth}")

    from dinov3.hub import backbones as hub_backbones  # type: ignore

    # 只构建结构，不触发下载
    model_fn = getattr(hub_backbones, backbone)
    model = model_fn(pretrained=False)

    ckpt = torch.load(weights_pth, map_location="cpu")
    # 常见格式：
    # - 直接 state_dict
    # - {"teacher": state_dict}
    # - {"model": state_dict}
    if isinstance(ckpt, dict) and "teacher" in ckpt:
        ckpt = ckpt["teacher"]
    if isinstance(ckpt, dict) and "model" in ckpt:
        ckpt = ckpt["model"]
    if not isinstance(ckpt, dict):
        raise ValueError("无法识别的 checkpoint 格式：期望 dict(state_dict) 或包含 teacher/model 键")

    ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
    ckpt = {k.replace("backbone.", ""): v for k, v in ckpt.items()}

    msg = model.load_state_dict(ckpt, strict=False)
    if len(msg.missing_keys) or len(msg.unexpected_keys):
        print(f"[提示] 小模型 load_state_dict non-strict: missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)}")

    model.eval()
    model.to(device)

    # 通用 224 预处理
    processor_cfg = {
        "size": {"height": 224, "width": 224},
        "image_mean": [0.485, 0.456, 0.406],
        "image_std": [0.229, 0.224, 0.225],
        "do_resize": True,
        "do_rescale": True,
        "rescale_factor": 1.0 / 255.0,
    }

    return model, processor_cfg


# -------------------------
# 单个窗口：读图 -> 编码 -> 取特征
# -------------------------


def load_images(window_paths: List[str]) -> List[Image.Image]:
    images: List[Image.Image] = []
    for p in window_paths:
        images.append(Image.open(p).convert("RGB"))
    return images


def _preprocess_images(images: List[Image.Image], processor_cfg: dict) -> torch.Tensor:
    """根据 preprocessor_config.json 中的字段做最小可用预处理。

    输出张量形状: [S, 3, H, W]
    """

    size = processor_cfg.get("size", {"height": 224, "width": 224})
    h = int(size.get("height", 224))
    w = int(size.get("width", 224))
    do_resize = bool(processor_cfg.get("do_resize", True))
    do_rescale = bool(processor_cfg.get("do_rescale", True))
    rescale_factor = float(processor_cfg.get("rescale_factor", 1.0 / 255.0))
    mean = torch.tensor(processor_cfg.get("image_mean", [0.485, 0.456, 0.406]), dtype=torch.float32)
    std = torch.tensor(processor_cfg.get("image_std", [0.229, 0.224, 0.225]), dtype=torch.float32)

    frames: List[torch.Tensor] = []
    for img in images:
        if do_resize:
            img = img.resize((w, h), resample=Image.BILINEAR)

        x = torch.from_numpy(__import__("numpy").array(img)).to(torch.float32)  # [H, W, 3]
        x = x.permute(2, 0, 1).contiguous()  # [3, H, W]
        if do_rescale:
            x = x * rescale_factor
        # normalize
        x = (x - mean[:, None, None]) / std[:, None, None]
        frames.append(x)

    return torch.stack(frames, dim=0)


def extract_per_frame_features(
    model: torch.nn.Module,
    processor_cfg: dict,
    images: List[Image.Image],
    device: str,
    feature_type: FeatureType,
) -> torch.Tensor:
    """提取窗口内每帧特征。

    返回：
    - feature_type=global: [S, C]
    - feature_type=patch:  [S, N, C]

    说明：
    - 对于 ViT 类模型，transformers 输出通常含 last_hidden_state。
    - pooler_output 不一定存在；若不存在，则用 CLS token 作为 global。
    """

    pixel_values = _preprocess_images(images, processor_cfg).to(device)

    amp_dtype = get_amp_dtype()

    def _forward(m: torch.nn.Module, x: torch.Tensor):
        """兼容两类 forward 签名：

        - dinov3 原生 DinoVisionTransformer: out = model(x)
        - transformers 风格:               out = model(pixel_values=x)
        """

        try:
            return m(x)
        except TypeError:
            return m(pixel_values=x)

    with torch.no_grad():
        if device == "cuda":
            with torch.cuda.amp.autocast(dtype=amp_dtype):
                out = _forward(model, pixel_values)
        else:
            out = _forward(model, pixel_values)

    # 兼容不同返回类型：
    # - transformers 风格：out.last_hidden_state / out.pooler_output
    # - dinov3 原生：可能直接返回 token 张量 [S, 1+N, C]，或 tuple(list)
    tokens = None
    pool = None

    if isinstance(out, torch.Tensor):
        tokens = out
    elif isinstance(out, (tuple, list)):
        # 取第一个张量作为 token（保守策略）
        for item in out:
            if isinstance(item, torch.Tensor):
                tokens = item
                break
    else:
        # 尝试 transformers 的字段
        if hasattr(out, "last_hidden_state"):
            tokens = out.last_hidden_state
        if hasattr(out, "pooler_output"):
            pool = out.pooler_output

    if tokens is None and pool is None:
        raise RuntimeError(
            "模型 forward 输出无法解析为 tokens/pooler_output。"
            "请打印 model(**inputs) 的返回类型并适配。"
        )

    if feature_type == "global":
        if isinstance(pool, torch.Tensor) and pool.numel() > 0:
            return pool  # [S, C]
        if tokens is None:
            raise RuntimeError("global 特征需要 tokens 或 pooler_output，但两者都不可用")
        # dinov3 原生有时会直接返回 [S, C] 的全局向量
        if tokens.dim() == 2:
            return tokens
        return tokens[:, 0, :]  # CLS

    if feature_type == "patch":
        if tokens is None:
            raise RuntimeError("patch 特征需要 tokens，但 tokens 不可用")
        # 若 forward 只返回了全局向量，则尝试调用 forward_features 拿 token
        if tokens.dim() == 2:
            if hasattr(model, "forward_features"):
                with torch.no_grad():
                    feats = model.forward_features(pixel_values)  # type: ignore[attr-defined]
                # forward_features 在 dinov3 中常见返回 dict，含 x_norm_patchtokens / x_norm_clstoken
                if isinstance(feats, dict) and "x_norm_patchtokens" in feats:
                    return feats["x_norm_patchtokens"]
            raise RuntimeError("patch 特征需要 token 序列，但模型 forward 仅返回了全局向量")
        return tokens[:, 1:, :]  # 去掉 CLS

    raise ValueError(f"不支持的 feature_type: {feature_type}")


# -------------------------
# Episode 级处理与保存
# -------------------------


@dataclass
class SavePack:
    """保存结构（方便你后续扩展到其他模型也保持一致）。"""

    # 每个窗口一个条目
    # - global + mean/attention/max: [num_windows, C]
    # - global + concat:            [num_windows, window_size*C]
    # - patch  + mean/...:          [num_windows, N, C]
    # - patch  + concat:            [num_windows, N, window_size*C]
    features: torch.Tensor

    # 元信息：用于追溯、调试与复现
    meta: dict

    # 可选：保留时间维的每帧特征（你要求的“真正多帧输入/不丢时间维”）
    # - global: [num_windows, S, C]
    # - patch:  [num_windows, S, Hf, Wf, C]
    per_frame_features: Optional[torch.Tensor] = None

    # 每个窗口对应的帧路径（用于跨任务/episode/窗口对齐与追溯）
    frame_paths: Optional[List[List[str]]] = None


def _infer_hw_from_num_patches(num_patches: int) -> tuple[int, int]:
    """把 patch token 数量 N 还原成 (Hf,Wf)。

    对于常见 ViT：N=14*14=196（224/16）。
    若 N 不是完全平方数，则尽量找一组接近正方形的因子。
    """

    sq = int(num_patches**0.5)
    if sq * sq == num_patches:
        return sq, sq

    best = (1, num_patches)
    best_aspect = float("inf")
    for h in range(1, sq + 1):
        if num_patches % h:
            continue
        w = num_patches // h
        aspect = max(w / h, h / w)
        if aspect < best_aspect:
            best_aspect = aspect
            best = (h, w)
    return best


def _patch_tokens_to_map(tokens: torch.Tensor) -> torch.Tensor:
    """tokens: [S, N, C] -> [S, Hf, Wf, C]"""

    if tokens.dim() != 3:
        raise ValueError(f"tokens_to_map 仅支持 [S,N,C]，但得到 {tuple(tokens.shape)}")
    s, n, c = tokens.shape
    hf, wf = _infer_hw_from_num_patches(n)
    if hf * wf != n:
        raise RuntimeError(f"无法把 N={n} 还原为网格 (Hf,Wf)")
    return tokens.view(s, hf, wf, c)


def process_one_episode(
    model: torch.nn.Module,
    processor_cfg: dict,
    episode_dir: str,
    output_root: str,
    window_size: int,
    stride: int,
    fusion: FusionMode,
    feature_type: FeatureType,
    device: str,
    max_windows: Optional[int] = None,
    keep_time_dim: bool = False,
    also_save_fused: bool = True,
    save_dtype: str = "fp32",
) -> Optional[str]:
    """处理单个 episode；返回保存路径（若无输出则返回 None）。"""

    img_paths = get_image_paths(episode_dir)
    if not img_paths:
        print(f"[跳过] {episode_dir} 下没有图片")
        return None

    windows = sliding_windows(img_paths, window_size=window_size, stride=stride)
    if not windows:
        print(f"[跳过] {episode_dir} 帧数 {len(img_paths)} < window_size {window_size}")
        return None

    if max_windows is not None:
        windows = windows[:max_windows]

    fused_list: List[torch.Tensor] = []
    per_frame_list: List[torch.Tensor] = []

    for w_idx, w_paths in enumerate(tqdm(windows, desc=f"Episode {os.path.basename(episode_dir)}", leave=False)):
        images = load_images(w_paths)
        per_frame = extract_per_frame_features(
            model=model,
            processor_cfg=processor_cfg,
            images=images,
            device=device,
            feature_type=feature_type,
        )

        if keep_time_dim:
            if feature_type == "global":
                # [S,C] -> [S,C]
                per_frame_list.append(per_frame.detach().cpu())
            else:
                # [S,N,C] -> [S,Hf,Wf,C]
                per_frame_list.append(_patch_tokens_to_map(per_frame.detach().cpu()))

        if also_save_fused:
            fused = fuse_window_features(per_frame, mode=fusion)
            fused_list.append(fused.cpu())

    features = torch.stack(fused_list, dim=0) if fused_list else None

    per_frame_features = None
    if keep_time_dim:
        per_frame_features = torch.stack(per_frame_list, dim=0)  # [num_windows, S, ...]

    def _cast(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if x is None:
            return None
        if save_dtype == "fp16":
            return x.to(torch.float16)
        if save_dtype == "bf16":
            return x.to(torch.bfloat16)
        if save_dtype == "fp32":
            return x.to(torch.float32)
        raise ValueError("save_dtype 仅支持 fp16/bf16/fp32")

    features = _cast(features)
    per_frame_features = _cast(per_frame_features)

    task_name = os.path.basename(os.path.dirname(episode_dir))
    episode_name = os.path.basename(episode_dir)
    save_dir = os.path.join(output_root, task_name)
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f"{episode_name}.pt")
    pack = SavePack(
        features=features if features is not None else torch.empty(0),
        per_frame_features=per_frame_features,
        frame_paths=windows,
        meta=dict(
            model_name="dinov3",
            episode_dir=episode_dir,
            num_frames=len(img_paths),
            num_windows=len(windows),
            window_size=window_size,
            stride=stride,
            fusion=fusion,
            feature_type=feature_type,
            img_size=int(processor_cfg.get("size", {}).get("height", 224)),
            keep_time_dim=keep_time_dim,
            also_save_fused=also_save_fused,
            save_dtype=save_dtype,
        ),
    )
    torch.save(pack, save_path)

    feat_shape = tuple(features.shape) if features is not None else None
    pf_shape = tuple(per_frame_features.shape) if per_frame_features is not None else None
    print(
        f"[保存] {save_path} | features={feat_shape} | per_frame={pf_shape} | fusion={fusion} | type={feature_type}"
    )
    return save_path


def process_episode(
    model_dir: str,
    episode_dir: str,
    out_root: str,
    window_size: int,
    stride: int,
    fuse: str,
    keep_time_dim: bool,
    also_save_fused: bool,
    save_dtype: str,
    device: str,
    amp_dtype: torch.dtype,
    overwrite: bool,
    model_cache: dict,
) -> None:
    task_name = os.path.basename(os.path.dirname(episode_dir))
    episode_name = os.path.basename(episode_dir)
    
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

    image_paths = get_image_paths(episode_dir)
    if not image_paths:
        return

    windows = sliding_windows(image_paths, window_size, stride)
    if not windows:
        return

    # 兼容：旧版本这里曾调用未实现的 load_model(model_dir, device)。
    # 当前脚本在 main() 里已经通过 load_local_hf_dinov3() 加载好模型，因此这里必须从 cache 取。
    if "model" not in model_cache:
        raise RuntimeError(
            "DINOv3 模型未在 model_cache 中初始化。请确保 main() 在调用 process_episode() 前已加载模型并写入 model_cache['model']。"
        )
    model = model_cache["model"]

    # 最小预处理：对齐 CroCo 的 224 输入与 ImageNet normalize。
    # DINOv3 patch grid 会随输入尺寸变化；为了“严格跟 CroCo 一样”，这里固定 224。
    image_size = 224
    patch_size = 16
    Hf = Wf = image_size // patch_size

    preprocess = T.Compose(
        [
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Determine channels C by running one forward
    with torch.no_grad():
        dummy_img = Image.open(image_paths[0]).convert("RGB")
        x = preprocess(dummy_img).unsqueeze(0).to(device)
        dummy_out = model(x)
        # DinoVisionTransformer forward returns dict-like or tensor depending on implementation.
        # We rely on get_intermediate_layers below for actual extraction, but here infer embed dim.
        if isinstance(dummy_out, dict) and "x" in dummy_out:
            C = int(dummy_out["x"].shape[-1])
        elif torch.is_tensor(dummy_out):
            C = int(dummy_out.shape[-1])
        else:
            # fallback to public attribute
            C = int(getattr(model, "embed_dim", 768))
    
    num_windows = len(windows)
    
    store = zarr.DirectoryStore(save_path)
    root = zarr.group(store=store, overwrite=True)
    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=zarr.Blosc.SHUFFLE)

    np_dtype = "float32"
    if save_dtype == "fp16":
        np_dtype = "float16"
    elif save_dtype == "bf16":
        # zarr/numpy 不支持 bfloat16，落盘用 float16；meta 仍记录 bf16
        np_dtype = "float16"
    elif save_dtype == "fp32":
        np_dtype = "float32"

    if keep_time_dim:
        ds_per_frame = root.create_dataset(
            "per_frame_features",
            shape=(num_windows, window_size, Hf, Wf, C),
            chunks=(1, window_size, Hf, Wf, C),
            dtype=np_dtype,
            compressor=compressor
        )

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
    
    print(f"[Process] {task_name}/{episode_name} -> {save_path}")

    for wi, w_paths in enumerate(windows):
        # Extract window
        # ... (logic to extract feats_tnc [T, N, C])
        # For brevity, I'll reuse the existing extraction logic structure but write to Zarr
        
        feats_list = []
        with torch.no_grad(), torch.amp.autocast(device_type="cuda" if "cuda" in device else "cpu", dtype=amp_dtype):
            for pth in w_paths:
                img = Image.open(pth).convert("RGB")
                x = preprocess(img).unsqueeze(0).to(device)

                # dinov3 模型支持 get_intermediate_layers；沿用仓库实现。
                # 返回 list[Tensor]，每个是 [B, N, C] (包含 CLS/寄存器)。我们取最后层。
                if hasattr(model, "get_intermediate_layers"):
                    y = model.get_intermediate_layers(x, n=1)[0]  # type: ignore[attr-defined]
                else:
                    y = model(x)
                # y: [1, N, C] -> 去掉前面的 CLS/寄存器，只保留 patch tokens。
                y = y[0]
                patch_tokens = y[-(Hf * Wf) :, :]  # [Hf*Wf, C]
                feats_list.append(patch_tokens)

        feats_tnc = torch.stack(feats_list, dim=0)  # [T, Hf*Wf, C]
        feats_map = feats_tnc.reshape(window_size, Hf, Wf, C)

        if keep_time_dim:
            ds_per_frame[wi] = feats_map.to(torch.float32).cpu().numpy().astype(np_dtype)
        
        if (not keep_time_dim) or also_save_fused:
            # Fuse
            if fuse == "mean":
                fused = feats_map.mean(dim=0)
            elif fuse == "max":
                fused = feats_map.max(dim=0)[0]
            elif fuse == "concat":
                fused = feats_map.reshape(Hf, Wf, -1)
            ds_features[wi] = fused.to(torch.float32).cpu().numpy().astype(np_dtype)

        all_window_frame_paths.append(w_paths)

    meta_dict = {
        "model_name": "dinov3",
        "episode_dir": episode_dir,
        "task_name": task_name,
        "episode_name": episode_name,
        "num_frames": len(image_paths),
        "window_size": window_size,
        "stride": stride,
        "num_windows": num_windows,
        "fuse": fuse,
        "keep_time_dim": keep_time_dim,
        "also_save_fused": also_save_fused,
        "img_size": image_size,
        "patch_size": patch_size,
        "enc_embed_dim": C,
        "save_dtype": save_dtype,
    }
    root.attrs["meta"] = meta_dict
    root.attrs["frame_paths"] = all_window_frame_paths

    # 严格对齐 CroCo：落盘三个 json 文件
    with open(os.path.join(save_path, "meta.json"), "w") as f:
        json.dump(meta_dict, f, indent=2)
    with open(os.path.join(save_path, "frame_paths.json"), "w") as f:
        json.dump(all_window_frame_paths, f)
    shape_dict = {"W": num_windows, "T": window_size, "Hf": Hf, "Wf": Wf, "C": C, "dtype": save_dtype}
    with open(os.path.join(save_path, "shape.json"), "w") as f:
        json.dump(shape_dict, f)

    print(f"[Done] {save_path}")

# -------------------------
# 主入口
# -------------------------


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "使用本地 DINOv3 权重，对机器人 RGB 数据集做多帧滑窗特征提取（离线，不联网）。"
        )
    )

    # 数据（统一命名 + 兼容旧命名）
    p.add_argument(
        "--rgb_root",
        type=str,
        default="/home/gl/features_model/rgb_dataset/RGB",
        help="数据集根目录，结构应为 rgb_root/task_name/episode_name/*.png",
    )
    p.add_argument(
        "--out_root",
        type=str,
        default="/home/gl/features_model/rgb_dataset/features_dinov3_encoder_dict",
        help=(
            "特征保存根目录（dict 格式，对齐 CroCo/VGGT/DA3），"
            "输出为 out_root/task_name/episode_name.pt"
        ),
    )
    p.add_argument(
        "--dataset_root",
        type=str,
        default=None,
        help="[兼容] 同 --rgb_root",
    )
    p.add_argument(
        "--output_root",
        type=str,
        default=None,
        help="[兼容] 同 --out_root",
    )

    # 模型
    p.add_argument(
        "--backbone",
        type=str,
        default="dinov3_vits16",
        choices=[
            "dinov3_vits16",
            "dinov3_vits16plus",
            "dinov3_vitb16",
            "dinov3_vitl16",
            "dinov3_vitl16plus",
        ],
        help=(
            "选择 dinov3 的更小 backbone 结构（仅结构，不会自动下载权重）。"
            "推荐先用 dinov3_vits16 跑通流程。"
        ),
    )
    p.add_argument(
        "--weights_pth",
        type=str,
        default=None,
        help=(
            "本地权重文件路径（.pth）。必须是你已经下载好的文件。"
            "注意：本脚本不会联网下载。"
        ),
    )

    # 兼容：你原先的 7B safetensors 目录（显存不够时不建议）
    p.add_argument(
        "--model_dir",
        type=str,
        default="/home/gl/features_model/dinov3/weight",
        help=(
            "本地 DINOv3 HF safetensors 目录（7B 权重）。"
            "默认保留仅用于 CPU 跑；GPU 16GB 上会 OOM。"
        ),
    )

    # 多帧切分
    p.add_argument("--window_size", type=int, default=8, help="滑动窗口大小 S（帧数）")
    p.add_argument("--stride", type=int, default=1, help="滑动窗口步长")

    # 统一控制（精确选择 + all/smoke 语义）
    p.add_argument("--task", type=str, default=None, help="只处理某一个 task 目录名")
    p.add_argument("--episode", type=str, default=None, help="只处理某一个 episode 目录名")
    p.add_argument("--all", action="store_true", help="全量导出（默认就是全量；加这个只是为了命令风格统一）")
    p.add_argument("--smoke", action="store_true", help="快速冒烟：只跑1个task/episode/window")

    # 多帧融合
    p.add_argument(
        "--fusion",
        type=str,
        default="mean",
        choices=["mean", "max", "attention", "concat"],
        help=(
            "窗口内融合方式：mean(默认)/max/attention/concat。"
            "注意 concat 会显著增大维度与文件体积。"
        ),
    )

    # 输出特征类型
    p.add_argument(
        "--feature_type",
        type=str,
        default="patch",
        choices=["global", "patch"],
        help=(
            "输出特征类型：global=每帧全局向量；patch=每帧 patch tokens（更大更慢）。"
        ),
    )

    # 你要求的“统一范式/保留时间维”
    p.add_argument(
        "--keep_time_dim",
        action="store_true",
        help="保存 per_frame_features（global: [W,S,C]；patch: [W,S,Hf,Wf,C]）",
    )
    p.add_argument(
        "--also_save_fused",
        action="store_true",
        help="同时保存融合后的 features（按 --fusion）",
    )
    p.add_argument(
        "--save_dtype",
        type=str,
        default="fp16",
        choices=["fp16", "bf16", "fp32"],
        help="保存到磁盘的 dtype（默认 fp16，节省空间）",
    )

    # 运行范围控制（调试参数，默认全量）
    p.add_argument("--max_tasks", type=int, default=None, help="[调试] 最多处理多少个 task")
    p.add_argument("--max_episodes", type=int, default=None, help="[调试] 每个 task 最多处理多少个 episode")
    p.add_argument("--max_windows", type=int, default=None, help="[调试] 每个 episode 最多处理多少个窗口")

    # CUDA
    p.add_argument(
        "--cuda_visible_devices",
        type=str,
        default=None,
        help=(
            "可选：设置 CUDA_VISIBLE_DEVICES（例如 '1'）。"
            "如果你已经在外部设了环境变量，可以不传。"
        ),
    )

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
    print(f"[设备] {device} | amp_dtype={get_amp_dtype() if device == 'cuda' else 'fp32'}")

    # 1) 加载模型（本地离线）
    # 优先：如果你提供 --weights_pth，则使用“小 backbone + 本地 .pth”，更适合在 GPU 上跑通流程。
    # 否则：尝试加载你现有的 7B safetensors（GPU 16GB 会 OOM，建议 CPU）。
    if args.weights_pth:
        model, processor_cfg = load_small_dinov3_from_pth(
            backbone=args.backbone,
            weights_pth=args.weights_pth,
            device=device,
        )
    else:
        model, processor_cfg = load_local_hf_dinov3(model_dir=args.model_dir, device=device)

    # 2) 扫描数据集
    if args.task is not None:
        task_dirs = [str(Path(args.rgb_root) / args.task)]
    else:
        task_dirs = list_task_dirs(args.rgb_root)
    if args.max_tasks is not None:
        task_dirs = task_dirs[: args.max_tasks]

    if not task_dirs:
        raise FileNotFoundError(f"rgb_root 下没有找到 task 目录: {args.rgb_root}")

    # 3) 遍历 task/episode
    for task_dir in task_dirs:
        eps = list_episode_dirs(task_dir)
        if args.episode is not None:
            eps = [p for p in eps if Path(p).name == args.episode]
        if args.max_episodes is not None:
            eps = eps[: args.max_episodes]

        for ep_dir in eps:
            print(f"\n[处理] task={os.path.basename(task_dir)} | episode={os.path.basename(ep_dir)}")
            process_episode(
                model_dir=args.model_dir,
                episode_dir=ep_dir,
                out_root=args.out_root,
                window_size=int(args.window_size),
                stride=int(args.stride),
                fuse=args.fusion,
                keep_time_dim=bool(args.keep_time_dim),
                also_save_fused=bool(args.also_save_fused),
                save_dtype=str(args.save_dtype),
                device=device,
                amp_dtype=get_amp_dtype(),
                overwrite=True,
                model_cache={"model": model},
            )


if __name__ == "__main__":
    main()
