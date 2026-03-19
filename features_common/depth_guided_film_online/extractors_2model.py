"""features_common/depth_guided_film_online/extractors_2model.py

2 模型在线特征提取器: DINOv3 + DA3
===================================

从 MultiGPUFeatureExtractors 简化而来, 去掉 CroCo / VGGT.
节省 ~40-50% 显存, 提速 ~40%.

模型顺序约定:
  index 0 -> DINOv3 (C=768)
  index 1 -> DA3    (C=2048)
"""

from __future__ import annotations

import sys
import os
import torch
import torch.nn as nn
import numpy as np
from typing import List
from PIL import Image


class TwoModelExtractors:
    """
    2 模型在线特征提取器: DINOv3 + DA3

    与 MultiGPUFeatureExtractors 的区别:
    - 只加载 DINOv3 + DA3 (vs 4 个模型)
    - 更省显存 (单卡可跑)
    - extract_batch_tokens 返回 List[2 × Tensor [B, K_i, C_i]]

    用法:
        extractors = TwoModelExtractors(gpu_id=0)
        # 提取 token 级别特征
        tokens_list = extractors.extract_batch_tokens(images)
        # tokens_list[0]: DINOv3 [B, K_d, 768]
        # tokens_list[1]: DA3    [B, K_a, 2048]
    """

    def __init__(self, gpu_id: int = 0):
        self.gpu_id = gpu_id
        self.device = f"cuda:{gpu_id}"
        self.extractors = {}

        print(f"[TwoModel] 加载 DINOv3 + DA3 到 GPU {gpu_id}...")
        self._load_extractors()

    def _load_extractors(self):
        """加载 DINOv3 和 DA3"""
        # __file__ = features_common/depth_guided_film_online/extractors_2model.py
        # 需要 features_model/ 目录 (往上3层)
        cwd = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        # ---- 1. DINOv3 ----
        print(f"[TwoModel] 加载 DINOv3 到 GPU {self.gpu_id}...")
        dinov3_outer_path = os.path.join(cwd, "dinov3")
        if dinov3_outer_path not in sys.path:
            sys.path.insert(0, dinov3_outer_path)

        dinov3_dir = os.path.join(cwd, "dinov3/weight/B16")

        import importlib
        load_local_hf_dinov3 = importlib.import_module(
            "extract_multi_frame_dinov3_features_local"
        ).load_local_hf_dinov3

        dinov3_model, processor_cfg = load_local_hf_dinov3(
            model_dir=dinov3_dir,
            device=self.device,
        )
        dinov3_model.eval()

        image_size = int(processor_cfg.get("size", {}).get("height", 224))
        patch_size = int(processor_cfg.get("patch_size", 16))
        image_mean = processor_cfg.get("image_mean", [0.485, 0.456, 0.406])
        image_std = processor_cfg.get("image_std", [0.229, 0.224, 0.225])
        dinov3_patch_tokens = (image_size // patch_size) ** 2

        self.dinov3_model = dinov3_model
        self.dinov3_image_size = image_size
        self.dinov3_image_mean = image_mean
        self.dinov3_image_std = image_std
        self.dinov3_patch_tokens = dinov3_patch_tokens
        self.dinov3_output_dim = 768
        print(f"[TwoModel] DINOv3 loaded: dim={self.dinov3_output_dim}, "
              f"patch_tokens={dinov3_patch_tokens}, image_size={image_size}")

        # ---- 2. DA3 (Depth-Anything-V3) ----
        print(f"[TwoModel] 加载 DA3 到 GPU {self.gpu_id}...")
        da3_path = os.path.join(cwd, "Depth-Anything-3/src")
        if da3_path not in sys.path:
            sys.path.insert(0, da3_path)

        os.environ.setdefault("DA3_LOG_LEVEL", "ERROR")

        da3_model_dir = os.path.join(cwd, "Depth-Anything-3/weight")

        try:
            import importlib.util
            da3_script = os.path.join(
                cwd, "Depth-Anything-3",
                "extract_multi_frame_depthanything3_features.py",
            )
            spec = importlib.util.spec_from_file_location("da3_extract", da3_script)
            da3_mod = importlib.util.module_from_spec(spec)
            assert spec and spec.loader
            spec.loader.exec_module(da3_mod)
            da3_model = da3_mod.build_da3(
                da3_model_dir, device=torch.device(self.device)
            )
        except Exception:
            from depth_anything_3.api import DepthAnything3
            da3_model = DepthAnything3.from_pretrained(da3_model_dir)
            da3_model = da3_model.to(self.device)
            da3_model.eval()

        self.da3_model = da3_model
        self.da3_output_dim = 2048
        self.da3_process_res = 504
        self.da3_process_res_method = "upper_bound_resize"
        print(f"[TwoModel] DA3 loaded: dim={self.da3_output_dim}")
        print("[TwoModel] ✓ 2 模型加载完成")

    # ------------------------------------------------------------------
    # Token 级别特征提取 (主要接口)
    # ------------------------------------------------------------------
    def extract_batch_tokens(
        self,
        images: List[Image.Image],
        *,
        max_tokens: int | None = None,
        return_torch: bool = True,
    ) -> list[torch.Tensor]:
        """
        批量提取 2 模型 token 特征 (不做均值池化).

        Args:
            images: List of PIL Image (RGB), length = B
            max_tokens: 若指定, 等间隔下采样到 K 个 token
            return_torch: True 返回 torch.Tensor; False 返回 numpy

        Returns:
            tokens_list: list[Tensor], len=2
                [0] DINOv3: [B, K_d, 768]
                [1] DA3:    [B, K_a, 2048]
        """
        B = len(images)
        tokens_list = []

        # ---- DINOv3 tokens ----
        dinov3_tokens = self._extract_dinov3_tokens(images)  # [B, K_d, 768]
        tokens_list.append(dinov3_tokens)

        # ---- DA3 tokens ----
        da3_tokens = self._extract_da3_tokens(images)  # [B, K_a, 2048]
        tokens_list.append(da3_tokens)

        # 可选: 下采样 token 数
        if max_tokens is not None:
            tokens_list = [
                self._subsample_tokens(t, max_tokens) for t in tokens_list
            ]

        if return_torch:
            return tokens_list
        return [t.detach().cpu().numpy() for t in tokens_list]

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _extract_dinov3_tokens(self, images: List[Image.Image]) -> torch.Tensor:
        """提取 DINOv3 patch tokens."""
        from torchvision import transforms

        transform = transforms.Compose([
            transforms.Resize((self.dinov3_image_size, self.dinov3_image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.dinov3_image_mean,
                std=self.dinov3_image_std,
            ),
        ])
        img_tensors = torch.stack([transform(img) for img in images]).to(self.device)

        if hasattr(self.dinov3_model, "get_intermediate_layers"):
            y = self.dinov3_model.get_intermediate_layers(img_tensors, n=1)[0]
        else:
            y = self.dinov3_model(img_tensors)

        patch_tokens = self.dinov3_patch_tokens or (y.shape[1] - 1)
        tokens = y[:, -patch_tokens:, :]  # [B, K_d, 768]
        return tokens.float()

    @torch.no_grad()
    def _extract_da3_tokens(self, images: List[Image.Image]) -> torch.Tensor:
        """提取 DA3 backbone tokens (不做均值池化)."""

        def _da3_forward(inputs: list[Image.Image | np.ndarray]) -> torch.Tensor:
            imgs_cpu, _, _ = self.da3_model._preprocess_inputs(
                inputs,
                extrinsics=None,
                intrinsics=None,
                process_res=self.da3_process_res,
                process_res_method=self.da3_process_res_method,
            )
            imgs, _, _ = self.da3_model._prepare_model_inputs(imgs_cpu, None, None)
            if isinstance(imgs, (list, tuple)):
                img_list = []
                for x in imgs:
                    if isinstance(x, torch.Tensor):
                        if x.ndim == 3:
                            x = x.unsqueeze(0)
                        img_list.append(x)
                if len(img_list) == 0:
                    raise RuntimeError("DA3 _prepare_model_inputs returned empty list")
                imgs = torch.cat(img_list, dim=0)
            if not isinstance(imgs, torch.Tensor):
                raise RuntimeError(
                    f"DA3 _prepare_model_inputs returned unexpected type: {type(imgs)}"
                )
            imgs = imgs.to(self.device)
            backbone = self.da3_model.model.backbone
            feats, _ = backbone(x=imgs, export_feat_layers=[23])
            if not feats:
                raise RuntimeError("backbone 返回空特征")
            tokens, _ = feats[0]
            if tokens.ndim == 4:
                tokens = tokens.flatten(1, 2)  # [B, H*W, C]
            return tokens

        B = len(images)
        tokens = _da3_forward(images)
        
        if tokens.ndim == 2:
            tokens = tokens.unsqueeze(0)
        if tokens.shape[0] != B:
            # 回退逐张
            per_tokens = []
            for img in images:
                t = _da3_forward([img])
                if t.ndim == 2:
                    t = t.unsqueeze(0)
                per_tokens.append(t[0])
            tokens = torch.stack(per_tokens, dim=0)

        return tokens.float()

    @staticmethod
    def _subsample_tokens(tokens: torch.Tensor, max_k: int) -> torch.Tensor:
        """确定性等间隔下采样."""
        B, K, C = tokens.shape
        if K <= max_k:
            return tokens
        idx = torch.linspace(0, K - 1, max_k, device=tokens.device).long()
        idx = idx.unsqueeze(0).expand(B, -1)  # [B, max_k]
        return tokens.gather(1, idx.unsqueeze(-1).expand(-1, -1, C))
