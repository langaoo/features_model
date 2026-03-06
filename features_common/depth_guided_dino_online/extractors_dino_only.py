"""features_common/depth_guided_dino_online/extractors_dino_only.py

DINOv3-only 在线特征提取器
============================

从 TwoModelExtractors 精简而来, 去掉 DA3 部分.
只加载 DINOv3 ViT-B/16 (768d), 显存节省约 60%.

接口与 TwoModelExtractors 兼容:
  extract_batch_tokens(images) -> List[1 × Tensor [B, K_d, 768]]
  (单元素列表, 兼容 DinoOnlyEncoder.forward(x) 输入)
"""

from __future__ import annotations
import sys
import os
import torch
import numpy as np
from typing import List
from PIL import Image


class DinoOnlyExtractors:
    """DINOv3-only 在线特征提取器.

    用法:
        extractors = DinoOnlyExtractors(gpu_id=0)
        tokens_list = extractors.extract_batch_tokens(images, max_tokens=196)
        # tokens_list[0]: DINOv3 [B, K_d, 768]
    """

    def __init__(self, gpu_id: int = 0):
        self.gpu_id = gpu_id
        self.device = f"cuda:{gpu_id}"
        print(f"[DinoOnly] 加载 DINOv3 到 GPU {gpu_id}...")
        self._load_dinov3()

    def _load_dinov3(self):
        # features_model/ 根目录
        cwd = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
        print(f"[DinoOnly] DINOv3 loaded: dim={self.dinov3_output_dim}, "
              f"patch_tokens={dinov3_patch_tokens}, image_size={image_size}")

    @staticmethod
    def _subsample_tokens(tokens: torch.Tensor, max_k: int) -> torch.Tensor:
        K = tokens.shape[1]
        if K <= max_k:
            return tokens
        idx = torch.linspace(0, K - 1, max_k, dtype=torch.long, device=tokens.device)
        return tokens[:, idx, :]

    def extract_batch_tokens(
        self,
        images: List[Image.Image],
        *,
        max_tokens: int | None = None,
        return_torch: bool = True,
    ) -> list:
        """
        批量提取 DINOv3 patch tokens.

        Returns:
            list[Tensor], len=1
                [0] DINOv3: [B, K_d, 768]
        """
        dino_tokens = self._extract_dinov3_tokens(images)  # [B, K_d, 768]
        if max_tokens is not None:
            dino_tokens = self._subsample_tokens(dino_tokens, max_tokens)
        if not return_torch:
            dino_tokens = dino_tokens.detach().cpu().numpy()
        return [dino_tokens]

    @torch.no_grad()
    def _extract_dinov3_tokens(self, images: List[Image.Image]) -> torch.Tensor:
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
        tokens = y[:, -patch_tokens:, :]   # [B, K_d, 768]
        return tokens.float()
