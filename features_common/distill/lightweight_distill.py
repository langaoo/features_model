#!/usr/bin/env python3
"""features_common/lightweight_distill.py

轻量化蒸馏模块：用ResNet类轻量网络替代4个大视觉模型

两种蒸馏方案：
方案A：仅蒸馏4模型融合逻辑 → 得到轻量化RGB特征提取模块 → 接入已训练的DP头
方案B：蒸馏完整链路(4模型+对齐) → 得到端到端轻量化模块 → 重新训练DP头

核心思路：
- Teacher: 4个大视觉模型 + 对齐encoder
- Student: ResNet18/34 + 轻量投影头
- Loss: MSE + InfoNCE (特征级蒸馏)

使用方法：
  # 训练轻量化模型
  python tools/train_lightweight_distill.py \
    --teacher_ckpt outputs/train_rgb2pc_runs/run_best_bs32/ckpt_step_0010000.pt \
    --student_backbone resnet18 \
    --scheme A \
    --save_dir outputs/lightweight_runs/run0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


@dataclass
class LightweightConfig:
    """轻量化模型配置"""
    backbone: str = "resnet18"  # resnet18, resnet34, mobilenetv3_small
    output_dim: int = 1280       # 与对齐encoder的fuse_dim一致
    hidden_dim: int = 512
    dropout: float = 0.1
    pretrained: bool = True
    freeze_backbone_layers: int = 0  # 冻结前N层，0=不冻结


class LightweightRGBEncoder(nn.Module):
    """轻量化RGB编码器
    
    用单个轻量backbone替代4个大视觉模型 + 对齐encoder
    输入: RGB图像 [B, 3, H, W]
    输出: 特征向量 [B, output_dim]
    """
    
    def __init__(self, config: LightweightConfig):
        super().__init__()
        self.config = config
        
        # 选择backbone
        if config.backbone == "resnet18":
            backbone = models.resnet18(pretrained=config.pretrained)
            backbone_dim = 512
        elif config.backbone == "resnet34":
            backbone = models.resnet34(pretrained=config.pretrained)
            backbone_dim = 512
        elif config.backbone == "resnet50":
            backbone = models.resnet50(pretrained=config.pretrained)
            backbone_dim = 2048
        elif config.backbone == "mobilenetv3_small":
            backbone = models.mobilenet_v3_small(pretrained=config.pretrained)
            backbone_dim = 576
        elif config.backbone == "mobilenetv3_large":
            backbone = models.mobilenet_v3_large(pretrained=config.pretrained)
            backbone_dim = 960
        elif config.backbone == "efficientnet_b0":
            backbone = models.efficientnet_b0(pretrained=config.pretrained)
            backbone_dim = 1280
        else:
            raise ValueError(f"Unknown backbone: {config.backbone}")
        
        # 移除分类头
        if "resnet" in config.backbone:
            self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        elif "mobilenet" in config.backbone:
            self.backbone = backbone.features
            self.pool = nn.AdaptiveAvgPool2d(1)
        elif "efficientnet" in config.backbone:
            self.backbone = backbone.features
            self.pool = nn.AdaptiveAvgPool2d(1)
        else:
            self.backbone = backbone
        
        self.backbone_dim = backbone_dim
        
        # 冻结部分层
        if config.freeze_backbone_layers > 0:
            self._freeze_layers(config.freeze_backbone_layers)
        
        # 投影头
        self.projector = nn.Sequential(
            nn.Linear(backbone_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.output_dim),
            nn.LayerNorm(config.output_dim),
        )
        
    def _freeze_layers(self, n_layers: int):
        """冻结backbone的前N层"""
        if "resnet" in self.config.backbone:
            layers = list(self.backbone.children())
            for layer in layers[:n_layers]:
                for param in layer.parameters():
                    param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, H, W] RGB图像 (ImageNet归一化)
        
        Returns:
            z: [B, output_dim] 特征向量
        """
        # Backbone特征提取
        feat = self.backbone(x)  # [B, C, H', W'] or [B, C, 1, 1]
        
        # 池化
        if hasattr(self, 'pool'):
            feat = self.pool(feat)
        
        feat = feat.flatten(1)  # [B, backbone_dim]
        
        # 投影
        z = self.projector(feat)  # [B, output_dim]
        
        return z


class LightweightSequenceEncoder(nn.Module):
    """支持时序输入的轻量化编码器
    
    输入: [B, To, 3, H, W] 连续To帧RGB
    输出: [B, To, output_dim]
    """
    
    def __init__(self, config: LightweightConfig, use_temporal_fusion: bool = False):
        super().__init__()
        self.config = config
        self.frame_encoder = LightweightRGBEncoder(config)
        self.use_temporal_fusion = use_temporal_fusion
        
        if use_temporal_fusion:
            # 简单的时序融合层
            self.temporal_attn = nn.MultiheadAttention(
                embed_dim=config.output_dim,
                num_heads=8,
                dropout=config.dropout,
                batch_first=True,
            )
            self.temporal_norm = nn.LayerNorm(config.output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, To, 3, H, W]
        
        Returns:
            z: [B, To, output_dim]
        """
        B, To, C, H, W = x.shape
        
        # 逐帧编码
        x_flat = x.reshape(B * To, C, H, W)
        z_flat = self.frame_encoder(x_flat)  # [B*To, output_dim]
        z = z_flat.reshape(B, To, -1)  # [B, To, output_dim]
        
        # 可选的时序融合
        if self.use_temporal_fusion:
            z_attn, _ = self.temporal_attn(z, z, z)
            z = self.temporal_norm(z + z_attn)
        
        return z


class DistillationLoss(nn.Module):
    """蒸馏损失函数
    
    组合多种损失：
    1. MSE: 特征级L2距离
    2. InfoNCE: 对比学习损失
    3. Cosine: 余弦相似度损失
    """
    
    def __init__(
        self,
        mse_weight: float = 1.0,
        infonce_weight: float = 0.1,
        cosine_weight: float = 0.5,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.mse_weight = mse_weight
        self.infonce_weight = infonce_weight
        self.cosine_weight = cosine_weight
        self.temperature = temperature
    
    def forward(
        self,
        student_feat: torch.Tensor,
        teacher_feat: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            student_feat: [B, D] or [B, T, D]
            teacher_feat: [B, D] or [B, T, D]
        
        Returns:
            total_loss: scalar
            loss_dict: 各项损失分解
        """
        # 展平时间维（如果有）
        if student_feat.ndim == 3:
            B, T, D = student_feat.shape
            student_feat = student_feat.reshape(B * T, D)
            teacher_feat = teacher_feat.reshape(B * T, D)
        
        losses = {}
        
        # MSE Loss
        if self.mse_weight > 0:
            mse = F.mse_loss(student_feat, teacher_feat)
            losses['mse'] = mse * self.mse_weight
        
        # Cosine Loss
        if self.cosine_weight > 0:
            s_norm = F.normalize(student_feat, dim=-1)
            t_norm = F.normalize(teacher_feat, dim=-1)
            cosine = 1 - (s_norm * t_norm).sum(dim=-1).mean()
            losses['cosine'] = cosine * self.cosine_weight
        
        # InfoNCE Loss
        if self.infonce_weight > 0:
            s_norm = F.normalize(student_feat, dim=-1)
            t_norm = F.normalize(teacher_feat, dim=-1)
            logits = (s_norm @ t_norm.t()) / self.temperature
            labels = torch.arange(logits.shape[0], device=logits.device)
            infonce = F.cross_entropy(logits, labels)
            losses['infonce'] = infonce * self.infonce_weight
        
        total_loss = sum(losses.values())
        losses['total'] = total_loss
        
        return total_loss, losses


class TeacherWrapper(nn.Module):
    """Teacher模型包装器
    
    封装4个视觉模型 + 对齐encoder的完整pipeline
    用于生成蒸馏目标
    """
    
    def __init__(
        self,
        encoder_ckpt_path: str,
        extractors,  # MultiGPUFeatureExtractors实例
        device: str = 'cuda',
    ):
        super().__init__()
        self.device = device
        self.extractors = extractors
        
        # 加载对齐encoder
        from features_common.rgb2pc_aligned_encoder_4models import RGB2PCAlignedEncoder4Models
        self.encoder = RGB2PCAlignedEncoder4Models.from_checkpoint(
            encoder_ckpt_path, freeze=True
        ).to(device)
        self.encoder.eval()
        
        # 冻结所有参数
        for param in self.parameters():
            param.requires_grad = False
    
    @torch.no_grad()
    def forward(self, rgb_images: List) -> torch.Tensor:
        """
        Args:
            rgb_images: List of PIL Images
        
        Returns:
            z: [B, output_dim] teacher特征
        """
        # 提取4模型特征
        feats_list = []
        for img in rgb_images:
            feat_4 = self.extractors(img)  # [4, 2048]
            feats_list.append(feat_4)
        
        feats = torch.from_numpy(
            __import__('numpy').stack(feats_list, axis=0)
        ).float().to(self.device)  # [B, 4, 2048]
        
        # 添加时间维
        feats = feats.unsqueeze(1)  # [B, 1, 4, 2048]
        
        # 通过对齐encoder
        z = self.encoder(feats)  # [B, 1, fuse_dim]
        
        return z.squeeze(1)  # [B, fuse_dim]


def create_lightweight_model(
    backbone: str = "resnet18",
    output_dim: int = 1280,
    pretrained: bool = True,
) -> LightweightRGBEncoder:
    """创建轻量化模型的便捷函数"""
    config = LightweightConfig(
        backbone=backbone,
        output_dim=output_dim,
        pretrained=pretrained,
    )
    return LightweightRGBEncoder(config)


def create_sequence_model(
    backbone: str = "resnet18",
    output_dim: int = 1280,
    use_temporal_fusion: bool = False,
) -> LightweightSequenceEncoder:
    """创建时序轻量化模型"""
    config = LightweightConfig(
        backbone=backbone,
        output_dim=output_dim,
    )
    return LightweightSequenceEncoder(config, use_temporal_fusion)


# ============ 性能对比 ============
"""
模型性能对比（推理时间 @ batch=1, 224x224）:

| 模型 | 参数量 | FLOPs | 推理时间 |
|------|--------|-------|----------|
| 4模型完整pipeline | ~1.5B | ~500G | ~200ms |
| ResNet18 | 11.7M | 1.8G | ~5ms |
| ResNet34 | 21.8M | 3.7G | ~8ms |
| MobileNetV3-Small | 2.5M | 0.06G | ~2ms |
| EfficientNet-B0 | 5.3M | 0.4G | ~4ms |

蒸馏后效果预期：
- 推理速度提升: 40-100x
- 精度损失: 5-15% (取决于蒸馏质量)
"""
