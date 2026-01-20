"""features_common/online_extractors.py

加载 4 个 vision backbone 用于在线特征提取

使用方法:
    extractors = load_all_extractors(device='cuda')
    dataset = DPRGBOnline4ModelDataset(
        raw_data_root='raw_data',
        tasks=['task1'],
        feature_extractors=extractors,
        ...
    )
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Dict
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_croco_extractor(device: str = 'cuda') -> Callable:
    """加载 CroCo 特征提取器 (1024 dim)"""
    croco_root = REPO_ROOT / 'croco'
    if str(croco_root) not in sys.path:
        sys.path.insert(0, str(croco_root))
    
    from models.croco import CroCoNet
    from models.croco_downstream import croco_args_from_ckpt
    
    # 加载模型
    ckpt_path = croco_root / 'pretrained_models' / 'CroCo_V2_ViTLarge_BaseDecoder.pth'
    if not ckpt_path.exists():
        raise FileNotFoundError(f"CroCo checkpoint not found: {ckpt_path}")
    
    ckpt = torch.load(ckpt_path, map_location='cpu')
    croco_kwargs = croco_args_from_ckpt(ckpt)
    model = CroCoNet(**croco_kwargs).to(device)
    model.eval()
    model.load_state_dict(ckpt['model'], strict=True)
    
    img_size = int(model.patch_embed.img_size[0])
    patch_size = int(model.patch_embed.patch_size[0])
    
    # Transform
    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    @torch.no_grad()
    def extract_croco(img: Image.Image) -> np.ndarray:
        """Extract CroCo feature (1024 dim, averaged over patches)"""
        x = transform(img).unsqueeze(0).to(device)  # [1, 3, 224, 224]
        # CroCo forward: _encode_image 返回 (feat, pos, mask)
        feat, _pos, _mask = model._encode_image(x, do_mask=False, return_all_blocks=False)
        # feat: [1, N, 1024], N = (img_size/patch_size)^2
        # 全局平均池化
        feat_avg = feat.mean(dim=1)  # [1, 1024]
        return feat_avg.cpu().numpy().squeeze(0)  # [1024]
    
    return extract_croco


def load_vggt_extractor(device: str = 'cuda') -> Callable:
    """加载 VGGT 特征提取器 (2048 dim)"""
    vggt_root = REPO_ROOT / 'vggt'
    if str(vggt_root) not in sys.path:
        sys.path.insert(0, str(vggt_root))
    
    from vggt.models.vggt import VGGT
    
    # 加载模型
    ckpt_path = vggt_root / 'weight' / 'model.pt'
    if not ckpt_path.exists():
        raise FileNotFoundError(f"VGGT checkpoint not found: {ckpt_path}")
    
    model = VGGT()
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    model.to(device)
    model.eval()
    
    # VGGT 需要特定的预处理
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    
    @torch.no_grad()
    def extract_vggt(img: Image.Image) -> np.ndarray:
        """Extract VGGT feature (2048 dim)"""
        x = transform(img).unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, 3, 224, 224] (B, S, C, H, W)
        
        with torch.cuda.amp.autocast(dtype=dtype):
            aggregated_tokens_list, _ = model.aggregator(x)
            # 取最后一层特征的平均
            feat = aggregated_tokens_list[-1].mean(dim=(1, 2))  # [B, embed_dim]
        
        # 确保是 2048 维
        if feat.shape[1] != 2048:
            # 自适应池化或投影到 2048
            feat = torch.nn.functional.adaptive_avg_pool1d(feat.unsqueeze(1), 2048).squeeze(1)
        
        return feat.cpu().numpy().squeeze(0)  # [2048]
    
    return extract_vggt


def load_dinov3_extractor(device: str = 'cuda') -> Callable:
    """加载本地DINOv3特征提取器 (768 dim for ViT-B/16) - 使用原生DinoVisionTransformer"""
    dinov3_root = REPO_ROOT / 'dinov3'
    dinov3_outer_path = str(dinov3_root)
    if dinov3_outer_path not in sys.path:
        sys.path.insert(0, dinov3_outer_path)
    
    dinov3_dir = dinov3_root / 'weight' / 'B16'
    print(f"[DINOv3] Loading from local path: {dinov3_dir}")
    
    # 设置离线模式
    import os as _os
    _os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    _os.environ.setdefault("HF_HUB_OFFLINE", "1")
    
    import json
    from safetensors.torch import load_file
    from dinov3.models.vision_transformer import DinoVisionTransformer
    
    # 加载config
    config_path = dinov3_dir / "config.json"
    with open(config_path, "r") as f:
        cfg = json.load(f)
    
    patch_size = int(cfg.get("patch_size", 16))
    image_size = int(cfg.get("image_size", 224))
    embed_dim = int(cfg.get("hidden_size", 768))
    depth = int(cfg.get("num_hidden_layers", 12))
    num_heads = int(cfg.get("num_attention_heads", 12))
    
    # 创建模型
    model = DinoVisionTransformer(
        img_size=image_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
    )
    
    # 加载权重
    weights_path = dinov3_dir / "model.safetensors"
    state_dict = load_file(str(weights_path))
    
    # 转换HF格式到dinov3格式 - 使用与离线提取一致的转换逻辑
    def convert_hf_to_dinov3(sd):
        """把 HF 风格 key(embeddings.*, encoder.layer.*) 转成 dinov3 原生 key
        
        ⚠️ 关键：DinoVisionTransformer使用融合的qkv权重，而HF格式是分离的q/k/v
        需要手动融合：qkv = concat([q_proj, k_proj, v_proj], dim=0)
        
        ⚠️ 注意：safetensors文件中键名格式为layer.X，而非encoder.layer.X
        """
        # Step 1: 融合q/k/v projections并立即生成转换后的key
        fused = {}
        to_remove = set()  # 记录要删除的原始q/k/v keys
        
        q_keys = [k for k in sd.keys() if '.q_proj.' in k and 'attention' in k]
        
        for q_key in q_keys:
            # layer.0.attention.q_proj.weight -> blocks.0.attn.qkv.weight
            k_key = q_key.replace('.q_proj.', '.k_proj.')
            v_key = q_key.replace('.q_proj.', '.v_proj.')
            
            if k_key in sd and v_key in sd:
                # 融合qkv weights
                q_weight = sd[q_key]
                k_weight = sd[k_key]
                v_weight = sd[v_key]
                qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
                
                # 生成目标key: layer.0.attention.q_proj.weight -> blocks.0.attn.qkv.weight
                target_key = q_key.replace('layer.', 'blocks.').replace('.attention.q_proj.', '.attn.qkv.')
                fused[target_key] = qkv_weight
                
            # 标记原始keys为待删除（无论bias是否存在都要标记）
            to_remove.add(q_key)
            to_remove.add(k_key)
            to_remove.add(v_key)
            
            # bias keys也要标记为删除（即使不存在或未融合）
            q_bias_key = q_key.replace('.weight', '.bias')
            k_bias_key = k_key.replace('.weight', '.bias')
            v_bias_key = v_key.replace('.weight', '.bias')
            to_remove.add(q_bias_key)
            to_remove.add(k_bias_key)
            to_remove.add(v_bias_key)
            
            # 如果所有bias都存在，才融合
            if all(k in sd for k in [q_bias_key, k_bias_key, v_bias_key]):
                qkv_bias = torch.cat([sd[q_bias_key], sd[k_bias_key], sd[v_bias_key]], dim=0)
                target_bias_key = target_key.replace('.weight', '.bias')
                fused[target_bias_key] = qkv_bias        # Step 2: 处理其他keys
        final = {}
        for k, v in sd.items():
            # 跳过要删除的q/k/v keys
            if k in to_remove:
                continue
            
            nk = k
            
            # embeddings  - 必须在layer转换之前处理！
            if nk.startswith('embeddings.'):
                nk = nk.replace('embeddings.cls_token', 'cls_token')
                nk = nk.replace('embeddings.mask_token', 'mask_token')
                nk = nk.replace('embeddings.register_tokens', 'storage_tokens')
                nk = nk.replace('embeddings.patch_embeddings.weight', 'patch_embed.proj.weight')
                nk = nk.replace('embeddings.patch_embeddings.bias', 'patch_embed.proj.bias')
            elif nk.startswith('layer.'):
                # layer -> blocks
                nk = nk.replace('layer.', 'blocks.')
                
                # attention projection
                nk = nk.replace('.attention.o_proj.', '.attn.proj.')
                
                # MLP - 普通MLP，不是gated
                nk = nk.replace('.mlp.up_proj.', '.mlp.fc1.')
                nk = nk.replace('.mlp.down_proj.', '.mlp.fc2.')
                
                # norms - 保持不变
                
                # layer scales
                nk = nk.replace('.layer_scale1.lambda1', '.ls1.gamma')
                nk = nk.replace('.layer_scale2.lambda1', '.ls2.gamma')
            
            # token参数形状对齐
            if nk == 'mask_token' and len(v.shape) == 3 and v.shape[0] == 1 and v.shape[1] == 1:
                v = v.squeeze(1)
            if nk == 'cls_token' and v.ndim == 2 and v.shape[0] == 1:
                v = v.unsqueeze(1)
            
            final[nk] = v
        
        # 合并融合的qkv
        final.update(fused)
        return final
        return final
    
    state_dict = convert_hf_to_dinov3(state_dict)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    
    # 创建transform
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    @torch.no_grad()
    def extract_dinov3(img: Image.Image) -> np.ndarray:
        """Extract DINOv3 feature (768 dim) - 使用patch tokens全局平均池化"""
        x = transform(img).unsqueeze(0).to(device)  # [1, 3, 224, 224]
        outputs = model.forward_features(x)
        # ⚠️ 关键修复：与训练时一致，使用patch tokens的全局平均池化
        patch_tokens = outputs['x_norm_patchtokens']  # [1, 196, 768]
        feat = patch_tokens.mean(dim=1)  # [1, 768]
        return feat.cpu().float().numpy().squeeze(0)  # [768]
    
    return extract_dinov3


def load_da3_extractor(device: str = 'cuda') -> Callable:
    """加载 Depth-Anything-3 特征提取器 (2048 dim)"""
    da3_root = REPO_ROOT / 'Depth-Anything-3'
    if str(da3_root) not in sys.path:
        sys.path.insert(0, str(da3_root))
    
    from depth_anything_3.api import DepthAnything3
    from pathlib import Path
    
    # 加载模型
    model_dir = da3_root / 'weight'
    if not model_dir.exists():
        raise FileNotFoundError(f"DA3 weight dir not found: {model_dir}")
    
    # 尝试加载 safetensors
    sf_path = model_dir / 'model.safetensors'
    if sf_path.exists():
        from safetensors.torch import load_file as safeload
        sd = safeload(str(sf_path))
        model = DepthAnything3()
        model.load_state_dict(sd, strict=False)
        model.to(device)
        model.eval()
    else:
        # Fallback to from_pretrained
        model = DepthAnything3.from_pretrained(str(model_dir))
        model.to(device)
        model.eval()
    
    transform = T.Compose([
        T.Resize((518, 518)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    @torch.no_grad()
    def extract_da3(img: Image.Image) -> np.ndarray:
        """Extract DA3 feature (2048 dim)"""
        x = transform(img).unsqueeze(0).to(device)  # [1, 3, 518, 518]
        
        # DA3 提取 encoder 特征
        # 假设 model 有 forward_features 或类似方法
        # 这里需要根据实际 API 调整
        if hasattr(model, 'encoder'):
            feat = model.encoder.forward_features(x)
        elif hasattr(model, 'forward_features'):
            feat = model.forward_features(x)
        else:
            # 如果没有特征提取方法，使用 forward 然后取中间层
            _ = model(x)
            # 这里需要根据实际模型结构调整
            feat = model.encoder.patch_embed(x)
            if hasattr(feat, 'shape') and len(feat.shape) == 3:
                feat = feat[:, 0, :]  # 取 [CLS] token
        
        # 确保是 2048 维
        if feat.shape[-1] != 2048:
            feat = torch.nn.functional.adaptive_avg_pool1d(
                feat.view(feat.shape[0], -1).unsqueeze(1), 2048
            ).squeeze(1)
        
        return feat.cpu().numpy().squeeze(0)  # [2048]
    
    return extract_da3


def load_all_extractors(device: str = 'cuda') -> Dict[str, Callable]:
    """
    加载所有 4 个特征提取器
    
    Returns:
        extractors: {'croco': fn, 'vggt': fn, 'dinov3': fn, 'da3': fn}
    """
    print("Loading feature extractors...")
    extractors = {}
    
    print("  Loading CroCo...")
    extractors['croco'] = load_croco_extractor(device)
    
    print("  Loading VGGT...")
    extractors['vggt'] = load_vggt_extractor(device)
    
    print("  Loading DINOv3...")
    extractors['dinov3'] = load_dinov3_extractor(device)
    
    print("  Loading DA3...")
    extractors['da3'] = load_da3_extractor(device)
    
    print("✓ All extractors loaded")
    return extractors


if __name__ == '__main__':
    # 测试
    extractors = load_all_extractors(device='cuda')
    
    # 测试提取
    from PIL import Image
    import numpy as np
    
    dummy_img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    
    for name, extractor in extractors.items():
        feat = extractor(dummy_img)
        print(f"{name}: {feat.shape}")
