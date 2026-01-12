"""
多GPU特征提取器管理器
支持将4个大模型分布到多个GPU上，解决单卡显存不足问题
"""
import torch
import torch.nn as nn
from typing import Dict, List
from PIL import Image
import numpy as np


class MultiGPUFeatureExtractors:
    """
    多GPU特征提取器管理
    
    策略:
    - GPU 0: CroCo + DINOv3 (较小模型)
    - GPU 1: VGGT + DA3 (较大模型)
    
    如果只有1张卡，fallback到单卡模式
    """
    
    def __init__(self, gpu_ids: list[int] = [0, 1]):
        self.gpu_ids = gpu_ids
        self.num_gpus = len(gpu_ids)
        self.extractors = {}
        
        # 检查可用GPU
        available_gpus = torch.cuda.device_count()
        if available_gpus < self.num_gpus:
            print(f"[MultiGPU] 警告: 请求{self.num_gpus}张卡，但只有{available_gpus}张可用")
            self.gpu_ids = list(range(available_gpus))
            self.num_gpus = available_gpus
        
        if self.num_gpus == 0:
            raise RuntimeError("没有可用GPU，在线训练需要GPU支持")
        
        print(f"[MultiGPU] 使用 {self.num_gpus} 张GPU: {self.gpu_ids}")
        
        self._load_extractors()
    
    def _load_extractors(self):
        """加载4个特征提取器到不同GPU"""
        import sys
        import os
        
        # 保存项目根目录（从multi_gpu_extractors.py的位置推断）
        # features_common/multi_gpu_extractors.py -> 项目根目录
        cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # 根据GPU数量分配模型
        if self.num_gpus >= 2:
            # 2卡或更多: CroCo+DINOv3 on GPU0, VGGT+DA3 on GPU1
            model_gpu_map = {
                'croco': self.gpu_ids[1],
                'dinov3': self.gpu_ids[1],
                'vggt': self.gpu_ids[0],
                'da3': self.gpu_ids[1],
            }
        else:
            # 单卡: 全部放GPU0 (显存紧张，可能OOM)
            print("[MultiGPU] 警告: 单GPU模式，显存可能不足！")
            model_gpu_map = {name: self.gpu_ids[0] for name in ['croco', 'dinov3', 'vggt', 'da3']}
        
        print("[MultiGPU] 模型分配:")
        for name, gpu_id in model_gpu_map.items():
            print(f"  {name} -> GPU {gpu_id}")
        
        # 1. CroCo
        print(f"[MultiGPU] 加载 CroCo 到 GPU {model_gpu_map['croco']}...")
        # 尝试 import croco (如果 cwd 在 path 中且 croco 是 namespace packge)
        # 或者添加 croco 目录到 path 并 import models
        try:
            from croco.models.croco import CroCoNet
            from croco.models.croco_downstream import croco_args_from_ckpt
        except ImportError:
            croco_path = os.path.join(cwd, 'croco')
            if croco_path not in sys.path:
                sys.path.insert(0, croco_path)
            from models.croco import CroCoNet
            from models.croco_downstream import croco_args_from_ckpt
        
        # 使用绝对路径（避免工作目录切换导致的问题）
        croco_ckpt_path = os.path.join(cwd, 'croco/pretrained_models/CroCo_V2_ViTLarge_BaseDecoder.pth')
        ckpt = torch.load(croco_ckpt_path, map_location='cpu')
        croco_kwargs = croco_args_from_ckpt(ckpt)
        croco_model = CroCoNet(**croco_kwargs)
        croco_model.load_state_dict(ckpt['model'], strict=True)
        croco_model = croco_model.to(f'cuda:{model_gpu_map["croco"]}')
        croco_model.eval()
        self.extractors['croco'] = FeatureExtractorWrapper(
            croco_model, model_gpu_map['croco'], 'croco', output_dim=1024
        )
        
        # 2. DINOv3
        print(f"[MultiGPU] 加载 DINOv3 到 GPU {model_gpu_map['dinov3']}...")
        # 为了 import dinov3.models.vision_transformer，需要把 dinov3 外层目录加到 path
        # 这样 import dinov3 就会找到 dinov3/dinov3 目录
        dinov3_outer_path = os.path.join(cwd, 'dinov3')
        if dinov3_outer_path not in sys.path:
            sys.path.insert(0, dinov3_outer_path)

        dinov3_dir = os.path.join(cwd, 'dinov3/weight/B16')  # 绝对路径
        
        import os as _os
        _os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        _os.environ.setdefault("HF_HUB_OFFLINE", "1")
        
        import json
        from safetensors.torch import load_file
        # 指向 dinov3/dinov3/models/vision_transformer.py
        from dinov3.models.vision_transformer import DinoVisionTransformer
        
        config_path = os.path.join(dinov3_dir, "config.json")
        with open(config_path, "r") as f:
            cfg = json.load(f)
        
        patch_size = int(cfg.get("patch_size", 16))
        image_size = int(cfg.get("image_size", 224))
        embed_dim = int(cfg.get("hidden_size", 768))
        depth = int(cfg.get("num_hidden_layers", 12))
        num_heads = int(cfg.get("num_attention_heads", 12))
        
        dinov3_model = DinoVisionTransformer(
            img_size=image_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
        )
        
        weights_path = os.path.join(dinov3_dir, "model.safetensors")
        state_dict = load_file(weights_path)
        
        def convert_hf_to_dinov3(sd):
            out = {}
            for k, v in sd.items():
                nk = k.replace("embeddings.cls_token", "cls_token")
                nk = nk.replace("embeddings.mask_token", "mask_token")
                nk = nk.replace("embeddings.patch_embeddings", "patch_embed")
                nk = nk.replace("encoder.layer.", "blocks.")
                nk = nk.replace(".attention.attention", ".attn")
                nk = nk.replace(".intermediate.dense", ".mlp.fc1")
                nk = nk.replace(".output.dense", ".mlp.fc2")
                nk = nk.replace(".layernorm_before", ".norm1")
                nk = nk.replace(".layernorm_after", ".norm2")
                
                # 修复mask_token维度: [1, 1, 768] -> [1, 768]
                if nk == "mask_token" and len(v.shape) == 3:
                    v = v.squeeze(1)
                
                out[nk] = v
            return out
        
        state_dict = convert_hf_to_dinov3(state_dict)
        dinov3_model.load_state_dict(state_dict, strict=False)
        dinov3_model = dinov3_model.to(f'cuda:{model_gpu_map["dinov3"]}')
        dinov3_model.eval()
        self.extractors['dinov3'] = FeatureExtractorWrapper(
            dinov3_model, model_gpu_map['dinov3'], 'dinov3', output_dim=1024
        )
        
        # 3. VGGT
        print(f"[MultiGPU] 加载 VGGT 到 GPU {model_gpu_map['vggt']}...")
        vggt_outer_path = os.path.join(cwd, 'vggt')
        if vggt_outer_path not in sys.path:
            sys.path.insert(0, vggt_outer_path)
        from vggt.models.vggt import VGGT
        
        vggt_model = VGGT()
        vggt_weight = os.path.join(cwd, 'vggt/weight/model.pt')  # 绝对路径
        state_dict = torch.load(vggt_weight, map_location='cpu')
        vggt_model.load_state_dict(state_dict)
        vggt_model = vggt_model.to(f'cuda:{model_gpu_map["vggt"]}')
        vggt_model.eval()
        self.extractors['vggt'] = FeatureExtractorWrapper(
            vggt_model, model_gpu_map['vggt'], 'vggt', output_dim=768
        )
        
        # 4. Depth-Anything-V3
        print(f"[MultiGPU] 加载 DA3 到 GPU {model_gpu_map['da3']}...")
        da3_path = os.path.join(cwd, 'Depth-Anything-3/src')
        if da3_path not in sys.path:
            sys.path.insert(0, da3_path)
        # Reduce noisy INFO logs from Depth-Anything-3 (e.g., "Processed Images Done ...")
        # Users can override by setting DA3_LOG_LEVEL in the environment.
        import os as _os
        _os.environ.setdefault("DA3_LOG_LEVEL", "ERROR")
        from depth_anything_3.api import DepthAnything3
        
        da3_model_dir = os.path.join(cwd, 'Depth-Anything-3/weight')  # 绝对路径
        da3_model = DepthAnything3.from_pretrained(da3_model_dir)
        da3_model = da3_model.to(f'cuda:{model_gpu_map["da3"]}')
        da3_model.eval()
        self.extractors['da3'] = FeatureExtractorWrapper(
            da3_model, model_gpu_map['da3'], 'da3', output_dim=1024
        )
        
        print("[MultiGPU] ✓ 所有模型加载完成")
    
    def extract(self, image: Image.Image) -> np.ndarray:
        """
        提取单帧的4模型特征
        
        Args:
            image: PIL Image (RGB)
        
        Returns:
            features: [4, 2048] numpy array
        """
        features = []
        for name in ['croco', 'vggt', 'dinov3', 'da3']:
            feat = self.extractors[name](image)
            features.append(feat)
        return np.stack(features, axis=0)  # [4, 2048]
    
    def __call__(self, image: Image.Image) -> np.ndarray:
        return self.extract(image)


class FeatureExtractorWrapper:
    """单个特征提取器的包装类"""
    
    def __init__(self, model: nn.Module, gpu_id: int, model_name: str, output_dim: int):
        self.model = model
        self.gpu_id = gpu_id
        self.model_name = model_name
        self.output_dim = output_dim
        self.device = f'cuda:{gpu_id}'
    
    def __call__(self, image: Image.Image) -> np.ndarray:
        """
        提取特征并pad到2048维
        
        Args:
            image: PIL Image
        
        Returns:
            feat: [2048] numpy array
        """
        with torch.no_grad():
            # 预处理（根据模型类型）
            if self.model_name == 'croco':
                from torchvision import transforms
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
                img_tensor = transform(image).unsqueeze(0).to(self.device)
                # 使用CroCo的_encode_image方法
                feat = self.model._encode_image(img_tensor, do_mask=False)[0]  # [1, N, C]
                feat = feat.mean(dim=1).squeeze(0).cpu().numpy()  # [C]
            
            elif self.model_name == 'dinov3':
                from torchvision import transforms
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
                img_tensor = transform(image).unsqueeze(0).to(self.device)
                feat = self.model(img_tensor)  # [1, 1024]
                feat = feat.squeeze(0).cpu().numpy()
            
            elif self.model_name == 'vggt':
                from torchvision import transforms
                from vggt.utils.load_fn import load_and_preprocess_images
                
                # VGGT需要从文件加载，先保存临时文件
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                    temp_path = f.name
                    image.save(temp_path)
                
                try:
                    imgs = load_and_preprocess_images([temp_path])  # [1, 3, 518, 518]
                    imgs = imgs.unsqueeze(0).to(self.device)  # [1, 1, 3, 518, 518]
                    
                    # 使用aggregator提取特征
                    aggregated_tokens_list, patch_start_idx = self.model.aggregator(imgs)
                    last_tokens = aggregated_tokens_list[-1]  # [1, 1, N, C]
                    patch_tokens = last_tokens[:, :, patch_start_idx:, :]  # [1, 1, Np, C]
                    
                    # 全局平均池化
                    feat = patch_tokens.mean(dim=[1, 2]).squeeze(0).cpu().numpy()  # [C]
                finally:
                    import os as _os
                    _os.unlink(temp_path)
            
            elif self.model_name == 'da3':
                # DA3使用backbone提取特征
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                    temp_path = f.name
                    image.save(temp_path)
                
                try:
                    imgs_cpu, _, _ = self.model._preprocess_inputs(
                        [temp_path],
                        extrinsics=None,
                        intrinsics=None,
                        process_res=518,
                        process_res_method='lower_bound_resize',
                    )
                    imgs, _, _ = self.model._prepare_model_inputs(imgs_cpu, None, None)
                    imgs = imgs.to(self.device)
                    
                    # 使用backbone提取特征（与extract_multi_frame脚本一致）
                    with torch.no_grad():
                        backbone = self.model.model.backbone
                        feats, _ = backbone(x=imgs, export_feat_layers=[-1])  # 导出最后一层
                        
                        if not feats:
                            raise RuntimeError("backbone返回空特征")
                        
                        tokens, _ = feats[0]  # [B, N, num_patches, C]
                        # 全局平均池化：对N和num_patches维度取平均
                        feat = tokens.mean(dim=[1, 2]).squeeze(0).cpu().numpy()  # [C]
                
                finally:
                    import os as _os
                    _os.unlink(temp_path)
            
            else:
                raise ValueError(f"Unknown model: {self.model_name}")
        
        # Pad到2048维
        if len(feat) < 2048:
            feat = np.pad(feat, (0, 2048 - len(feat)), mode='constant')
        elif len(feat) > 2048:
            feat = feat[:2048]
        
        return feat
