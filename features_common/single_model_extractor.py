"""
单模型特征提取器 - 只加载一个视觉模型
用于单模型推理,避免加载所有4个模型
"""
import torch
import torch.nn as nn
from typing import List
from PIL import Image
import numpy as np
import sys
import os

class SingleModelFeatureExtractor:
    """单模型特征提取器"""
    
    def __init__(self, model_name: str, gpu_id: int = 0):
        """
        Args:
            model_name: 模型名称 ('croco', 'vggt', 'dinov3', 'da3')
            gpu_id: GPU ID
        """
        self.model_name = model_name.lower()
        self.gpu_id = gpu_id
        self.device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
        
        # 输出维度
        self.output_dims = {
            'croco': 1024,
            'vggt': 2048,
            'dinov3': 768,
            'da3': 2048
        }
        self.output_dim = self.output_dims[self.model_name]
        
        print(f"[SingleModel] Loading {self.model_name} on GPU {gpu_id}...")
        self._load_model()
        print(f"[SingleModel] {self.model_name} loaded successfully!")
    
    def _load_model(self):
        """加载指定的单个模型"""
        cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        if self.model_name == 'croco':
            self._load_croco(cwd)
        elif self.model_name == 'vggt':
            self._load_vggt(cwd)
        elif self.model_name == 'dinov3':
            self._load_dinov3(cwd)
        elif self.model_name == 'da3':
            self._load_da3(cwd)
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
    
    def _load_croco(self, cwd):
        """加载CroCo"""
        try:
            from croco.models.croco import CroCoNet
            from croco.models.croco_downstream import croco_args_from_ckpt
        except ImportError:
            croco_path = os.path.join(cwd, 'croco')
            if croco_path not in sys.path:
                sys.path.insert(0, croco_path)
            from models.croco import CroCoNet
            from models.croco_downstream import croco_args_from_ckpt
        
        croco_ckpt_path = os.path.join(cwd, 'croco/pretrained_models/CroCo_V2_ViTLarge_BaseDecoder.pth')
        ckpt = torch.load(croco_ckpt_path, map_location='cpu')
        croco_kwargs = croco_args_from_ckpt(ckpt)
        self.model = CroCoNet(**croco_kwargs)
        self.model.load_state_dict(ckpt['model'], strict=True)
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def _load_vggt(self, cwd):
        """加载VGGT"""
        from transformers import AutoImageProcessor, AutoModel
        vggt_ckpt = "Idiot-Scientist/VGGT"
        self.processor = AutoImageProcessor.from_pretrained(vggt_ckpt)
        self.model = AutoModel.from_pretrained(vggt_ckpt, torch_dtype=torch.bfloat16).to(self.device)
        self.model.eval()
    
    def _load_dinov3(self, cwd):
        """加载本地DINOv3 ViT-B/16 (768 dim) - 使用原生DinoVisionTransformer"""
        # 添加dinov3外层目录到path
        dinov3_outer_path = os.path.join(cwd, 'dinov3')
        if dinov3_outer_path not in sys.path:
            sys.path.insert(0, dinov3_outer_path)
        
        dinov3_dir = os.path.join(cwd, 'dinov3/weight/B16')
        print(f"[SingleModel] Loading DINOv3 from local path: {dinov3_dir}")
        
        # 设置离线模式
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        
        import json
        from safetensors.torch import load_file
        from dinov3.models.vision_transformer import DinoVisionTransformer
        
        # 加载config
        config_path = os.path.join(dinov3_dir, "config.json")
        with open(config_path, "r") as f:
            cfg = json.load(f)
        
        patch_size = int(cfg.get("patch_size", 16))
        image_size = int(cfg.get("image_size", 224))
        embed_dim = int(cfg.get("hidden_size", 768))
        depth = int(cfg.get("num_hidden_layers", 12))
        num_heads = int(cfg.get("num_attention_heads", 12))
        
        # 创建模型
        self.model = DinoVisionTransformer(
            img_size=image_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
        )
        
        # 加载权重
        weights_path = os.path.join(dinov3_dir, "model.safetensors")
        state_dict = load_file(weights_path)
        
        # 转换HF格式到dinov3格式 - 关键：需要融合q/k/v projections
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
                    fused[target_bias_key] = qkv_bias            # Step 2: 处理其他keys
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
        
        state_dict = convert_hf_to_dinov3(state_dict)
        msg = self.model.load_state_dict(state_dict, strict=False)
        
        # 打印加载状态 - 调试用
        if len(msg.missing_keys) > 0 or len(msg.unexpected_keys) > 0:
            print(f"[SingleModel] load_state_dict non-strict: missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)}")
            if len(msg.missing_keys) > 0:
                print(f"  Missing keys (前10个): {msg.missing_keys[:10]}")
            if len(msg.unexpected_keys) > 0:
                print(f"  Unexpected keys (前10个): {msg.unexpected_keys[:10]}")
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # 创建transform
        import torchvision.transforms as T
        self.dinov3_transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def _load_da3(self, cwd):
        """加载Depth-Anything v2"""
        depth_anything_path = os.path.join(cwd, 'Depth-Anything-V2')
        if depth_anything_path not in sys.path:
            sys.path.insert(0, depth_anything_path)
        
        from depth_anything_v2.dpt import DepthAnythingV2
        
        model_configs = {
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
        }
        
        self.model = DepthAnythingV2(**model_configs['vitl'])
        ckpt_path = os.path.join(cwd, 'Depth-Anything-V2/checkpoints/depth_anything_v2_vitl.pth')
        self.model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def extract_batch(self, images: List[Image.Image]) -> np.ndarray:
        """
        提取一批图像的特征
        Args:
            images: PIL Image列表
        Returns:
            features: [B, D] numpy array
        """
        with torch.no_grad():
            if self.model_name == 'croco':
                return self._extract_croco(images)
            elif self.model_name == 'vggt':
                return self._extract_vggt(images)
            elif self.model_name == 'dinov3':
                return self._extract_dinov3(images)
            elif self.model_name == 'da3':
                return self._extract_da3(images)
    
    def _extract_croco(self, images):
        """CroCo特征提取"""
        import torchvision.transforms as T
        transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        imgs_tensor = torch.stack([transform(img) for img in images]).to(self.device)
        out = self.model._encode_image(imgs_tensor, do_mask=False, do_aggregate=False)
        feats = out[:, 0]  # CLS token [B, 1024]
        return feats.cpu().numpy()
    
    def _extract_vggt(self, images):
        """VGGT特征提取"""
        inputs = self.processor(images=images, return_tensors="pt", do_rescale=True).to(self.device)
        out = self.model(**inputs)
        feats = out.last_hidden_state[:, 0]  # [B, 2048]
        return feats.cpu().float().numpy()
    
    def _extract_dinov3(self, images):
        """DINOv3特征提取 - 使用patch tokens全局平均池化（与训练时一致）"""
        # 使用手动transform预处理
        imgs_tensor = torch.stack([self.dinov3_transform(img) for img in images]).to(self.device)
        
        with torch.no_grad():
            # DinoVisionTransformer的forward返回特征字典
            outputs = self.model.forward_features(imgs_tensor)
            # ⚠️ 关键修复：训练时用的是patch tokens的全局平均池化，不是CLS token！
            # 离线特征: [T, 8, 14, 14, 768] -> mean(axis=(1,2,3)) -> [T, 768]
            # 在线特征: 应该也用patch tokens平均，不是CLS token
            patch_tokens = outputs['x_norm_patchtokens']  # [B, 196, 768] (14x14=196)
            # 全局平均池化
            feats = patch_tokens.mean(dim=1)  # [B, 768]
        
        return feats.cpu().float().numpy()
    
    def _extract_da3(self, images):
        """Depth-Anything v2特征提取"""
        import torchvision.transforms as T
        transform = T.Compose([
            T.Resize((518, 518)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        imgs_tensor = torch.stack([transform(img) for img in images]).to(self.device)
        feats = self.model.forward_features(imgs_tensor)['x_prenorm'][:, 0]  # [B, 1024]
        
        # DA3输出1024d,需要投影到2048d
        if not hasattr(self, 'da3_proj'):
            self.da3_proj = nn.Linear(1024, 2048).to(self.device)
            self.da3_proj.eval()
        
        feats = self.da3_proj(feats)  # [B, 2048]
        return feats.cpu().numpy()
