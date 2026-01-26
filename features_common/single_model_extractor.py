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
        # 优先使用本地权重（与离线导出一致）
        vggt_outer = os.path.join(cwd, 'vggt')
        if vggt_outer not in sys.path:
            sys.path.insert(0, vggt_outer)

        weight_path = os.path.join(cwd, 'vggt/weight/model.pt')
        if os.path.exists(weight_path):
            from vggt.models.vggt import VGGT
            self.model = VGGT()
            state_dict = torch.load(weight_path, map_location='cpu')
            self.model.load_state_dict(state_dict)
            self.model = self.model.to(self.device)
            self.model.eval()
            self.processor = None
        else:
            from transformers import AutoImageProcessor, AutoModel
            vggt_ckpt = "Idiot-Scientist/VGGT"
            self.processor = AutoImageProcessor.from_pretrained(vggt_ckpt)
            self.model = AutoModel.from_pretrained(vggt_ckpt, torch_dtype=torch.bfloat16).to(self.device)
            self.model.eval()
    
    def _load_dinov3(self, cwd):
        """加载本地DINOv3 ViT-B/16 (768 dim) - 使用原生DinoVisionTransformer"""
        dinov3_dir = os.path.join(cwd, 'dinov3/weight/B16')
        # 添加dinov3外层目录到path
        dinov3_outer_path = os.path.join(cwd, 'dinov3')
        if dinov3_outer_path not in sys.path:
            sys.path.insert(0, dinov3_outer_path)
        
        # 优先复用离线导出脚本的加载逻辑
        try:
            from extract_multi_frame_dinov3_features_local import load_local_hf_dinov3
            model, processor_cfg = load_local_hf_dinov3(model_dir=dinov3_dir, device=str(self.device))
            self.model = model
            self.model.eval()
            self.dinov3_mode = "offline"
            image_size = int(processor_cfg.get("size", {}).get("height", 224))
            patch_size = int(processor_cfg.get("patch_size", 16))
            image_mean = processor_cfg.get("image_mean", [0.485, 0.456, 0.406])
            image_std = processor_cfg.get("image_std", [0.229, 0.224, 0.225])
            self.dinov3_image_size = image_size
            self.dinov3_patch_size = patch_size
            import torchvision.transforms as T
            self.dinov3_transform = T.Compose([
                T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=image_mean, std=image_std),
            ])
            return
        except Exception as e:
            print(f"[SingleModel] Offline DINOv3 loader failed, fallback to custom: {e}")
            self.dinov3_mode = "custom"

        print(f"[SingleModel] Loading DINOv3 from local path (custom): {dinov3_dir}")
        
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
        self.dinov3_patch_size = patch_size
        self.dinov3_image_size = image_size
        
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
        self.dinov3_mode = "custom"
        
        # 创建transform
        import torchvision.transforms as T
        self.dinov3_transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def _load_da3(self, cwd):
        """加载Depth-Anything-3（与离线特征一致）"""
        da3_path = os.path.join(cwd, 'Depth-Anything-3/src')
        if da3_path not in sys.path:
            sys.path.insert(0, da3_path)

        import os as _os
        _os.environ.setdefault("DA3_LOG_LEVEL", "ERROR")

        from depth_anything_3.api import DepthAnything3

        da3_model_dir = os.path.join(cwd, 'Depth-Anything-3/weight')
        # 优先复用离线导出脚本的build_da3（保证权重加载一致）
        try:
            import importlib.util
            da3_script = os.path.join(cwd, 'Depth-Anything-3', 'extract_multi_frame_depthanything3_features.py')
            spec = importlib.util.spec_from_file_location('da3_extract', da3_script)
            da3_mod = importlib.util.module_from_spec(spec)
            assert spec and spec.loader
            spec.loader.exec_module(da3_mod)
            self.model = da3_mod.build_da3(da3_model_dir, device=self.device)
        except Exception:
            self.model = DepthAnything3.from_pretrained(da3_model_dir)
            self.model = self.model.to(self.device)
            self.model.eval()
        # 与离线导出脚本保持一致
        self.da3_process_res = 504
        self.da3_process_res_method = 'upper_bound_resize'
    
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
        from torchvision.transforms import InterpolationMode
        transform = T.Compose([
            T.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        imgs_tensor = torch.stack([transform(img) for img in images]).to(self.device)
        out = self.model._encode_image(imgs_tensor, do_mask=False)[0]
        feats = out.mean(dim=1)  # mean pool [B, 1024]
        return feats.cpu().numpy()
    
    def _extract_vggt(self, images):
        """VGGT特征提取"""
        if self.processor is None:
            from vggt.utils.load_fn import load_and_preprocess_images
            import tempfile
            import os as _os

            temp_paths = []
            try:
                for img in images:
                    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                        temp_paths.append(f.name)
                        img.save(f.name)

                imgs = load_and_preprocess_images(temp_paths)
                imgs = imgs.unsqueeze(1).to(self.device)  # [B, 1, 3, 518, 518]
                amp_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8) else torch.float16
                with torch.no_grad(), torch.cuda.amp.autocast(dtype=amp_dtype):
                    aggregated_tokens_list, patch_start_idx = self.model.aggregator(imgs)
                last_tokens = aggregated_tokens_list[-1]  # [B, 1, N, C]
                patch_tokens = last_tokens[:, :, patch_start_idx:, :]
                feat = patch_tokens.mean(dim=[1, 2])
                return feat.detach().cpu().float().numpy()
            finally:
                for p in temp_paths:
                    if _os.path.exists(p):
                        _os.unlink(p)

        inputs = self.processor(images=images, return_tensors="pt", do_rescale=True).to(self.device)
        out = self.model(**inputs)
        feats = out.last_hidden_state[:, 0]  # [B, 2048]
        return feats.cpu().float().numpy()
    
    def _extract_dinov3(self, images):
        """DINOv3特征提取 - 使用patch tokens全局平均池化（与训练时一致）"""
        if getattr(self, "dinov3_mode", "custom") == "hf" and self.processor is not None:
            inputs = self.processor(images=images, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
            # 使用patch tokens均值（去掉CLS）
            patch_tokens = outputs.last_hidden_state[:, 1:, :]
            feats = patch_tokens.mean(dim=1)
            return feats.cpu().float().numpy()

        # custom fallback: 对齐离线导出逻辑
        imgs_tensor = torch.stack([self.dinov3_transform(img) for img in images]).to(self.device)
        hfwf = (int(self.dinov3_image_size) // int(self.dinov3_patch_size)) ** 2
        amp_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8) else torch.float16
        with torch.no_grad(), torch.amp.autocast(device_type="cuda" if self.device.type == "cuda" else "cpu", dtype=amp_dtype):
            if hasattr(self.model, "get_intermediate_layers"):
                y = self.model.get_intermediate_layers(imgs_tensor, n=1)[0]
            else:
                y = self.model(imgs_tensor)
            # 去掉CLS/寄存器，只保留最后Hf*Wf patch tokens
            patch_tokens = y[:, -hfwf:, :]
            feats = patch_tokens.mean(dim=1)
        return feats.cpu().float().numpy()
    
    def _extract_da3(self, images):
        """Depth-Anything-3特征提取（与离线脚本一致）"""
        import tempfile
        import os as _os

        def _da3_forward_from_paths(paths: list[str]) -> np.ndarray:
            imgs_cpu, _, _ = self.model._preprocess_inputs(
                paths,
                extrinsics=None,
                intrinsics=None,
                process_res=self.da3_process_res,
                process_res_method=self.da3_process_res_method,
            )
            imgs, _, _ = self.model._prepare_model_inputs(imgs_cpu, None, None)
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
                raise RuntimeError(f"DA3 _prepare_model_inputs returned unexpected type: {type(imgs)}")

            imgs = imgs.to(self.device)
            backbone = self.model.model.backbone
            feats, _ = backbone(x=imgs, export_feat_layers=[23])
            if not feats:
                raise RuntimeError("backbone返回空特征")
            tokens, _ = feats[0]
            return tokens.mean(dim=[1, 2]).detach().cpu().numpy()

        temp_paths = []
        try:
            for img in images:
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                    temp_paths.append(f.name)
                    img.save(f.name)

            feat = _da3_forward_from_paths(temp_paths)
            if feat.ndim == 1:
                feat = feat[None, :]
            return feat
        finally:
            for p in temp_paths:
                if _os.path.exists(p):
                    _os.unlink(p)
