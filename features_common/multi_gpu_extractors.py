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
        self.extractors = {}
        
        # 检查可用GPU（CUDA_VISIBLE_DEVICES已经重新编号，所以总是从0开始）
        available_gpus = torch.cuda.device_count()
        
        # 使用相对索引：即使原始是[1]，在CUDA_VISIBLE_DEVICES=1环境下也变成了[0]
        if len(gpu_ids) > available_gpus:
            print(f"[MultiGPU] 警告: 请求{len(gpu_ids)}张卡，但只有{available_gpus}张可用")
            self.gpu_ids = list(range(available_gpus))
        else:
            # 映射到可用的相对索引
            self.gpu_ids = list(range(min(len(gpu_ids), available_gpus)))
        
        self.num_gpus = len(self.gpu_ids)
        
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
        # 约定：
        # - 2+ GPU: GPU0 跑 CroCo + DINOv3，GPU1 跑 VGGT + DA3
        # - 1 GPU : 全部模型放同一张卡
        if self.num_gpus >= 2:
            model_gpu_map = {
                'croco': self.gpu_ids[0],
                'dinov3': self.gpu_ids[0],
                'vggt': self.gpu_ids[1],
                'da3': self.gpu_ids[0],
            }
        else:
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
        
        # 2. DINOv3 (复用离线导出逻辑，保证权重映射一致)
        print(f"[MultiGPU] 加载 DINOv3 到 GPU {model_gpu_map['dinov3']}...")
        dinov3_outer_path = os.path.join(cwd, 'dinov3')
        if dinov3_outer_path not in sys.path:
            sys.path.insert(0, dinov3_outer_path)

        dinov3_dir = os.path.join(cwd, 'dinov3/weight/B16')  # 绝对路径

        # 复用离线导出脚本的本地加载逻辑（包含qkv融合与rope buffer修复）
        import importlib
        load_local_hf_dinov3 = importlib.import_module(
            "extract_multi_frame_dinov3_features_local"
        ).load_local_hf_dinov3

        dinov3_model, processor_cfg = load_local_hf_dinov3(
            model_dir=dinov3_dir,
            device=f'cuda:{model_gpu_map["dinov3"]}'
        )
        dinov3_model.eval()

        image_size = int(processor_cfg.get("size", {}).get("height", 224))
        patch_size = int(processor_cfg.get("patch_size", 16))
        image_mean = processor_cfg.get("image_mean", [0.485, 0.456, 0.406])
        image_std = processor_cfg.get("image_std", [0.229, 0.224, 0.225])
        dinov3_patch_tokens = (image_size // patch_size) ** 2

        self.extractors['dinov3'] = FeatureExtractorWrapper(
            dinov3_model,
            model_gpu_map['dinov3'],
            'dinov3',
            output_dim=768,
            patch_tokens=dinov3_patch_tokens,
            dinov3_image_size=image_size,
            dinov3_image_mean=image_mean,
            dinov3_image_std=image_std,
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
            vggt_model, model_gpu_map['vggt'], 'vggt', output_dim=2048
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
        # 优先复用离线导出脚本的build_da3（保证权重加载一致）
        try:
            import importlib.util
            da3_script = os.path.join(cwd, 'Depth-Anything-3', 'extract_multi_frame_depthanything3_features.py')
            spec = importlib.util.spec_from_file_location('da3_extract', da3_script)
            da3_mod = importlib.util.module_from_spec(spec)
            assert spec and spec.loader
            spec.loader.exec_module(da3_mod)
            da3_model = da3_mod.build_da3(da3_model_dir, device=torch.device(f'cuda:{model_gpu_map["da3"]}'))
        except Exception:
            da3_model = DepthAnything3.from_pretrained(da3_model_dir)
            da3_model = da3_model.to(f'cuda:{model_gpu_map["da3"]}')
            da3_model.eval()
        self.extractors['da3'] = FeatureExtractorWrapper(
            da3_model,
            model_gpu_map['da3'],
            'da3',
            output_dim=2048,
            da3_process_res=504,
            da3_process_res_method='upper_bound_resize',
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
        return self.extract_batch([image])[0]
    
    def extract_batch(self, images: List[Image.Image]) -> np.ndarray:
        """
        批量提取多帧特征
        
        Args:
            images: List of PIL Image (RGB)
            
        Returns:
            features: [B, 4, 2048] numpy array
        """
        # 收集各模型的 batch features: {'name': [B,2048]}
        features_dict: dict[str, np.ndarray] = {}
        B = len(images)
        for name in ['croco', 'vggt', 'dinov3', 'da3']:
            feats = self.extractors[name].extract_batch(images)
            feats = np.asarray(feats)

            # 兼容错误返回：如果返回的是 [2048]，自动扩展为 [1,2048] 并检查 B==1
            if feats.ndim == 1:
                feats = feats[None, :]

            if feats.shape[0] != B:
                raise RuntimeError(
                    f"Extractor '{name}' batch size mismatch: expected {B}, got {feats.shape[0]}"
                )
            if feats.shape[1] != 2048:
                raise RuntimeError(
                    f"Extractor '{name}' feature dim mismatch: expected 2048, got {feats.shape[1]}"
                )
            features_dict[name] = feats

        # 直接 stack 出 [B, 4, 2048]
        return np.stack(
            [features_dict['croco'], features_dict['vggt'], features_dict['dinov3'], features_dict['da3']],
            axis=1,
        )

    def __call__(self, image: Image.Image) -> np.ndarray:
        return self.extract(image)


class FeatureExtractorWrapper:
    """单个特征提取器的包装类"""

    def __init__(
        self,
        model: nn.Module,
        gpu_id: int,
        model_name: str,
        output_dim: int,
        patch_tokens: int | None = None,
        dinov3_image_size: int | None = None,
        dinov3_image_mean: list[float] | None = None,
        dinov3_image_std: list[float] | None = None,
        da3_process_res: int | None = None,
        da3_process_res_method: str | None = None,
    ):
        self.model = model
        self.gpu_id = gpu_id
        self.model_name = model_name
        self.output_dim = output_dim
        self.device = f'cuda:{gpu_id}'
        self.patch_tokens = patch_tokens
        self.dinov3_image_size = dinov3_image_size
        self.dinov3_image_mean = dinov3_image_mean
        self.dinov3_image_std = dinov3_image_std
        self.da3_process_res = da3_process_res
        self.da3_process_res_method = da3_process_res_method or 'upper_bound_resize'
    
    def __call__(self, image: Image.Image) -> np.ndarray:
        return self.extract_batch([image])[0]

    def extract_batch(self, images: List[Image.Image]) -> np.ndarray:
        """
        批量提取特征并pad到2048维
        
        Args:
            images: List of PIL Image
        
        Returns:
            feats: [B, 2048] numpy array
        """
        B = len(images)
        features = []
        
        with torch.no_grad():
            # 预处理（根据模型类型）
            if self.model_name == 'croco':
                from torchvision import transforms
                from torchvision.transforms import InterpolationMode
                transform = transforms.Compose([
                    transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
                # 批量转换
                img_tensors = torch.stack([transform(img) for img in images]).to(self.device) # [B, 3, 224, 224]
                
                # CroCo 批量推理
                # _encode_image: [B, 3, H, W] -> [B, N, C]
                feat = self.model._encode_image(img_tensors, do_mask=False)[0]  # [B, N, C]
                feat = feat.mean(dim=1).cpu().numpy()  # [B, C]
            
            elif self.model_name == 'dinov3':
                from torchvision import transforms
                image_size = self.dinov3_image_size or 224
                image_mean = self.dinov3_image_mean or [0.485, 0.456, 0.406]
                image_std = self.dinov3_image_std or [0.229, 0.224, 0.225]
                transform = transforms.Compose([
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=image_mean, std=image_std),
                ])
                img_tensors = torch.stack([transform(img) for img in images]).to(self.device)
                
                # DINOv3 批量推理（对齐离线导出逻辑）
                if hasattr(self.model, "get_intermediate_layers"):
                    y = self.model.get_intermediate_layers(img_tensors, n=1)[0]
                else:
                    y = self.model(img_tensors)
                patch_tokens = self.patch_tokens or (y.shape[1] - 1)
                y = y[:, -patch_tokens:, :]
                feat = y.mean(dim=1).cpu().numpy()  # [B, 1024]
            
            elif self.model_name == 'vggt':
                from torchvision import transforms
                from vggt.utils.load_fn import load_and_preprocess_images
                import tempfile
                import os as _os
                
                temp_paths = []
                try:
                    # 批量保存临时文件
                    for img in images:
                        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                            temp_paths.append(f.name)
                            img.save(f.name)
                    
                    # 批量加载和预处理
                    # load_and_preprocess_images returns [B, 3, 518, 518]
                    imgs = load_and_preprocess_images(temp_paths)
                    imgs = imgs.unsqueeze(1).to(self.device) # [B, 1, 3, 518, 518] (VGGT expects [B, T, ...])
                    
                    # 使用aggregator提取特征
                    # output: [List of [B, T, N, C], patch_start_idx]
                    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
                        amp_dtype = torch.bfloat16
                    else:
                        amp_dtype = torch.float16
                    with torch.cuda.amp.autocast(dtype=amp_dtype):
                        aggregated_tokens_list, patch_start_idx = self.model.aggregator(imgs)
                    last_tokens = aggregated_tokens_list[-1]  # [B, 1, N, C]
                    patch_tokens = last_tokens[:, :, patch_start_idx:, :]  # [B, 1, Np, C]
                    
                    # 全局平均池化
                    feat = patch_tokens.mean(dim=[1, 2]).cpu().numpy()  # [B, C]

                finally:
                    for p in temp_paths:
                        if _os.path.exists(p):
                            _os.unlink(p)
            
            elif self.model_name == 'da3':
                import tempfile
                import os as _os
                
                # DA3 的 API 在不同版本里 batch 行为不一致。
                # 为了保证稳定性：
                # 1) 尝试真正 batch；
                # 2) 如果 batch 后得到的输出行数 != B，则退化为逐张处理（但仍在 GPU 上）。

                def _da3_forward_from_paths(paths: list[str]) -> np.ndarray:
                    imgs_cpu, _, _ = self.model._preprocess_inputs(
                        paths,
                        extrinsics=None,
                        intrinsics=None,
                        process_res=self.da3_process_res or 504,
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

                    # 先尝试 batch
                    feat = _da3_forward_from_paths(temp_paths)  # 期望 [B, C]
                    if feat.ndim == 1:
                        feat = feat[None, :]

                    if feat.shape[0] != B:
                        # 回退逐张
                        per_feats = []
                        for p in temp_paths:
                            f = _da3_forward_from_paths([p])
                            if f.ndim == 1:
                                f = f[None, :]
                            per_feats.append(f[0])
                        feat = np.stack(per_feats, axis=0)

                finally:
                    for p in temp_paths:
                        if _os.path.exists(p):
                            _os.unlink(p)
            
            else:
                raise ValueError(f"Unknown model: {self.model_name}")
        
        # 将特征调整到目标维度（如 VGGT/DA3 强制 2048），再 pad 到 2048
        if feat.shape[1] != self.output_dim:
            feat_t = torch.from_numpy(feat).to(self.device)
            feat_t = torch.nn.functional.adaptive_avg_pool1d(
                feat_t.unsqueeze(1), self.output_dim
            ).squeeze(1)
            feat = feat_t.detach().cpu().numpy()

        # 批量 Pad 到 2048 维
        batch_feats = []
        for f in feat:
            if len(f) < 2048:
                f = np.pad(f, (0, 2048 - len(f)), mode='constant')
            elif len(f) > 2048:
                f = f[:2048]
            batch_feats.append(f)
            
        return np.stack(batch_feats, axis=0)

