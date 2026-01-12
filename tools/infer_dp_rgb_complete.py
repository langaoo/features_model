#!/usr/bin/env python3
"""tools/infer_dp_rgb_complete.py

完整的RGB->DP动作头推理脚本
支持两种模式：
1. 离线模式：从预提取的zarr特征读取（快速验证）
2. 在线模式：实时从RGB图像提取4模型特征（部署场景）

用法：
  # 离线模式（推荐先用此模式验证）
  python tools/infer_dp_rgb_complete.py \
    --head_ckpt outputs/dp_rgb_runs/task/final_head.pt \
    --mode offline \
    --zarr_roots rgb_dataset/features_croco_encoder_dict_unified_zarr \
                 rgb_dataset/features_vggt_encoder_dict_unified_zarr \
                 rgb_dataset/features_dinov3_encoder_dict_unified_zarr \
                 rgb_dataset/features_da3_encoder_dict_unified_zarr \
    --task beat_block_hammer-demo_randomized-20_sapien_head_camera \
    --episode episode_0 \
    --start 0

  # 在线模式（需要4个视觉backbone，显存需求约16GB）
  python tools/infer_dp_rgb_complete.py \
    --head_ckpt outputs/dp_rgb_runs/task/final_head.pt \
    --mode online \
    --rgb_image /path/to/image.png

输出：
  action: [horizon, action_dim] 动作序列
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from features_common.rgb2pc_aligned_encoder_4models import RGB2PCAlignedEncoder4Models
from features_common.zarr_pack import load_zarr_pack


class OnlineFeatureExtractor:
    """在线4模型特征提取器
    
    将4个视觉backbone分布到多个GPU以节省单卡显存
    """
    
    def __init__(self, gpu_ids: list[int] = [0, 1], device: str = 'cuda'):
        self.device = torch.device(device)
        self.gpu_ids = gpu_ids
        self.models_loaded = False
        
        # 延迟加载以节省内存
        self.extractors = {}
        
    def load_models(self):
        """加载4个视觉backbone"""
        if self.models_loaded:
            return
            
        print("[OnlineExtractor] 加载4个视觉backbone...")
        
        # 使用multi_gpu_extractors中的逻辑
        from features_common.multi_gpu_extractors import MultiGPUFeatureExtractors
        self._multi_gpu = MultiGPUFeatureExtractors(gpu_ids=self.gpu_ids)
        
        self.models_loaded = True
        print("[OnlineExtractor] ✓ 加载完成")
        
    def extract_from_rgb(self, rgb_image) -> np.ndarray:
        """从RGB图像提取4模型特征
        
        Args:
            rgb_image: PIL Image 或 numpy array [H,W,3]
        
        Returns:
            features: [4, 2048] numpy array (padded to max dim)
        """
        if not self.models_loaded:
            self.load_models()
        
        from PIL import Image
        if isinstance(rgb_image, np.ndarray):
            rgb_image = Image.fromarray(rgb_image)
        
        return self._multi_gpu.extract(rgb_image)


def load_head_and_encoder(
    head_ckpt_path: str,
    device: str = 'cuda'
) -> Tuple[nn.Module, RGB2PCAlignedEncoder4Models, Dict[str, Any], Dict[str, Any]]:
    """加载DP Head和Encoder
    
    Returns:
        head: DiffusionRGBHead
        encoder: RGB2PCAlignedEncoder4Models
        normalizer: dict
        config: dict
    """
    ckpt = torch.load(head_ckpt_path, map_location='cpu', weights_only=False)
    
    # 1. 加载encoder
    encoder_ckpt_path = ckpt.get('encoder_ckpt')
    if encoder_ckpt_path is None:
        raise ValueError(f"Checkpoint缺少encoder_ckpt字段")
    
    encoder = RGB2PCAlignedEncoder4Models.from_checkpoint(
        encoder_ckpt_path, freeze=True
    ).to(device)
    encoder.eval()
    
    # 2. 加载head
    from features_common.dp_rgb_policy_multitask import DiffusionRGBHead, HeadSpec
    
    encoder_spec = ckpt.get('encoder_spec', {})
    config = ckpt.get('config', {})
    
    head_spec = HeadSpec(
        action_dim=ckpt['action_dim'],
        horizon=config.get('horizon', 8),
        n_obs_steps=config.get('n_obs_steps', 2),
        n_action_steps=config.get('n_action_steps', 4),
        obs_feature_dim=encoder_spec.get('fuse_dim', 1280),
        obs_as_global_cond=True,
    )
    
    head = DiffusionRGBHead(spec=head_spec)
    head.load_state_dict(ckpt['head_state'])
    head.to(device)
    head.eval()
    
    # 3. normalizer
    normalizer = ckpt.get('normalizer', {})
    
    return head, encoder, normalizer, config


def load_obs_from_zarr(
    zarr_roots: list[Path],
    task: str,
    episode: str,
    start_frame: int,
    n_obs_steps: int,
    expect_dims: tuple = (1024, 2048, 768, 2048),
) -> torch.Tensor:
    """从zarr加载观测特征
    
    Returns:
        obs: [1, To, 4, C_max] tensor
    """
    max_dim = max(expect_dims)
    packs = [load_zarr_pack(root / task / f"{episode}.zarr") for root in zarr_roots]
    
    W, T = packs[0].shape[0], packs[0].shape[1]
    total_steps = W * T
    
    # 边界处理
    if start_frame + n_obs_steps > total_steps:
        start_frame = max(0, total_steps - n_obs_steps)
    
    frames = []
    for s in range(start_frame, start_frame + n_obs_steps):
        wi = s // T
        ti = s % T
        
        per_model = []
        for mi, pack in enumerate(packs):
            f = pack.get_frame(wi, ti)  # [Hf,Wf,C]
            # Mean pool spatial dims
            f = f.reshape(-1, f.shape[-1]).mean(axis=0)  # [C]
            
            # Pad to expected dim
            ed = expect_dims[mi]
            if f.shape[0] >= ed:
                f_ed = f[:ed]
            else:
                f_ed = np.zeros((ed,), dtype=f.dtype)
                f_ed[:f.shape[0]] = f
            
            # Pad to max_dim
            if ed < max_dim:
                f2 = np.zeros((max_dim,), dtype=f.dtype)
                f2[:ed] = f_ed
            else:
                f2 = f_ed
            
            per_model.append(f2)
        
        frames.append(np.stack(per_model, axis=0))  # [4, C_max]
    
    obs = torch.from_numpy(np.stack(frames, axis=0)).unsqueeze(0).float()  # [1, To, 4, C_max]
    return obs


def predict_action(
    head: nn.Module,
    encoder: RGB2PCAlignedEncoder4Models,
    normalizer: dict,
    obs_4models: torch.Tensor,
    device: str = 'cuda',
) -> np.ndarray:
    """推理动作
    
    Args:
        obs_4models: [1, To, 4, C_max] tensor
    
    Returns:
        action: [horizon, action_dim] numpy array
    """
    obs_4models = obs_4models.to(device)
    
    with torch.no_grad():
        # Encoder: [1, To, 4, C_max] -> [1, To, fuse_dim]
        obs_encoded = encoder(obs_4models)
        
        # Head: predict action
        action_pred = head.predict_action(
            obs_features=obs_encoded,
            normalizer_obs=normalizer.get('obs'),
            normalizer_action=normalizer.get('action'),
        )
        
        action = action_pred['action_pred'].cpu().numpy()[0]  # [horizon, action_dim]
    
    return action


def pad_action_to_14d(action: np.ndarray) -> np.ndarray:
    """将12维动作补齐到14维（添加gripper）
    
    RoBoTwin标准: [left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)] = 14
    """
    if action.shape[-1] == 14:
        return action
    
    if action.shape[-1] == 12:
        # 12 = left_arm(6) + right_arm(6)
        left_arm = action[:, :6]
        right_arm = action[:, 6:12]
        left_grip = np.full((action.shape[0], 1), 0.5, dtype=np.float32)
        right_grip = np.full((action.shape[0], 1), 0.5, dtype=np.float32)
        return np.concatenate([left_arm, left_grip, right_arm, right_grip], axis=-1)
    
    raise ValueError(f"Unsupported action dim: {action.shape[-1]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--head_ckpt', type=str, required=True,
                       help='DP Head checkpoint路径')
    parser.add_argument('--mode', type=str, default='offline',
                       choices=['offline', 'online'],
                       help='推理模式: offline=读zarr特征, online=实时提取')
    
    # 离线模式参数
    parser.add_argument('--zarr_roots', nargs=4, type=str, default=None,
                       help='4个zarr特征根目录 (croco vggt dinov3 da3)')
    parser.add_argument('--task', type=str, default=None)
    parser.add_argument('--episode', type=str, default=None)
    parser.add_argument('--start', type=int, default=0)
    
    # 在线模式参数
    parser.add_argument('--rgb_image', type=str, default=None,
                       help='RGB图像路径(在线模式)')
    parser.add_argument('--gpu_ids', type=str, default='0,1',
                       help='GPU IDs for online extraction')
    
    # 通用参数
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--exec_steps', type=int, default=1,
                       help='只执行前K步(receding horizon)')
    parser.add_argument('--pad_to_14d', action='store_true',
                       help='将动作补齐到14维(RoBoTwin标准)')
    
    args = parser.parse_args()
    
    # 加载模型
    print(f"加载模型: {args.head_ckpt}")
    head, encoder, normalizer, config = load_head_and_encoder(
        args.head_ckpt, args.device
    )
    
    n_obs_steps = config.get('n_obs_steps', 2)
    horizon = config.get('horizon', 8)
    
    print(f"模型配置: n_obs_steps={n_obs_steps}, horizon={horizon}")
    print(f"Encoder fuse_dim={encoder.spec.fuse_dim}")
    
    # 获取观测
    if args.mode == 'offline':
        if args.zarr_roots is None or args.task is None or args.episode is None:
            raise ValueError("离线模式需要指定 --zarr_roots, --task, --episode")
        
        zarr_roots = [Path(r) for r in args.zarr_roots]
        obs = load_obs_from_zarr(
            zarr_roots, args.task, args.episode, args.start, n_obs_steps
        )
        print(f"从zarr加载观测: {obs.shape}")
        
    else:  # online
        if args.rgb_image is None:
            raise ValueError("在线模式需要指定 --rgb_image")
        
        from PIL import Image
        
        gpu_ids = [int(x) for x in args.gpu_ids.split(',')]
        extractor = OnlineFeatureExtractor(gpu_ids=gpu_ids)
        
        # 加载图像
        img = Image.open(args.rgb_image).convert('RGB')
        
        # 提取特征 (简化: 使用同一帧重复n_obs_steps次)
        feat_single = extractor.extract_from_rgb(img)  # [4, 2048]
        obs = torch.from_numpy(feat_single).unsqueeze(0).unsqueeze(0).float()  # [1, 1, 4, 2048]
        obs = obs.repeat(1, n_obs_steps, 1, 1)  # [1, To, 4, 2048]
        print(f"在线提取特征: {obs.shape}")
    
    # 推理
    action = predict_action(head, encoder, normalizer, obs, args.device)
    print(f"\n预测动作: {action.shape}")
    
    if args.pad_to_14d:
        action = pad_action_to_14d(action)
        print(f"补齐到14维: {action.shape}")
    
    # 输出执行动作
    exec_action = action[:args.exec_steps]
    print(f"\n执行动作 (前{args.exec_steps}步):")
    for i, a in enumerate(exec_action):
        print(f"  Step {i}: {a}")
    
    return action


if __name__ == '__main__':
    main()
