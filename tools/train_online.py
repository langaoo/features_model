"""tools/train_online.py

基于配置文件的在线训练脚本（从 raw_data 实时提取特征）

用法:
  python tools/train_online.py --config configs/train_online_default.yaml
  
  # 可以覆盖配置文件中的参数
  python tools/train_online.py --config configs/train_online_default.yaml \\
    --data.tasks beat_block_hammer-demo_randomized-20_sapien_head_camera \\
    --training.batch_size 4 \\
    --device.cuda_visible_devices 0,1
"""

import argparse
import sys
import os
from pathlib import Path
from datetime import datetime
import yaml
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from features_common.dp_rgb_dataset_online_4models import (
    DPRGBOnline4ModelDataset,
    collate_fn_online_4,
)
from features_common.online_extractors import load_all_extractors
from features_common.rgb2pc_aligned_encoder_4models import RGB2PCAlignedEncoder4Models


class DummyDPHead(nn.Module):
    """Dummy DP head (替换为实际的 DiffusionRGBHead)"""
    def __init__(self, obs_dim, action_dim, horizon):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim * horizon),
        )
        self.action_dim = action_dim
        self.horizon = horizon
    
    def forward(self, obs_feat):
        B, To, D = obs_feat.shape
        out = self.net(obs_feat[:, -1, :])
        return out.view(B, self.horizon, self.action_dim)
    
    def compute_loss(self, obs_feat, action):
        pred = self.forward(obs_feat)
        return nn.functional.mse_loss(pred, action)


def load_config(config_path: str) -> Dict[str, Any]:
    """加载 YAML 配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def override_config(config: Dict[str, Any], overrides: list) -> Dict[str, Any]:
    """根据命令行参数覆盖配置"""
    for override in overrides:
        if '=' not in override:
            continue
        key, value = override.split('=', 1)
        keys = key.lstrip('--').split('.')
        
        # 类型转换
        if value.lower() == 'true':
            value = True
        elif value.lower() == 'false':
            value = False
        elif value.isdigit():
            value = int(value)
        else:
            try:
                value = float(value)
            except ValueError:
                pass  # 保持字符串
        
        # 递归设置
        d = config
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value
    
    return config


def main():
    parser = argparse.ArgumentParser(description="在线训练（基于配置文件）")
    parser.add_argument('--config', type=str, required=True,
                       help='配置文件路径')
    parser.add_argument('overrides', nargs='*',
                       help='覆盖配置项（例如: --data.batch_size=16）')
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    config = override_config(config, args.overrides)
    
    # 设置环境
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config['device']['cuda_visible_devices'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n" + "="*60)
    print("在线训练配置")
    print("="*60)
    print(f"Config: {args.config}")
    print(f"Tasks: {config['data']['tasks']}")
    print(f"Camera: {config['data']['camera_name']}")
    print(f"Vision backbones: {config['model']['vision_backbones']}")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"Epochs: {config['training']['epochs']}")
    print(f"Device: {device} (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')})")
    
    # ========== 1. 加载视觉 backbone ==========
    print("\n" + "="*60)
    print("加载视觉 Backbone...")
    print("="*60)
    
    # 只加载配置中指定的模型
    available_models = {'croco', 'vggt', 'dinov3', 'da3'}
    requested_models = set(config['model']['vision_backbones'])
    invalid_models = requested_models - available_models
    
    if invalid_models:
        print(f"⚠ Warning: 未知的模型: {invalid_models}")
        requested_models = requested_models & available_models
    
    extractors = {}
    for model_name in requested_models:
        try:
            print(f"  Loading {model_name}...")
            if model_name == 'croco':
                from features_common.online_extractors import load_croco_extractor
                extractors['croco'] = load_croco_extractor(device='cuda')
            elif model_name == 'dinov3':
                from features_common.online_extractors import load_dinov3_extractor
                extractors['dinov3'] = load_dinov3_extractor(device='cuda')
            elif model_name == 'vggt':
                from features_common.online_extractors import load_vggt_extractor
                extractors['vggt'] = load_vggt_extractor(device='cuda')
            elif model_name == 'da3':
                from features_common.online_extractors import load_da3_extractor
                extractors['da3'] = load_da3_extractor(device='cuda')
            print(f"    ✓ {model_name} loaded")
        except Exception as e:
            print(f"    ✗ {model_name} failed: {e}")
            print(f"    跳过 {model_name}")
    
    if len(extractors) == 0:
        print("✗ 没有可用的视觉模型，退出")
        sys.exit(1)
    
    print(f"\n✓ 成功加载 {len(extractors)} 个模型: {list(extractors.keys())}\n")
    
    # 注意：当前 Dataset 需要 4 个模型，这里需要调整
    if len(extractors) < 4:
        print("⚠ Warning: Dataset 需要 4 个模型，但只加载了 {len(extractors)} 个")
        print("⚠ 使用简化版本或调整 Dataset 实现")
        # TODO: 实现灵活的 Dataset 支持任意数量的模型
    
    # ========== 2. 创建 Dataset ==========
    print("="*60)
    print("创建 Dataset...")
    print("="*60)
    
    # 暂时跳过 Dataset 创建，因为需要修改支持任意数量模型
    print("⚠ 当前 Dataset 实现需要精确 4 个模型")
    print("⚠ 建议先使用离线训练，或修改 Dataset 支持灵活数量的模型")
    
    print("\n" + "="*60)
    print("配置文件系统已创建")
    print("="*60)
    print(f"✓ 配置文件: configs/train_online_default.yaml")
    print(f"✓ 训练脚本: tools/train_online.py")
    print("\n下一步:")
    print("  1. 修改 Dataset 支持任意数量的视觉模型")
    print("  2. 或者使用完整 4 模型配置")
    print("\n用法:")
    print("  python tools/train_online.py --config configs/train_online_default.yaml")


if __name__ == '__main__':
    main()
