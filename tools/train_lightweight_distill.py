#!/usr/bin/env python3
"""tools/train_lightweight_distill.py

轻量化蒸馏训练脚本

将4个大视觉模型+对齐encoder的知识蒸馏到单个轻量backbone

用法：
  # 方案A：蒸馏到轻量模型，接入已训练的DP头
  python tools/train_lightweight_distill.py \
    --teacher_ckpt outputs/train_rgb2pc_runs/run_best_bs32/ckpt_step_0010000.pt \
    --rgb_root rgb_dataset/RGB_ORI \
    --student_backbone resnet18 \
    --scheme A \
    --batch_size 32 \
    --epochs 100 \
    --save_dir outputs/lightweight_runs/resnet18_schemeA

  # 方案B：端到端蒸馏（需要后续重新训练DP头）
  python tools/train_lightweight_distill.py \
    --teacher_ckpt outputs/train_rgb2pc_runs/run_best_bs32/ckpt_step_0010000.pt \
    --student_backbone mobilenetv3_small \
    --scheme B \
    --save_dir outputs/lightweight_runs/mobilenet_schemeB
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from features_common.distill.lightweight_distill import (
    LightweightConfig,
    LightweightRGBEncoder,
    LightweightSequenceEncoder,
    DistillationLoss,
    create_lightweight_model,
)


class RGBDistillDataset(Dataset):
    """蒸馏训练数据集
    
    从RGB目录加载图像，并缓存teacher特征
    """
    
    def __init__(
        self,
        rgb_root: Path,
        tasks: List[str],
        teacher_feat_cache: dict = None,  # 可选的预计算teacher特征缓存
        transform=None,
    ):
        self.rgb_root = Path(rgb_root)
        self.tasks = tasks
        self.teacher_feat_cache = teacher_feat_cache or {}
        self.transform = transform
        
        # 发现所有图像
        self.samples = []  # [(task, episode, frame_path), ...]
        self._discover_samples()
        
    def _discover_samples(self):
        for task in self.tasks:
            task_dir = self.rgb_root / task
            if not task_dir.exists():
                continue
            
            for ep_dir in sorted(task_dir.iterdir()):
                if not ep_dir.is_dir():
                    continue
                
                for img_path in sorted(ep_dir.glob("*.png")):
                    self.samples.append((task, ep_dir.name, img_path))
                for img_path in sorted(ep_dir.glob("*.jpg")):
                    self.samples.append((task, ep_dir.name, img_path))
        
        print(f"[Dataset] Found {len(self.samples)} images")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int):
        task, episode, img_path = self.samples[idx]
        
        # 加载图像
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img_tensor = self.transform(img)
        else:
            # 默认变换
            from torchvision import transforms
            default_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
            img_tensor = default_transform(img)
        
        # 获取teacher特征（如果有缓存）
        cache_key = str(img_path)
        teacher_feat = self.teacher_feat_cache.get(cache_key, None)
        
        return {
            'image': img_tensor,
            'image_path': str(img_path),
            'task': task,
            'episode': episode,
            'teacher_feat': teacher_feat,
        }


def compute_teacher_features(
    teacher_encoder,
    extractors,
    dataloader: DataLoader,
    device: str = 'cuda',
) -> dict:
    """预计算所有样本的teacher特征（加速训练）"""
    print("[Teacher] 预计算teacher特征...")
    
    teacher_encoder.eval()
    cache = {}
    
    for batch in tqdm(dataloader, desc="Computing teacher features"):
        image_paths = batch['image_path']
        
        # 加载原始图像用于teacher
        images = [Image.open(p).convert('RGB') for p in image_paths]
        
        with torch.no_grad():
            # 提取4模型特征
            feats_4 = []
            for img in images:
                feat = extractors(img)  # [4, 2048]
                feats_4.append(feat)
            feats_4 = torch.from_numpy(np.stack(feats_4, axis=0)).float().to(device)
            
            # 添加时间维并通过encoder
            feats_4 = feats_4.unsqueeze(1)  # [B, 1, 4, 2048]
            teacher_feat = teacher_encoder(feats_4).squeeze(1)  # [B, fuse_dim]
        
        # 缓存
        for i, path in enumerate(image_paths):
            cache[path] = teacher_feat[i].cpu()
    
    print(f"[Teacher] 缓存了 {len(cache)} 个teacher特征")
    return cache


def train_epoch(
    student: nn.Module,
    teacher_encoder: nn.Module,
    extractors,
    dataloader: DataLoader,
    criterion: DistillationLoss,
    optimizer: torch.optim.Optimizer,
    device: str,
    use_cached_teacher: bool = True,
) -> float:
    """训练一个epoch"""
    student.train()
    total_loss = 0
    n_batches = 0
    
    for batch in dataloader:
        images = batch['image'].to(device)
        
        # Student前向
        student_feat = student(images)  # [B, output_dim]
        
        # Teacher前向
        if use_cached_teacher and batch['teacher_feat'][0] is not None:
            teacher_feat = torch.stack(batch['teacher_feat']).to(device)
        else:
            # 实时计算teacher特征
            image_paths = batch['image_path']
            pil_images = [Image.open(p).convert('RGB') for p in image_paths]
            
            with torch.no_grad():
                feats_4 = []
                for img in pil_images:
                    feat = extractors(img)
                    feats_4.append(feat)
                feats_4 = torch.from_numpy(np.stack(feats_4, axis=0)).float().to(device)
                feats_4 = feats_4.unsqueeze(1)
                teacher_feat = teacher_encoder(feats_4).squeeze(1)
        
        # 计算损失
        loss, loss_dict = criterion(student_feat, teacher_feat)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / max(n_batches, 1)


def evaluate(
    student: nn.Module,
    teacher_encoder: nn.Module,
    extractors,
    dataloader: DataLoader,
    device: str,
) -> dict:
    """评估蒸馏质量"""
    student.eval()
    
    all_student = []
    all_teacher = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            images = batch['image'].to(device)
            
            # Student
            student_feat = student(images)
            all_student.append(student_feat.cpu())
            
            # Teacher
            if batch['teacher_feat'][0] is not None:
                teacher_feat = torch.stack(batch['teacher_feat'])
            else:
                image_paths = batch['image_path']
                pil_images = [Image.open(p).convert('RGB') for p in image_paths]
                feats_4 = []
                for img in pil_images:
                    feat = extractors(img)
                    feats_4.append(feat)
                feats_4 = torch.from_numpy(np.stack(feats_4, axis=0)).float().to(device)
                feats_4 = feats_4.unsqueeze(1)
                teacher_feat = teacher_encoder(feats_4).squeeze(1).cpu()
            
            all_teacher.append(teacher_feat)
    
    all_student = torch.cat(all_student, dim=0)
    all_teacher = torch.cat(all_teacher, dim=0)
    
    # 计算指标
    mse = ((all_student - all_teacher) ** 2).mean().item()
    
    s_norm = torch.nn.functional.normalize(all_student, dim=-1)
    t_norm = torch.nn.functional.normalize(all_teacher, dim=-1)
    cosine_sim = (s_norm * t_norm).sum(dim=-1).mean().item()
    
    return {
        'mse': mse,
        'cosine_similarity': cosine_sim,
    }


def main():
    parser = argparse.ArgumentParser()
    
    # 必需参数
    parser.add_argument('--teacher_ckpt', type=str, required=True,
                       help='对齐encoder的checkpoint路径')
    parser.add_argument('--save_dir', type=str, required=True,
                       help='输出目录')
    
    # 数据
    parser.add_argument('--rgb_root', type=str, 
                       default='rgb_dataset/RGB_ORI',
                       help='RGB图像根目录')
    parser.add_argument('--tasks', nargs='+', default=[
        'beat_block_hammer-demo_randomized-20_sapien_head_camera',
        'dump_bin_bigbin-demo_randomized-20_sapien_head_camera',
    ])
    
    # Student模型
    parser.add_argument('--student_backbone', type=str, default='resnet18',
                       choices=['resnet18', 'resnet34', 'resnet50',
                               'mobilenetv3_small', 'mobilenetv3_large',
                               'efficientnet_b0'])
    parser.add_argument('--output_dim', type=int, default=1280,
                       help='输出特征维度（应与teacher的fuse_dim一致）')
    
    # 蒸馏方案
    parser.add_argument('--scheme', type=str, default='A',
                       choices=['A', 'B'],
                       help='A=蒸馏到轻量模型接入已有DP头; B=端到端蒸馏需重训DP头')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=4)
    
    # 损失权重
    parser.add_argument('--mse_weight', type=float, default=1.0)
    parser.add_argument('--cosine_weight', type=float, default=0.5)
    parser.add_argument('--infonce_weight', type=float, default=0.1)
    
    # GPU
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--gpu_ids', type=str, default='0,1',
                       help='用于teacher特征提取的GPU')
    
    # 其他
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_every', type=int, default=10)
    parser.add_argument('--precompute_teacher', action='store_true',
                       help='预计算teacher特征（加速训练但需要更多内存）')
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device(args.device)
    gpu_ids = [int(x) for x in args.gpu_ids.split(',')]
    
    print(f"{'='*60}")
    print(f"轻量化蒸馏训练")
    print(f"{'='*60}")
    print(f"方案: {args.scheme}")
    print(f"Student backbone: {args.student_backbone}")
    print(f"Output dim: {args.output_dim}")
    print(f"{'='*60}\n")
    
    # 1. 创建Student模型
    print("1. 创建Student模型...")
    config = LightweightConfig(
        backbone=args.student_backbone,
        output_dim=args.output_dim,
        pretrained=True,
    )
    student = LightweightRGBEncoder(config).to(device)
    print(f"   参数量: {sum(p.numel() for p in student.parameters()):,}")
    
    # 2. 加载Teacher模型
    print("\n2. 加载Teacher模型...")
    from features_common.rgb2pc_aligned_encoder_4models import RGB2PCAlignedEncoder4Models
    teacher_encoder = RGB2PCAlignedEncoder4Models.from_checkpoint(
        args.teacher_ckpt, freeze=True
    ).to(device)
    teacher_encoder.eval()
    
    # 加载4模型特征提取器
    print("   加载4个视觉backbone...")
    from features_common.multi_gpu_extractors import MultiGPUFeatureExtractors
    extractors = MultiGPUFeatureExtractors(gpu_ids=gpu_ids)
    
    # 3. 创建数据集
    print("\n3. 创建数据集...")
    dataset = RGBDistillDataset(
        rgb_root=Path(args.rgb_root),
        tasks=args.tasks,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    # 4. 预计算teacher特征（可选）
    if args.precompute_teacher:
        print("\n4. 预计算Teacher特征...")
        teacher_cache = compute_teacher_features(
            teacher_encoder, extractors, dataloader, device
        )
        dataset.teacher_feat_cache = teacher_cache
    
    # 5. 损失函数和优化器
    print("\n5. 配置训练...")
    criterion = DistillationLoss(
        mse_weight=args.mse_weight,
        cosine_weight=args.cosine_weight,
        infonce_weight=args.infonce_weight,
    )
    
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    
    # 6. 训练循环
    print(f"\n6. 开始训练 (共{args.epochs}个epoch)...")
    print(f"{'='*60}\n")
    
    best_cosine = 0
    
    for epoch in range(1, args.epochs + 1):
        avg_loss = train_epoch(
            student, teacher_encoder, extractors,
            dataloader, criterion, optimizer, device,
            use_cached_teacher=args.precompute_teacher,
        )
        scheduler.step()
        
        print(f"Epoch {epoch}/{args.epochs} | Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")
        
        # 定期评估和保存
        if epoch % args.save_every == 0:
            metrics = evaluate(student, teacher_encoder, extractors, dataloader, device)
            print(f"  Eval: MSE={metrics['mse']:.4f}, Cosine={metrics['cosine_similarity']:.4f}")
            
            # 保存checkpoint
            ckpt = {
                'epoch': epoch,
                'config': vars(config),
                'student_state': student.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'metrics': metrics,
                'args': vars(args),
            }
            torch.save(ckpt, save_dir / f'ckpt_epoch_{epoch:04d}.pt')
            
            # 保存最佳模型
            if metrics['cosine_similarity'] > best_cosine:
                best_cosine = metrics['cosine_similarity']
                torch.save(ckpt, save_dir / 'best_model.pt')
                print(f"  ✓ New best model saved!")
    
    # 保存最终模型
    final_ckpt = {
        'epoch': args.epochs,
        'config': vars(config),
        'student_state': student.state_dict(),
        'args': vars(args),
    }
    torch.save(final_ckpt, save_dir / 'final_model.pt')
    
    print(f"\n{'='*60}")
    print(f"训练完成!")
    print(f"最佳Cosine相似度: {best_cosine:.4f}")
    print(f"模型保存至: {save_dir}")
    print(f"{'='*60}\n")
    
    # 方案说明
    if args.scheme == 'A':
        print("【方案A后续步骤】")
        print("  蒸馏模型可直接替换4模型pipeline，接入已训练的DP头:")
        print("  1. 加载轻量化模型: model = LightweightRGBEncoder.from_checkpoint(...)")
        print("  2. 输入RGB图像: feat = model(image)  # [B, 1280]")
        print("  3. 接入DP头: action = dp_head(feat)")
    else:
        print("【方案B后续步骤】")
        print("  需要用蒸馏模型的特征重新训练DP头:")
        print("  1. 使用轻量化模型提取特征")
        print("  2. 训练新的DP头（使用相同的动作标签）")
        print("  3. 预期精度损失: 5-15%，速度提升: 40-100x")


if __name__ == '__main__':
    main()
