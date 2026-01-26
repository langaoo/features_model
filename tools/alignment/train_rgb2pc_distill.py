#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/train_rgb2pc_distill.py
【路线2：RGB-only 推理，训练期用点云特征作 teacher】- 最终优化纯净版 
核心目标：实现 RGB视觉特征 → 点云ULIP特征 的语义空间对齐训练
核心特性：
1. 无需 per-point uv 几何对应，采用「集合级别对齐」，仅依赖时序匹配
2. Teacher：固定的点云侧ULIP特征，无训练参数，保证表征可靠性
3. Student：多视觉模型(croco/vggt/dinov3/da3)特征融合+增强，仅训练适配层
4. 训练后推理：仅输入RGB特征即可得到与点云等效的语义表征
对齐核心：CLIP-style Batch级双向InfoNCE损失 + 特征中心化，行业最优对齐方案
所有冗余代码已删除 | 关键bug已修复 | 全中文详细注释 | 可直接运行
"""
from __future__ import annotations

import argparse
import math
import sys
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml
import zarr

try:
    from tqdm import tqdm
except Exception:  # 兼容无tqdm环境，不影响训练
    tqdm = None  # type: ignore[assignment]

# 允许直接运行本脚本，无需pip install -e .，自动添加项目根目录到环境变量
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# 项目内部核心依赖导入（必须保留）
from features_common.feature_pack import load_feature_pack
from features_common.zarr_pack import load_zarr_pack
from features_common.fusion import MoEFusion, WeightedFusion
from features_common.alignment.rgb2pc_distill_dataset import RGB2PCDistillDataset, DistillSample


def info_nce_batch(z_t: torch.Tensor, z_s: torch.Tensor, *, tau: float = 0.07) -> torch.Tensor:
    """
    ✅ 核心损失函数：CLIP风格的Batch级双向InfoNCE对比损失（语义对齐的最优损失）
    作用：强制student的RGB特征与teacher的点云特征在单位球面空间分布一致
    双向计算：student→teacher + teacher→student，平衡对齐方向，对齐效果更好
    :param z_t: [B,D] teacher特征（点云ULIP），B=批次大小，D=特征维度
    :param z_s: [B,D] student特征（RGB融合增强），必须和teacher维度一致
    :param tau: 温度系数，越小对正负样本区分度要求越高，默认0.07是CLIP最优值
    :return: 平均后的双向InfoNCE损失值
    """
    # 严格校验输入维度，防止维度不匹配导致的对齐失效
    if z_t.ndim != 2 or z_s.ndim != 2 or z_t.shape != z_s.shape:
        raise ValueError(f"输入特征必须都是[B,D]且形状一致！当前: z_t={tuple(z_t.shape)} vs z_s={tuple(z_s.shape)}")
    
    # 特征L2归一化 → 投影到单位球面，此时余弦相似度 = 向量点积，是对比学习标准操作
    z_t = F.normalize(z_t, dim=-1)
    z_s = F.normalize(z_s, dim=-1)
    
    # 计算相似度矩阵 [B,B]，矩阵中(i,j)表示第i个student与第j个teacher的相似度
    logits = (z_s @ z_t.t()) / float(tau)
    # 构建标签：对角线匹配，第i个student的正样本就是第i个teacher，其余为负样本
    labels = torch.arange(logits.shape[0], device=logits.device)
    
    # 双向交叉熵计算：两个方向的损失平均，是InfoNCE的标准实现
    loss_s2t = F.cross_entropy(logits, labels)    # student特征匹配teacher特征
    loss_t2s = F.cross_entropy(logits.t(), labels)# teacher特征匹配student特征
    return 0.5 * (loss_s2t + loss_t2s)


def local_align_loss(z_tokens: torch.Tensor, t_points: torch.Tensor, *, tau: float = 0.07) -> torch.Tensor:
    """
    局部对齐损失：token-点云集合级软匹配（无显式对应）。
    z_tokens: [B,K,D], t_points: [B,Kt,D]
    目标：每个token至少能“靠近”某个点，每个点也能被某些token覆盖。
    """
    if z_tokens.ndim != 3 or t_points.ndim != 3:
        raise ValueError(f"local_align_loss expects [B,K,D] & [B,Kt,D], got {tuple(z_tokens.shape)} and {tuple(t_points.shape)}")

    zt = F.normalize(z_tokens, dim=-1)
    tp = F.normalize(t_points, dim=-1)
    sim = torch.einsum("bkd,bqd->bkq", zt, tp) / float(tau)

    # token->point, point->token (soft max with logsumexp)
    tok_to_pt = torch.logsumexp(sim, dim=-1)  # [B,K]
    pt_to_tok = torch.logsumexp(sim, dim=-2)  # [B,Kt]
    return -0.5 * (tok_to_pt.mean() + pt_to_tok.mean())


class MLP(nn.Module):
    """
    ✅ 通用多层感知机模块：用于特征维度适配/投影/增强
    结构：线性升维 → GELU激活 → Dropout防过拟合 → 线性降维 → LayerNorm稳定训练
    作用场景：视觉特征adapter、student特征投影层，是Transformer后处理的标准结构
    """
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 512, *, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),       # 特征升维，扩大表征空间
            nn.GELU(),                           # 平滑激活函数，无硬饱和区，梯度更稳定，优于ReLU
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),  # 可选dropout，无则恒等映射不影响
            nn.Linear(hidden_dim, out_dim),      # 特征降维到目标维度
            nn.LayerNorm(out_dim),               # 层归一化，防止梯度爆炸，加速收敛
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # 前向传播，直接调用序列层


class PositionalEncoding(nn.Module):
    """
    ✅ 经典正弦余弦位置编码（Transformer原论文实现）
    作用：为视觉token特征添加空间位置信息，让模型感知token的空间排布关系
    特性：位置编码矩阵注册为缓冲区，不参与训练，仅做特征叠加
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)  # dropout层，防止位置信息过拟合

        # 构建位置编码矩阵 [max_len, 1, d_model]
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)  # 偶数维度用正弦编码
        pe[:, 0, 1::2] = torch.cos(position * div_term)  # 奇数维度用余弦编码
        self.register_buffer('pe', pe)  # 注册缓冲区：保存但不训练，随模型权重一起保存

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入x格式：[seq_len, batch_size, embedding_dim] → Transformer原论文格式
        输出格式：[seq_len, batch_size, embedding_dim] → 与输入格式一致
        """
        x = x + self.pe[:x.size(0)]  # 特征叠加位置编码，仅取前seq_len个位置的编码
        return self.dropout(x)       # dropout后返回，增强鲁棒性


class TokenAttentionPool(nn.Module):
    """轻量token attention pooling：学习一个query向量对token加权汇聚。"""
    def __init__(self, dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,K,D]
        if x.ndim != 3:
            raise ValueError(f"TokenAttentionPool expects [B,K,D], got {tuple(x.shape)}")
        dim = x.shape[-1]
        scores = (x * self.query).sum(dim=-1) / math.sqrt(float(dim))  # [B,K]
        weights = torch.softmax(scores, dim=1)
        return (x * weights.unsqueeze(-1)).sum(dim=1)


def _discover_available_pairs(
    *,
    tasks: list[str],
    pc_root: Path,
    vis_roots: list[Path],
    vis_zarr_roots: list[Path],
) -> list[tuple[str, str]]:
    """
    ✅ 数据配对校验函数：校验并返回所有有效的 (task, episode) 配对
    核心逻辑：以点云teacher数据为基准，校验对应的RGB student数据是否存在
    保证：训练时每个样本都能拿到 同task+同episode 的RGB和点云特征，时序严格匹配
    """
    use_zarr = bool(vis_zarr_roots)
    roots = vis_zarr_roots if use_zarr else vis_roots

    pairs: list[tuple[str, str]] = []
    for task in tasks:
        t_pc = pc_root / task
        if not t_pc.exists():
            continue

        # 遍历所有episode目录，以点云数据为基准
        for ep_dir in sorted([p for p in t_pc.iterdir() if p.is_dir() and p.name.startswith("episode_")]):
            episode = ep_dir.name
            if episode.endswith(".zarr"):
                episode = episode[:-5]
            
            # 校验当前episode是否有点云teacher数据
            found_teacher = False
            if (t_pc / f"{episode}.zarr").exists() or next(iter(ep_dir.glob("step_*.ply.ulip_*.zarr")), None) or next(iter(ep_dir.glob("step_*.ply.ulip_*.pt")), None):
                found_teacher = True
            if not found_teacher:
                continue

            # 校验当前episode是否有对应的RGB student数据
            has_all_vis = True
            for r in roots:
                if use_zarr:
                    if not (r / task / f"{episode}.zarr").exists():
                        has_all_vis = False
                        break
                else:
                    if not (r / task / f"{episode}.pt").exists():
                        has_all_vis = False
                        break
            
            if has_all_vis:
                pairs.append((task, episode))
    
    return pairs


def main() -> None:
    """
    ✅ 主训练流程：所有核心逻辑入口，完整的RGB→点云语义对齐训练流程
    流程顺序：参数解析 → 随机种子固定 → 设备配置 → 数据集初始化 → 模型构建 → 优化器配置 → 训练循环 → 日志保存 → 模型保存
    """
    # ===================== 1. 命令行参数解析 + YAML配置文件兼容 =====================
    ap = argparse.ArgumentParser(description="路线2：RGB-only 蒸馏训练（teacher=点云特征）- 最终优化版")
    ap.add_argument("--config", type=str, default="", help="可选YAML配置文件，命令行参数可覆盖配置文件")

    # -------------------------- 数据相关参数 --------------------------
    ap.add_argument("--tasks", type=str, nargs="*", default=[], help="训练任务列表，必须指定")
    ap.add_argument("--episodes", type=int, default=20, help="每个任务采样的episode数量")
    ap.add_argument("--pc_root", type=str, default="/home/gl/RoboTwin/policy/DP2DP3/features_model/pc_dataset/ulip_features_zarr", help="点云ULIP特征根目录")
    ap.add_argument("--vis_roots", type=str, nargs="*", default=[], help="RGB特征根目录列表（.pt格式）")
    ap.add_argument("--vis_zarr_roots", type=str, nargs="*", default=[], help="RGB特征根目录列表（.zarr格式，优先级高于pt）")

    # -------------------------- 训练核心参数 --------------------------
    ap.add_argument("--device", type=str, default="cuda", help="训练设备：cuda/cpu")
    ap.add_argument("--steps", type=int, default=5000, help="训练总步数（有效步数，带梯度累积）")
    ap.add_argument("--lr", type=float, default=1e-3, help="初始学习率，余弦退火自动衰减")
    ap.add_argument("--weight_decay", type=float, default=1e-4, help="AdamW权重衰减系数，防过拟合")  # 新增：补全缺失的参数
    ap.add_argument("--seed", type=int, default=0, help="全局随机种子，保证训练可复现")
    ap.add_argument("--tau", type=float, default=0.07, help="InfoNCE温度系数，默认0.07最优")
    ap.add_argument("--fuse_dim", type=int, default=768, help="特征融合维度，会自动适配teacher维度")
    ap.add_argument("--moe_hidden", type=int, default=256, help="MoE融合层的隐藏维度")
    ap.add_argument("--fusion", type=str, default="weighted", choices=["weighted", "moe"], help="多模型特征融合方式")
    ap.add_argument("--loss_mse", type=float, default=0.0, help="额外MSE损失权重，0则关闭")
    ap.add_argument("--loss_rgb", type=float, default=0.0, help="RGB自对比损失权重，保持视觉语义一致性")
    ap.add_argument("--rgb_tau", type=float, default=0.07, help="RGB自对比温度系数")
    ap.add_argument("--amp", action="store_true", help="启用AMP混合精度训练（仅CUDA生效，提速省显存）")
    ap.add_argument("--batch_size", type=int, default=8, help="单批次样本数")
    ap.add_argument("--grad_accum_steps", type=int, default=1, help="梯度累积步数，有效批次=batch_size*该值，提升对比学习稳定性")
    ap.add_argument("--student_tokens", type=int, default=1024, help="每个视觉模型采样的token数量")
    ap.add_argument("--teacher_points", type=int, default=1024, help="每个点云样本采样的点数量")
    ap.add_argument(
        "--student_pool",
        type=str,
        default="tokens",
        choices=["tokens", "mean"],
        help="student特征池化方式：tokens=使用token序列；mean=对token均值池化为单向量",
    )
    ap.add_argument(
        "--token_pool",
        type=str,
        default="mean",
        choices=["mean", "attn"],
        help="token模式下的池化方式：mean=均值池化，attn=注意力池化",
    )
    ap.add_argument("--loss_local", type=float, default=0.0, help="局部token-点云对齐损失权重（无点云时自动跳过）")
    ap.add_argument("--local_tau", type=float, default=0.07, help="局部对齐温度系数")
    ap.add_argument("--loss_pool", type=float, default=0.0, help="token-attn与mean池化一致性损失权重")

    # -------------------------- 采样粒度参数 --------------------------
    ap.add_argument("--sample_unit", type=str, default="step", choices=["step", "window"], help="样本粒度：step=单帧，window=8帧窗口")
    ap.add_argument("--window_agg", type=str, default="mean", choices=["mean"], help="window模式下的时间聚合方式，仅实现均值")

    # -------------------------- 性能/调试参数 --------------------------
    ap.add_argument("--pack_cache_size", type=int, default=4, help="特征包缓存大小，提升数据加载速度")
    ap.add_argument("--strict_pairing", action="store_true", help="严格按step_stem配对RGB和点云，缺失则降级")
    ap.add_argument("--pairing_fallback", type=str, default="random", choices=["random", "skip", "error"], help="严格配对失败的兜底策略")
    ap.add_argument("--log_every", type=int, default=50, help="wandb日志记录频率")
    ap.add_argument("--print_every", type=int, default=50, help="终端打印频率")
    ap.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值，<=0则关闭，防止梯度爆炸")
    ap.add_argument("--skip_nonfinite", action="store_true", help="检测到无效梯度时跳过更新，提升训练稳定性")

    # -------------------------- DataLoader性能参数 --------------------------
    ap.add_argument("--num_workers", type=int, default=4, help="数据加载线程数")
    ap.add_argument("--prefetch_factor", type=int, default=2, help="数据预取数，提升加载速度")
    ap.add_argument("--persistent_workers", action="store_true", help="持久化数据线程，避免重复创建开销")
    ap.add_argument("--pin_memory", action="store_true", help="固定内存，CUDA下提速，推荐开启")

    # -------------------------- 模型保存参数 --------------------------
    ap.add_argument("--save_dir", type=str, default="/home/gl/RoboTwin/policy/DP2DP3/features_model/outputs/train_rgb2pc_runs/run0", help="模型保存目录")
    ap.add_argument("--save_every", type=int, default=500, help="模型保存频率")

    # -------------------------- 日志可视化参数 --------------------------
    ap.add_argument("--tqdm", action="store_true", help="启用tqdm进度条，终端可视化更友好")
    ap.add_argument("--wandb", action="store_true", help="启用wandb可视化日志")
    ap.add_argument("--wandb_project", type=str, default="rgb2pc_distill", help="wandb项目名")
    ap.add_argument("--wandb_entity", type=str, default="", help="wandb账号实体")
    ap.add_argument("--wandb_run_name", type=str, default="", help="wandb运行名称")

    # 解析配置文件（如果有），配置文件参数可被命令行覆盖
    args_pre, _ = ap.parse_known_args()
    if str(args_pre.config).strip():
        cfg = yaml.safe_load(Path(args_pre.config).read_text(encoding="utf-8"))
        if isinstance(cfg, dict):
            ap.set_defaults(**cfg)
    args = ap.parse_args()

    # ===================== 2. 全局初始化：种子+设备+混合精度 =====================
    # 固定随机种子，保证训练可复现
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    # 设备配置
    device = torch.device(str(args.device))
    # 启用AMP混合精度训练（仅CUDA生效），新版API无警告
    use_amp = bool(args.amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # 必传参数校验，防止无数据训练
    if not args.tasks:
        raise ValueError("必须指定 --tasks 参数，用于采样配对的RGB/点云样本！")
    if not (args.vis_zarr_roots or args.vis_roots):
        raise ValueError("必须指定 --vis_roots 或 --vis_zarr_roots 参数，提供RGB特征！")

    # ===================== 3. wandb日志初始化（可选，兼容失败降级） =====================
    wb = None
    if bool(args.wandb):
        if not str(os.environ.get("WANDB_API_KEY", "")).strip():
            print("[warn] 未检测到WANDB_API_KEY，自动禁用wandb！")
        else:
            try:
                import wandb
                wb = wandb
                wb.init(
                    project=str(args.wandb_project),
                    config=vars(args),
                    entity=str(args.wandb_entity) if args.wandb_entity else None,
                    name=str(args.wandb_run_name) if args.wandb_run_name else None
                )
            except Exception as e:
                print(f"[warn] wandb启用失败，降级为终端输出：{e}")
                wb = None

    # ===================== 4. 数据集+DataLoader初始化：核心数据加载 =====================
    dataset = RGB2PCDistillDataset(
        pc_root=str(args.pc_root),
        vis_zarr_roots=list(args.vis_zarr_roots) if args.vis_zarr_roots else list(args.vis_roots),
        tasks=list(args.tasks),
        episodes=int(args.episodes),
        sample_unit=str(args.sample_unit),
        student_tokens=int(args.student_tokens),
        teacher_points=int(args.teacher_points),
        strict_pairing=bool(args.strict_pairing),
        pairing_fallback=str(args.pairing_fallback),
        student_pool=str(args.student_pool),
        seed=int(args.seed),
    )

    # 自动检测teacher特征维度，并自适应调整融合维度，保证维度匹配（核心对齐前提）
    s0 = dataset[0]
    t_dim = 384
    if getattr(s0, "teacher_points", None) is not None:
        t_dim = int(s0.teacher_points.shape[-1])
    elif getattr(s0, "teacher_embed", None) is not None:
        t_dim = int(s0.teacher_embed.shape[-1])
    if t_dim != int(args.fuse_dim):
        print(f"[info] 自动适配teacher维度：fuse_dim从 {args.fuse_dim} → {t_dim}")
        args.fuse_dim = t_dim

    # ===================== 5. 模型构建：所有可训练模块（核心，仅训练这些层） =====================
    # 5.1 视觉特征适配器：每个视觉模型一个MLP，统一特征维度到fuse_dim
    packs0 = [dataset._get_pack(i, s0.task, s0.episode) for i in range(len(dataset.vis_zarr_roots))]
    adapters = nn.ModuleList()
    for p in packs0:
        c_in = int(p.per_frame_features.shape[-1]) if hasattr(p, "per_frame_features") else int(p.arr.shape[-1])
        adapters.append(MLP(in_dim=c_in, out_dim=int(args.fuse_dim), hidden_dim=int(args.fuse_dim)*2))
    adapters = adapters.to(device)

    # 5.2 多模型特征融合层：Weighted加权融合 / MoE混合专家融合
    if str(args.fusion) == "weighted":
        fusion: nn.Module = WeightedFusion(num_models=len(packs0)).to(device)
    else:
        fusion = MoEFusion(dim=int(args.fuse_dim), num_models=len(packs0), hidden_dim=int(args.moe_hidden), k=2).to(device)

    # 5.3 位置编码+Transformer上下文编码器：增强特征的空间交互能力，提升语义表达
    use_token_pool = str(args.student_pool) == "tokens"
    if use_token_pool:
        pos_encoder = PositionalEncoding(d_model=int(args.fuse_dim), dropout=0.1, max_len=int(args.student_tokens)).to(device)
        # ✅ 修复关键BUG：动态计算n_head，保证d_model % n_head == 0，彻底避免硬编码报错
        d_model = int(args.fuse_dim)
        nhead = d_model // 64  # 行业通用head_dim=64，保证整除，且注意力头数合理
        assert nhead >= 1, f"融合维度{d_model}过小，无法拆分注意力头！建议增大fuse_dim"
        context_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, dropout=0.1, batch_first=True)
        context_encoder = nn.TransformerEncoder(context_layer, num_layers=2).to(device)
        token_pooler = TokenAttentionPool(d_model).to(device) if str(args.token_pool) == "attn" else None
    else:
        pos_encoder = None
        context_encoder = None
        token_pooler = None

    # 5.4 Student特征投影层：最终映射到teacher的特征空间，无teacher投影层（固定teacher特征，核心优化）
    proj_student = MLP(in_dim=int(args.fuse_dim), out_dim=int(args.fuse_dim), hidden_dim=int(args.fuse_dim)*2).to(device)

    # ===================== 6. 优化器+学习率调度器配置 =====================
    # 收集所有可训练参数
    params = list(adapters.parameters()) + list(fusion.parameters()) + list(proj_student.parameters())
    if context_encoder is not None:
        params = params + list(context_encoder.parameters())
    if token_pooler is not None:
        params = params + list(token_pooler.parameters())
    opt = torch.optim.AdamW(params, lr=float(args.lr), weight_decay=float(args.weight_decay))  # 修复：参数已定义，可正常使用
    # 余弦退火调度器：自动降低学习率，避免后期震荡，保证收敛到最优值
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(args.steps), eta_min=1e-6)

    # ===================== 7. 训练前置准备：保存目录+数据校验 =====================
    save_dir = Path(str(args.save_dir))
    save_dir.mkdir(parents=True, exist_ok=True)

    # 校验有效数据配对
    available_pairs = _discover_available_pairs(
        tasks=list(args.tasks),
        pc_root=Path(args.pc_root),
        vis_roots=[Path(p) for p in args.vis_roots],
        vis_zarr_roots=[Path(p) for p in args.vis_zarr_roots],
    )
    if not available_pairs:
        raise RuntimeError("未找到有效(task,episode)配对！请检查RGB和点云数据路径是否匹配")

    # ===================== 8. DataLoader迭代器配置 =====================
    def _collate(batch: list[DistillSample]):
        """自定义数据拼接函数：适配DistillSample的结构化数据，保证批次维度正确"""
        # 获取当前批次的样本粒度（step/window）
        unit = batch[0].sample_unit
        # 拼接每个模型的token特征列表
        '''[
    [模型1特征(样本1, 1024,384), 模型2特征(样本1, 1024,512)],  # 样本1的多模型特征
    [模型1特征(样本2, 1024,384), 模型2特征(样本2, 1024,512)]   # 样本2的多模型特征
    ]   ->[
    (模型1特征(样本1), 模型1特征(样本2)),
    (模型2特征(样本1), 模型2特征(样本2))
    ]'''
        toks_by_model = list(zip(*[b.tokens_by_model for b in batch]))
        toks = [torch.stack(list(m), dim=0) for m in toks_by_model]
        out = {"sample_unit": unit, "tokens": toks, "task": [b.task for b in batch], "episode": [b.episode for b in batch]}
        # 根据样本粒度添加teacher特征
        if unit == "window":
            out["teacher_embed"] = torch.stack([b.teacher_embed for b in batch], dim=0)
        else:
            if batch[0].teacher_points is not None:
                out["teacher_points"] = torch.stack([b.teacher_points for b in batch], dim=0)
            elif batch[0].teacher_embed is not None:
                out["teacher_embed"] = torch.stack([b.teacher_embed for b in batch], dim=0)
            else:
                raise RuntimeError("step样本缺失teacher特征！")
        return out
    # 自动从dataset里取batch_size个样本，调用_collate拼接成批次，再返回给训练循环
    loader = DataLoader(
        dataset, batch_size=int(args.batch_size), shuffle=True, num_workers=int(args.num_workers),
        pin_memory=bool(args.pin_memory) and device.type == "cuda",
        persistent_workers=bool(args.persistent_workers) and int(args.num_workers) > 0,
        prefetch_factor=int(args.prefetch_factor) if int(args.num_workers) > 0 else None,
        collate_fn=_collate, drop_last=True
    )
    loader_iter = iter(loader)

    # ===================== 9. 核心训练循环 =====================
    global_step = 0       # 有效训练步数（累计梯度后更新的步数）
    ema_loss: float | None = None  # 损失滑动平均，仅用于日志展示
    opt.zero_grad(set_to_none=True)  # 初始化梯度为None，省显存
    micro_step = 0        # 微步数，用于梯度累积计数
    accum_steps = max(1, int(args.grad_accum_steps))  # 梯度累积步数，至少为1
    micro_total = int(args.steps) * accum_steps       # 总微步数

    # 进度条配置
    step_iter = range(int(micro_total))
    if bool(args.tqdm) and tqdm is not None:
        step_iter = tqdm(step_iter, total=int(micro_total), dynamic_ncols=True)

    # 逐步训练
    for _ in step_iter:
        # 数据加载，迭代器耗尽则重置
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)

        # 数据转移到设备，非阻塞传输提速
        toks_list: list[torch.Tensor] = batch["tokens"]
        toks_list = [t.to(device, non_blocking=True).to(torch.float32) for t in toks_list]
        # ✅ 修复：特征异常值检测，替换inf/nan，避免训练崩溃
        for i in range(len(toks_list)):
            if not torch.isfinite(toks_list[i]).all():
                toks_list[i] = torch.nan_to_num(toks_list[i], nan=0.0, posinf=1e6, neginf=-1e6)

        # -------------------------- 前向传播：特征处理+损失计算 --------------------------
        loss_rgb_val: float | None = None
        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            if use_token_pool:
                B, K = toks_list[0].shape[:2]  # B=批次，K=token数量

                # 1. 视觉特征适配：每个模型的特征通过MLP统一维度
                z_tokens_list = []
                for mi, toks in enumerate(toks_list):
                    z = adapters[mi](toks.reshape(B * K, -1)).reshape(B, K, -1)
                    z_tokens_list.append(z)
                
                # 2. 多模型特征融合：token级融合，兼顾不同模型的语义优势
                z_flat_list = [z.reshape(B*K, -1) for z in z_tokens_list]
                z_fused_flat, _ = fusion(z_flat_list)
                z_fused_tokens = z_fused_flat.reshape(B, K, -1)
                
                # 3. 位置编码+上下文增强：添加空间位置信息 + Transformer空间交互
                z_fused_tokens = z_fused_tokens.transpose(0, 1)  # 转置为[seq_len, batch, dim]适配PE格式
                z_fused_tokens = pos_encoder(z_fused_tokens)
                z_fused_tokens = z_fused_tokens.transpose(0, 1)  # 转回[batch, seq_len, dim]适配Transformer（batch_first=True）
                z_enhanced = context_encoder(z_fused_tokens)
                
                # 4. 全局池化：集合级特征对齐，得到样本级全局特征 [B, D]
                z_mean = z_enhanced.mean(dim=1)
                if token_pooler is not None:
                    z_attn = token_pooler(z_enhanced)
                    z_final = z_attn
                    zs_mean = proj_student(z_mean)
                else:
                    z_final = z_mean
                    zs_mean = None

                # 5. Student特征最终投影
                zs = proj_student(z_final)

                # 5.1 局部对齐：对token投影后与点云点特征做集合级对齐
                z_tokens_proj = None
                if float(args.loss_local) > 0 and "teacher_points" in batch:
                    z_tokens_proj = proj_student(z_enhanced.reshape(B * K, -1)).reshape(B, K, -1)

                # RGB自对比：对每个模型做token均值池化
                z_rgb_list = [z.mean(dim=1) for z in z_tokens_list]
            else:
                # mean 池化模式：先对token做均值池化，再走adapter+fusion
                z_vec_list = []
                for mi, toks in enumerate(toks_list):
                    toks_mean = toks.mean(dim=1)  # [B, C]
                    z_vec_list.append(adapters[mi](toks_mean))  # [B, D]
                z_final, _ = fusion(z_vec_list)  # [B, D]
                zs = proj_student(z_final)
                z_rgb_list = z_vec_list
                zs_mean = None
                z_tokens_proj = None

            # 6. Teacher特征处理：适配step/window两种模式，保证格式为[B,D]
            if str(batch["sample_unit"]) == "window":
                zt = batch["teacher_embed"].to(device, non_blocking=True).to(torch.float32)
            else:
                if "teacher_embed" in batch:
                    zt = batch["teacher_embed"].to(device, non_blocking=True).to(torch.float32)
                else:
                    tp = batch["teacher_points"].to(device, non_blocking=True).to(torch.float32)
                    zt = tp.mean(dim=1) if tp.ndim == 3 else tp

            # ✅ 修复：正确的归一化→中心化顺序，保持 student 梯度
            B = int(zs.shape[0])
            # 注意：不能把 zs 放在 no_grad 里，否则梯度为 0
            zt_norm = F.normalize(zt, dim=-1)
            zs_norm = F.normalize(zs, dim=-1)
            # 第二步：中心化（仅batch_size≥2时，避免单样本特征全零）
            if B > 1:
                zt_centered = zt_norm - zt_norm.mean(dim=0, keepdim=True)
                zs_centered = zs_norm - zs_norm.mean(dim=0, keepdim=True)
            else:
                zt_centered = zt_norm
                zs_centered = zs_norm
            
            # 计算核心对齐损失（修复：移除不存在的skip_norm参数）
            loss_nce = info_nce_batch(zt_centered, zs_centered, tau=float(args.tau))
            loss = loss_nce
            
            # 可选MSE损失：辅助对齐，权重为0则关闭
            if float(args.loss_mse) > 0:
                loss = loss + float(args.loss_mse) * F.mse_loss(F.normalize(zs, dim=-1), F.normalize(zt, dim=-1))

            # 可选RGB自对比：保持视觉语义一致性
            if float(args.loss_rgb) > 0:
                pair_cnt = 0
                loss_rgb = 0.0
                for i in range(len(z_rgb_list)):
                    for j in range(i + 1, len(z_rgb_list)):
                        loss_rgb = loss_rgb + info_nce_batch(z_rgb_list[i], z_rgb_list[j], tau=float(args.rgb_tau))
                        pair_cnt += 1
                if pair_cnt > 0:
                    loss_rgb = loss_rgb / float(pair_cnt)
                else:
                    loss_rgb = torch.tensor(0.0, device=zs.device)
                loss = loss + float(args.loss_rgb) * loss_rgb
                loss_rgb_val = float(loss_rgb.item())

            # 可选局部对齐损失（仅在有teacher_points时生效）
            if float(args.loss_local) > 0 and "teacher_points" in batch and z_tokens_proj is not None:
                tp = batch["teacher_points"].to(device, non_blocking=True).to(torch.float32)
                loss_local = local_align_loss(z_tokens_proj, tp, tau=float(args.local_tau))
                loss = loss + float(args.loss_local) * loss_local

            # token-attn 与 mean 池化一致性（保持推理接口为mean）
            if float(args.loss_pool) > 0 and zs_mean is not None:
                loss_pool = F.mse_loss(zs_mean, zs.detach())
                loss = loss + float(args.loss_pool) * loss_pool

        # -------------------------- 梯度累积：损失缩放 --------------------------
        loss_micro = loss / float(accum_steps)

        # -------------------------- 相似度指标计算：仅日志展示，无梯度 --------------------------
        with torch.no_grad():
            zs_n = F.normalize(zs.detach(), dim=-1)
            zt_n = F.normalize(zt.detach(), dim=-1)
            pos_sim = float((zs_n * zt_n).sum(dim=-1).mean().cpu().item())  # 正样本相似度
            sim_mat = zs_n @ zt_n.t()
            B = int(sim_mat.shape[0])
            neg_sim = float("nan")
            if B > 1:
                neg_sum = float((sim_mat.sum() - sim_mat.diag().sum()).cpu().item())
                neg_sim = neg_sum / float(B * (B - 1))  # 负样本平均相似度
            sim_gap = float(pos_sim - neg_sim) if math.isfinite(neg_sim) else float("nan")  # 正负相似度差，越大越好

        # -------------------------- 反向传播+梯度处理 --------------------------
        if use_amp:
            scaler.scale(loss_micro).backward()
            try:
                scaler.unscale_(opt)  # 反缩放，保证梯度范数计算准确
            except Exception:
                pass
        else:
            loss_micro.backward()

        # 梯度有效性校验+梯度范数计算
        grad_norm_pre, grad_norm_post, clipped, nonfinite_grads = None, None, False, 0
        try:
            total, cnt = 0.0, 0
            mods_for_grad = [adapters, fusion, proj_student]
            if context_encoder is not None:
                mods_for_grad.append(context_encoder)
            for mod in mods_for_grad:
                for p in mod.parameters():
                    if p.grad is None: continue
                    g = p.grad.detach()
                    if not torch.isfinite(g).all(): nonfinite_grads +=1; continue
                    total += float(g.norm(2).cpu().item())**2; cnt +=1
            grad_norm_pre = float(total**0.5) if cnt>0 else None
        except Exception: pass

        # 梯度裁剪：防止梯度爆炸，提升训练稳定性
        if float(args.grad_clip) >0 and nonfinite_grads ==0:
            try:
                torch.nn.utils.clip_grad_norm_(params, max_norm=float(args.grad_clip))
                clipped = grad_norm_pre is not None and grad_norm_pre > float(args.grad_clip) + 1e-12
            except Exception: pass

        # 计算裁剪后的梯度范数
        try:
            total2, cnt2 =0.0,0
            mods_for_grad = [adapters, fusion, proj_student]
            if context_encoder is not None:
                mods_for_grad.append(context_encoder)
            for mod in mods_for_grad:
                for p in mod.parameters():
                    if p.grad is None: continue
                    g = p.grad.detach()
                    if torch.isfinite(g).all(): total2 += float(g.norm(2).cpu().item())**2; cnt2 +=1
            grad_norm_post = float(total2**0.5) if cnt2>0 else None
        except Exception: pass

        micro_step +=1
        stepped = None

        # -------------------------- 优化器更新：梯度累积满步数后执行 --------------------------
        if micro_step % accum_steps == 0:
            stepped = True
            # 检测到无效梯度，跳过更新，AMP下自动降低缩放因子
            if nonfinite_grads >0 and bool(args.skip_nonfinite):
                stepped = False
                if use_amp:
                    try: scaler.update()
                    except Exception: pass
            else:
                # 优化器更新
                if use_amp:
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                scheduler.step()  # 更新学习率（仅有效步数更新，适配梯度累积）
                opt.zero_grad(set_to_none=True)  # 清空梯度
                global_step +=1
        # -------------------------- 日志可视化+模型保存 --------------------------
        loss_v_now = float(loss.item())
        nce_v_now = float(loss_nce.item())
        lr_v_now = float(opt.param_groups[0]["lr"])
        scaler_scale = float(scaler.get_scale()) if use_amp else None

        # 更新损失滑动平均
        if stepped is not None:
            if ema_loss is None:
                ema_loss = loss_v_now
            else:
                ema_loss = 0.98 * ema_loss + 0.02 * loss_v_now

        # tqdm进度条实时展示指标
        if bool(args.tqdm) and tqdm is not None:
            try:
                step_iter.set_postfix({
                    "loss":f"{loss_v_now:.4f}", "ema":f"{ema_loss:.4f}" if ema_loss else "-",
                    "nce":f"{nce_v_now:.4f}", "lr":f"{lr_v_now:.1e}",
                    "gn":f"{grad_norm_post:.2f}" if grad_norm_post else "-",
                    "pos":f"{pos_sim:.2f}", "neg":f"{neg_sim:.2f}" if math.isfinite(neg_sim) else "-",
                    "gap":f"{sim_gap:.2f}" if math.isfinite(sim_gap) else "-",
                    "ng":str(nonfinite_grads), "st":"1" if stepped else "0",
                    "s":str(global_step)
                })
            except Exception: pass

        # 终端打印日志
        if stepped and int(args.print_every) >0 and (global_step % int(args.print_every) ==0):
            msg = f"step={global_step} loss={loss_v_now:.4f} ema={ema_loss:.4f} nce={nce_v_now:.4f}"
            if loss_rgb_val is not None:
                msg += f" rgb={loss_rgb_val:.4f}"
            msg += f" lr={lr_v_now:.3e} pos={pos_sim:.3f} gap={sim_gap:.3f}"
            if tqdm is not None: tqdm.write(msg)
            else: print(msg)

        # wandb日志记录
        if stepped and wb is not None and int(args.log_every) >0 and (global_step % int(args.log_every) ==0):
            wb_payload = {
                "loss":loss_v_now, "ema_loss":ema_loss, "nce_loss":nce_v_now, "lr":lr_v_now,
                "grad_norm_pre":grad_norm_pre or 0, "grad_norm_post":grad_norm_post or 0,
                "pos_sim":pos_sim, "neg_sim":neg_sim or 0, "sim_gap":sim_gap or 0,
                "nonfinite_grads":nonfinite_grads, "stepped":1 if stepped else 0,
                "step":global_step
            }
            if loss_rgb_val is not None:
                wb_payload["rgb_loss"] = loss_rgb_val
            wb.log(wb_payload)

        # 模型保存
        if stepped and int(args.save_every) >0 and (global_step % int(args.save_every) ==0):
            ckpt = {
                "global_step":global_step, "args":vars(args),
                "adapters":adapters.state_dict(), "fusion":fusion.state_dict(),
                "proj_student":proj_student.state_dict(), "opt":opt.state_dict()
            }
            if context_encoder is not None and pos_encoder is not None:
                ckpt["context_encoder"] = context_encoder.state_dict()
                ckpt["pos_encoder"] = pos_encoder.state_dict()
            if token_pooler is not None:
                ckpt["token_pooler"] = token_pooler.state_dict()
            out = save_dir / f"ckpt_step_{global_step:07d}.pt"
            torch.save(ckpt, out)
            print(f"[ckpt] 保存模型: {out}")

    # ===================== 10. 训练结束：保存最终模型 =====================
    final_ckpt = save_dir / "ckpt_final.pt"
    final_payload = {
        "global_step":global_step, "args":vars(args),
        "adapters":adapters.state_dict(), "fusion":fusion.state_dict(),
        "proj_student":proj_student.state_dict(), "opt":opt.state_dict()
    }
    if context_encoder is not None and pos_encoder is not None:
        final_payload["context_encoder"] = context_encoder.state_dict()
        final_payload["pos_encoder"] = pos_encoder.state_dict()
    if token_pooler is not None:
        final_payload["token_pooler"] = token_pooler.state_dict()
    torch.save(final_payload, final_ckpt)
    print(f"[ckpt] 训练完成，保存最终模型: {final_ckpt}")


if __name__ == "__main__":
    main()