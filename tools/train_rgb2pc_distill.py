"""tools/train_rgb2pc_distill.py

路线2：RGB-only 推理，训练期用点云特征作 teacher。

你当前仓库的 `tools/train_alignment_skeleton.py` 属于几何对齐：需要 per-point uv 才能做 point↔patch。
本脚本做的是：

- Teacher：点云侧 ULIP 特征（从 ULIP .pt 读取 pc_feat，[N,256]）。
- Student：多视觉模型的特征（croco/vggt/dinov3/da3）在同一 episode/window 内抽样 token，
  经过 adapter -> 融合(Weighted/MoE) -> projector 得到 student embedding。

训练目标：让 student embedding 在语义空间上“像 teacher 一样好用”，推理时只需要 RGB 特征。

重要说明
- 本脚本不使用 per-point uv，因此不做严格几何对应；监督来自“同一时刻/同一 step 的点云 teacher”。
- 为了避免把 teacher 的 N 个点特征与 M 个 patch token 一一对应，我们使用集合级别的对齐：
  - 从点云 pc_feat 随机采样 K 个点
  - 从每个视觉模型的 [Hf,Wf] token 随机采样 K 个 token
  - 通过 CLIP-style InfoNCE 在 batch 内做对比（每个样本一个全局 embedding）

这是一份可跑通骨架，重点是把数据、shape、loss 跑通，并提供可扩展的融合/蒸馏接口。
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import os
from collections import OrderedDict
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
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore[assignment]

# 允许直接用 `python tools/train_rgb2pc_distill.py ...` 运行（不要求 `pip install -e .`）
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from features_common.feature_pack import load_feature_pack
from features_common.zarr_pack import load_zarr_pack
from features_common.fusion import MoEFusion, WeightedFusion
from features_common.rgb2pc_distill_dataset import RGB2PCDistillDataset, DistillSample


def info_nce_batch(z_t: torch.Tensor, z_s: torch.Tensor, *, tau: float = 0.07) -> torch.Tensor:
    """Batch-level InfoNCE (CLIP-style).

    z_t: [B,D] teacher
    z_s: [B,D] student
    """

    if z_t.ndim != 2 or z_s.ndim != 2 or z_t.shape != z_s.shape:
        raise ValueError(f"Expect z_t,z_s both [B,D] same shape, got {tuple(z_t.shape)} vs {tuple(z_s.shape)}")
    z_t = F.normalize(z_t, dim=-1)
    z_s = F.normalize(z_s, dim=-1)
    logits = (z_s @ z_t.t()) / float(tau)
    labels = torch.arange(logits.shape[0], device=logits.device)
    loss_s2t = F.cross_entropy(logits, labels)
    loss_t2s = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_s2t + loss_t2s)


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 512, *, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


def _sorted_steps_in_dir_by_stem(ep_dir: Path, *, stem_contains: str) -> list[Path]:
    items = sorted(ep_dir.glob(f"step_*{stem_contains}*"))
    if not items:
        raise FileNotFoundError(f"目录下没有 step_*{stem_contains}*: {ep_dir}")

    def _key(p: Path) -> int:
        name = p.name
        try:
            after = name.split("step_")[-1]
            digits = after.split(".")[0]
            return int(digits)
        except Exception:
            return 0

    return sorted(items, key=_key)


def _discover_available_pairs(
    *,
    tasks: list[str],
    pc_root: Path,
    vis_roots: list[Path],
    vis_zarr_roots: list[Path],
) -> list[tuple[str, str]]:
    print(f"Discovering pairs with pc_root={pc_root}")
    print(f"Tasks: {tasks}")

    use_zarr = bool(vis_zarr_roots)
    roots = vis_zarr_roots if use_zarr else vis_roots

    pairs: list[tuple[str, str]] = []
    for task in tasks:
        t = str(task)
        t_pc = pc_root / t
        if not t_pc.exists():
            print(f"Task PC root not found: {t_pc}")
            continue

        # 以 teacher 侧 episode 目录为基准
        for ep_dir in sorted([p for p in t_pc.iterdir() if p.is_dir() and p.name.startswith("episode_")]):
            episode = ep_dir.name
            if episode.endswith(".zarr"):
                episode = episode[:-5]
            
            # print(f"Checking {task}/{episode}")
            try:
                # 至少要有一个 ulip step (zarr or pt) 或者 episode.zarr
                found = False
                if (t_pc / f"{episode}.zarr").exists():
                    found = True
                elif next(iter(ep_dir.glob("step_*.ply.ulip_*.zarr")), None):
                    found = True
                elif next(iter(ep_dir.glob("step_*.ply.ulip_*.pt")), None):
                    found = True
                
                if not found:
                    print(f"Teacher data not found for {task}/{episode}")
                    continue
            except Exception:
                continue

            ok = True
            for r in roots:
                if use_zarr:
                    if not (r / t / f"{episode}.zarr").exists():
                        print(f"Student Zarr not found: {r}/{t}/{episode}.zarr")
                        ok = False
                        break
                else:
                    if not (r / t / f"{episode}.pt").exists():
                        print(f"Student PT not found: {r}/{t}/{episode}.pt")
                        ok = False
                        break
            
            if ok:
                pairs.append((t, episode))
    
    print(f"Found {len(pairs)} pairs")
    return pairs


def _load_ulip_step(path: Path) -> tuple[np.ndarray | None, torch.Tensor]:
    """读取 ULIP step (.pt 或 .zarr).

    返回: (pc, pc_feat)
    - pc: [N,3] (optional, None if zarr)
    - pc_feat: [N,D]
    """
    pstr = str(path)
    if pstr.endswith(".zarr"):
        # zarr mode: [N, D] array
        arr = zarr.open(pstr, mode="r")
        pc_feat = torch.from_numpy(arr[:]).to(torch.float32)
        return None, pc_feat
    
    # pt mode
    obj = torch.load(pstr, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict):
        raise TypeError(f"ULIP pt 格式不支持（期望 dict）: {path}")
    pc = obj.get("pc", None)
    pc_feat = obj.get("pc_feat", None)
    if pc is None or pc_feat is None:
        raise KeyError(f"ULIP pt 缺少 pc/pc_feat: {path}")
    pc = np.asarray(pc)
    pc_feat_t = torch.as_tensor(pc_feat).to(torch.float32)
    # if pc_feat_t.ndim != 2 or pc_feat_t.shape[1] != 256:
    #     raise ValueError(f"pc_feat 形状不对，应为 [N,256]，但收到 {tuple(pc_feat_t.shape)}: {path}")
    return pc, pc_feat_t


def _sample_tokens_from_frame(frame: np.ndarray | torch.Tensor, *, k: int, device: torch.device) -> torch.Tensor:
    """从单帧特征 [Hf,Wf,C] 采样 k 个 token，返回 [k,C] float32 on device."""

    if isinstance(frame, torch.Tensor):
        if frame.ndim != 3:
            raise ValueError(f"expect [Hf,Wf,C], got {tuple(frame.shape)}")
        Hf, Wf, C = frame.shape
        n_tokens = int(Hf * Wf)
        kk = min(int(k), n_tokens)
        flat = frame.reshape(n_tokens, C)
        sel = torch.randint(0, n_tokens, (kk,), device="cpu")
        tok = flat[sel]
        tok = torch.as_tensor(tok).to(torch.float32).to(device)
    else:
        if frame.ndim != 3:
            raise ValueError(f"expect [Hf,Wf,C], got {tuple(frame.shape)}")
        Hf, Wf, C = frame.shape
        n_tokens = int(Hf * Wf)
        kk = min(int(k), n_tokens)
        flat = frame.reshape(n_tokens, C)
        sel = np.random.randint(0, n_tokens, size=(kk,))
        tok = torch.from_numpy(flat[sel]).to(torch.float32).to(device)

    tok = _pad_or_trunc_tokens(tok, int(k))
    return tok


def _sample_tokens_from_pack(pack, *, k: int, device: torch.device) -> torch.Tensor:
    """从一个 FeaturePack 里随机采样 k 个 token（student 侧）。

    pack.per_frame_features: [W,T,Hf,Wf,C]
    这里为简单起见：随机选 (wi,ti)，并从 [Hf,Wf] 随机采 token。
    """

    pf = pack.per_frame_features
    if pf is None:
        raise RuntimeError("pack 缺少 per_frame_features")
    W, T, Hf, Wf, C = pf.shape

    wi = random.randrange(W)
    ti = random.randrange(T)

    # 在 CPU 上先取出这一帧 [Hf,Wf,C]，再只采样少量 token 搬到 GPU
    frame = pf[wi, ti]  # [Hf,Wf,C]  (通常在 CPU)
    return _sample_tokens_from_frame(frame, k=int(k), device=device)


def _step_stem_from_path(p: str) -> str:
    return Path(p).stem


def _build_step_index(frame_paths: list[list[str]]) -> dict[str, tuple[int, int]]:
    out: dict[str, tuple[int, int]] = {}
    for wi, win in enumerate(frame_paths):
        for ti, fp in enumerate(win):
            out[_step_stem_from_path(fp)] = (wi, ti)
    return out


def _pad_or_trunc_tokens(x: torch.Tensor, k: int) -> torch.Tensor:
    """把 [n,d] 调整到 [k,d]，用重复采样做 pad。"""
    if x.ndim != 2:
        raise ValueError(f"expect [n,d], got {tuple(x.shape)}")
    n, d = x.shape
    if n == k:
        return x
    if n > k:
        return x[:k]
    # n < k: repeat indices
    idx = torch.randint(0, n, (k - n,), device=x.device)
    return torch.cat([x, x[idx]], dim=0)


def _pool_tokens(x: torch.Tensor, *, method: str = "mean") -> torch.Tensor:
    """把 token 序列 [N,D] pool 成 [D]。"""

    if x.ndim != 2:
        raise ValueError(f"expect [N,D], got {tuple(x.shape)}")
    if method == "mean":
        return x.mean(dim=0)
    raise ValueError(f"unknown pool method: {method}")


def main() -> None:
    ap = argparse.ArgumentParser(description="路线2：RGB-only 蒸馏训练（teacher=点云特征）")

    ap.add_argument("--config", type=str, default="", help="可选 YAML 配置，命令行参数可覆盖")

    # data
    ap.add_argument("--tasks", type=str, nargs="*", default=[], help="任务列表")
    ap.add_argument("--episodes", type=int, default=20)

    ap.add_argument(
        "--pc_root",
        type=str,
        default="/home/gl/features_model/pc_dataset/PC/ULIP_FEAT_PT_POINT",
        help="ULIP step 特征根目录：<task>/<episode>/step_*.ply.ulip_*.pt",
    )
    ap.add_argument(
        "--vis_roots",
        type=str,
        nargs="*",
        default=[],
        help="视觉特征根目录列表（四模型或两模型都可以），每个 root 下应有 <task>/<episode>.pt",
    )

    ap.add_argument(
        "--vis_zarr_roots",
        type=str,
        nargs="*",
        default=[],
        help=(
            "可选：Zarr 视觉特征根目录列表（优先于 --vis_roots）。"
            "每个 root 下应有 <task>/<episode>.zarr（由 tools/convert_episode_pt_to_zarr.py 生成）。"
        ),
    )

    # training
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tau", type=float, default=0.07)
    ap.add_argument("--fuse_dim", type=int, default=384, help="融合/投影维度（应与 Teacher 维度一致，如 384）")
    ap.add_argument("--moe_hidden", type=int, default=256, help="MoE gate hidden dim")
    ap.add_argument("--fusion", type=str, default="weighted", choices=["weighted", "moe"])
    ap.add_argument("--loss_mse", type=float, default=0.0, help="额外 MSE loss 权重")
    ap.add_argument(
        "--amp",
        action="store_true",
        help="启用 AMP（仅 CUDA 生效）。当出现 nonfinite_grads 时可配合 --skip_nonfinite 让 scaler 自动降 scale。",
    )
    ap.add_argument("--batch_size", type=int, default=8, help="batch 内样本数（task/episode/step 的组合）")
    ap.add_argument(
        "--grad_accum_steps",
        type=int,
        default=1,
        help="梯度累积步数。effective_batch = batch_size * grad_accum_steps。>1 时会降低对比学习的噪声（更多负样本/更稳定），但吞吐会按比例下降。",
    )
    ap.add_argument("--student_tokens", type=int, default=1024, help="每个样本从每个视觉模型采样多少 tokens")
    ap.add_argument("--teacher_points", type=int, default=1024, help="每个样本从点云采样多少 teacher 点")

    # sampling granularity
    ap.add_argument(
        "--sample_unit",
        type=str,
        default="step",
        choices=["step", "window"],
        help="样本粒度：step=一个样本对应一个step(单帧)；window=一个样本对应一个window(8帧)",
    )
    ap.add_argument(
        "--window_agg",
        type=str,
        default="mean",
        choices=["mean"],
        help="window 模式下时间聚合方式（当前实现 mean）",
    )

    # perf/debug
    ap.add_argument("--pack_cache_size", type=int, default=4)
    ap.add_argument(
        "--strict_pairing",
        action="store_true",
        help="启用严格按 step_stem 配对（teacher step_XXXX ↔ student frame_paths 的 step_XXXX）",
    )
    ap.add_argument(
        "--pairing_fallback",
        type=str,
        default="random",
        choices=["random", "skip", "error"],
        help="严格配对缺失 step 时的处理：random=回退随机帧；skip=跳过该样本并重采；error=直接报错",
    )
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument(
        "--print_every",
        type=int,
        default=50,
        help="终端额外打印频率（步）。tqdm postfix 会每步更新；该参数控制 tqdm.write/print 的频率。",
    )
    ap.add_argument(
        "--grad_clip",
        type=float,
        default=1.0,
        help="梯度裁剪阈值（global norm）。<=0 表示不裁剪。",
    )
    ap.add_argument(
        "--skip_nonfinite",
        action="store_true",
        help="若检测到非有限梯度（inf/nan），跳过该步 optimizer 更新（建议与 AMP 一起使用）。",
    )

    # dataloader (for performance)
    ap.add_argument("--num_workers", type=int, default=4, help="DataLoader worker 数")
    ap.add_argument("--prefetch_factor", type=int, default=2, help="DataLoader prefetch_factor（仅 num_workers>0 生效）")
    ap.add_argument("--persistent_workers", action="store_true", help="DataLoader 持久 worker（仅 num_workers>0 生效）")
    ap.add_argument("--pin_memory", action="store_true", help="DataLoader pin_memory（CUDA 推荐开启）")
    ap.add_argument("--mem_every", type=int, default=0)

    # ckpt
    ap.add_argument("--save_dir", type=str, default="/home/gl/features_model/outputs/train_rgb2pc_runs/run0")
    ap.add_argument("--save_every", type=int, default=500)

    # logging
    ap.add_argument("--tqdm", action="store_true", help="使用 tqdm 进度条（终端显示更友好）")
    ap.add_argument("--wandb", action="store_true", help="启用 Weights & Biases 日志")
    ap.add_argument("--wandb_project", type=str, default="rgb2pc_distill")
    ap.add_argument("--wandb_entity", type=str, default="")
    ap.add_argument("--wandb_run_name", type=str, default="")

    # 先 parse config
    args_pre, _ = ap.parse_known_args()
    if str(getattr(args_pre, "config", "")).strip():
        cfg_path = Path(str(args_pre.config))
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        if isinstance(cfg, dict):
            ap.set_defaults(**cfg)
    args = ap.parse_args()

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    device = torch.device(str(args.device))
    use_amp = bool(getattr(args, "amp", False)) and device.type == "cuda"
    # 新版 AMP API（消除 FutureWarning）；CPU 下等价禁用
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    if not args.tasks:
        raise ValueError("路线2需要 --tasks（用于采样 paired RGB/点云样本）")
    if not (args.vis_zarr_roots or args.vis_roots):
        raise ValueError("路线2需要 --vis_roots 或 --vis_zarr_roots")

    # optional wandb
    wb = None
    if bool(getattr(args, "wandb", False)):
        # 避免在无交互环境下 wandb 进入 login prompt 卡住训练。
        # 若用户没设置 WANDB_API_KEY，则自动降级为仅终端输出。
        if not str(os.environ.get("WANDB_API_KEY", "")).strip():
            print("[warn] 未检测到 WANDB_API_KEY，已自动禁用 wandb（避免交互式登录阻塞训练）。")
        else:
            try:
                import wandb  # type: ignore

                wb = wandb
                init_kwargs = {
                    "project": str(getattr(args, "wandb_project", "rgb2pc_distill")),
                    "config": vars(args),
                }
                if str(getattr(args, "wandb_entity", "")).strip():
                    init_kwargs["entity"] = str(getattr(args, "wandb_entity"))
                if str(getattr(args, "wandb_run_name", "")).strip():
                    init_kwargs["name"] = str(getattr(args, "wandb_run_name"))
                wb.init(**init_kwargs)
            except Exception as e:
                # print frequency rules
                if int(getattr(args, "print_every", 0)) > 0:
                    print_every = int(getattr(args, "print_every", 0))
                else:
                    if bool(getattr(args, "tqdm", False)) and tqdm is not None:
                        print_every = 50
                    else:
                        print_every = int(getattr(args, "log_every", 50))

                print(f"[warn] wandb 启用失败，将降级为仅终端输出：{e}")
                wb = None

    # -----------------
    # DataLoader dataset (streaming random sampler)
    # -----------------
    dataset = RGB2PCDistillDataset(
        pc_root=str(getattr(args, "pc_root")),
        vis_zarr_roots=list(getattr(args, "vis_zarr_roots", []))
        if list(getattr(args, "vis_zarr_roots", []))
        else list(getattr(args, "vis_roots", [])),
        tasks=list(getattr(args, "tasks", [])),
        episodes=int(getattr(args, "episodes", 0)),
        sample_unit=str(getattr(args, "sample_unit", "step")),
        student_tokens=int(getattr(args, "student_tokens", 1024)),
        teacher_points=int(getattr(args, "teacher_points", 1024)),
        strict_pairing=bool(getattr(args, "strict_pairing", False)),
        pairing_fallback=str(getattr(args, "pairing_fallback", "random")),
        seed=int(getattr(args, "seed", 0)),
    )

    # build student modules based on first sample
    s0 = dataset[0]
    
    # Auto-detect teacher dim
    t_dim = 384
    if getattr(s0, "teacher_points", None) is not None:
        t_dim = int(s0.teacher_points.shape[-1])
    elif getattr(s0, "teacher_embed", None) is not None:
        t_dim = int(s0.teacher_embed.shape[-1])
    
    if t_dim != int(args.fuse_dim):
        print(f"[info] Auto-adjusting fuse_dim from {args.fuse_dim} to {t_dim} to match Teacher")
        args.fuse_dim = t_dim

    packs0 = [dataset._get_pack(i, s0.task, s0.episode) for i in range(len(dataset.vis_zarr_roots))]  # type: ignore[attr-defined]

    adapters = nn.ModuleList()
    for p in packs0:
        if hasattr(p, "per_frame_features"):
            pf = p.per_frame_features
            if pf is None:
                raise RuntimeError("vis pack 缺少 per_frame_features")
            c_in = int(pf.shape[-1])
        else:
            # zarr pack
            c_in = int(p.arr.shape[-1])
        adapters.append(MLP(in_dim=c_in, out_dim=int(args.fuse_dim), hidden_dim=int(args.fuse_dim) * 2))
    adapters = adapters.to(device)

    if str(args.fusion) == "weighted":
        fusion: nn.Module = WeightedFusion(num_models=len(packs0)).to(device)
    else:
        # MoE with Top-2
        fusion = MoEFusion(dim=int(args.fuse_dim), num_models=len(packs0), hidden_dim=int(args.moe_hidden), k=2).to(device)

    # Context Encoder (Transformer) to enhance complexity
    # [B, K, D] -> [B, K, D] with spatial interaction
    # Add Positional Encoding
    pos_encoder = PositionalEncoding(d_model=int(args.fuse_dim), dropout=0.1, max_len=int(args.student_tokens)).to(device)
    
    context_layer = nn.TransformerEncoderLayer(d_model=int(args.fuse_dim), nhead=8, dim_feedforward=int(args.fuse_dim)*4, dropout=0.1, batch_first=True)
    context_encoder = nn.TransformerEncoder(context_layer, num_layers=2).to(device)

    # teacher projector and student projector (both -> D)
    # [FIX] 移除 proj_teacher：Teacher 应该是固定的 ULIP 特征，不应通过随机初始化的 MLP 扭曲。
    # proj_teacher = MLP(in_dim=256, out_dim=int(args.fuse_dim), hidden_dim=int(args.fuse_dim) * 2).to(device)
    proj_student = MLP(in_dim=int(args.fuse_dim), out_dim=int(args.fuse_dim), hidden_dim=int(args.fuse_dim) * 2).to(device)

    params = list(adapters.parameters()) + list(fusion.parameters()) + list(context_encoder.parameters()) + list(proj_student.parameters())
    opt = torch.optim.AdamW(params, lr=float(args.lr))
    
    # Learning Rate Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(args.steps), eta_min=1e-6)

    save_dir = Path(str(args.save_dir))
    save_dir.mkdir(parents=True, exist_ok=True)

    # helper sampling
    pc_root = Path(str(args.pc_root))
    vis_roots = [Path(p) for p in (list(args.vis_roots) if args.vis_roots else [])]
    vis_zarr_roots = [Path(p) for p in (list(args.vis_zarr_roots) if args.vis_zarr_roots else [])]

    available_pairs = _discover_available_pairs(
        tasks=list(args.tasks),
        pc_root=pc_root,
        vis_roots=vis_roots,
        vis_zarr_roots=vis_zarr_roots,
    )
    if not available_pairs:
        raise RuntimeError(
            "没有找到可训练的 (task,episode) 对：请检查 pc_root 与 vis_roots/vis_zarr_roots 是否包含相同 task/episode。"
        )
    print(f"[data] available_pairs={len(available_pairs)} (sampling uniformly)")

    def _sample_task_episode() -> tuple[str, str]:
        # 若用户显式提供 --episodes>0，则仍可走旧逻辑（便于快速 smoke）；否则用发现列表。
        if int(getattr(args, "episodes", 0)) > 0:
            t = random.choice(list(args.tasks))
            ei = random.randrange(int(args.episodes))
            return str(t), f"episode_{ei}"
        return random.choice(available_pairs)

    def _sample_teacher_step(task: str, episode: str) -> Path:
        # prefer ulip pt steps: step_0000.ply.ulip_*.pt
        ep_dir = pc_root / task / episode
        # try zarr
        steps = _sorted_steps_in_dir_by_stem(ep_dir, stem_contains=".ply.ulip_")
        # filter for zarr or pt
        valid = [p for p in steps if p.name.endswith(".zarr") or p.name.endswith(".pt")]
        if not valid:
             raise FileNotFoundError(f"no valid teacher steps in {ep_dir}")
        return random.choice(valid)

    def _step_stem_from_ulip_path(p: Path) -> str:
        name = p.name
        if name.startswith("step_"):
            return "step_" + name.split("step_")[-1].split(".")[0]
        return p.stem

    def _sample_student_frame_tokens(pack, *, step_stem: str, k: int) -> torch.Tensor:
        """严格按 step_stem 找到 (wi,ti)，并从该帧采样 tokens。"""

        idx = getattr(pack, "_step_index", None)
        if not isinstance(idx, dict) or step_stem not in idx:
            raise KeyError(f"step {step_stem} not found in pack.frame_paths")
        wi, ti = idx[step_stem]

        if hasattr(pack, "per_frame_features"):
            pf = pack.per_frame_features
            assert pf is not None
            frame = pf[wi, ti]
            return _sample_tokens_from_frame(frame, k=int(k), device=device)

        # zarr: numpy
        frame_np = pack.get_frame(wi, ti)  # [Hf,Wf,C]
        return _sample_tokens_from_frame(frame_np, k=int(k), device=device)

    def _get_frame_by_index(pack, *, wi: int, ti: int):
        if hasattr(pack, "per_frame_features"):
            pf = pack.per_frame_features
            assert pf is not None
            return pf[wi, ti]
        return pack.get_frame(wi, ti)

    def _sample_student_window_tokens(pack, *, wi: int, k_per_frame: int) -> torch.Tensor:
        """window级 student：对该window内每帧采样 K tokens，拼接为 [T*K, C]。"""

        # 假设 pack.shape == [W,T,Hf,Wf,C]
        if hasattr(pack, "per_frame_features"):
            pf = pack.per_frame_features
            assert pf is not None
            _W, T, _Hf, _Wf, _C = pf.shape
        else:
            _W, T, _Hf, _Wf, _C = pack.arr.shape

        toks_all: list[torch.Tensor] = []
        for ti in range(int(T)):
            frame = _get_frame_by_index(pack, wi=wi, ti=ti)
            toks_all.append(_sample_tokens_from_frame(frame, k=int(k_per_frame), device=device))
        return torch.cat(toks_all, dim=0)

    def _sample_teacher_window_embedding(task: str, episode: str, *, wi: int, frame_paths: list[list[str]]) -> torch.Tensor:
        """window级 teacher：取该window内每个 step 的 ULIP pc_feat -> proj_teacher -> pool，再时间聚合。"""

        steps_in_win = frame_paths[wi]
        z_steps: list[torch.Tensor] = []
        ep_dir = pc_root / task / episode
        for fp in steps_in_win:
            step_stem = _step_stem_from_path(fp)
            # 允许 frame_paths 里带相对路径；我们只取 stem 后拼 ULIP 模式 glob
            # try zarr first
            cand = sorted(ep_dir.glob(f"{step_stem}.ply.ulip_*.zarr"))
            if not cand:
                cand = sorted(ep_dir.glob(f"{step_stem}.ply.ulip_*.pt"))
            
            if not cand:
                raise FileNotFoundError(f"missing ulip step for {task}/{episode}/{step_stem}")
            ulip_step = cand[0]
            _pc, pc_feat_cpu = _load_ulip_step(ulip_step)
            n = int(pc_feat_cpu.shape[0])
            k_t = min(int(args.teacher_points), n)
            sel_t = torch.randint(0, n, (k_t,), device="cpu")
            pc_feat = pc_feat_cpu[sel_t].to(torch.float32).to(device)
            # with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            #     z_ti = proj_teacher(pc_feat)
            # z_steps.append(_pool_tokens(z_ti, method="mean"))
            # [FIX] 移除 proj_teacher
            z_steps.append(_pool_tokens(pc_feat, method="mean"))

        z_stack = torch.stack(z_steps, dim=0)  # [T,D]
        if str(getattr(args, "window_agg", "mean")) == "mean":
            return z_stack.mean(dim=0)
        raise ValueError(f"unknown window_agg: {args.window_agg}")

    global_step = 0  # optimizer step count (effective step)
    ema_loss: float | None = None
    ema_beta = 0.98
    prev_ema: float | None = None
    opt.zero_grad(set_to_none=True)
    micro_step = 0

    def _collate(batch: list[DistillSample]):
        # all samples in a batch share same unit
        unit = batch[0].sample_unit
        toks_by_model = list(zip(*[b.tokens_by_model for b in batch]))  # type: ignore[arg-type]
        toks = [torch.stack(list(m), dim=0) for m in toks_by_model]  # M x [B,K,C]
        out = {
            "sample_unit": unit,
            "tokens": toks,
            "task": [b.task for b in batch],
            "episode": [b.episode for b in batch],
        }
        if unit == "window":
            out["teacher_embed"] = torch.stack([b.teacher_embed for b in batch], dim=0)  # type: ignore[arg-type]
        else:
            # Step samples can provide either teacher_points ([Kt,D]) OR teacher_embed ([D])
            # depending on teacher storage format.
            if batch[0].teacher_points is not None:
                out["teacher_points"] = torch.stack([b.teacher_points for b in batch], dim=0)  # type: ignore[arg-type]
            elif batch[0].teacher_embed is not None:
                out["teacher_embed"] = torch.stack([b.teacher_embed for b in batch], dim=0)  # type: ignore[arg-type]
            else:
                raise RuntimeError("step sample missing teacher_points/teacher_embed")
        return out

    loader = DataLoader(
        dataset,
        batch_size=int(getattr(args, "batch_size", 8)),
        shuffle=False,
        num_workers=int(getattr(args, "num_workers", 4)),
        pin_memory=bool(getattr(args, "pin_memory", False)) and device.type == "cuda",
        persistent_workers=bool(getattr(args, "persistent_workers", False)) and int(getattr(args, "num_workers", 0)) > 0,
        prefetch_factor=int(getattr(args, "prefetch_factor", 2)) if int(getattr(args, "num_workers", 0)) > 0 else None,
        collate_fn=_collate,
        drop_last=True,
    )
    loader_iter = iter(loader)

    accum_steps = max(1, int(getattr(args, "grad_accum_steps", 1)))
    micro_total = int(args.steps) * accum_steps

    step_iter = range(int(micro_total))
    if bool(getattr(args, "tqdm", False)) and tqdm is not None:
        step_iter = tqdm(step_iter, total=int(micro_total), dynamic_ncols=True)

    for _ in step_iter:
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)

        toks_list: list[torch.Tensor] = batch["tokens"]
        toks_list = [t.to(device, non_blocking=True).to(torch.float32) for t in toks_list]

        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            # REFACTORING FOR TOKEN-LEVEL FUSION & CONTEXT
            # 1. Adapters -> [B, K, D]
            z_tokens_list = []
            for mi, toks in enumerate(toks_list):
                B, K, C = toks.shape
                z = adapters[mi](toks.reshape(B * K, C)).reshape(B, K, -1) # [B, K, D]
                z_tokens_list.append(z)
            
            # 2. Fusion (Token-wise)
            # MoE expects [N, D]. Flatten B*K.
            z_flat_list = [z.reshape(B*K, -1) for z in z_tokens_list]
            z_fused_flat, _ = fusion(z_flat_list)
            
            z_fused_tokens = z_fused_flat.reshape(B, K, -1) # [B, K, D]
            
            # 3. Context Encoder (Spatial Interaction)
            # Add Positional Encoding
            # PE expects [Seq, Batch, Dim], so transpose
            z_fused_tokens = z_fused_tokens.transpose(0, 1) # [K, B, D]
            z_fused_tokens = pos_encoder(z_fused_tokens)
            z_fused_tokens = z_fused_tokens.transpose(0, 1) # [B, K, D]
            
            z_enhanced = context_encoder(z_fused_tokens) # [B, K, D]
            
            # 4. Pooling
            z_final = z_enhanced.mean(dim=1) # [B, D]

            zs = proj_student(z_final)  # [B, D]

            if str(batch["sample_unit"]) == "window":
                te = batch["teacher_embed"].to(device, non_blocking=True).to(torch.float32)
                zt = te
            else:
                # step mode: teacher may be points or already an embedding
                if "teacher_embed" in batch:
                    zt = batch["teacher_embed"].to(device, non_blocking=True).to(torch.float32)
                else:
                    tp = batch["teacher_points"].to(device, non_blocking=True).to(torch.float32)
                    # If tp is (B, N, D), mean. If (B, D), use directly.
                    if tp.ndim == 3:
                        zt = tp.mean(dim=1)
                    else:
                        zt = tp
            
            # [CRITICAL FIX] Remove Common Mode (Batch Mean) to force diversity learning
            # Teacher embeddings are very similar (sim ~ 1.0), so we must subtract the mean
            # to let InfoNCE focus on the differences.
            # zt_centered = zt - zt.mean(dim=0, keepdim=True)
            # zs_centered = zs - zs.mean(dim=0, keepdim=True)
            
            # Use centered features for InfoNCE
            loss_nce = info_nce_batch(zt, zs, tau=float(args.tau))
            loss = loss_nce
            
            if float(args.loss_mse) > 0:
                loss = loss + float(args.loss_mse) * F.mse_loss(
                    F.normalize(zs.to(torch.float32), dim=-1),
                    F.normalize(zt.to(torch.float32), dim=-1),
                )

        # scale loss for gradient accumulation
        loss_micro = loss / float(accum_steps)

        # -----------------
        # interpretable similarity stats (fp32)
        # -----------------
        with torch.no_grad():
            zs_n = F.normalize(zs.detach().to(torch.float32), dim=-1)
            zt_n = F.normalize(zt.detach().to(torch.float32), dim=-1)
            # pos cosine: diag
            pos_sim = float((zs_n * zt_n).sum(dim=-1).mean().cpu().item())
            # neg cosine: off-diagonal mean
            sim_mat = zs_n @ zt_n.t()  # [B,B]
            B = int(sim_mat.shape[0])
            if B > 1:
                neg_sum = float((sim_mat.sum() - sim_mat.diag().sum()).cpu().item())
                neg_sim = neg_sum / float(B * (B - 1))
            else:
                neg_sim = float("nan")
            sim_gap = float(pos_sim - neg_sim) if math.isfinite(neg_sim) else float("nan")

        # -----------------
        # backward + grad stats (must be before step/zero_grad)
        # -----------------
        if use_amp:
            scaler.scale(loss_micro).backward()
            # unscale so grad_norm reflects true fp32 grads
            try:
                scaler.unscale_(opt)
            except Exception:
                pass
        else:
            loss_micro.backward()

        # grad norms (pre/post clip)
        grad_norm_pre = None
        grad_norm_post = None
        clipped = False
        nonfinite_grads = 0
        try:
            total = 0.0
            cnt = 0
            # [FIX] 移除 proj_teacher
            for mod in (adapters, fusion, context_encoder, proj_student):
                for p in mod.parameters():
                    if p.grad is None:
                        continue
                    g = p.grad.detach()
                    if not torch.isfinite(g).all():
                        nonfinite_grads += 1
                        continue
                    total += float(g.to(torch.float32).norm(2).cpu().item()) ** 2
                    cnt += 1
            if cnt > 0:
                grad_norm_pre = float(total**0.5)
        except Exception:
            grad_norm_pre = None

        # optional grad clip (after unscale in AMP) + recompute post norm
        if float(getattr(args, "grad_clip", 0.0)) > 0 and int(nonfinite_grads) == 0:
            try:
                gn_ret = torch.nn.utils.clip_grad_norm_(params, max_norm=float(args.grad_clip))
                # clip_grad_norm_ returns pre-clip total norm
                if grad_norm_pre is None:
                    grad_norm_pre = float(gn_ret.detach().to(torch.float32).cpu().item())
                clipped = bool(grad_norm_pre is not None and float(grad_norm_pre) > float(args.grad_clip) + 1e-12)
            except Exception:
                pass

        # post-clip norm (what actually goes into optimizer)
        try:
            total2 = 0.0
            cnt2 = 0
            # [FIX] 移除 proj_teacher
            for mod in (adapters, fusion, context_encoder, proj_student):
                for p in mod.parameters():
                    if p.grad is None:
                        continue
                    g = p.grad.detach()
                    if not torch.isfinite(g).all():
                        continue
                    total2 += float(g.to(torch.float32).norm(2).cpu().item()) ** 2
                    cnt2 += 1
            if cnt2 > 0:
                grad_norm_post = float(total2**0.5)
        except Exception:
            grad_norm_post = None

        micro_step += 1

        # only step optimizer every accum_steps micro-batches
        stepped = None
        if micro_step % accum_steps == 0:
            stepped = True
            if int(nonfinite_grads) > 0 and bool(getattr(args, "skip_nonfinite", False)):
                stepped = False
                # AMP 下遇到非有限梯度：即便跳过 opt.step，也要 update 让 GradScaler 降低 scale
                if use_amp:
                    try:
                        scaler.update()
                    except Exception:
                        pass
            else:
                if use_amp:
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
            
            # Update LR Scheduler
            scheduler.step()

            opt.zero_grad(set_to_none=True)
            global_step += 1

            if (stepped is False) and (int(getattr(args, "print_every", 0)) > 0) and (global_step % int(args.print_every) == 0):
                # 用 tqdm.write 避免破坏进度条
                msg = f"[warn] step={global_step} detected nonfinite grads ({int(nonfinite_grads)} params); skipped optimizer step"
                if bool(getattr(args, "tqdm", False)) and tqdm is not None:
                    try:
                        tqdm.write(msg)
                    except Exception:
                        pass
                else:
                    print(msg)

        # -----------------
        # live metrics
        # -----------------
        loss_v_now = float(loss.item())
        nce_v_now = float(loss_nce.item())
        lr_v_now = float(opt.param_groups[0]["lr"])
        scaler_scale = None
        if use_amp:
            try:
                scaler_scale = float(scaler.get_scale())
            except Exception:
                scaler_scale = None

        # Only update EMA on optimizer steps to keep semantics consistent
        if stepped is not None:
            if ema_loss is None:
                ema_loss = loss_v_now
            else:
                ema_loss = ema_beta * float(ema_loss) + (1.0 - ema_beta) * loss_v_now
            delta_ema = None if prev_ema is None else float(ema_loss - prev_ema)
            prev_ema = float(ema_loss)
        else:
            delta_ema = None

        # tqdm postfix every micro-step so you can "see" changes immediately
        if bool(getattr(args, "tqdm", False)) and tqdm is not None:
            try:
                # step_iter is the tqdm iterator
                step_iter.set_postfix(
                    {
                        "loss": f"{loss_v_now:.4f}",
                        "ema": "-" if ema_loss is None else f"{float(ema_loss):.4f}",
                        "dE": "-" if delta_ema is None else f"{delta_ema:+.2e}",
                        "nce": f"{nce_v_now:.4f}",
                        "lr": f"{lr_v_now:.1e}",
                        "gn": "-" if grad_norm_post is None else f"{grad_norm_post:.2f}",
                        "g0": "-" if grad_norm_pre is None else f"{grad_norm_pre:.2f}",
                        "cl": "1" if clipped else "0",
                        "pos": f"{pos_sim:.2f}",
                        "neg": "-" if not math.isfinite(neg_sim) else f"{neg_sim:.2f}",
                        "gap": "-" if not math.isfinite(sim_gap) else f"{sim_gap:.2f}",
                        "ng": str(int(nonfinite_grads)),
                        "st": "-" if stepped is None else ("1" if stepped else "0"),
                        "sc": "-" if scaler_scale is None else f"{scaler_scale:.0f}",
                        "ma": f"{micro_step%accum_steps}/{accum_steps}" if accum_steps > 1 else "-",
                        "s": str(int(global_step)),
                    }
                )  # type: ignore[attr-defined]
            except Exception:
                pass

        # periodic printing (separate from wandb logging)
        if stepped is not None and int(getattr(args, "print_every", 0)) > 0 and (global_step % int(args.print_every) == 0):
            msg = (
                f"step={global_step} loss={loss_v_now:.4f} ema={float(ema_loss):.4f} "
                f"nce={nce_v_now:.4f} lr={lr_v_now:.3e} "
                f"gn_pre={'-' if grad_norm_pre is None else f'{grad_norm_pre:.3f}'} "
                f"gn_post={'-' if grad_norm_post is None else f'{grad_norm_post:.3f}'} clipped={int(clipped)} "
                f"pos={pos_sim:.3f} neg={'-' if not math.isfinite(neg_sim) else f'{neg_sim:.3f}'} gap={'-' if not math.isfinite(sim_gap) else f'{sim_gap:.3f}'} "
                f"scale={'-' if scaler_scale is None else f'{scaler_scale:.0f}'} nonfinite_grads={int(nonfinite_grads)} stepped={int(stepped)}"
            )
            if bool(getattr(args, "tqdm", False)) and tqdm is not None:
                try:
                    tqdm.write(msg)
                except Exception:
                    pass
            else:
                print(msg)

        if stepped is not None and int(args.log_every) > 0 and (global_step % int(args.log_every) == 0):
            if wb is not None:
                wb.log(
                    {
                        "loss": loss_v_now,
                        "ema_loss": float(ema_loss),
                        "delta_ema": 0.0 if delta_ema is None else float(delta_ema),
                        "nce": nce_v_now,
                        "lr": lr_v_now,
                        "grad_norm_pre": 0.0 if grad_norm_pre is None else float(grad_norm_pre),
                        "grad_norm_post": 0.0 if grad_norm_post is None else float(grad_norm_post),
                        "clipped": int(clipped),
                        "pos_sim": float(pos_sim),
                        "neg_sim": 0.0 if not math.isfinite(neg_sim) else float(neg_sim),
                        "sim_gap": 0.0 if not math.isfinite(sim_gap) else float(sim_gap),
                        "scaler_scale": 0.0 if scaler_scale is None else float(scaler_scale),
                        "nonfinite_grads": int(nonfinite_grads),
                        "stepped": int(stepped),
                        "step": int(global_step),
                    },
                    step=int(global_step),
                )

        if stepped is not None and int(args.mem_every) > 0 and (global_step % int(args.mem_every) == 0):
            rss_mib = None
            try:
                import resource

                rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                rss_mib = float(rss_kb) / 1024.0
            except Exception:
                pass
            if device.type == "cuda" and torch.cuda.is_available():
                alloc = float(torch.cuda.memory_allocated(device)) / (1024.0**2)
                reserved = float(torch.cuda.memory_reserved(device)) / (1024.0**2)
                peak = float(torch.cuda.max_memory_allocated(device)) / (1024.0**2)
                print(f"[mem] step={global_step} rss_mib={rss_mib} cuda_alloc={alloc:.1f} reserved={reserved:.1f} peak={peak:.1f}")
            else:
                print(f"[mem] step={global_step} rss_mib={rss_mib}")

        if stepped is not None and int(args.save_every) > 0 and (global_step % int(args.save_every) == 0):
            ckpt = {
                "global_step": int(global_step),
                "args": vars(args),
                "adapters": adapters.state_dict(),
                "fusion": fusion.state_dict(),
                "context_encoder": context_encoder.state_dict(),
                "pos_encoder": pos_encoder.state_dict(),
                "proj_student": proj_student.state_dict(),
                "opt": opt.state_dict(),
            }
            out = save_dir / f"ckpt_step_{global_step:07d}.pt"
            torch.save(ckpt, out)
            print(f"[ckpt] saved: {out}")

    # final
    out = save_dir / "ckpt_final.pt"
    torch.save(
        {
            "global_step": int(global_step),
            "args": vars(args),
            "adapters": adapters.state_dict(),
            "fusion": fusion.state_dict(),
            "context_encoder": context_encoder.state_dict(),
            "pos_encoder": pos_encoder.state_dict(),
            "proj_student": proj_student.state_dict(),
            "opt": opt.state_dict(),
        },
        out,
    )
    print(f"[ckpt] saved final: {out}")


if __name__ == "__main__":
    main()
