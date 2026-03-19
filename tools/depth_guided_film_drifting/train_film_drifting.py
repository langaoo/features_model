"""
DA3-FiLM + Drifting Head 训练脚本
===================================

Innovation 2: 把 DP的 DDPM head 替换为 Drifting Model (arXiv:2602.04770)。
  - 推理时只需 1次前向传播 (1 NFE)，而非 DDPM 的 100 步迭代。
  - 训练目标：drifting loss (均值漂移 MSE)，不再预测噪声 epsilon。

使用方法:
  python tools/depth_guided_film_drifting/train_film_drifting.py \
      --config configs/depth_guided_film_drifting/train_film_drifting.yaml
"""

import os
import sys
import argparse
from pathlib import Path
from collections import defaultdict

import yaml
import zarr
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler
from tqdm.auto import tqdm


# ============================================================
# Per-episode Grouped Batch Sampler
# (论文 Section 4: "For each label, we perform Alg. 1 independently")
# ============================================================

class EpisodeGroupedBatchSampler(Sampler):
    """
    每个 batch 从 n_episodes_per_batch 个 episode 中采样，
    确保 drift loss 在同一 episode 内计算（类似论文的 per-class batching）。

    每个 mini-batch = n_episodes_per_batch 个 episode,
    每个 episode 采 samples_per_episode 个样本。
    总 batch_size = n_episodes_per_batch * samples_per_episode.
    """

    def __init__(self, ep_ids, n_episodes_per_batch=8, samples_per_episode=16,
                 n_batches_per_epoch=None, drop_small_episodes=4):
        self.ep_ids = ep_ids
        self.n_episodes_per_batch = n_episodes_per_batch
        self.samples_per_episode = samples_per_episode
        self.drop_small = drop_small_episodes

        # Build episode → sample indices mapping
        self.ep_to_indices = defaultdict(list)
        for sample_idx, ep_id in enumerate(ep_ids):
            self.ep_to_indices[ep_id].append(sample_idx)

        # Filter out episodes with too few samples
        self.valid_episodes = [
            ep for ep, indices in self.ep_to_indices.items()
            if len(indices) >= self.drop_small
        ]
        assert len(self.valid_episodes) > 0, "No episodes with enough samples"

        if n_batches_per_epoch is None:
            total_samples = sum(len(self.ep_to_indices[ep]) for ep in self.valid_episodes)
            self.n_batches = max(1, total_samples // (n_episodes_per_batch * samples_per_episode))
        else:
            self.n_batches = n_batches_per_epoch

    def __iter__(self):
        for _ in range(self.n_batches):
            # Randomly select episodes
            chosen_eps = np.random.choice(
                self.valid_episodes,
                size=min(self.n_episodes_per_batch, len(self.valid_episodes)),
                replace=False,
            )
            batch = []
            for ep in chosen_eps:
                pool = self.ep_to_indices[ep]
                n_take = min(self.samples_per_episode, len(pool))
                selected = np.random.choice(pool, size=n_take, replace=False)
                batch.extend(selected.tolist())
            yield batch

    def __len__(self):
        return self.n_batches

# -------- 路径设置 --------
current_file_path = os.path.abspath(__file__)
tools_dir         = os.path.dirname(current_file_path)
tools_parent      = os.path.dirname(tools_dir)  # tools/
features_model_dir = os.path.dirname(tools_parent)  # features_model/
sys.path.insert(0, features_model_dir)

# 官方 DP 路径（normalizer）
# features_model/DP/diffusion_policy/ 是 git repo 根目录，
# 其内部的 diffusion_policy/ 才是 Python package，需直接添加 repo 根目录
DP_OUTER = Path(features_model_dir) / "DP" / "diffusion_policy"
if DP_OUTER.exists():
    sys.path.insert(0, str(DP_OUTER))
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.common.lr_scheduler import get_scheduler

# 本项目模块
from features_common.depth_guided_film_online.extractors_2model import TwoModelExtractors
from features_common.depth_guided_film_online.encoder_film_2model import DA3Film2ModelEncoder
from features_common.depth_guided_film_drifting.policy_drifting import DA3FilmDriftingPolicy


def _init_wandb(config: dict, save_dir: Path | None = None):
    wandb_cfg = config.get("wandb", {})
    if not bool(wandb_cfg.get("enable", False)):
        return None

    try:
        import wandb
    except Exception as e:
        print(f"[wandb] import 失败，自动禁用: {e}")
        return None

    task_name = config.get("data", {}).get("tasks", "unknown")
    if isinstance(task_name, list):
        task_name = task_name[0] if task_name else "unknown"
    run_name = wandb_cfg.get("run_name", "")
    if not run_name:
        run_name = f"film_drifting_{task_name}"

    init_kwargs = {
        "project": wandb_cfg.get("project", "geo_drift_policy"),
        "name": run_name,
        "config": config,
    }
    if wandb_cfg.get("entity", ""):
        init_kwargs["entity"] = wandb_cfg["entity"]
    if wandb_cfg.get("mode", ""):
        init_kwargs["mode"] = wandb_cfg["mode"]
    if save_dir is not None:
        init_kwargs["dir"] = str(save_dir)

    try:
        run = wandb.init(**init_kwargs)
        print(
            f"[wandb] 已启用: project={init_kwargs['project']} "
            f"name={run_name} mode={init_kwargs.get('mode', 'online')}"
        )
        return run
    except Exception as e:
        print(f"[wandb] 初始化失败，自动禁用: {e}")
        return None


def _try_load_checkpoint(path: Path):
    try:
        ckpt = torch.load(path, map_location="cpu")
        return ckpt
    except Exception as e:
        print(f"[resume] 跳过损坏/不可读 checkpoint: {path} ({e})")
        return None


def _find_resume_checkpoint(save_dir: Path, resume_cfg):
    if resume_cfg in (None, False, "", "none", "None", "false", "False"):
        return None, None

    if resume_cfg is True:
        resume_cfg = "auto"

    if isinstance(resume_cfg, str) and resume_cfg.lower() == "auto":
        candidates = []
        for path in save_dir.glob("*.ckpt"):
            stem = path.stem
            if stem.isdigit():
                candidates.append((int(stem), path))
        candidates.sort(key=lambda x: x[0], reverse=True)
        ordered_paths = [path for _, path in candidates]
        best_path = save_dir / "best.ckpt"
        if best_path.exists():
            ordered_paths.append(best_path)

        for path in ordered_paths:
            ckpt = _try_load_checkpoint(path)
            if ckpt is not None:
                return path, ckpt
        return None, None

    resume_path = Path(resume_cfg)
    if not resume_path.is_absolute():
        resume_path = save_dir / resume_path
    if not resume_path.exists():
        print(f"[resume] 指定 checkpoint 不存在: {resume_path}")
        return None, None

    ckpt = _try_load_checkpoint(resume_path)
    if ckpt is None:
        return None, None
    return resume_path, ckpt


# ============================================================
# 预缓存 token 提取（与 train_film_online.py 完全相同）
# ============================================================

def _precompute_tokens(
    zarr_path: str,
    camera_name: str,
    extractors: TwoModelExtractors,
    max_tokens: int,
    batch_size: int = 32,
) -> tuple:
    from PIL import Image as PILImage
    z       = zarr.open(zarr_path, "r")
    imgs_np = z[f"data/{camera_name}"][:]
    N       = imgs_np.shape[0]
    print(f"  [precompute] {N} 帧图像 {imgs_np.shape}, batch={batch_size} ...")

    all_dino, all_da3 = [], []
    for i in tqdm(range(0, N, batch_size), desc="  预提取 tokens", leave=False):
        chunk    = imgs_np[i : i + batch_size]
        pil_list = [
            PILImage.fromarray(chunk[j].transpose(1, 2, 0))
            for j in range(chunk.shape[0])
        ]
        with torch.no_grad():
            tokens = extractors.extract_batch_tokens(
                pil_list, max_tokens=max_tokens, return_torch=True
            )
        all_dino.append(tokens[0].half().cpu())
        all_da3.append(tokens[1].half().cpu())

    dino_all = torch.cat(all_dino, dim=0)
    da3_all  = torch.cat(all_da3,  dim=0)
    mem_gb   = (dino_all.nbytes + da3_all.nbytes) / 1e9
    print(
        f"  [precompute] DINOv3 {tuple(dino_all.shape)}, "
        f"DA3 {tuple(da3_all.shape)}, 共 {mem_gb:.2f} GB (float16 CPU)"
    )
    return dino_all, da3_all


# ============================================================
# 缓存数据集（与 train_film_online.py 相同）
# ============================================================

class CachedTokenDataset(Dataset):
    def __init__(
        self,
        dino_tokens:   torch.Tensor,
        da3_tokens:    torch.Tensor,
        actions:       np.ndarray,
        states:        np.ndarray,
        episode_ends:  np.ndarray,
        horizon:       int = 8,
        n_obs_steps:   int = 3,
        positive_pool_size: int = 1,
        positive_radius: int = 0,
    ):
        super().__init__()
        self.dino_tokens = dino_tokens
        self.da3_tokens  = da3_tokens
        self.actions     = torch.from_numpy(actions).float()
        self.states      = torch.from_numpy(states).float()
        self.horizon     = horizon
        self.n_obs_steps = n_obs_steps
        self.positive_pool_size = max(1, int(positive_pool_size))
        self.positive_radius = max(0, int(positive_radius))
        self.indices     = self._build_indices(episode_ends)
        self.positive_pools = self._build_positive_pools()
        print(f"  [Dataset] {len(self.indices)} 个训练样本")
        if self.positive_pools is not None:
            print(
                f"  [Dataset] temporal multi-positive: P={self.positive_pool_size}, "
                f"radius={self.positive_radius}"
            )

    def _build_indices(self, episode_ends):
        indices  = []
        ep_ids   = []
        ep_local_ids = []
        ep_to_dataset_ids = defaultdict(list)
        ep_start = 0
        dataset_idx = 0
        for ep_idx, ep_end in enumerate(episode_ends):
            ep_len = int(ep_end) - ep_start
            max_t  = ep_len - self.n_obs_steps - self.horizon + 1
            n_samples = max(0, max_t)
            for t in range(n_samples):
                indices.append(ep_start + t)
                ep_ids.append(ep_idx)
                ep_local_ids.append(t)
                ep_to_dataset_ids[ep_idx].append(dataset_idx)
                dataset_idx += 1
            ep_start = int(ep_end)
        self.ep_ids = ep_ids
        self.ep_local_ids = ep_local_ids
        self.ep_to_dataset_ids = ep_to_dataset_ids
        return indices

    def _build_positive_pools(self):
        if self.positive_pool_size <= 1:
            return None

        positive_pools = []
        for dataset_idx, ep_id in enumerate(self.ep_ids):
            ep_dataset_ids = self.ep_to_dataset_ids[ep_id]
            center_local = self.ep_local_ids[dataset_idx]
            start = max(0, center_local - self.positive_radius)
            end = min(len(ep_dataset_ids), center_local + self.positive_radius + 1)
            candidate_ids = ep_dataset_ids[start:end]
            candidate_ids = sorted(
                candidate_ids,
                key=lambda did: (abs(self.ep_local_ids[did] - center_local), did),
            )
            chosen = candidate_ids[: self.positive_pool_size]
            while len(chosen) < self.positive_pool_size:
                chosen.append(dataset_idx)
            positive_pools.append(chosen)
        return positive_pools

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t0        = self.indices[idx]
        obs_slice = slice(t0, t0 + self.n_obs_steps)
        act_start = t0 + self.n_obs_steps
        act_slice = slice(act_start, act_start + self.horizon)
        sample = {
            "dino_tokens": self.dino_tokens[obs_slice].float(),
            "da3_tokens":  self.da3_tokens[obs_slice].float(),
            "agent_pos":   self.states[obs_slice],
            "action":      self.actions[act_slice],
            "episode_id":  self.ep_ids[idx],
        }
        if self.positive_pools is not None:
            positive_actions = []
            for pos_dataset_idx in self.positive_pools[idx]:
                pos_t0 = self.indices[pos_dataset_idx]
                pos_act_start = pos_t0 + self.n_obs_steps
                pos_act_slice = slice(pos_act_start, pos_act_start + self.horizon)
                positive_actions.append(self.actions[pos_act_slice])
            sample["positive_action"] = torch.stack(positive_actions, dim=0)
        return sample


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="DA3-FiLM + Drifting head 训练")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    print("=" * 65)
    print("DA3-FiLM + Drifting Head 训练 (Innovation 2 — 1-step inference)")
    print("=" * 65)

    gpu_ids  = config["device"]["gpu_ids"]
    if isinstance(gpu_ids, list):
        gpu_id = gpu_ids[0]
    else:
        gpu_ids = [int(gpu_ids)]
        gpu_id  = gpu_ids[0]
    device   = f"cuda:{gpu_id}"
    multi_gpu = len(gpu_ids) > 1
    print(f"Device: {device}" + (f"  (DataParallel: {gpu_ids})" if multi_gpu else ""))

    task_name      = config["data"]["tasks"]
    if isinstance(task_name, list):
        task_name = task_name[0]
    task_config    = config["data"].get("task_config", "demo_clean")
    expert_num     = int(config.get("checkpoint", {}).get("expert_data_num", 50))
    horizon        = int(config["data"]["horizon"])
    n_obs_steps    = int(config["data"]["n_obs_steps"])
    n_action_steps = int(config["data"].get("n_action_steps", 6))
    max_tokens     = int(config.get("encoder", {}).get("max_tokens", 196))
    camera_name    = config["data"].get("camera_name", "head_camera")

    zarr_base = config["data"].get("zarr_base", "/home/gl/RoboTwin/policy/DP/data")
    zarr_name = f"{task_name}-{task_config}-{expert_num}_multi_cam.zarr"
    zarr_path = os.path.join(zarr_base, zarr_name)
    if not os.path.exists(zarr_path):
        raise FileNotFoundError(f"Zarr not found: {zarr_path}")
    print(f"Zarr: {zarr_path}")

    # ---- 1. 加载 backbone ----
    print("\n1. 加载 DINOv3 + DA3 backbone (冻结)...")
    extractors = TwoModelExtractors(gpu_id=gpu_id)
    for m in [extractors.dinov3_model, extractors.da3_model]:
        m.requires_grad_(False)
        m.eval()
    print("   ✓ 两个 backbone 已冻结")

    # ---- 2. 预提取 tokens ----
    print("\n2. 预提取所有帧 tokens...")
    precompute_bs = int(config.get("precompute_batch_size", 32))
    dino_tokens, da3_tokens = _precompute_tokens(
        zarr_path, camera_name, extractors, max_tokens, batch_size=precompute_bs,
    )
    if config.get("offload_backbone_after_precompute", True):
        extractors.dinov3_model.cpu()
        extractors.da3_model.cpu()
        torch.cuda.empty_cache()
        print("   ✓ Backbone 已转到 CPU, GPU 显存已释放")

    # ---- 3. 读 zarr ----
    print("\n3. 读取 state / action / episode_ends ...")
    z               = zarr.open(zarr_path, "r")
    actions_np      = z["data/action"][:]
    states_np       = z["data/state"][:]
    episode_ends_np = z["meta/episode_ends"][:]
    print(f"   总帧数: {len(actions_np)}, episodes: {len(episode_ends_np)}")

    # ---- 4. 数据集 ----
    print("\n4. 创建 CachedTokenDataset ...")
    drifting_cfg = config.get("drifting", {})
    dataset = CachedTokenDataset(
        dino_tokens=dino_tokens,
        da3_tokens=da3_tokens,
        actions=actions_np,
        states=states_np,
        episode_ends=episode_ends_np,
        horizon=horizon,
        n_obs_steps=n_obs_steps,
        positive_pool_size=int(drifting_cfg.get("positive_pool_size", 1)),
        positive_radius=int(drifting_cfg.get("positive_radius", 0)),
    )

    train_cfg  = config["train"]

    # ---- DataLoader ----
    batch_size = int(train_cfg.get("batch_size", 64))
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 4)),
        pin_memory=True,
        persistent_workers=int(train_cfg.get("num_workers", 4)) > 0,
        drop_last=True,
    )
    print(
        f"   batch_size={batch_size}, shuffle=True, "
        f"num_workers={train_cfg.get('num_workers', 4)}, "
        f"batches/epoch={len(dataloader)}"
    )

    # ---- 5. 模型 ----
    print("\n5. 创建 DA3Film2ModelEncoder + DriftingPolicy ...")
    enc_cfg = config.get("encoder", {})
    fusion_encoder = DA3Film2ModelEncoder(
        semantic_in_dim=int(enc_cfg.get("semantic_in_dim", 768)),
        geometric_in_dim=int(enc_cfg.get("geometric_in_dim", 2048)),
        proj_dim=int(enc_cfg.get("proj_dim", 256)),
        film_hidden=int(enc_cfg.get("film_hidden", 256)),
        out_dim=int(enc_cfg.get("out_dim", 1280)),
        with_pos_enc=bool(enc_cfg.get("with_pos_enc", True)),
        dropout=float(enc_cfg.get("dropout", 0.1)),
        max_tokens=int(enc_cfg.get("max_tokens", 196)),
    ).to(device)

    drifting_cfg = config.get("drifting", {})
    drifting_temp_scale = float(drifting_cfg.get("temp_scale", 0.1))
    drifting_K = int(drifting_cfg.get("K", 8))
    drift_alpha = drifting_cfg.get("drift_alpha", None)
    if drift_alpha is not None:
        drift_scale = float(drift_alpha) * drifting_K
    else:
        drift_scale = float(drifting_cfg.get("drift_scale", 50.0))
    drift_scale_min = float(drifting_cfg.get("drift_scale_min", 1.0))
    drift_scale_schedule = drifting_cfg.get("drift_scale_schedule", "constant") # constant, linear, cosine
    drift_use_scale = bool(drifting_cfg.get("drift_use_scale", True))
    drift_normalize = bool(drifting_cfg.get("drift_normalize", False))
    drift_norm_mode = drifting_cfg.get("drift_norm_mode", "per_obs")
    drift_norm_eps = float(drifting_cfg.get("drift_norm_eps", 1e-6))
    drift_norm_ema_decay = float(drifting_cfg.get("drift_norm_ema_decay", 0.99))
    drift_target_rms = float(drifting_cfg.get("drift_target_rms", 1.0))
    bc_mode = drifting_cfg.get("bc_mode", "fixed")
    
    policy = DA3FilmDriftingPolicy(
        fusion_encoder=fusion_encoder,
        proprio_dim=14,
        action_dim=14,
        horizon=horizon,
        n_obs_steps=n_obs_steps,
        n_action_steps=n_action_steps,
        drifting_temp_scale=drifting_temp_scale,
        drift_normalize=drift_normalize,
        drift_norm_mode=drift_norm_mode,
        drift_norm_eps=drift_norm_eps,
        drift_norm_ema_decay=drift_norm_ema_decay,
        drift_target_rms=drift_target_rms,
        drift_use_scale=drift_use_scale,
        bc_mode=bc_mode,
    ).to(device)
    policy.drift_scale = drift_scale if drift_use_scale else 1.0
    bc_lambda = float(drifting_cfg.get("bc_lambda", 0.0))
    policy.bc_lambda = bc_lambda

    # ---- DataParallel for UNet (memory bottleneck) ----
    if multi_gpu:
        policy.noise_pred_net = nn.DataParallel(
            policy.noise_pred_net, device_ids=gpu_ids
        )
        print(f"   ✓ UNet wrapped in DataParallel on GPUs {gpu_ids}")

    total_params   = sum(p.numel() for p in policy.parameters()) / 1e6
    encoder_params = sum(p.numel() for p in fusion_encoder.parameters()) / 1e6
    gen_params     = sum(p.numel() for p in policy.noise_pred_net.parameters()) / 1e6
    print(
        f"   模型总参数: {total_params:.2f}M  "
        f"(FiLM encoder: {encoder_params:.2f}M, "
        f"ConditionalUnet1D: {gen_params:.2f}M)"
    )

    # ---- 6. Fit normalizer ----
    print("\n6. Fit normalizer ...")
    policy.normalizer.fit(
        {
            "action":    torch.from_numpy(actions_np).float(),
            "agent_pos": torch.from_numpy(states_np).float(),
        },
        last_n_dims=1,
        mode="limits",
        output_min=-1.0,
        output_max=1.0,
    )
    try:
        policy.normalizer.to(device)
    except Exception:
        pass

    # ---- 7. 优化器 ----
    total_epochs = int(train_cfg["epochs"])
    grad_accum_steps = int(train_cfg.get("grad_accum_steps", 1))
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg.get("weight_decay", 1e-6)),
        betas=(0.95, 0.999),
    )
    total_train_steps = max(
        1,
        (len(dataloader) * total_epochs + grad_accum_steps - 1) // grad_accum_steps,
    )
    lr_scheduler_name = train_cfg.get("lr_scheduler", "cosine")
    lr_warmup_steps = int(train_cfg.get("lr_warmup_steps", 500))
    ema_decay = float(train_cfg.get("ema_decay", 0.0))
    ema_enabled = 0.0 < ema_decay < 1.0
    ema_state = None

    # ---- 8. 训练循环 ----
    save_dir   = Path(config["checkpoint"]["save_dir"])
    save_every = int(config["checkpoint"].get("save_every", 100))
    save_dir.mkdir(parents=True, exist_ok=True)
    resume_cfg = config["checkpoint"].get("resume", None)

    start_epoch = 0
    best_loss  = float("inf")
    resume_path, resume_ckpt = _find_resume_checkpoint(save_dir, resume_cfg)

    def _restore_state_dict_for_model(sd):
        current_keys = policy.state_dict().keys()
        if any(k.startswith("noise_pred_net.module.") for k in current_keys):
            return {
                (k.replace("noise_pred_net.", "noise_pred_net.module.", 1)
                 if k.startswith("noise_pred_net.") else k): v
                for k, v in sd.items()
            }
        return sd

    if resume_ckpt is not None:
        print(f"[resume] 从 checkpoint 恢复: {resume_path}")
        policy.load_state_dict(_restore_state_dict_for_model(resume_ckpt["policy"]), strict=True)
        if "normalizer" in resume_ckpt:
            policy.normalizer.load_state_dict(resume_ckpt["normalizer"])
        if "optimizer" in resume_ckpt:
            optimizer.load_state_dict(resume_ckpt["optimizer"])
        start_epoch = int(resume_ckpt.get("epoch", 0))
        best_loss = float(resume_ckpt.get("best_loss", resume_ckpt.get("loss", float("inf"))))
        if ema_enabled:
            ema_policy = _restore_state_dict_for_model(
                resume_ckpt.get("ema_policy", resume_ckpt["policy"])
            )
            current_state = policy.state_dict()
            ema_state = {}
            for k, v in ema_policy.items():
                ref = current_state[k]
                ema_state[k] = v.detach().clone().to(
                    device=ref.device,
                    dtype=ref.dtype,
                )
        print(f"[resume] 恢复 epoch={start_epoch}, best_loss={best_loss:.6f}")
    elif ema_enabled:
        ema_state = {k: v.detach().clone() for k, v in policy.state_dict().items()}
        print(f"   EMA enabled: decay={ema_decay}")

    completed_train_steps = max(
        0,
        (len(dataloader) * start_epoch + grad_accum_steps - 1) // grad_accum_steps,
    )
    scheduler = get_scheduler(
        lr_scheduler_name,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps,
        num_training_steps=total_train_steps,
        last_epoch=completed_train_steps - 1,
    )
    if resume_ckpt is not None and "scheduler" in resume_ckpt:
        scheduler.load_state_dict(resume_ckpt["scheduler"])

    best_ckpt_path = save_dir / "best.ckpt"
    if best_ckpt_path.exists():
        best_ckpt = _try_load_checkpoint(best_ckpt_path)
        if best_ckpt is not None:
            best_loss = min(best_loss, float(best_ckpt.get("loss", float("inf"))))

    wandb_run = _init_wandb(config, save_dir=save_dir)
    avg_loss   = float("inf")

    _enc_cfg_save = {
        "type":             "film_2model",
        "semantic_in_dim":  int(enc_cfg.get("semantic_in_dim", 768)),
        "geometric_in_dim": int(enc_cfg.get("geometric_in_dim", 2048)),
        "proj_dim":         int(enc_cfg.get("proj_dim", 256)),
        "film_hidden":      int(enc_cfg.get("film_hidden", 256)),
        "out_dim":          int(enc_cfg.get("out_dim", 1280)),
        "with_pos_enc":     bool(enc_cfg.get("with_pos_enc", True)),
        "dropout":          float(enc_cfg.get("dropout", 0.1)),
        "max_tokens":       int(enc_cfg.get("max_tokens", 196)),
    }

    # ---- 8a. 打印 drifting 配置和数据统计 ----
    print(f"\n7a. Drifting 配置: temp_scale={drifting_temp_scale}, drift_scale={drift_scale}")
    print(
        f"   drift_use_scale={drift_use_scale}, drift_normalize={drift_normalize}, drift_norm_mode={drift_norm_mode}, "
        f"drift_norm_ema_decay={drift_norm_ema_decay}, drift_target_rms={drift_target_rms}, "
        f"ema_decay={ema_decay if ema_enabled else 0.0}, bc_lambda={bc_lambda}, bc_mode={bc_mode}"
    )
    if drift_alpha is not None:
        print(f"   drift_alpha={float(drift_alpha):.4f} → effective drift_scale={drift_scale:.4f}")
    print(
        f"   lr_scheduler={lr_scheduler_name}, lr_warmup_steps={lr_warmup_steps}, "
        f"total_train_steps={total_train_steps}"
    )
    with torch.no_grad():
        first_batch = next(iter(dataloader))
        _act = first_batch["action"].to(device)
        _nact = policy.normalizer.normalize({"action": _act})["action"].to(device)
        _nact_flat = _nact.reshape(_nact.shape[0], -1)  # [B, H*D]
        _dists = torch.cdist(_nact_flat, _nact_flat)     # [B, B]
        _mask = ~torch.eye(_dists.shape[0], dtype=torch.bool, device=device)
        _mean_dist = _dists[_mask].mean().item()
        _D = _nact_flat.shape[-1]
        print(f"   归一化动作: D={_D}, mean_pairwise_dist={_mean_dist:.3f}")
        print(
            f"   Drifting temp_scale={drifting_temp_scale}, drift_scale={drift_scale}"
            f" (effective={policy.drift_scale})"
        )
    del first_batch, _act, _nact, _nact_flat, _dists, _mask

    drifting_sub_K = drifting_cfg.get("sub_K", None)
    if drifting_sub_K is not None:
        drifting_sub_K = int(drifting_sub_K)

    print(
        f"\n7b. 开始训练 ({total_epochs} epochs, save_every={save_every}) ..."
        + (f"\n   resume_from_epoch={start_epoch}" if start_epoch > 0 else "")
        + f"\n   Drifting temp_scale={drifting_temp_scale}, drift_scale={drift_scale}, "
        f"drift_use_scale={drift_use_scale}, K={drifting_K}"
        + (f", sub_K={drifting_sub_K}" if drifting_sub_K else "")
        + f"\n   batch_size={batch_size}, shuffle=True"
        + f"\n   horizon={horizon}, n_obs_steps={n_obs_steps}"
    )

    if grad_accum_steps > 1:
        effective_batch = int(train_cfg["batch_size"]) * grad_accum_steps
        print(f"   Gradient accumulation: {grad_accum_steps} steps → effective batch={effective_batch}")
    grad_norm = torch.tensor(0.0)

    def _clean_state_dict(sd):
        """Strip DataParallel 'module.' prefix for clean checkpoint."""
        return {k.replace("noise_pred_net.module.", "noise_pred_net."): v
                for k, v in sd.items()}

    @torch.no_grad()
    def _update_ema():
        if ema_state is None:
            return
        for key, value in policy.state_dict().items():
            if (
                ema_state[key].device != value.device
                or ema_state[key].dtype != value.dtype
            ):
                ema_state[key] = ema_state[key].to(
                    device=value.device,
                    dtype=value.dtype,
                )
            ema_state[key].mul_(ema_decay).add_(value.detach(), alpha=1.0 - ema_decay)

    def _build_ckpt(epoch_idx, loss_value):
        ckpt = {
            "policy":        _clean_state_dict(policy.state_dict()),
            "normalizer":    policy.normalizer.state_dict(),
            "optimizer":     optimizer.state_dict(),
            "scheduler":     scheduler.state_dict(),
            "epoch":         epoch_idx,
            "config":        config,
            "loss":          loss_value,
            "best_loss":     best_loss,
            "policy_class":  "DA3FilmDriftingPolicy",
            "policy_type":   "depth_guided_film_drifting",
            "encoder_cfg":   _enc_cfg_save,
            "drifting_cfg":  drifting_cfg,
        }
        if ema_state is not None:
            ckpt["ema_policy"] = _clean_state_dict(ema_state)
            ckpt["ema_decay"] = ema_decay
        return ckpt

    if start_epoch >= total_epochs:
        print(f"[resume] start_epoch={start_epoch} 已达到 total_epochs={total_epochs}，无需继续训练。")
        return

    for epoch in range(start_epoch, total_epochs):
        diag_gen_pos_dist = None
        diag_gen_spread = None
        diag_gen_std = None

        # Optional progressive drift_scale (legacy mode).
        if not drift_use_scale:
            current_drift_scale = 1.0
        elif drift_scale_schedule == "linear":
            current_drift_scale = drift_scale - (drift_scale - drift_scale_min) * (epoch / max(1, total_epochs - 1))
        elif drift_scale_schedule == "cosine":
            import math
            current_drift_scale = drift_scale_min + 0.5 * (drift_scale - drift_scale_min) * (1 + math.cos(math.pi * epoch / max(1, total_epochs - 1)))
        else:
            current_drift_scale = drift_scale
            
        policy.drift_scale = current_drift_scale

        policy.train()
        epoch_loss = 0.0
        n_batches  = 0

        pbar = tqdm(
            dataloader,
            desc=f"Epoch {epoch+1:4d}/{total_epochs} [step_scale: {current_drift_scale:.2f}]",
            leave=False
        )
        optimizer.zero_grad(set_to_none=True)
        for step_idx, batch in enumerate(pbar):
            tokens_list = [
                batch["dino_tokens"].to(device, non_blocking=True),
                batch["da3_tokens"].to(device, non_blocking=True),
            ]
            agent_pos = batch["agent_pos"].to(device, non_blocking=True)
            action    = batch["action"].to(device, non_blocking=True)
            episode_ids = torch.tensor(batch["episode_id"], device=device)
            positive_action = None
            if "positive_action" in batch:
                positive_action = batch["positive_action"].to(device, non_blocking=True)

            loss = policy.compute_loss(
                tokens_list,
                agent_pos,
                action,
                K=drifting_K,
                sub_K=drifting_sub_K,
                positive_actions=positive_action,
            )
            scaled_loss = loss / grad_accum_steps
            scaled_loss.backward()

            if (step_idx + 1) % grad_accum_steps == 0 or (step_idx + 1) == len(dataloader):
                grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=2.0)
                optimizer.step()
                scheduler.step()
                _update_ema()
                optimizer.zero_grad(set_to_none=True)

            epoch_loss += loss.item()
            n_batches  += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "gnorm": f"{grad_norm:.3f}"})

        avg_loss = epoch_loss / max(n_batches, 1)
        lr_now = scheduler.get_last_lr()[0]
        drift_raw_rms = float(policy.drift_last_raw_rms.detach().cpu())
        drift_norm_rms = float(policy.drift_last_norm_rms.detach().cpu())
        drift_scale_ema = float(policy.drift_norm_rms_ema.detach().cpu())
        drift_step_scale = float(policy.drift_last_step_scale.detach().cpu())
        bc_last_weight = float(policy.bc_last_weight.detach().cpu())
        bc_last_pos_spread = float(policy.bc_last_pos_spread.detach().cpu())
        print(
            f"Epoch {epoch+1:4d}/{total_epochs}:  "
            f"Loss={avg_loss:.6f}  LR={lr_now:.2e}  GradNorm={grad_norm:.3f}  "
            f"DriftRawRMS={drift_raw_rms:.4f}  DriftNormRMS={drift_norm_rms:.4f}  "
            f"DriftNormEMA={drift_scale_ema:.4f}  StepScale={drift_step_scale:.4f}  "
            f"BCw={bc_last_weight:.4f}  PosSpread={bc_last_pos_spread:.4f}"
        )

        # 每 50 epoch 打印一次诊断信息
        if (epoch + 1) % 50 == 0 or epoch == 0:
            _K_diag = int(drifting_cfg.get("K", 8))
            with torch.no_grad():
                _diag_batch = next(iter(dataloader))
                _d_tokens = [
                    _diag_batch["dino_tokens"].to(device),
                    _diag_batch["da3_tokens"].to(device),
                ]
                _d_act = _diag_batch["action"].to(device)
                _d_nact = policy.normalizer.normalize({"action": _d_act})["action"].to(device)
                _d_pos = _diag_batch["agent_pos"].to(device)
                _d_npos = policy.normalizer.normalize({"agent_pos": _d_pos})["agent_pos"].to(device)
                _B_diag = _d_act.shape[0]
                fused = policy.fusion_encoder(_d_tokens)
                obs_cond = policy.obs_encoder(
                    torch.cat([fused, _d_npos], dim=-1).reshape(_B_diag, -1)
                )
                # K-sampling: generate K samples per observation
                obs_cond_k = obs_cond.repeat_interleave(_K_diag, dim=0)
                noise = torch.randn((_B_diag * _K_diag, policy.horizon, policy.action_dim), device=device)
                timestep_zero = torch.zeros(_B_diag * _K_diag, dtype=torch.long, device=device)
                gen = policy.noise_pred_net(noise, timestep_zero, global_cond=obs_cond_k)
                gen_flat = gen.reshape(_B_diag, _K_diag, -1)  # [B, K, D]
                pos_flat = _d_nact.reshape(_B_diag, -1)       # [B, D]
                # Per-observation gen→pos distance
                _gp_dist = (gen_flat - pos_flat.unsqueeze(1)).norm(dim=-1).mean()
                # Intra-observation gen spread (K samples diversity)
                _gen_spread = torch.cdist(gen_flat, gen_flat).mean()
                diag_gen_pos_dist = float(_gp_dist.detach().cpu())
                diag_gen_spread = float(_gen_spread.detach().cpu())
                diag_gen_std = float(gen_flat.std().detach().cpu())
                print(
                    f"  [诊断] B={_B_diag} K={_K_diag}  gen→pos_dist={_gp_dist:.4f}  "
                    f"intra_obs_spread={_gen_spread:.4f}  gen_std={gen_flat.std():.4f}"
                )
                del _diag_batch, _d_tokens, _d_act, _d_nact, _d_pos, _d_npos

        if wandb_run is not None:
            log_data = {
                "epoch": epoch + 1,
                "train/loss": avg_loss,
                "train/lr": lr_now,
                "train/grad_norm": float(grad_norm),
                "drift/raw_rms": drift_raw_rms,
                "drift/norm_rms": drift_norm_rms,
                "drift/norm_ema": drift_scale_ema,
                "drift/step_scale": drift_step_scale,
                "drift/temp_scale": drifting_temp_scale,
                "drift/K": drifting_K,
                "bc/weight": bc_last_weight,
                "bc/pos_spread": bc_last_pos_spread,
                "train/batch_size": batch_size,
                "train/grad_accum_steps": grad_accum_steps,
            }
            if diag_gen_pos_dist is not None:
                log_data["diag/gen_pos_dist"] = diag_gen_pos_dist
            if diag_gen_spread is not None:
                log_data["diag/intra_obs_spread"] = diag_gen_spread
            if diag_gen_std is not None:
                log_data["diag/gen_std"] = diag_gen_std
            wandb_run.log(log_data, step=epoch + 1)

        if (epoch + 1) % save_every == 0:
            ckpt = _build_ckpt(epoch + 1, avg_loss)
            ckpt_path = save_dir / f"{epoch+1}.ckpt"
            torch.save(ckpt, ckpt_path)
            print(f"  → Saved: {ckpt_path}")
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(ckpt, save_dir / "best.ckpt")
                print(f"  → New best!")
                if wandb_run is not None:
                    wandb_run.log({"best/loss": best_loss, "best/epoch": epoch + 1}, step=epoch + 1)

    final_ckpt = _build_ckpt(total_epochs, avg_loss)
    torch.save(final_ckpt, save_dir / f"{total_epochs}.ckpt")
    print(f"\n最终 ckpt: {save_dir}/{total_epochs}.ckpt")
    print(f"训练完成! 最佳 Loss: {best_loss:.6f}")
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
