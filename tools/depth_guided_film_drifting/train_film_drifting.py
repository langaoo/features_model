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

import yaml
import zarr
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

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

# 本项目模块
from features_common.depth_guided_film_online.extractors_2model import TwoModelExtractors
from features_common.depth_guided_film_online.encoder_film_2model import DA3Film2ModelEncoder
from features_common.depth_guided_film_drifting.policy_drifting import DA3FilmDriftingPolicy


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
    ):
        super().__init__()
        self.dino_tokens = dino_tokens
        self.da3_tokens  = da3_tokens
        self.actions     = torch.from_numpy(actions).float()
        self.states      = torch.from_numpy(states).float()
        self.horizon     = horizon
        self.n_obs_steps = n_obs_steps
        self.indices     = self._build_indices(episode_ends)
        print(f"  [Dataset] {len(self.indices)} 个训练样本")

    def _build_indices(self, episode_ends):
        indices  = []
        ep_start = 0
        for ep_end in episode_ends:
            ep_len = int(ep_end) - ep_start
            max_t  = ep_len - self.n_obs_steps - self.horizon + 1
            for t in range(max(0, max_t)):
                indices.append(ep_start + t)
            ep_start = int(ep_end)
        return indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t0        = self.indices[idx]
        obs_slice = slice(t0, t0 + self.n_obs_steps)
        act_start = t0 + self.n_obs_steps
        act_slice = slice(act_start, act_start + self.horizon)
        return {
            "dino_tokens": self.dino_tokens[obs_slice].float(),
            "da3_tokens":  self.da3_tokens[obs_slice].float(),
            "agent_pos":   self.states[obs_slice],
            "action":      self.actions[act_slice],
        }


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
    gpu_id   = gpu_ids[0] if isinstance(gpu_ids, list) else int(gpu_ids)
    device   = f"cuda:{gpu_id}"
    print(f"Device: {device}")

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
    dataset = CachedTokenDataset(
        dino_tokens=dino_tokens,
        da3_tokens=da3_tokens,
        actions=actions_np,
        states=states_np,
        episode_ends=episode_ends_np,
        horizon=horizon,
        n_obs_steps=n_obs_steps,
    )

    train_cfg  = config["train"]
    dataloader = DataLoader(
        dataset,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 4)),
        pin_memory=True,
        drop_last=True,
        persistent_workers=int(train_cfg.get("num_workers", 4)) > 0,
    )
    print(
        f"   batch={train_cfg['batch_size']}, "
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
    policy = DA3FilmDriftingPolicy(
        fusion_encoder=fusion_encoder,
        proprio_dim=14,
        action_dim=14,
        horizon=horizon,
        n_obs_steps=n_obs_steps,
        n_action_steps=n_action_steps,
        drifting_temp=float(drifting_cfg.get("temp", 0.05)),
        hidden_dim=int(drifting_cfg.get("hidden_dim", 1024)),
    ).to(device)

    total_params   = sum(p.numel() for p in policy.parameters()) / 1e6
    encoder_params = sum(p.numel() for p in fusion_encoder.parameters()) / 1e6
    gen_params     = sum(p.numel() for p in policy.action_generator.parameters()) / 1e6
    print(
        f"   模型总参数: {total_params:.2f}M  "
        f"(FiLM encoder: {encoder_params:.2f}M, "
        f"DriftingGenerator: {gen_params:.2f}M)"
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
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg.get("weight_decay", 1e-6)),
        betas=(0.95, 0.999),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_epochs, eta_min=float(train_cfg["lr"]) * 0.01
    )

    # ---- 8. 训练循环 ----
    save_dir   = Path(config["checkpoint"]["save_dir"])
    save_every = int(config["checkpoint"].get("save_every", 100))
    save_dir.mkdir(parents=True, exist_ok=True)
    best_loss  = float("inf")
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

    print(
        f"\n7. 开始训练 ({total_epochs} epochs, save_every={save_every}) ..."
        f"\n   Drifting temp={drifting_cfg.get('temp', 0.05)}"
        f"\n   horizon={horizon}, n_obs_steps={n_obs_steps}"
    )

    for epoch in range(total_epochs):
        policy.train()
        epoch_loss = 0.0
        n_batches  = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1:4d}/{total_epochs}", leave=False)
        for batch in pbar:
            tokens_list = [
                batch["dino_tokens"].to(device, non_blocking=True),
                batch["da3_tokens"].to(device, non_blocking=True),
            ]
            agent_pos = batch["agent_pos"].to(device, non_blocking=True)
            action    = batch["action"].to(device, non_blocking=True)

            loss = policy.compute_loss(tokens_list, agent_pos, action)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches  += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        print(
            f"Epoch {epoch+1:4d}/{total_epochs}:  "
            f"Loss={avg_loss:.6f}  LR={scheduler.get_last_lr()[0]:.2e}"
        )

        if (epoch + 1) % save_every == 0:
            ckpt = {
                "policy":        policy.state_dict(),
                "normalizer":    policy.normalizer.state_dict(),
                "optimizer":     optimizer.state_dict(),
                "epoch":         epoch + 1,
                "config":        config,
                "loss":          avg_loss,
                "policy_class":  "DA3FilmDriftingPolicy",
                "policy_type":   "depth_guided_film_drifting",
                "encoder_cfg":   _enc_cfg_save,
                "drifting_cfg":  drifting_cfg,
            }
            ckpt_path = save_dir / f"{epoch+1}.ckpt"
            torch.save(ckpt, ckpt_path)
            print(f"  → Saved: {ckpt_path}")
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(ckpt, save_dir / "best.ckpt")
                print(f"  → New best!")

    final_ckpt = {
        "policy":        policy.state_dict(),
        "normalizer":    policy.normalizer.state_dict(),
        "epoch":         total_epochs,
        "config":        config,
        "loss":          avg_loss,
        "policy_class":  "DA3FilmDriftingPolicy",
        "policy_type":   "depth_guided_film_drifting",
        "encoder_cfg":   _enc_cfg_save,
        "drifting_cfg":  drifting_cfg,
    }
    torch.save(final_ckpt, save_dir / f"{total_epochs}.ckpt")
    print(f"\n最终 ckpt: {save_dir}/{total_epochs}.ckpt")
    print(f"训练完成! 最佳 Loss: {best_loss:.6f}")


if __name__ == "__main__":
    main()
