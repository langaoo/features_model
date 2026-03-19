#!/usr/bin/env python3
"""tools/depth_guided_film_online/train_film_online.py

半离线 2 模型 DA3-FiLM 训练 (DINOv3 + DA3) — 快速版
=====================================================

核心流程:
  1. 加载 DINOv3 + DA3 两个 backbone (完全冻结, no_grad)
  2. 从 zarr 批量读 RGB → 一次性提取所有 tokens → 缓存到 CPU RAM
  3. 训练时直接从缓存读 Tensor, num_workers 可开多个, 无在线提取开销
  4. Diffusion Policy loss → 反向传播 (只训练 FiLM encoder + DP head)

与原 "在线" 版的区别:
  - 不再逐 batch 在线提取 (消除 DA3 临时文件写盘 + 逐 iter 前向瓶颈)
  - 预缓存后每 iter 只有纯 Tensor GPU 操作 (~0.05s vs ~3.5s)
  - zarr 已含 head_camera uint8 图像, 不需要再读 HDF5
  - backbone 显式 requires_grad_(False) + eval(), 节省显存

用法:
  python tools/depth_guided_film_online/train_film_online.py \\
    --config configs/depth_guided_film_online/train_film_online.yaml
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import argparse
import sys
import os
from tqdm import tqdm
import warnings
import yaml
import numpy as np
import zarr

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

# ---- 正版 DP ----
HAS_OFFICIAL_DP = False
try:
    DP_OUTER = REPO_ROOT / "third_party" / "DP" / "diffusion_policy"
    if DP_OUTER.exists():
        sys.path.insert(0, str(DP_OUTER))
    from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
    from diffusion_policy.model.common.normalizer import LinearNormalizer
    HAS_OFFICIAL_DP = True
    print("[INFO] 正版 DP 已加载")
except ImportError as e:
    print(f"[ERROR] 正版 DP 导入失败: {e}")
    sys.exit(1)

from features_common.depth_guided_film_online.encoder_film_2model import DA3Film2ModelEncoder
from features_common.depth_guided_film_online.extractors_2model import TwoModelExtractors


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
        run_name = f"film_online_{task_name}"

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


# ============================================================
# Policy
# ============================================================

class DA3Film2ModelPolicy(nn.Module):
    """2 模型 DA3-FiLM + proprio + Diffusion Policy."""

    def __init__(
        self,
        fusion_encoder: DA3Film2ModelEncoder,
        proprio_dim: int = 14,
        action_dim: int = 14,
        horizon: int = 8,
        n_obs_steps: int = 3,
        n_action_steps: int = 6,          # ★ 对齐 robot_dp_14.yaml
        num_inference_steps: int = 100,
    ):
        super().__init__()
        self.fusion_encoder = fusion_encoder
        self.proprio_dim = proprio_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.num_inference_steps = num_inference_steps

        self.normalizer = LinearNormalizer()

        per_step_dim = fusion_encoder.out_dim + proprio_dim
        obs_encoder_dim = per_step_dim * n_obs_steps

        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_encoder_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.noise_pred_net = ConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=256,
            diffusion_step_embed_dim=128,
            down_dims=[256, 512, 1024],
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
        )
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=100,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )

    def compute_loss(self, tokens_list, agent_pos, actions):
        """
        tokens_list: List[2 × Tensor [B, To, K_i, C_i]]
        agent_pos:   [B, To, 14]
        actions:     [B, horizon, 14]
        """
        B = actions.shape[0]
        device = actions.device
        nactions   = self.normalizer.normalize({"action": actions})["action"].to(device)
        nagent_pos = self.normalizer.normalize({"agent_pos": agent_pos})["agent_pos"].to(device)

        fused = self.fusion_encoder(tokens_list)          # [B, To, out_dim]
        obs_cond = self.obs_encoder(
            torch.cat([fused, nagent_pos], dim=-1).reshape(B, -1)
        )
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (B,), device=device
        ).long()
        noise = torch.randn_like(nactions)
        noisy = self.noise_scheduler.add_noise(nactions, noise, timesteps)
        noise_pred = self.noise_pred_net(noisy, timesteps, global_cond=obs_cond)
        return nn.functional.mse_loss(noise_pred, noise)

    @torch.no_grad()
    def predict_action(self, tokens_list, agent_pos=None):
        B = tokens_list[0].shape[0]
        device = tokens_list[0].device
        fused = self.fusion_encoder(tokens_list)
        if agent_pos is not None:
            nagent_pos = self.normalizer.normalize({"agent_pos": agent_pos})["agent_pos"].to(device)
            obs_combined = torch.cat([fused, nagent_pos], dim=-1)
        else:
            obs_combined = torch.cat(
                [fused, torch.zeros(B, fused.shape[1], self.proprio_dim, device=device)], dim=-1
            )
        obs_cond = self.obs_encoder(obs_combined.reshape(B, -1))
        action = torch.randn((B, self.horizon, self.action_dim), device=device)
        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        for t in self.noise_scheduler.timesteps:
            noise_pred = self.noise_pred_net(
                action, t.unsqueeze(0).expand(B).to(device), global_cond=obs_cond
            )
            action = self.noise_scheduler.step(noise_pred, t, action).prev_sample
        return self.normalizer.unnormalize({"action": action})["action"]


# ============================================================
# 预缓存: 一次性提取所有帧的 tokens
# ============================================================

def _precompute_tokens(
    zarr_path: str,
    camera_name: str,
    extractors: TwoModelExtractors,
    max_tokens: int,
    batch_size: int = 32,
) -> tuple:
    """
    从 zarr head_camera 批量提取 DINOv3 + DA3 tokens, 缓存到 CPU RAM (float16).

    Returns:
        dino_tokens: [N, K_d, 768]  float16 CPU
        da3_tokens:  [N, K_a, 2048] float16 CPU
    """
    from PIL import Image

    z = zarr.open(zarr_path, "r")
    imgs_np = z[f"data/{camera_name}"][:]   # [N, 3, H, W] uint8
    N = imgs_np.shape[0]
    print(f"  [precompute] {N} 帧图像 {imgs_np.shape}, batch={batch_size} ...")

    all_dino, all_da3 = [], []
    for i in tqdm(range(0, N, batch_size), desc="  预提取 tokens", leave=False):
        chunk = imgs_np[i : i + batch_size]              # [B, 3, H, W] uint8
        pil_list = [
            Image.fromarray(chunk[j].transpose(1, 2, 0))  # CHW→HWC RGB
            for j in range(chunk.shape[0])
        ]
        with torch.no_grad():
            tokens = extractors.extract_batch_tokens(
                pil_list, max_tokens=max_tokens, return_torch=True
            )
        all_dino.append(tokens[0].half().cpu())
        all_da3.append(tokens[1].half().cpu())

    dino_all = torch.cat(all_dino, dim=0)   # [N, K_d, 768]
    da3_all  = torch.cat(all_da3,  dim=0)   # [N, K_a, 2048]
    mem_gb = (dino_all.nbytes + da3_all.nbytes) / 1e9
    print(f"  [precompute] DINOv3 {tuple(dino_all.shape)}, DA3 {tuple(da3_all.shape)}, "
          f"共 {mem_gb:.2f} GB (float16 CPU)")
    return dino_all, da3_all


# ============================================================
# 缓存数据集 (训练时无任何 backbone 调用)
# ============================================================

class CachedTokenDataset(Dataset):
    """
    从预缓存的 token Tensor + zarr state/action 构建滑窗训练样本.
    训练时完全不调用 backbone, 每个 __getitem__ 只是 Tensor 切片.

    每个样本:
        dino_tokens:  [To, K_d, 768]  float16  (训练时转 float32)
        da3_tokens:   [To, K_a, 2048] float16
        agent_pos:    [To, 14]        float32
        action:       [horizon, 14]   float32
    """

    def __init__(
        self,
        dino_tokens:   torch.Tensor,   # [N, K_d, 768]
        da3_tokens:    torch.Tensor,   # [N, K_a, 2048]
        actions:       np.ndarray,     # [N, 14]
        states:        np.ndarray,     # [N, 14]
        episode_ends:  np.ndarray,     # [n_ep] exclusive end 索引
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
        """构建起始帧索引列表, 保证窗口不跨 episode."""
        indices = []
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
            "dino_tokens": self.dino_tokens[obs_slice].float(),  # [To, K_d, 768]
            "da3_tokens":  self.da3_tokens[obs_slice].float(),   # [To, K_a, 2048]
            "agent_pos":   self.states[obs_slice],               # [To, 14]
            "action":      self.actions[act_slice],              # [horizon, 14]
        }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="半离线 2 模型 DA3-FiLM 训练")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    print("=" * 65)
    print("半离线 2 模型 DA3-FiLM 训练 (DINOv3 + DA3) — 快速版")
    print("=" * 65)

    gpu_ids  = config["device"]["gpu_ids"]
    gpu_id   = gpu_ids[0] if isinstance(gpu_ids, list) else int(gpu_ids)
    device   = f"cuda:{gpu_id}"
    print(f"Device: {device}")

    task_name = config["data"]["tasks"]
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
        raise FileNotFoundError(
            f"Zarr not found: {zarr_path}\n"
            "请先运行 collect_data.sh 或 pkl2zarr 生成 zarr 数据."
        )
    print(f"Zarr: {zarr_path}")

    # ---- 1. 加载 backbone (完全冻结) ----
    print("\n1. 加载 DINOv3 + DA3 backbone (冻结)...")
    extractors = TwoModelExtractors(gpu_id=gpu_id)
    for m in [extractors.dinov3_model, extractors.da3_model]:
        m.requires_grad_(False)
        m.eval()
    print("   ✓ 两个 backbone 已冻结 (requires_grad=False, eval mode)")

    # ---- 2. 预提取所有帧 tokens ----
    print("\n2. 预提取所有帧 tokens (一次性, 之后训练不再调用 backbone)...")
    precompute_bs = int(config.get("precompute_batch_size", 32))
    dino_tokens, da3_tokens = _precompute_tokens(
        zarr_path, camera_name, extractors, max_tokens, batch_size=precompute_bs,
    )
    # 提取完释放 backbone 显存
    if config.get("offload_backbone_after_precompute", True):
        extractors.dinov3_model.cpu()
        extractors.da3_model.cpu()
        torch.cuda.empty_cache()
        print("   ✓ Backbone 已转到 CPU, GPU 显存已释放")

    # ---- 3. 读 zarr state/action/episode_ends ----
    print("\n3. 读取 state / action / episode_ends ...")
    z               = zarr.open(zarr_path, "r")
    actions_np      = z["data/action"][:]        # [N, 14]
    states_np       = z["data/state"][:]         # [N, 14]
    episode_ends_np = z["meta/episode_ends"][:]  # [n_ep]
    print(f"   总帧数: {len(actions_np)}, episodes: {len(episode_ends_np)}")

    # ---- 4. 数据集 + DataLoader ----
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
    print(f"   batch={train_cfg['batch_size']}, "
          f"num_workers={train_cfg.get('num_workers', 4)}, "
          f"batches/epoch={len(dataloader)}")

    # ---- 5. 模型 ----
    print("\n5. 创建 DA3Film2ModelEncoder + Policy ...")
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

    policy = DA3Film2ModelPolicy(
        fusion_encoder=fusion_encoder,
        proprio_dim=14,
        action_dim=14,
        horizon=horizon,
        n_obs_steps=n_obs_steps,
        n_action_steps=n_action_steps,
        num_inference_steps=int(config.get("policy", {}).get("num_inference_steps", 100)),
    ).to(device)

    total_params   = sum(p.numel() for p in policy.parameters()) / 1e6
    encoder_params = sum(p.numel() for p in fusion_encoder.parameters()) / 1e6
    print(f"   模型总参数: {total_params:.2f}M  (FiLM encoder: {encoder_params:.2f}M)")

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
    wandb_run = _init_wandb(config, save_dir=save_dir)
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

    print(f"\n7. 开始训练 ({total_epochs} epochs, save_every={save_every}) ...")
    print(f"   horizon={horizon}, n_obs_steps={n_obs_steps}, n_action_steps={n_action_steps}")

    for epoch in range(total_epochs):
        policy.train()
        epoch_loss = 0.0
        n_batches  = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1:4d}/{total_epochs}", leave=False)
        for batch in pbar:
            tokens_list = [
                batch["dino_tokens"].to(device, non_blocking=True),  # [B, To, K_d, 768]
                batch["da3_tokens"].to(device, non_blocking=True),   # [B, To, K_a, 2048]
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
        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": epoch + 1,
                    "train/loss": avg_loss,
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/batch_size": int(train_cfg["batch_size"]),
                },
                step=epoch + 1,
            )

        if (epoch + 1) % save_every == 0:
            ckpt = {
                "policy":       policy.state_dict(),
                "normalizer":   policy.normalizer.state_dict(),
                "optimizer":    optimizer.state_dict(),
                "epoch":        epoch + 1,
                "config":       config,
                "loss":         avg_loss,
                "policy_class": "DA3Film2ModelPolicy",
                "policy_type":  "depth_guided_film_online_2model",
                "encoder_cfg":  _enc_cfg_save,
            }
            ckpt_path = save_dir / f"{epoch+1}.ckpt"
            torch.save(ckpt, ckpt_path)
            print(f"  → Saved: {ckpt_path}")
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(ckpt, save_dir / "best.ckpt")
                print(f"  → New best!")
                if wandb_run is not None:
                    wandb_run.log({"best/loss": best_loss, "best/epoch": epoch + 1}, step=epoch + 1)

    final_ckpt = {
        "policy":       policy.state_dict(),
        "normalizer":   policy.normalizer.state_dict(),
        "epoch":        total_epochs,
        "config":       config,
        "loss":         avg_loss,
        "policy_class": "DA3Film2ModelPolicy",
        "policy_type":  "depth_guided_film_online_2model",
        "encoder_cfg":  _enc_cfg_save,
    }
    torch.save(final_ckpt, save_dir / f"{total_epochs}.ckpt")
    print(f"\n最终 ckpt: {save_dir}/{total_epochs}.ckpt")
    print(f"训练完成! 最佳 Loss: {best_loss:.6f}")
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
