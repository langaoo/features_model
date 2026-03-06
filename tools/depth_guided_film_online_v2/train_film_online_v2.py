#!/usr/bin/env python3
"""tools/depth_guided_film_online_v2/train_film_online_v2.py

半离线 2 模型 DA3-FiLM v2 训练 (DINOv3 + DA3, AttentionPooling)
=================================================================

与 v1 (train_film_online.py) 的核心区别:
  1. 使用 DA3Film2ModelEncoderV2 (AttentionPooling 替代 mean_pool)
  2. DA3 tokens 保留更多 (max_geo_tokens=256, 比 v1 的 196 多)
  3. DINOv3 tokens 仍为 max_sem_tokens=196
  4. checkpoint encoder_cfg 中 type="film_2model_v2"

核心流程 (与 v1 相同):
  1. 加载 DINOv3 + DA3 两个 backbone (完全冻结, no_grad)
  2. 从 zarr 批量读 RGB → 一次性提取所有 tokens → 缓存到 CPU RAM
  3. 训练时直接从缓存读 Tensor, num_workers 可开多个, 无在线提取开销
  4. Diffusion Policy loss → 反向传播 (只训练 FiLM encoder v2 + DP head)

用法:
  python tools/depth_guided_film_online_v2/train_film_online_v2.py \\
    --config configs/depth_guided_film_online_v2/train_film_online_v2.yaml
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

# ---- v2 Encoder (AttentionPooling 替代 mean_pool) ----
from features_common.depth_guided_film_online_v2.encoder_film_2model_v2 import DA3Film2ModelEncoderV2
# ---- 共用 backbone 提取器 (与 v1 相同, 无需修改) ----
from features_common.depth_guided_film_online.extractors_2model import TwoModelExtractors


# ============================================================
# Policy (与 v1 完全相同, 只是 fusion_encoder 类型不同)
# ============================================================

class DA3Film2ModelPolicyV2(nn.Module):
    """2 模型 DA3-FiLM v2 + proprio + Diffusion Policy."""

    def __init__(
        self,
        fusion_encoder: DA3Film2ModelEncoderV2,
        proprio_dim: int = 14,
        action_dim: int = 14,
        horizon: int = 8,
        n_obs_steps: int = 3,
        n_action_steps: int = 6,
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
          tokens_list[0]: DINOv3 [B, To, K_sem, 768]
          tokens_list[1]: DA3    [B, To, K_geo, 2048]
        agent_pos:   [B, To, 14]
        actions:     [B, horizon, 14]
        """
        B = actions.shape[0]
        device = actions.device

        norm_inputs = self.normalizer.normalize(
            {"action": actions, "agent_pos": agent_pos}
        )
        norm_actions  = norm_inputs["action"]
        norm_agent_pos = norm_inputs["agent_pos"]

        fused = self.fusion_encoder(tokens_list)  # [B, To, out_dim]
        obs_combined = torch.cat([fused, norm_agent_pos], dim=-1)  # [B, To, out_dim+14]
        obs_flat = obs_combined.reshape(B, -1)
        obs_cond = self.obs_encoder(obs_flat)     # [B, 256]

        noise = torch.randn_like(norm_actions)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (B,), device=device
        ).long()
        noisy_actions = self.noise_scheduler.add_noise(norm_actions, noise, timesteps)

        noise_pred = self.noise_pred_net(noisy_actions, timesteps, global_cond=obs_cond)
        loss = nn.functional.mse_loss(noise_pred, noise)
        return loss

    @torch.no_grad()
    def predict_action(self, tokens_list, agent_pos=None):
        B = tokens_list[0].shape[0]
        device = tokens_list[0].device
        fused = self.fusion_encoder(tokens_list)  # [B, To, out_dim]
        if agent_pos is not None:
            nagent_pos = (
                self.normalizer.normalize({"agent_pos": agent_pos})["agent_pos"].to(device)
            )
            obs_combined = torch.cat([fused, nagent_pos], dim=-1)
        else:
            zeros = torch.zeros(B, fused.shape[1], self.proprio_dim, device=device)
            obs_combined = torch.cat([fused, zeros], dim=-1)

        obs_flat = obs_combined.reshape(B, -1)
        obs_cond = self.obs_encoder(obs_flat)

        action = torch.randn((B, self.horizon, self.action_dim), device=device)
        self.noise_scheduler.set_timesteps(self.num_inference_steps)

        for t in self.noise_scheduler.timesteps:
            noise_pred = self.noise_pred_net(
                action, t.unsqueeze(0).expand(B).to(device), global_cond=obs_cond
            )
            action = self.noise_scheduler.step(noise_pred, t, action).prev_sample

        return self.normalizer.unnormalize({"action": action})["action"]


# ============================================================
# 预提取 tokens (注意: DA3 使用 max_geo_tokens, DINOv3 使用 max_sem_tokens)
# ============================================================

def _precompute_tokens(
    zarr_path: str,
    camera_name: str,
    extractors: TwoModelExtractors,
    max_sem_tokens: int,
    max_geo_tokens: int,
    batch_size: int = 32,
):
    """从 zarr 中读取所有 RGB 帧, 一次性提取 DINOv3 + DA3 tokens.

    Args:
        max_sem_tokens: DINOv3 保留的 token 数 (default=196, 即 14×14 全部)
        max_geo_tokens: DA3 保留的 token 数 (default=256; v1 限制在 196,
                        v2 允许更多以保留更多空间信息)

    Returns:
        dino_all: CPU float16 Tensor [N, max_sem_tokens, 768]
        da3_all:  CPU float16 Tensor [N, max_geo_tokens, 2048]
    """
    from PIL import Image
    z = zarr.open(zarr_path, "r")
    imgs = z[f"data/{camera_name}"][:]  # [N, H, W, 3] uint8
    N = len(imgs)
    print(f"   共 {N} 帧, 分 batch={batch_size} 提取...")

    dino_list = []
    da3_list  = []

    for start in tqdm(range(0, N, batch_size), desc="precompute tokens"):
        end = min(start + batch_size, N)
        # imgs: [N, 3, H, W] uint8 (CHW) → 转换为 PIL (HWC)
        pil_imgs = [
            Image.fromarray(imgs[i].transpose(1, 2, 0))  # CHW → HWC
            for i in range(start, end)
        ]

        # 一次性提取全量 tokens (不限制数量, 避免两次 backbone 调用)
        tokens_torch = extractors.extract_batch_tokens(
            pil_imgs, max_tokens=None, return_torch=True
        )
        # tokens_torch[0]: [bs, K_dino_full, 768]
        # tokens_torch[1]: [bs, K_da3_full,  2048]

        # 用 linspace 等间隔采样 (与 deploy 时 extractors._subsample_tokens 一致)
        dino_t = extractors._subsample_tokens(tokens_torch[0], max_sem_tokens).half().cpu()
        da3_t  = extractors._subsample_tokens(tokens_torch[1], max_geo_tokens).half().cpu()

        dino_list.append(dino_t)
        da3_list.append(da3_t)

    dino_all = torch.cat(dino_list, dim=0)  # [N, max_sem_tokens, 768]
    da3_all  = torch.cat(da3_list,  dim=0)  # [N, max_geo_tokens, 2048]

    sem_gb  = dino_all.numel() * 2 / 1e9
    geo_gb  = da3_all.numel()  * 2 / 1e9
    print(f"   DINOv3 cache: {dino_all.shape}  {sem_gb:.2f} GB")
    print(f"   DA3    cache: {da3_all.shape}   {geo_gb:.2f} GB")
    return dino_all, da3_all


# ============================================================
# Dataset
# ============================================================

class CachedTokenDataset(Dataset):
    """基于缓存 tokens 的滑动窗口数据集 (与 v1 相同逻辑)."""

    def __init__(
        self,
        dino_tokens: torch.Tensor,  # [N, K_sem, 768]  float16
        da3_tokens:  torch.Tensor,  # [N, K_geo, 2048] float16
        actions:     np.ndarray,    # [N, 14]
        states:      np.ndarray,    # [N, 14]
        episode_ends: np.ndarray,   # [n_ep]
        horizon: int = 8,
        n_obs_steps: int = 3,
    ):
        self.dino_tokens  = dino_tokens
        self.da3_tokens   = da3_tokens
        self.actions      = torch.from_numpy(actions).float()
        self.states       = torch.from_numpy(states).float()
        self.horizon      = horizon
        self.n_obs_steps  = n_obs_steps

        # 构建合法采样 indices (episode 边界内)
        self.valid_starts = []
        prev_end = 0
        for ep_end in episode_ends:
            ep_len = ep_end - prev_end
            for s in range(prev_end, ep_end):
                # obs 窗口: [s - n_obs_steps + 1, s]
                # action 窗口: [s, s + horizon - 1]
                if s - n_obs_steps + 1 >= prev_end and s + horizon <= ep_end:
                    self.valid_starts.append(s)
            prev_end = ep_end

        print(f"   Dataset: {len(self.valid_starts)} 有效样本")

    def __len__(self):
        return len(self.valid_starts)

    def __getitem__(self, idx):
        s = self.valid_starts[idx]
        obs_start = s - self.n_obs_steps + 1

        # obs 窗口
        dino = self.dino_tokens[obs_start : s + 1].float()   # [To, K_sem, 768]
        da3  = self.da3_tokens[obs_start : s + 1].float()    # [To, K_geo, 2048]

        agent_pos = self.states[obs_start : s + 1]   # [To, 14]
        action    = self.actions[s : s + self.horizon]  # [horizon, 14]

        return {
            "dino_tokens": dino,
            "da3_tokens":  da3,
            "agent_pos":   agent_pos,
            "action":      action,
        }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="YAML config path")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    print("=" * 65)
    print("半离线 2 模型 DA3-FiLM v2 训练 (AttentionPooling)")
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
    camera_name    = config["data"].get("camera_name", "head_camera")

    # v2: 区分 sem/geo token 数量
    enc_cfg = config.get("encoder", {})
    max_sem_tokens = int(enc_cfg.get("max_sem_tokens", 196))
    max_geo_tokens = int(enc_cfg.get("max_geo_tokens", 256))

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
    print(f"\n2. 预提取所有帧 tokens "
          f"(DINOv3 max={max_sem_tokens}, DA3 max={max_geo_tokens})...")
    precompute_bs = int(config.get("precompute_batch_size", 32))
    dino_tokens, da3_tokens = _precompute_tokens(
        zarr_path, camera_name, extractors,
        max_sem_tokens=max_sem_tokens,
        max_geo_tokens=max_geo_tokens,
        batch_size=precompute_bs,
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

    # ---- 5. 模型 (v2 Encoder with AttentionPooling) ----
    print("\n5. 创建 DA3Film2ModelEncoderV2 + Policy ...")
    fusion_encoder = DA3Film2ModelEncoderV2(
        semantic_in_dim=int(enc_cfg.get("semantic_in_dim", 768)),
        geometric_in_dim=int(enc_cfg.get("geometric_in_dim", 2048)),
        proj_dim=int(enc_cfg.get("proj_dim", 256)),
        film_hidden=int(enc_cfg.get("film_hidden", 256)),
        out_dim=int(enc_cfg.get("out_dim", 1280)),
        with_pos_enc=bool(enc_cfg.get("with_pos_enc", True)),
        dropout=float(enc_cfg.get("dropout", 0.1)),
        max_sem_tokens=max_sem_tokens,
        max_geo_tokens=max_geo_tokens,
    ).to(device)

    policy = DA3Film2ModelPolicyV2(
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
    print(f"   模型总参数: {total_params:.2f}M  (FiLM encoder v2: {encoder_params:.2f}M)")

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

    # v2: encoder_cfg 中写入新类型和分离的 token 数
    _enc_cfg_save = {
        "type":             "film_2model_v2",
        "semantic_in_dim":  int(enc_cfg.get("semantic_in_dim", 768)),
        "geometric_in_dim": int(enc_cfg.get("geometric_in_dim", 2048)),
        "proj_dim":         int(enc_cfg.get("proj_dim", 256)),
        "film_hidden":      int(enc_cfg.get("film_hidden", 256)),
        "out_dim":          int(enc_cfg.get("out_dim", 1280)),
        "with_pos_enc":     bool(enc_cfg.get("with_pos_enc", True)),
        "dropout":          float(enc_cfg.get("dropout", 0.1)),
        "max_sem_tokens":   max_sem_tokens,
        "max_geo_tokens":   max_geo_tokens,
    }

    print(f"\n7. 开始训练 ({total_epochs} epochs, save_every={save_every}) ...")
    print(f"   horizon={horizon}, n_obs_steps={n_obs_steps}, n_action_steps={n_action_steps}")
    print(f"   AttentionPooling: DA3 {max_geo_tokens} tokens, DINOv3 {max_sem_tokens} tokens")

    for epoch in range(total_epochs):
        policy.train()
        epoch_loss = 0.0
        n_batches  = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1:4d}/{total_epochs}", leave=False)
        for batch in pbar:
            tokens_list = [
                batch["dino_tokens"].to(device, non_blocking=True),  # [B, To, K_sem, 768]
                batch["da3_tokens"].to(device, non_blocking=True),   # [B, To, K_geo, 2048]
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
                "policy":       policy.state_dict(),
                "normalizer":   policy.normalizer.state_dict(),
                "optimizer":    optimizer.state_dict(),
                "epoch":        epoch + 1,
                "config":       config,
                "loss":         avg_loss,
                "policy_class": "DA3Film2ModelPolicyV2",
                "policy_type":  "depth_guided_film_online_2model_v2",
                "encoder_cfg":  _enc_cfg_save,
            }
            ckpt_path = save_dir / f"{epoch+1}.ckpt"
            torch.save(ckpt, ckpt_path)
            print(f"  → Saved: {ckpt_path}")
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(ckpt, save_dir / "best.ckpt")
                print(f"  → New best!")

    final_ckpt = {
        "policy":       policy.state_dict(),
        "normalizer":   policy.normalizer.state_dict(),
        "epoch":        total_epochs,
        "config":       config,
        "loss":         avg_loss,
        "policy_class": "DA3Film2ModelPolicyV2",
        "policy_type":  "depth_guided_film_online_2model_v2",
        "encoder_cfg":  _enc_cfg_save,
    }
    torch.save(final_ckpt, save_dir / f"{total_epochs}.ckpt")
    print(f"\n最终 ckpt: {save_dir}/{total_epochs}.ckpt")
    print(f"训练完成! 最佳 Loss: {best_loss:.6f}")


if __name__ == "__main__":
    main()
