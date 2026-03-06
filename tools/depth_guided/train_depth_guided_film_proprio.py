#!/usr/bin/env python3
"""
tools/depth_guided/train_depth_guided_film_proprio.py

DA3-FiLM Depth-Guided Fusion + proprio 训练 (离线版)
=====================================================
改进动机
--------
原 DepthGuidedFusionEncoder Cross-Attention ~14M 参数, 50 条 demo 过拟合, 实测 71%.
本脚本改用 DA3FilmFusionEncoder (~1.5M 参数):
  - DA3 几何 mean_pool -> FiLM MLP -> scale/shift 调制语义 tokens
  - 初始化为恒等, 训练稳定
  - 语义完整性更好

与 train_depth_guided_offline_proprio.py 的区别
  - DepthGuidedFusionEncoder -> DA3FilmFusionEncoder
  - proj_dim 256 (vs 原 512), 参数更少
  - Dataset / collate_fn / Normalizer / DP head 完全相同
"""
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import argparse, sys, os
from tqdm import tqdm
import warnings, yaml
from typing import Dict, Any, List, Tuple
import numpy as np
import zarr
import h5py

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

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
    print(f"[WARNING] 正版 DP 导入失败: {e}")
    sys.exit(1)

from features_common.depth_guided.encoder_film import DA3FilmFusionEncoder


# ============================================================
# Policy
# ============================================================

class DA3FilmPolicy(nn.Module):
    """DA3-FiLM Fusion + proprio + Diffusion Policy."""

    def __init__(self, fusion_encoder, proprio_dim=14, action_dim=14,
                 horizon=8, n_obs_steps=3, num_inference_steps=100):
        super().__init__()
        if not HAS_OFFICIAL_DP:
            raise RuntimeError("正版 DP 未加载")

        self.fusion_encoder = fusion_encoder
        self.proprio_dim = proprio_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
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
        B = actions.shape[0]
        device = actions.device
        nactions   = self.normalizer.normalize({"action": actions})["action"].to(device)
        nagent_pos = self.normalizer.normalize({"agent_pos": agent_pos})["agent_pos"].to(device)
        fused      = self.fusion_encoder(tokens_list)
        obs_cond   = self.obs_encoder(
            torch.cat([fused, nagent_pos], dim=-1).reshape(B, -1)
        )
        timesteps    = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (B,), device=device).long()
        noise        = torch.randn_like(nactions)
        noisy_actions = self.noise_scheduler.add_noise(nactions, noise, timesteps)
        noise_pred   = self.noise_pred_net(noisy_actions, timesteps, global_cond=obs_cond)
        return nn.functional.mse_loss(noise_pred, noise)

    @torch.no_grad()
    def predict_action(self, tokens_list, agent_pos=None):
        B = tokens_list[0].shape[0]
        device = tokens_list[0].device
        fused  = self.fusion_encoder(tokens_list)
        if agent_pos is not None:
            nagent_pos = self.normalizer.normalize({"agent_pos": agent_pos})["agent_pos"].to(device)
            obs_combined = torch.cat([fused, nagent_pos], dim=-1)
        else:
            obs_combined = torch.cat([fused, torch.zeros(B, fused.shape[1], self.proprio_dim, device=device)], dim=-1)
        obs_cond = self.obs_encoder(obs_combined.reshape(B, -1))
        action   = torch.randn((B, self.horizon, self.action_dim), device=device)
        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        for t in self.noise_scheduler.timesteps:
            noise_pred = self.noise_pred_net(action, t.unsqueeze(0).expand(B).to(device), global_cond=obs_cond)
            action = self.noise_scheduler.step(noise_pred, t, action).prev_sample
        return self.normalizer.unnormalize({"action": action})["action"]


# ============================================================
# Dataset  (与 train_depth_guided_offline_proprio.py 完全相同)
# ============================================================

class DepthGuidedDataset(Dataset):
    def __init__(self, vis_zarr_roots, robotwin_data_root, task_name, task_config,
                 horizon=8, n_obs_steps=3, expert_data_num=50, camera_name="head_camera",
                 max_tokens=None, zarr_expert_num=None):
        super().__init__()
        assert len(vis_zarr_roots) == 4
        self.vis_zarr_roots = vis_zarr_roots
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.task_name = task_name
        self.task_config = task_config
        self.camera_name = camera_name
        self.expert_data_num = expert_data_num
        self.max_tokens = max_tokens
        _zarr_num = zarr_expert_num if zarr_expert_num is not None else expert_data_num
        self.raw_data_root = os.path.join(robotwin_data_root, task_name, task_config, "data")
        self.episodes = [f"episode_{i}" for i in range(expert_data_num)]
        self.zarr_subdir = f"{task_name}-{task_config}-{_zarr_num}_sapien_{camera_name}"
        print(f"[DA3Film] 收集数据样本...")
        self.samples: List[Tuple] = self._collect_samples()
        print(f"[DA3Film] 共 {len(self.samples)} 个样本")

    def _load_vector(self, ep_num):
        hdf5_path = os.path.join(self.raw_data_root, f"episode{ep_num}.hdf5")
        with h5py.File(hdf5_path, "r") as f:
            if "joint_action/vector" in f:
                vec = f["joint_action/vector"][:]
            else:
                left = f["joint_action/left_arm"][:]
                right = f["joint_action/right_arm"][:]
                lg = f["joint_action/left_gripper"][:] if "joint_action/left_gripper" in f else np.zeros((left.shape[0], 1))
                rg = f["joint_action/right_gripper"][:] if "joint_action/right_gripper" in f else np.zeros((right.shape[0], 1))
                if lg.ndim == 1: lg = lg[:, None]
                if rg.ndim == 1: rg = rg[:, None]
                vec = np.concatenate([left, lg, right, rg], axis=-1)
        return torch.from_numpy(vec).float()

    def _load_episode_tokens(self, episode):
        tokens_by_model = []
        for mi, zarr_root in enumerate(self.vis_zarr_roots):
            zarr_path = os.path.join(zarr_root, self.zarr_subdir, f"{episode}.zarr")
            z = zarr.open(zarr_path, mode="r")
            pf = z["per_frame_features"]
            if pf.ndim == 5:
                T_frames, ws, Hf, Wf, C = pf.shape
                arr = pf[:, 0, :, :, :].reshape(T_frames, Hf * Wf, C)
            elif pf.ndim == 4:
                T_frames, Hf, Wf, C = pf.shape
                arr = pf[:].reshape(T_frames, Hf * Wf, C)
            elif pf.ndim == 3:
                arr = pf[:]
            else:
                raise ValueError(f"未知 zarr 维度: {pf.ndim}")
            t = torch.from_numpy(arr.astype(np.float32))
            if self.max_tokens is not None and t.shape[1] > self.max_tokens:
                idx = torch.linspace(0, t.shape[1] - 1, self.max_tokens).long()
                t = t[:, idx, :]
            tokens_by_model.append(t)
        return tokens_by_model

    def _collect_samples(self):
        samples = []
        for ep_idx, ep in enumerate(self.episodes):
            try:
                if ep_idx % 10 == 0:
                    print(f"  {ep_idx}/{len(self.episodes)}: {ep}")
                ep_num = int(ep.split("_")[1])
                tokens_list = self._load_episode_tokens(ep)
                vector = self._load_vector(ep_num)
                T = min(tokens_list[0].shape[0], len(vector))
                states  = vector[:T-1]
                actions = vector[1:T]
                toks = [t[:T-1] for t in tokens_list]
                T_eff = T - 1
                if T_eff < self.n_obs_steps + self.horizon:
                    continue
                for t_idx in range(T_eff - self.n_obs_steps - self.horizon + 1):
                    obs_toks   = [tok[t_idx:t_idx + self.n_obs_steps] for tok in toks]
                    obs_state  = states[t_idx:t_idx + self.n_obs_steps]
                    act_window = actions[t_idx + self.n_obs_steps:t_idx + self.n_obs_steps + self.horizon]
                    samples.append((obs_toks, obs_state, act_window))
            except Exception as e:
                print(f"  跳过 {ep}: {e}")
                continue
        return samples

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        obs_toks, agent_pos, action = self.samples[idx]
        return {"obs_tokens": obs_toks, "agent_pos": agent_pos, "action": action}


def collate_fn(batch):
    B = len(batch)
    n_models = len(batch[0]["obs_tokens"])
    obs_tokens_batched = [
        torch.stack([batch[b]["obs_tokens"][mi] for b in range(B)], dim=0)
        for mi in range(n_models)
    ]
    agent_pos = torch.stack([batch[b]["agent_pos"] for b in range(B)], dim=0)
    action    = torch.stack([batch[b]["action"]    for b in range(B)], dim=0)
    return {"obs_tokens": obs_tokens_batched, "agent_pos": agent_pos, "action": action}


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="DA3-FiLM Depth-Guided 训练")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    print("=" * 65)
    print("DA3-FiLM Depth-Guided Fusion + proprio 训练 (离线版)")
    print("=" * 65)

    gpu_ids = config["device"]["gpu_ids"]
    gpu_id  = gpu_ids[0] if isinstance(gpu_ids, list) else int(gpu_ids)
    device  = f"cuda:{gpu_id}"
    print(f"Device: {device}")

    task_name   = config["data"]["tasks"]
    if isinstance(task_name, list): task_name = task_name[0]
    task_config = config["data"].get("task_config", "demo_clean")
    expert_num  = config.get("checkpoint", {}).get("expert_data_num", 50)
    data_root   = config["data"].get("robotwin_data_root", "/home/gl/RoboTwin/data")
    horizon     = int(config["data"]["horizon"])
    n_obs_steps = int(config["data"]["n_obs_steps"])
    max_tokens  = config.get("encoder", {}).get("max_tokens", 196)

    # 1. 数据集
    print("\n1. 创建数据集...")
    dataset = DepthGuidedDataset(
        vis_zarr_roots=config["data"]["vis_zarr_roots"],
        robotwin_data_root=data_root,
        task_name=task_name,
        task_config=task_config,
        horizon=horizon,
        n_obs_steps=n_obs_steps,
        expert_data_num=expert_num,
        camera_name=config["data"].get("camera_name", "head_camera"),
        max_tokens=max_tokens,
        zarr_expert_num=config["checkpoint"].get("zarr_expert_num", expert_num),
    )
    print(f"Dataset: {len(dataset)} 样本")

    dataloader = DataLoader(
        dataset, batch_size=config["train"]["batch_size"],
        shuffle=True, num_workers=config["train"].get("num_workers", 4),
        pin_memory=True, drop_last=True, collate_fn=collate_fn,
    )

    # 2. 模型
    print("\n2. 创建 DA3-FiLM Encoder + Policy...")
    enc_cfg = config.get("encoder", {})
    fusion_encoder = DA3FilmFusionEncoder(
        semantic_in_dims=tuple(enc_cfg.get("semantic_in_dims", [1024, 2048, 768])),
        geometric_in_dim=int(enc_cfg.get("geometric_in_dim", 2048)),
        proj_dim=int(enc_cfg.get("proj_dim", 256)),
        film_hidden=int(enc_cfg.get("film_hidden", 256)),
        out_dim=int(enc_cfg.get("out_dim", 1280)),
        semantic_fusion=str(enc_cfg.get("semantic_fusion", "concat_proj")),
        with_pos_enc=bool(enc_cfg.get("with_pos_enc", True)),
        dropout=float(enc_cfg.get("dropout", 0.1)),
        max_tokens=int(enc_cfg.get("max_tokens", 196)),
    ).to(device)

    policy = DA3FilmPolicy(
        fusion_encoder=fusion_encoder, proprio_dim=14, action_dim=14,
        horizon=horizon, n_obs_steps=n_obs_steps,
        num_inference_steps=int(config["policy"].get("num_inference_steps", 100)),
    ).to(device)

    total_params   = sum(p.numel() for p in policy.parameters()) / 1e6
    encoder_params = sum(p.numel() for p in fusion_encoder.parameters()) / 1e6
    print(f"模型总参数: {total_params:.2f}M  (encoder: {encoder_params:.2f}M)")

    # 3. Fit normalizer
    print("\n3. Fit normalizer...")
    all_actions   = torch.stack([s[2] for s in dataset.samples])
    all_agent_pos = torch.stack([s[1] for s in dataset.samples])
    policy.normalizer.fit(
        {"action": all_actions, "agent_pos": all_agent_pos},
        last_n_dims=1, mode="limits", output_min=-1.0, output_max=1.0,
    )
    try:
        policy.normalizer.to(device)
    except Exception:
        pass

    diff = (dataset.samples[0][1][0] - dataset.samples[0][2][0]).abs().mean().item()
    print(f"  state[0] vs action[0] 差异={diff:.6f} ({'正确' if diff > 1e-4 else '警告: 时间偏移可能有问题'})")

    # 4. 优化器
    train_cfg    = config["train"]
    total_epochs = int(train_cfg["epochs"])
    optimizer    = torch.optim.AdamW(
        policy.parameters(), lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg.get("weight_decay", 1e-6)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_epochs, eta_min=float(train_cfg["lr"]) * 0.01
    )

    # 5. 训练
    print("\n4. 开始训练...")
    save_dir   = Path(config["checkpoint"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    save_every = int(config["checkpoint"].get("save_every", 100))
    best_loss  = float("inf")
    avg_loss   = float("inf")

    _enc_cfg_save = {
        "type": "film",
        "semantic_in_dims": list(enc_cfg.get("semantic_in_dims", [1024, 2048, 768])),
        "geometric_in_dim": int(enc_cfg.get("geometric_in_dim", 2048)),
        "proj_dim": int(enc_cfg.get("proj_dim", 256)),
        "film_hidden": int(enc_cfg.get("film_hidden", 256)),
        "out_dim": int(enc_cfg.get("out_dim", 1280)),
        "semantic_fusion": str(enc_cfg.get("semantic_fusion", "concat_proj")),
        "with_pos_enc": bool(enc_cfg.get("with_pos_enc", True)),
        "dropout": float(enc_cfg.get("dropout", 0.1)),
        "max_tokens": int(enc_cfg.get("max_tokens", 196)),
    }

    for epoch in range(total_epochs):
        policy.train()
        epoch_loss = 0.0
        n_batches  = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{total_epochs}", leave=False)
        for batch in pbar:
            tokens_list = [t.to(device, non_blocking=True) for t in batch["obs_tokens"]]
            agent_pos   = batch["agent_pos"].to(device, non_blocking=True)
            action      = batch["action"].to(device, non_blocking=True)
            loss = policy.compute_loss(tokens_list, agent_pos, action)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches  += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        print(f"Epoch {epoch+1:4d}/{total_epochs}: Loss={avg_loss:.6f}  LR={scheduler.get_last_lr()[0]:.2e}")

        if (epoch + 1) % save_every == 0:
            ckpt = {
                "policy": policy.state_dict(),
                "normalizer": policy.normalizer.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "config": config,
                "loss": avg_loss,
                "policy_class": "DA3FilmPolicy",
                "policy_type": "depth_guided_film",
                "encoder_cfg": _enc_cfg_save,
            }
            ckpt_path = save_dir / f"{epoch+1}.ckpt"
            torch.save(ckpt, ckpt_path)
            print(f"  Saved: {ckpt_path}")
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(ckpt, save_dir / "best.ckpt")
                print(f"  Best: {save_dir}/best.ckpt")

    # 最终保存
    final_ckpt = {
        "policy": policy.state_dict(),
        "normalizer": policy.normalizer.state_dict(),
        "epoch": total_epochs - 1,
        "config": config,
        "loss": avg_loss,
        "policy_class": "DA3FilmPolicy",
        "policy_type": "depth_guided_film",
        "encoder_cfg": _enc_cfg_save,
    }
    final_path = save_dir / f"{total_epochs}.ckpt"
    torch.save(final_ckpt, final_path)
    print(f"\n最终 ckpt: {final_path}")
    print(f"训练完成! 最佳 Loss: {best_loss:.6f}")


if __name__ == "__main__":
    main()
