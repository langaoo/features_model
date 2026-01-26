#!/usr/bin/env python3
"""
离线Head训练脚本 - 读取预提取特征
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import argparse
import sys
import os
from tqdm import tqdm
import logging
import yaml
import numpy as np
import time
import zarr

# 关闭日志
logging.basicConfig(level=logging.ERROR)
os.environ['PYTHONWARNINGS'] = 'ignore'

# 添加路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from path_config import DP_ROOT

# 导入正版DP
HAS_OFFICIAL_DP = False
try:
    DP_OUTER = DP_ROOT
    if DP_OUTER.exists() and (DP_OUTER / "diffusion_policy").exists():
        sys.path.insert(0, str(DP_OUTER))
        from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
        from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
        from diffusion_policy.model.common.normalizer import LinearNormalizer  # ✅ 新增
        HAS_OFFICIAL_DP = True
    else:
        print(f"[WARNING] DP not found at {DP_OUTER}")
except ImportError as e:
    print(f"[WARNING] DP import failed: {e}")


class DPRGBPolicy(nn.Module):
    """正版Diffusion Policy"""
    def __init__(self, obs_dim, action_dim, horizon, n_obs_steps, n_action_steps, num_inference_steps=100):
        super().__init__()
        if not HAS_OFFICIAL_DP:
            raise RuntimeError("DP not loaded")
        
        # ✅ 新增: 使用LinearNormalizer
        self.normalizer = LinearNormalizer()
        
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 512),
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
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon'
        )
        self.num_inference_steps = num_inference_steps
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps

    def compute_loss(self, obs, action_gt):
        """
        训练时的loss计算
        Args:
            obs: [B, To, D] 观测特征
            action_gt: [B, Ta, A] ground truth动作 (未归一化)
        """
        B = obs.shape[0]
        device = obs.device
        
        # ✅ 使用normalizer归一化action (模仿RoboTwin原版DP)
        # 归一化结果可能在CPU上，显式转到训练device
        nactions = self.normalizer['action'].normalize(action_gt).to(device)
        
        obs_flat = obs.reshape(B, -1)
        obs_cond = self.obs_encoder(obs_flat)
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (B,), device=device).long()
        noise = torch.randn(nactions.shape, device=device)
        noisy_actions = self.noise_scheduler.add_noise(nactions, noise, timesteps)
        noise_pred = self.noise_pred_net(noisy_actions, timesteps, global_cond=obs_cond)
        return nn.functional.mse_loss(noise_pred, noise)


class OfflineFeatureDataset(Dataset):
    def __init__(self, dataset_dir, tasks, horizon=8, n_obs_steps=2):
        self.dataset_dir = Path(dataset_dir)
        self.tasks = tasks
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        
        self.samples = [] # (task, ep_name, start_idx, zarr_path)
        
        for task in tasks:
            task_dir = self.dataset_dir / task
            if not task_dir.exists():
                print(f"[Dataset] Warning: {task_dir} not found")
                continue
            
            zarr_files = sorted(list(task_dir.glob("*.zarr")))
            for zf in zarr_files:
                try:
                    # 使用 mode='r' 只读打开，快速读取 attributes
                    root = zarr.open(str(zf), mode='r')
                    n_frames = root.attrs['num_frames']
                    ep_name = zf.stem
                    
                    # 生成样本索引
                    valid_starts = max(0, n_frames - (horizon + n_obs_steps) + 1)
                    for i in range(valid_starts):
                        self.samples.append((task, ep_name, i, str(zf)))
                        
                except Exception as e:
                    print(f"Error reading {zf}: {e}")

        print(f"[Dataset] Loaded {len(self.samples)} samples from {len(tasks)} tasks")
        
        # 缓存打开的zarr文件句柄 (path -> root)
        # 注意：多进程 DataLoader 下不能共享 zarr 句柄，需要在 getitem 或 worker_init 打开
        # 为简单起见，这里每次 open (Zarr cache 机制会帮忙) 或者使用 chunk store
        # 实际上 zarr.open 开销很小
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        task, ep_name, start_idx, zpath = self.samples[idx]
        
        # 打开 Zarr (Read Only)
        # 建议使用 Cache
        root = zarr.open(zpath, mode='r')
        
        # 读取 Obs [T, 1280]
        # 需要读取 start_idx 到 start_idx + n_obs_steps
        # 注意边界处理
        # 实际上 start_idx 是当前步。观测是历史。
        # 通常 obs_steps=2 意味着 [Current, Current-1] 还是 [Current, Current+1]?
        # 在 OnlineDataset 中：
        # for i in range(start_idx, start_idx + n_obs_steps): ...
        # 这意味着观测 feature 是未来的？ 不，start_idx 是时间轴上的点。
        # 如果 start_idx 是 t，那么 obs 是 t, t+1。 action 是 t...t+H。
        # 这是一个常见的误区。但我们要和 Train Online 对齐。
        # train_online: range(start_idx, start_idx + self.n_obs_steps)
        # 所以是 [t, t+1] (如果 n_obs=2)
        
        obs_indices = []
        n_frames = root.attrs['num_frames']
        for i in range(start_idx, start_idx + self.n_obs_steps):
            # Clamp to max frame
            idx_clamped = min(i, n_frames - 1)
            obs_indices.append(idx_clamped)
        
        # 读取数据 (Zarr 支持切片，但不一定支持非连续索引高效读取，这里范围很小所以还好)
        # 如果是连续的：
        slice_start = start_idx
        slice_end = min(start_idx + self.n_obs_steps, n_frames)
        obs_data = root['obs_aligned'][slice_start:slice_end]
        
        # 如果不够长 Pad last
        if obs_data.shape[0] < self.n_obs_steps:
             pad_len = self.n_obs_steps - obs_data.shape[0]
             last_frame = obs_data[-1:]
             obs_data = np.concatenate([obs_data, np.tile(last_frame, (pad_len, 1))], axis=0)
             
        # Action [T, 14]
        # action 从 obs 窗口之后开始
        act_slice_start = start_idx + self.n_obs_steps
        act_slice_end = min(act_slice_start + self.horizon, n_frames)
        action_data = root['action'][act_slice_start:act_slice_end]
        
        if action_data.shape[0] < self.horizon:
            pad_len = self.horizon - action_data.shape[0]
            last_act = action_data[-1:]
            action_data = np.concatenate([action_data, np.tile(last_act, (pad_len, 1))], axis=0)
            
        return {
            'obs': torch.from_numpy(obs_data).float(),     # [To, 1280]
            'action': torch.from_numpy(action_data).float() # [Ta, 14]
        }


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    
    config = load_config(args.config)
    gpu_ids = config['device']['gpu_ids']
    device = torch.device(f"cuda:{gpu_ids[0]}")
    
    # 1. Dataset
    print("Creating Datasets...")
    dataset = OfflineFeatureDataset(
        dataset_dir=config['data']['features_dataset_dir'],
        tasks=config['data']['tasks'],
        horizon=config['data']['horizon'],
        n_obs_steps=config['data']['n_obs_steps']
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config['train']['batch_size'],
        shuffle=True,
        num_workers=config['train']['num_workers'],
        pin_memory=True,
        prefetch_factor=2 if config['train']['num_workers']>0 else None
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # 2. Policy
    # 确定 Action Dim
    # 从数据集中拿一个样本看
    sample = dataset[0]
    action_dim = sample['action'].shape[-1]
    obs_dim = config['data']['n_obs_steps'] * 1280
    
    print(f"Obs Dim: {obs_dim}, Action Dim: {action_dim}")
    
    policy = DPRGBPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        horizon=config['data']['horizon'],
        n_obs_steps=config['data']['n_obs_steps'],
        n_action_steps=config['data']['horizon'],
        num_inference_steps=config['policy']['num_inference_steps']
    ).to(device)
    
    # ✅ 新增: Fit normalizer (模仿RoboTwin原版DP)
    print("Fitting normalizer...")
    all_actions = []
    for i in range(min(len(dataset), 1000)):  # 使用前1000个样本统计
        all_actions.append(dataset[i]['action'])
    all_actions = torch.stack(all_actions)  # [N, Ta, A]
    policy.normalizer.fit(
        {'action': all_actions},
        last_n_dims=1,
        mode='limits',
        output_min=-1.0,
        output_max=1.0
    )
    print(f"  Action stats: min={policy.normalizer['action'].params_dict['input_stats'].min[:3]}...")
    print(f"  Action stats: max={policy.normalizer['action'].params_dict['input_stats'].max[:3]}...")
    
    # 3. Optimizer
    lr_value = float(config['train']['lr'])
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr_value)
    
    # 4. Train Loop
    print("Starting Offline Training...")
    best_loss = float('inf')
    
    global_step = 0
    for epoch in range(config['train']['epochs']):
        policy.train()
        epoch_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}", ncols=100)
        
        for batch in pbar:
            try:
                obs = batch['obs'].to(device)
                action = batch['action'].to(device)
                
                loss = policy.compute_loss(obs, action)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            except RuntimeError as e:
                if 'out of memory' in str(e).lower() and device.type == 'cuda':
                    print("[WARNING] CUDA OOM, switching to CPU for training.")
                    torch.cuda.empty_cache()
                    device = torch.device('cpu')
                    policy = policy.to(device)
                    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr_value)

                    obs = batch['obs'].to(device)
                    action = batch['action'].to(device)
                    loss = policy.compute_loss(obs, action)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                else:
                    raise
            
            epoch_loss += loss.item()
            global_step += 1
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} done. Avg Loss: {avg_loss:.4f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            
        # Save Checkpoint
        if (epoch+1) % config['output']['save_every_n_epochs'] == 0:
            task_name = config['data']['tasks'][0]
            ckpt_setting = config['checkpoint']['ckpt_setting']
            num = config['checkpoint']['expert_data_num']
            seed = config['checkpoint']['seed']
            
            # checkpoints/{task}-{setting}-{num}-{seed}/{epoch}.ckpt
            save_dir_name = f"{task_name}-{ckpt_setting}-{num}-{seed}"
            output_root = config.get('output', {}).get('dir')
            if output_root:
                save_path = Path(output_root) / save_dir_name
            else:
                save_path = Path(__file__).parent.parent / "checkpoints" / save_dir_name
            save_path.mkdir(parents=True, exist_ok=True)
            
            ckpt_file = save_path / f"{epoch+1}.ckpt"
            
            torch.save({
                'policy': policy.state_dict(),
                'normalizer': policy.normalizer.state_dict(),  # ✅ 保存normalizer
                'optimizer': optimizer.state_dict(),
                'epoch': epoch+1,
                'config': config,
                'loss': avg_loss
            }, ckpt_file)
            print(f"Saved checkpoint: {ckpt_file}")

if __name__ == "__main__":
    main()
