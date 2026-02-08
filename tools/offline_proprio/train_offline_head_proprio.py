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
    """正版Diffusion Policy (可选本体感知)"""
    def __init__(
        self,
        obs_dim,
        action_dim,
        horizon,
        n_obs_steps,
        n_action_steps,
        num_inference_steps=100,
        use_proprio: bool = False,
        proprio_dim: int = 14,
        proprio_mode: str = "concat",
        proprio_hidden: int = 256,
    ):
        super().__init__()
        if not HAS_OFFICIAL_DP:
            raise RuntimeError("DP not loaded")
        
        # ✅ 新增: 使用LinearNormalizer
        self.normalizer = LinearNormalizer()
        
        self.use_proprio = bool(use_proprio)
        self.proprio_mode = str(proprio_mode)

        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        if self.use_proprio:
            self.proprio_encoder = nn.Sequential(
                nn.Linear(int(proprio_dim) * int(n_obs_steps), proprio_hidden),
                nn.ReLU(),
                nn.Linear(proprio_hidden, 256),
                nn.ReLU(),
            )
            if self.proprio_mode == "concat":
                self.proprio_fuse = nn.Sequential(
                    nn.Linear(256 + 256, 256),
                    nn.ReLU(),
                )
            elif self.proprio_mode != "add":
                print(f"[Warning] unknown proprio_mode={self.proprio_mode}, fallback to 'concat'")
                self.proprio_mode = "concat"
                self.proprio_fuse = nn.Sequential(
                    nn.Linear(256 + 256, 256),
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

    def compute_loss(self, obs, action_gt, agent_pos=None):
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
        if self.use_proprio and agent_pos is not None:
            if hasattr(self, "normalizer") and "agent_pos" in self.normalizer.params_dict:
                agent_pos = self.normalizer["agent_pos"].normalize(agent_pos).to(device)
            proprio_flat = agent_pos.reshape(B, -1)
            proprio_cond = self.proprio_encoder(proprio_flat)
            if self.proprio_mode == "add":
                obs_cond = obs_cond + proprio_cond
            else:
                obs_cond = self.proprio_fuse(torch.cat([obs_cond, proprio_cond], dim=-1))
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (B,), device=device).long()
        noise = torch.randn(nactions.shape, device=device)
        noisy_actions = self.noise_scheduler.add_noise(nactions, noise, timesteps)
        noise_pred = self.noise_pred_net(noisy_actions, timesteps, global_cond=obs_cond)
        return nn.functional.mse_loss(noise_pred, noise)


class DPRGBDualStreamPolicy(nn.Module):
    """Dual-stream DP: global + token cross-attention conditioning (可选本体感知)."""
    def __init__(
        self,
        obs_dim,
        token_dim,
        action_dim,
        horizon,
        n_obs_steps,
        n_action_steps,
        num_inference_steps=100,
        token_dropout=0.0,
        ctx_dropout=0.0,
        token_gate_init=-4.0,
        use_proprio: bool = False,
        proprio_dim: int = 14,
        proprio_mode: str = "concat",
        proprio_hidden: int = 256,
    ):
        super().__init__()
        if not HAS_OFFICIAL_DP:
            raise RuntimeError("DP not loaded")

        self.normalizer = LinearNormalizer()
        self.use_proprio = bool(use_proprio)
        self.proprio_mode = str(proprio_mode)
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        if self.use_proprio:
            self.proprio_encoder = nn.Sequential(
                nn.Linear(int(proprio_dim) * int(n_obs_steps), proprio_hidden),
                nn.ReLU(),
                nn.Linear(proprio_hidden, 256),
                nn.ReLU(),
            )
            if self.proprio_mode == "concat":
                self.proprio_fuse = nn.Sequential(
                    nn.Linear(256 + 256, 256),
                    nn.ReLU(),
                )
            elif self.proprio_mode != "add":
                print(f"[Warning] unknown proprio_mode={self.proprio_mode}, fallback to 'concat'")
                self.proprio_mode = "concat"
                self.proprio_fuse = nn.Sequential(
                    nn.Linear(256 + 256, 256),
                    nn.ReLU(),
                )
        self.token_proj = nn.Linear(token_dim, 256)
        self.query_proj = nn.Linear(256, 256)
        self.cross_attn = nn.MultiheadAttention(256, num_heads=8, batch_first=True)
        self.token_dropout = nn.Dropout(float(token_dropout)) if float(token_dropout) > 0 else nn.Identity()
        self.ctx_dropout = nn.Dropout(float(ctx_dropout)) if float(ctx_dropout) > 0 else nn.Identity()
        self.token_gate = nn.Parameter(torch.tensor(float(token_gate_init)))

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
        self.n_obs_steps = n_obs_steps
        self.use_normalizer = False

    def _build_cond(self, obs_global, obs_tokens, force_gate: float = None, agent_pos=None):
        """
        构建条件特征
        Args:
            obs_global: [B, To, D] global特征
            obs_tokens: [B, To, K, D] token特征
            force_gate: 如果指定，强制使用该gate值（用于训练初期）
        """
        B = obs_global.shape[0]
        obs_flat = obs_global.reshape(B, -1)
        global_cond = self.obs_encoder(obs_flat)
        if self.use_proprio and agent_pos is not None:
            if hasattr(self, "normalizer") and "agent_pos" in self.normalizer.params_dict:
                agent_pos = self.normalizer["agent_pos"].normalize(agent_pos).to(obs_global.device)
            proprio_flat = agent_pos.reshape(B, -1)
            proprio_cond = self.proprio_encoder(proprio_flat)
            if self.proprio_mode == "add":
                global_cond = global_cond + proprio_cond
            else:
                global_cond = self.proprio_fuse(torch.cat([global_cond, proprio_cond], dim=-1))
        tokens = obs_tokens.reshape(B, -1, obs_tokens.shape[-1])
        tokens = self.token_proj(tokens)
        tokens = self.token_dropout(tokens)
        query = self.query_proj(global_cond).unsqueeze(1)
        ctx, _ = self.cross_attn(query, tokens, tokens)
        ctx = self.ctx_dropout(ctx.squeeze(1))
        
        # 🔧 修复: 支持强制gate值，防止gate塌缩
        if force_gate is not None:
            gate = force_gate
        else:
            gate = torch.sigmoid(self.token_gate)
        return global_cond + gate * ctx

    def compute_loss(self, obs_global, obs_tokens, action_gt, agent_pos=None,
                     force_gate: float = None,
                     gate_regularization: float = 0.0):
        """
        计算训练损失
        Args:
            force_gate: 强制gate值（训练初期使用0.5强制token分支学习）
            gate_regularization: gate正则化系数，鼓励gate保持在合理范围
        """
        B = obs_global.shape[0]
        device = obs_global.device

        nactions = self.normalizer['action'].normalize(action_gt).to(device)
        obs_cond = self._build_cond(obs_global, obs_tokens, force_gate=force_gate, agent_pos=agent_pos)

        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (B,), device=device).long()
        noise = torch.randn(nactions.shape, device=device)
        noisy_actions = self.noise_scheduler.add_noise(nactions, noise, timesteps)
        noise_pred = self.noise_pred_net(noisy_actions, timesteps, global_cond=obs_cond)
        
        mse_loss = nn.functional.mse_loss(noise_pred, noise)
        
        # 🔧 新增: Gate正则化，防止gate塌缩到0
        if gate_regularization > 0:
            # 鼓励gate sigmoid在[0.3, 0.7]范围内
            gate_sigmoid = torch.sigmoid(self.token_gate)
            # 惩罚gate偏离0.5
            gate_reg_loss = gate_regularization * (gate_sigmoid - 0.5).pow(2)
            return mse_loss + gate_reg_loss
        
        return mse_loss

    def forward(self, obs_global, obs_tokens, agent_pos=None):
        """推理：生成动作序列 [B, Ta, A]。"""
        B = obs_global.shape[0]
        device = obs_global.device
        obs_cond = self._build_cond(obs_global, obs_tokens, agent_pos=agent_pos)

        action = torch.randn((B, self.n_action_steps, self.action_dim), device=device)
        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        for t in self.noise_scheduler.timesteps:
            noise_pred = self.noise_pred_net(
                action,
                t.unsqueeze(0).expand(B).to(device),
                global_cond=obs_cond,
            )
            action = self.noise_scheduler.step(noise_pred, t, action).prev_sample

        if self.use_normalizer:
            action = self.normalizer.unnormalize({'action': action})['action']
        return action


class OfflineFeatureDataset(Dataset):
    def __init__(
        self,
        dataset_dir,
        tasks,
        horizon=8,
        n_obs_steps=2,
        use_proprio=False,
        action_offset=0,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.tasks = tasks
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.use_proprio = bool(use_proprio)
        # action_offset=1 表示用 t+1 的动作作为监督，避免 agent_pos 与 action 同步导致的捷径
        self.action_offset = int(action_offset)
        
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
                    effective_frames = max(0, int(n_frames) - self.action_offset)
                    valid_starts = max(0, effective_frames - (horizon + n_obs_steps) + 1)
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
        # 训练对齐推理：使用“历史帧”窗口 [t-n_obs_steps+1, ..., t]
        n_frames = root.attrs['num_frames']
        obs_end = min(start_idx + 1, n_frames)
        obs_start = max(0, obs_end - self.n_obs_steps)

        # 读取数据 (Zarr 支持切片，这里使用连续切片)
        obs_data = root['obs_aligned'][obs_start:obs_end]
        obs_tokens = root['obs_tokens'][obs_start:obs_end] if 'obs_tokens' in root else None
        agent_pos = None
        if self.use_proprio and 'agent_pos' in root:
            agent_pos = root['agent_pos'][obs_start:obs_end]
        
        # 如果不够长，前向补齐（复制第一帧），保持时序从旧到新
        if obs_data.shape[0] < self.n_obs_steps:
            pad_len = self.n_obs_steps - obs_data.shape[0]
            first_frame = obs_data[:1]
            obs_data = np.concatenate([np.tile(first_frame, (pad_len, 1)), obs_data], axis=0)
            if obs_tokens is not None:
                first_tok = obs_tokens[:1]
                obs_tokens = np.concatenate([np.tile(first_tok, (pad_len, 1, 1)), obs_tokens], axis=0)
            if agent_pos is not None:
                first_ap = agent_pos[:1]
                agent_pos = np.concatenate([np.tile(first_ap, (pad_len, 1)), agent_pos], axis=0)
             
        # Action [T, 14]
        # action 从当前帧开始，预测未来 horizon 步
        act_slice_start = start_idx + self.action_offset
        act_slice_end = min(act_slice_start + self.horizon, n_frames)
        action_data = root['action'][act_slice_start:act_slice_end]
        
        if action_data.shape[0] < self.horizon:
            pad_len = self.horizon - action_data.shape[0]
            last_act = action_data[-1:]
            action_data = np.concatenate([action_data, np.tile(last_act, (pad_len, 1))], axis=0)
            
        payload = {
            'obs': torch.from_numpy(obs_data).float(),     # [To, 1280]
            'action': torch.from_numpy(action_data).float() # [Ta, 14]
        }
        if obs_tokens is not None:
            payload['obs_tokens'] = torch.from_numpy(obs_tokens).float()  # [To, K, 1280]
        if agent_pos is not None:
            payload['agent_pos'] = torch.from_numpy(agent_pos).float()  # [To, D]
        return payload


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
    use_proprio = bool(config.get('policy', {}).get('use_proprio', False))
    proprio_dim = int(config.get('policy', {}).get('proprio_dim', 14))
    proprio_mode = str(config.get('policy', {}).get('proprio_mode', 'concat'))
    proprio_hidden = int(config.get('policy', {}).get('proprio_hidden', 256))
    
    # 1. Dataset
    print("Creating Datasets...")
    dataset = OfflineFeatureDataset(
        dataset_dir=config['data']['features_dataset_dir'],
        tasks=config['data']['tasks'],
        horizon=config['data']['horizon'],
        n_obs_steps=config['data']['n_obs_steps'],
        use_proprio=use_proprio,
        action_offset=int(config.get('data', {}).get('action_offset', 0)),
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
    use_tokens = bool(config.get('policy', {}).get('use_tokens', False)) and 'obs_tokens' in sample
    if use_proprio and 'agent_pos' not in sample:
        print("[Warning] use_proprio=True but agent_pos not found in dataset. Fallback to pure visual.")
        use_proprio = False
    
    print(f"Obs Dim: {obs_dim}, Action Dim: {action_dim}")
    
    if use_tokens:
        token_dim = sample['obs_tokens'].shape[-1]
        token_dropout = float(config.get('policy', {}).get('token_dropout', 0.0))
        ctx_dropout = float(config.get('policy', {}).get('ctx_dropout', 0.0))
        token_gate_init = float(config.get('policy', {}).get('token_gate_init', -4.0))
        policy = DPRGBDualStreamPolicy(
            obs_dim=obs_dim,
            token_dim=token_dim,
            action_dim=action_dim,
            horizon=config['data']['horizon'],
            n_obs_steps=config['data']['n_obs_steps'],
            n_action_steps=config['data']['horizon'],
            num_inference_steps=config['policy']['num_inference_steps'],
            token_dropout=token_dropout,
            ctx_dropout=ctx_dropout,
            token_gate_init=token_gate_init,
            use_proprio=use_proprio,
            proprio_dim=proprio_dim,
            proprio_mode=proprio_mode,
            proprio_hidden=proprio_hidden,
        ).to(device)
    else:
        policy = DPRGBPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            horizon=config['data']['horizon'],
            n_obs_steps=config['data']['n_obs_steps'],
            n_action_steps=config['data']['horizon'],
            num_inference_steps=config['policy']['num_inference_steps'],
            use_proprio=use_proprio,
            proprio_dim=proprio_dim,
            proprio_mode=proprio_mode,
            proprio_hidden=proprio_hidden,
        ).to(device)
    
    # ✅ 新增: Fit normalizer (模仿RoboTwin原版DP)
    print("Fitting normalizer...")
    all_actions = []
    all_agent_pos = [] if use_proprio else None
    for i in range(min(len(dataset), 1000)):  # 使用前1000个样本统计
        sample_i = dataset[i]
        all_actions.append(sample_i['action'])
        if use_proprio and 'agent_pos' in sample_i:
            all_agent_pos.append(sample_i['agent_pos'])
    all_actions = torch.stack(all_actions)  # [N, Ta, A]
    norm_data = {'action': all_actions}
    if use_proprio and all_agent_pos:
        all_agent_pos = torch.stack(all_agent_pos)  # [N, To, D]
        norm_data['agent_pos'] = all_agent_pos
    policy.normalizer.fit(
        norm_data,
        last_n_dims=1,
        mode='limits',
        output_min=-1.0,
        output_max=1.0
    )
    print(f"  Action stats: min={policy.normalizer['action'].params_dict['input_stats'].min[:3]}...")
    print(f"  Action stats: max={policy.normalizer['action'].params_dict['input_stats'].max[:3]}...")
    if use_proprio and "agent_pos" in policy.normalizer.params_dict:
        ap_stats = policy.normalizer['agent_pos'].params_dict['input_stats']
        print(f"  Agent_pos stats: min={ap_stats.min[:3]}...")
        print(f"  Agent_pos stats: max={ap_stats.max[:3]}...")
    
    # 3. Optimizer
    lr_value = float(config['train']['lr'])
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr_value)

    # 3.1 Resume (optional)
    resume_ckpt = config.get('train', {}).get('resume_ckpt')
    start_epoch = 0
    if resume_ckpt:
        resume_path = Path(str(resume_ckpt))
        if resume_path.exists():
            payload = torch.load(resume_path, map_location=device)
            if 'policy' in payload:
                missing, unexpected = policy.load_state_dict(payload['policy'], strict=False)
                if missing:
                    print(f"[Resume] Missing keys: {missing}")
                if unexpected:
                    print(f"[Resume] Unexpected keys: {unexpected}")
            if 'normalizer' in payload and hasattr(policy, 'normalizer'):
                policy.normalizer.load_state_dict(payload['normalizer'])
            if 'optimizer' in payload:
                try:
                    optimizer.load_state_dict(payload['optimizer'])
                except Exception:
                    print("[WARN] Optimizer state incompatible, continuing with fresh optimizer.")
            start_epoch = int(payload.get('epoch', 0))
            print(f"[Resume] Loaded {resume_path}, start_epoch={start_epoch}")
        else:
            print(f"[WARN] resume_ckpt not found: {resume_path}")
    
    # 4. Train Loop
    print("Starting Offline Training...")
    best_loss = float('inf')
    
    # 🔧 新增: Gate训练策略配置
    total_epochs = int(config['train']['epochs'])
    gate_warmup_epochs = int(config.get('train', {}).get('gate_warmup_epochs', total_epochs // 3))  # 前1/3强制gate=0.5
    gate_regularization = float(config.get('train', {}).get('gate_regularization', 0.01))  # gate正则化系数
    print(f"[Gate Strategy] warmup_epochs={gate_warmup_epochs}, regularization={gate_regularization}")
    
    global_step = 0
    for epoch in range(start_epoch, int(config['train']['epochs'])):
        policy.train()
        epoch_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}", ncols=100)
        
        # 🔧 计算当前epoch的gate策略
        if epoch < gate_warmup_epochs:
            # 预热阶段: 强制gate=0.5，让token分支必须学习
            force_gate = 0.5
            current_gate_reg = 0.0  # 预热阶段不需要正则化
        else:
            # 正常阶段: 使用可学习gate，加正则化防止塌缩
            force_gate = None
            current_gate_reg = gate_regularization
        
        for batch in pbar:
            try:
                obs = batch['obs'].to(device)
                action = batch['action'].to(device)
                agent_pos = batch.get('agent_pos', None)
                if agent_pos is not None:
                    agent_pos = agent_pos.to(device)
                if use_tokens:
                    obs_tokens = batch['obs_tokens'].to(device)
                    loss = policy.compute_loss(
                        obs, obs_tokens, action,
                        agent_pos=agent_pos,
                        force_gate=force_gate,
                        gate_regularization=current_gate_reg
                    )
                else:
                    loss = policy.compute_loss(obs, action, agent_pos=agent_pos)
                
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
                    agent_pos = batch.get('agent_pos', None)
                    if agent_pos is not None:
                        agent_pos = agent_pos.to(device)
                    if use_tokens:
                        obs_tokens = batch['obs_tokens'].to(device)
                        loss = policy.compute_loss(obs, obs_tokens, action, agent_pos=agent_pos)
                    else:
                        loss = policy.compute_loss(obs, action, agent_pos=agent_pos)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                else:
                    raise
            
            epoch_loss += loss.item()
            global_step += 1
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        avg_loss = epoch_loss / len(dataloader)
        
        # 🔧 新增: 打印gate状态
        if use_tokens and hasattr(policy, 'token_gate'):
            gate_val = policy.token_gate.item()
            gate_sigmoid = torch.sigmoid(policy.token_gate).item()
            gate_status = "FORCED=0.5" if force_gate is not None else f"learnable"
            print(f"Epoch {epoch+1} done. Avg Loss: {avg_loss:.4f}, Gate: {gate_val:.4f} (sigmoid={gate_sigmoid:.4f}, {gate_status})")
        else:
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
                'policy_type': 'dual_stream' if use_tokens else 'global_only',
                'token_dim': int(token_dim) if use_tokens else None,
                'optimizer': optimizer.state_dict(),
                'epoch': epoch+1,
                'config': config,
                'loss': avg_loss
            }, ckpt_file)
            print(f"Saved checkpoint: {ckpt_file}")

if __name__ == "__main__":
    main()
