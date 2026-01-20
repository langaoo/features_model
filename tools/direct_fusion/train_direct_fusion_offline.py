#!/usr/bin/env python3
"""
直接融合训练【离线版】: 跳过对齐模块+跳过在线提取,加载离线4模型特征 → 融合 → Head → 动作
适配：已有离线特征文件，无需任何RGB图像→特征的前向，彻底解决大模型前向慢问题
目的: 
1. 测试是否是对齐模块破坏了特征
2. 验证融合本身的有效性
3. 提供更简单快速的baseline

流程:
  离线zarr特征 → 简单融合(weighted/concat) → Head → 动作
  
核心不变:
- 无对齐模块 (no context_encoder, no projection)
- 只训练fusion权重和Head
- 4个backbone特征完全冻结，无任何更新
核心改动:
- 彻底删除在线特征提取，加载预生成的4模型zarr特征文件
"""
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import argparse
import sys
import os
from tqdm import tqdm
import logging
import warnings
import yaml
from typing import Dict, Any, List, Tuple
import numpy as np
import zarr
import time
import h5py

# 关闭不必要的日志
logging.basicConfig(level=logging.ERROR)
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

# 设置multiprocessing
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

# 添加路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# ✅ 核心改动1: 删除所有在线提取相关导入，无需再加载在线数据集和提取器
# from features_common.dp_rgb_dataset_from_hdf5 import DPRGBOnlineDataset, make_batch_collate_fn
# from features_common.multi_gpu_extractors import MultiGPUFeatureExtractors

# 导入正版DP (完全保留,一行不改)
HAS_OFFICIAL_DP = False
try:
    # 修正路径: third_party/DP/diffusion_policy (其中diffusion_policy是项目根目录)
    DP_OUTER = Path(__file__).parent.parent.parent / "third_party" / "DP" / "diffusion_policy"
    if DP_OUTER.exists():
        sys.path.insert(0, str(DP_OUTER))
        from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
        from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
        from diffusion_policy.model.common.normalizer import LinearNormalizer  # ✅ 新增
        HAS_OFFICIAL_DP = True
        print("[INFO] 正版DP已加载")
except ImportError as e:
    print(f"[WARNING] 正版DP导入失败: {e}")


# ✅ 核心不变: 融合编码器完全复用，一行代码不改！！！
class SimpleFusionEncoder(nn.Module):
    """简单融合编码器: 4模型特征 → 融合 → 输出"""
    
    def __init__(
        self,
        in_dims=(1024, 2048, 768, 2048),
        fusion_type='weighted',  # 'weighted', 'concat', 'mean'
        out_dim=1280,
    ):
        super().__init__()
        self.in_dims = in_dims
        self.fusion_type = fusion_type
        self.out_dim = out_dim
        
        if fusion_type == 'weighted':
            # 可学习的加权融合
            # 先投影到统一维度,再加权
            self.projectors = nn.ModuleList([
                nn.Linear(dim, out_dim) for dim in in_dims
            ])
            self.fusion_weights = nn.Parameter(torch.ones(len(in_dims)) / len(in_dims))
            
        elif fusion_type == 'concat':
            # 拼接后降维
            total_dim = sum(in_dims)
            self.projector = nn.Sequential(
                nn.Linear(total_dim, out_dim * 2),
                nn.ReLU(),
                nn.Linear(out_dim * 2, out_dim),
                nn.LayerNorm(out_dim),
            )
            
        elif fusion_type == 'mean':
            # 简单平均(先投影)
            self.projectors = nn.ModuleList([
                nn.Linear(dim, out_dim) for dim in in_dims
            ])
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")
    
    def forward(self, x):
        """
        Args:
            x: [B, To, M, C_max] 其中M=4
        Returns:
            out: [B, To, out_dim]
        """
        B, To, M, _ = x.shape
        
        if self.fusion_type == 'weighted':
            # 投影 + 加权
            feats = []
            for i, proj in enumerate(self.projectors):
                feat_i = x[:, :, i, :self.in_dims[i]]  # [B, To, C_i]
                feat_i = feat_i.reshape(B * To, self.in_dims[i])
                feat_i = proj(feat_i)  # [B*To, out_dim]
                feat_i = feat_i.reshape(B, To, self.out_dim)
                feats.append(feat_i)
            
            feats = torch.stack(feats, dim=2)  # [B, To, M, out_dim]
            weights = torch.softmax(self.fusion_weights, dim=0)  # [M]
            out = (feats * weights.view(1, 1, M, 1)).sum(dim=2)  # [B, To, out_dim]
            
        elif self.fusion_type == 'concat':
            # 拼接
            feats = []
            for i in range(M):
                feat_i = x[:, :, i, :self.in_dims[i]]
                feats.append(feat_i)
            feats_concat = torch.cat(feats, dim=-1)  # [B, To, sum(dims)]
            feats_concat = feats_concat.reshape(B * To, -1)
            out = self.projector(feats_concat)
            out = out.reshape(B, To, self.out_dim)
            
        elif self.fusion_type == 'mean':
            # 简单平均
            feats = []
            for i, proj in enumerate(self.projectors):
                feat_i = x[:, :, i, :self.in_dims[i]]
                feat_i = feat_i.reshape(B * To, self.in_dims[i])
                feat_i = proj(feat_i)
                feat_i = feat_i.reshape(B, To, self.out_dim)
                feats.append(feat_i)
            out = torch.stack(feats, dim=0).mean(dim=0)  # [B, To, out_dim]
        
        return out


# ✅ 核心不变: DP Policy网络完全复用，一行代码不改！！！
class DirectFusionDPPolicy(nn.Module):
    """直接融合 + Diffusion Policy"""
    
    def __init__(
        self,
        fusion_encoder: SimpleFusionEncoder,
        action_dim: int = 14,
        horizon: int = 4,
        n_obs_steps: int = 4,
        num_inference_steps: int = 100,
    ):
        super().__init__()
        
        if not HAS_OFFICIAL_DP:
            raise RuntimeError("正版DP未加载")
        
        self.fusion_encoder = fusion_encoder
        self.action_dim = action_dim
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.num_inference_steps = num_inference_steps
        
        # ✅ 新增: 使用LinearNormalizer
        self.normalizer = LinearNormalizer()
        
        obs_dim = n_obs_steps * fusion_encoder.out_dim
        
        # 观测编码器
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        
        # Diffusion UNet
        self.noise_pred_net = ConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=256,
            diffusion_step_embed_dim=128,
            down_dims=[256, 512, 1024],
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
        )
        
        # 噪声调度器
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=100,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon'
        )
    
    def compute_loss(self, rgb_feats, actions):
        """
        训练时的loss计算
        Args:
            rgb_feats: [B, To, M, C] 4模型特征
            actions: [B, Ta, A] ground truth动作 (未归一化)
        """
        B = rgb_feats.shape[0]
        device = rgb_feats.device
        
        # ✅ 使用normalizer归一化action (模仿RoboTwin原版DP)
        # 统一使用LinearNormalizer的dict接口，确保device一致
        nactions = self.normalizer.normalize({'action': actions})['action']
        nactions = nactions.to(device=device, dtype=actions.dtype)
        
        # 融合特征
        obs_fused = self.fusion_encoder(rgb_feats)  # [B, To, D]
        
        # 编码观测
        obs_flat = obs_fused.reshape(B, -1)
        obs_cond = self.obs_encoder(obs_flat)  # [B, 256]
        
        # 随机采样时间步
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (B,), device=device
        ).long()
        
        # 添加噪声
        noise = torch.randn_like(nactions)
        noisy_actions = self.noise_scheduler.add_noise(nactions, noise, timesteps)
        
        # 预测噪声
        noise_pred = self.noise_pred_net(
            noisy_actions, timesteps, global_cond=obs_cond
        )
        
        # MSE loss
        loss = nn.functional.mse_loss(noise_pred, noise)
        return loss
    
    def forward(self, rgb_feats):
        """
        推理时的去噪采样
        Args:
            rgb_feats: [B, To, M, C]
        Returns:
            actions: [B, Ta, A] (已反归一化)
        """
        B = rgb_feats.shape[0]
        device = rgb_feats.device
        
        # 融合特征
        obs_fused = self.fusion_encoder(rgb_feats)
        
        # 编码观测
        obs_flat = obs_fused.reshape(B, -1)
        obs_cond = self.obs_encoder(obs_flat)
        
        # 初始化随机噪声
        action = torch.randn((B, self.horizon, self.action_dim), device=device)
        
        # 设置推理步数
        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        
        # 逐步去噪
        for t in self.noise_scheduler.timesteps:
            noise_pred = self.noise_pred_net(
                action,
                t.unsqueeze(0).expand(B).to(device),
                global_cond=obs_cond
            )
            action = self.noise_scheduler.step(noise_pred, t, action).prev_sample
        
        # ✅ 使用normalizer反归一化 (模仿RoboTwin原版DP)
        action = self.normalizer.unnormalize({'action': action})['action']
        
        return action

# ✅ 核心改动2: 新增【离线Zarr特征数据集】 核心！！！
# 专门加载你的4个离线特征文件 + 动作标签，输出格式和在线版完全一致
class DPRGBOfflineZarrDataset(Dataset):
    def __init__(
        self,
        vis_zarr_roots: List[str],
        robotwin_data_root: str,
        task_name: str,
        task_config: str,
        horizon: int =4,
        n_obs_steps: int=4,
        use_left_arm: bool=True,
        use_right_arm: bool=True,
        fuse_arms: bool=True,
        include_gripper: bool=True,
        expert_data_num: int=50,
        camera_name: str='head_camera'
    ):
        super().__init__()
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.in_dims = [1024,2048,768,2048]
        self.C_max = 2048
        self.M = 4
        
        # 动作维度固定
        self.action_dim = 14
        
        # 加载4个模型的离线zarr特征根路径
        # zarr结构: root/task_name-task_config-expert_num_sapien_camera/episode_X.zarr
        self.vis_zarr_roots = vis_zarr_roots
        self.task_name = task_name
        self.task_config = task_config
        self.camera_name = camera_name
        self.expert_data_num = expert_data_num
        
        # 加载动作标签路径
        self.raw_data_root = os.path.join(robotwin_data_root, task_name, task_config, 'data')
        self.episodes = [f'episode_{i}' for i in range(expert_data_num)]
        
        # ✅ 预处理：收集所有有效样本 (特征窗口 + 对应动作)
        # 不再手动归一化,留给normalizer处理
        self.samples = self._collect_samples()
        print(f"[INFO] Dataset初始化完成: {len(self.samples)} samples")

    def _load_episode_action(self, episode: str):
        """加载单episode的动作标签"""
        # episode格式: episode_X -> episodeX.hdf5
        ep_num = episode.split('_')[1]  # 'episode_0' -> '0'
        hdf5_path = os.path.join(self.raw_data_root, f'episode{ep_num}.hdf5')
        with h5py.File(hdf5_path, 'r') as f:
            if 'joint_action/left_arm' in f:
                left = f['joint_action/left_arm'][:]
                right = f['joint_action/right_arm'][:]
                left_g = f['joint_action/left_gripper'][:] if 'joint_action/left_gripper' in f else np.zeros((left.shape[0], 1))
                right_g = f['joint_action/right_gripper'][:] if 'joint_action/right_gripper' in f else np.zeros((right.shape[0], 1))
                if left_g.ndim == 1:
                    left_g = left_g[:, None]
                if right_g.ndim == 1:
                    right_g = right_g[:, None]
                actions = np.concatenate([left, left_g, right, right_g], axis=-1)
            elif 'joint_action/vector' in f:
                actions = f['joint_action/vector'][:]
            else:
                raise KeyError("HDF5中缺少joint_action数据")
        return torch.from_numpy(actions).float()
    
    def _load_episode_feat(self, episode: str):
        """加载单episode的4模型特征 [T, M, C_max]"""
        feat_list = []
        # 构建zarr路径: root/task-config-num_sapien_camera/episode.zarr
        zarr_subdir = f"{self.task_name}-{self.task_config}-{self.expert_data_num}_sapien_{self.camera_name}"
        
        for mi in range(self.M):
            zarr_root = self.vis_zarr_roots[mi]
            zarr_path = os.path.join(zarr_root, zarr_subdir, f"{episode}.zarr")
            
            # 读取单个模型的特征 zarr Group
            feat_zarr = zarr.open(zarr_path, mode='r')
            # 从Group中提取per_frame_features数组
            feat = feat_zarr['per_frame_features'][:]
            
            # ✅ 修复：正确处理5D特征，保留时序信息
            # 实际格式是 [W, T, Hf, Wf, C]，其中W=windows, T=8 frames per window
            # 由于stride=1，每个窗口的第一帧就是连续的帧序列
            if feat.ndim == 5:  # [W, T, Hf, Wf, C]
                # 取每个窗口的第一帧（stride=1保证连续性）
                feat = feat[:, 0, :, :, :]  # [W, Hf, Wf, C]
                W, Hf, Wf, C = feat.shape
                # 空间平均：只对Hf和Wf维度求平均，保留W（时间）维度
                feat = feat.reshape(W, Hf * Wf, C).mean(axis=1)  # [W, C]
            elif feat.ndim == 4:  # [T, h, w, c]
                feat = feat.mean(axis=(1, 2))  # 对h/w维度平均
            elif feat.ndim == 3:  # [T, tokens, c]
                feat = feat.mean(axis=1)  # 对tokens维度平均
            # 现在feat应该是 [T, c] (这里T实际是W，即帧数)
            
            feat = torch.from_numpy(feat).float()
            
            # padding到C_max=2048，和在线版格式一致
            pad_len = self.C_max - feat.shape[-1]
            if pad_len > 0:
                feat = torch.nn.functional.pad(feat, (0, pad_len))
            feat_list.append(feat)
        
        # 堆叠：将list转换为张量 [M, T, C_max]
        feats = torch.stack(feat_list, dim=0)  # [M, T, C_max]
        # 转置: [M, T, C_max] → [T, M, C_max]
        feats = feats.transpose(0, 1)  # [T, M, C_max]
        return feats
    
    def _collect_samples(self):
        """收集所有时序对齐的样本 (特征窗口+动作)"""
        samples = []
        print(f"[INFO] 开始收集samples，共{len(self.episodes)}个episodes...")
        for ep_idx, ep in enumerate(self.episodes):
            try:
                if ep_idx % 10 == 0:
                    print(f"  处理进度: {ep_idx}/{len(self.episodes)}")
                feats = self._load_episode_feat(ep)  # [T_feat, M, C_max]
                acts = self._load_episode_action(ep)  # [T_act, A]
                
                # 对齐时间步：取较小的长度
                T = min(feats.shape[0], acts.shape[0])
                feats = feats[:T]
                acts = acts[:T]
                
                # 滑动窗口采样：和在线版逻辑一致
                for t in range(T - self.n_obs_steps - self.horizon + 1):
                    feat_window = feats[t:t+self.n_obs_steps]  # [To, M, C_max]
                    act_window = acts[t+self.n_obs_steps : t+self.n_obs_steps+self.horizon] # [Ta, A]
                    # ✅ 不再手动归一化,留给normalizer处理
                    samples.append((feat_window, act_window))
            except Exception as e:
                print(f"[WARN] 跳过episode {ep}: {e}")
                import traceback
                traceback.print_exc()
                continue
        print(f"[INFO] 收集完成，共{len(samples)}个samples")
        return samples

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        feat_win, act_win = self.samples[idx]
        return {'obs': feat_win, 'action': act_win}


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


# ✅ 核心改动3: 修改main函数，删除在线提取，加载离线特征
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                       help='配置文件路径')
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    print("="*60)
    print("直接融合训练 (离线版 ✔️ 无在线特征提取)")
    print("="*60)
    print(f"配置文件: {args.config}")
    print(f"任务: {config['data']['tasks']}")
    print(f"融合方式: {config['fusion']['type']}")
    print(f"Horizon: {config['data']['horizon']}, N_obs: {config['data']['n_obs_steps']}")
    print("✅ 已加载离线特征，无任何模型前向提取，训练速度拉满！")
    print("="*60)
    
    gpu_ids = config['device']['gpu_ids']
    device = f'cuda:{gpu_ids[0]}'
    
    # ✅ 删掉: 不再加载特征提取器
    # print("\n1. 加载4个特征提取器...")
    # extractors = MultiGPUFeatureExtractors(gpu_ids=gpu_ids)
    
    # ✅ 修改: 创建【离线Zarr数据集】
    print("\n1. 创建离线特征Dataset...")
    robotwin_data_root = config['data'].get('robotwin_data_root', '/home/gl/RoboTwin/data')
    task_config = config['data'].get('task_config', 'demo_clean')
    expert_data_num = config.get('checkpoint', {}).get('expert_data_num', 50)
    task_name = config['data']['tasks'][0]
    
    dataset = DPRGBOfflineZarrDataset(
        vis_zarr_roots=config['data']['vis_zarr_roots'], # 4个离线特征路径
        robotwin_data_root=robotwin_data_root,
        task_name=task_name,
        task_config=task_config,
        horizon=config['data']['horizon'],
        n_obs_steps=config['data']['n_obs_steps'],
        use_left_arm=config['data']['use_left_arm'],
        use_right_arm=config['data']['use_right_arm'],
        fuse_arms=config['data']['fuse_arms'],
        include_gripper=config['data']['include_gripper'],
        expert_data_num=expert_data_num,
        camera_name=config['data']['camera_name'],
    )
    print(f"✓ Dataset: {len(dataset)} samples")
    
    # ✅ 修改: DataLoader 删掉在线提取的collate_fn，无需任何处理
    dataloader = DataLoader(
        dataset,
        batch_size=config['train']['batch_size'],
        shuffle=True,
        num_workers=4, # 离线特征加载可以开多线程加速
        pin_memory=True,
        drop_last=True
    )
    
    # ✅ 完全不变: 创建模型，复用原有逻辑
    print("\n2. 创建直接融合模型...")
    fusion_encoder = SimpleFusionEncoder(
        in_dims=(1024, 2048, 768, 2048),
        fusion_type=config['fusion']['type'],
        out_dim=config['fusion']['out_dim'],
    ).to(device)
    
    policy = DirectFusionDPPolicy(
        fusion_encoder=fusion_encoder,
        action_dim=14,
        horizon=config['data']['horizon'],
        n_obs_steps=config['data']['n_obs_steps'],
        num_inference_steps=config['policy']['num_inference_steps'],
    ).to(device)
    
    print(f"✓ 模型参数: {sum(p.numel() for p in policy.parameters()) / 1e6:.2f}M")
    
    # ✅ 新增: Fit normalizer (模仿RoboTwin原版DP)
    print("\n2.5. Fit normalizer...")
    all_actions = torch.stack([s[1] for s in dataset.samples])  # [N, Ta, A], s[1]已经是tensor
    policy.normalizer.fit(
        {'action': all_actions},
        last_n_dims=1,
        mode='limits',
        output_min=-1.0,
        output_max=1.0
    )
    # 将normalizer移动到同一device，避免dtype/device不一致
    try:
        policy.normalizer.to(device)
    except Exception:
        pass
    print(f"  Action stats: min={policy.normalizer['action'].params_dict['input_stats'].min[:3]}...")
    print(f"  Action stats: max={policy.normalizer['action'].params_dict['input_stats'].max[:3]}...")
    
    # ✅ 完全不变: 优化器
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=config['train']['lr'],
        weight_decay=config['train']['weight_decay'],
    )
    
    # ✅ 完全不变: 训练循环 + 保存逻辑，一行不改！！！
    print("\n3. 开始训练...")
    save_dir = Path(config['checkpoint']['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    
    global_step = 0
    best_loss = float('inf')
    
    for epoch in range(config['train']['epochs']):
        policy.train()
        epoch_loss = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['train']['epochs']}")
        
        for batch in pbar:
            obs = batch['obs'].to(device, non_blocking=True)  # [B, To, M, C]
            action_gt = batch['action'].to(device, non_blocking=True)  # [B, Ta, A]
            
            # 前向
            loss = policy.compute_loss(obs, action_gt)
            
            # 反向
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            global_step += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.6f}")
        
        # 保存
        if (epoch + 1) % config['checkpoint']['save_every'] == 0:
            ckpt_path = save_dir / f"{epoch+1}.ckpt"
            torch.save({
                'policy': policy.state_dict(),
                'normalizer': policy.normalizer.state_dict(),  # ✅ 保存normalizer
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'config': config,
                'loss': avg_loss,
            }, ckpt_path)
            print(f"✓ Saved: {ckpt_path}")
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_path = save_dir / "best.ckpt"
                torch.save({
                    'policy': policy.state_dict(),
                    'normalizer': policy.normalizer.state_dict(),  # ✅ 保存normalizer
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'config': config,
                    'loss': avg_loss,
                }, best_path)
                print(f"✓ Best model saved: {best_path}")
    
    print("\n训练完成!")


if __name__ == '__main__':
    main()