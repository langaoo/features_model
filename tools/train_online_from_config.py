#!/usr/bin/env python3
"""
在线训练脚本 - 基于配置文件
支持多GPU并行 + 从HDF5读取数据
"""
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
import sys
import os
from tqdm import tqdm
import logging
import warnings
import yaml
from typing import Dict, Any

# 关闭所有不必要的日志
logging.basicConfig(level=logging.ERROR)
logging.getLogger("depth_anything_3").setLevel(logging.ERROR)
logging.getLogger("depth_anything_3.api").setLevel(logging.ERROR)
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

# 设置multiprocessing启动方法为spawn（CUDA兼容）
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

# 添加路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from features_common.dp_rgb_dataset_from_hdf5 import DPRGBOnlineDataset, collate_fn_online_4, make_batch_collate_fn
from features_common.multi_gpu_extractors import MultiGPUFeatureExtractors
from features_common.rgb2pc_aligned_encoder_4models import RGB2PCAlignedEncoder4Models

# 导入正版DP
HAS_OFFICIAL_DP = False
try:
    # 添加正版DP外层路径（DP/diffusion_policy包含diffusion_policy子目录）
    DP_OUTER = Path(__file__).parent.parent / "DP" / "diffusion_policy"
    if DP_OUTER.exists() and (DP_OUTER / "diffusion_policy").exists():
        sys.path.insert(0, str(DP_OUTER))
        from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
        from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
        HAS_OFFICIAL_DP = True
        print("[INFO] 正版DP已加载")
    else:
        print(f"[WARNING] 正版DP路径不存在: {DP_OUTER}")
except ImportError as e:
    print(f"[WARNING] 正版DP导入失败: {e}")


# 定义正版DP Policy类
class DPRGBPolicy(nn.Module):
    """正版Diffusion Policy for RGB特征"""
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        horizon: int = 8,
        n_obs_steps: int = 2,
        n_action_steps: int = 8,
        num_inference_steps: int = 100,
    ):
        super().__init__()
        
        if not HAS_OFFICIAL_DP:
            raise RuntimeError("正版DP未加载，无法使用DPRGBPolicy")
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.num_inference_steps = num_inference_steps
        
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
        
    def compute_loss(self, obs, action_gt):
        """计算扩散损失"""
        B = obs.shape[0]
        device = obs.device
        
        # 编码观测
        obs_flat = obs.reshape(B, -1)
        obs_cond = self.obs_encoder(obs_flat)
        
        # 随机采样时间步
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (B,), device=device
        ).long()
        
        # 添加噪声
        noise = torch.randn(action_gt.shape, device=device)
        noisy_actions = self.noise_scheduler.add_noise(action_gt, noise, timesteps)
        
        # 预测噪声
        noise_pred = self.noise_pred_net(noisy_actions, timesteps, global_cond=obs_cond)
        
        # MSE损失
        loss = nn.functional.mse_loss(noise_pred, noise)
        return loss
    
    def forward(self, obs):
        """推理模式"""
        B = obs.shape[0]
        device = obs.device
        
        # 编码观测
        obs_flat = obs.reshape(B, -1)
        obs_cond = self.obs_encoder(obs_flat)
        
        # 初始化随机噪声
        action = torch.randn((B, self.n_action_steps, self.action_dim), device=device)
        
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
        
        return action


def load_config(config_path: str) -> Dict[str, Any]:
    """加载YAML配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def print_config(config: Dict[str, Any]):
    """打印配置信息"""
    print(f"\n{'='*60}")
    print("训练配置")
    print(f"{'='*60}")
    print(f"数据:")
    print(f"  任务: {', '.join(config['data']['tasks'])}")
    print(f"  相机: {config['data']['camera_name']}")
    print(f"  Horizon: {config['data']['horizon']}, Obs steps: {config['data']['n_obs_steps']}")
    print(f"  使用左臂: {config['data']['use_left_arm']}, 右臂: {config['data']['use_right_arm']}")
    print(f"  融合双臂: {config['data']['fuse_arms']}")
    print(f"\n编码器:")
    print(f"  Checkpoint: {config['encoder']['checkpoint']}")
    print(f"  冻结: {config['encoder']['freeze']}")
    print(f"\n训练:")
    print(f"  轮数: {config['train']['epochs']}")
    print(f"  批大小: {config['train']['batch_size']}")
    print(f"  学习率: {config['train']['lr']}")
    print(f"  Workers: {config['train']['num_workers']}")
    print(f"\nGPU:")
    print(f"  设备: {config['device']['gpu_ids']}")
    print(f"\n输出:")
    print(f"  目录: {config['output']['dir']}")
    print(f"  保存间隔: 每{config['output']['save_every_n_epochs']}轮")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                       help='配置文件路径')
    parser.add_argument('--override', type=str, nargs='*',
                       help='覆盖配置项，格式: train.epochs=100 data.batch_size=4')
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 处理覆盖参数
    if args.override:
        for override in args.override:
            keys, value = override.split('=')
            keys = keys.split('.')
            # 简单的嵌套字典更新
            d = config
            for key in keys[:-1]:
                d = d[key]
            # 尝试转换类型
            try:
                d[keys[-1]] = eval(value)
            except:
                d[keys[-1]] = value
    
    # 打印配置
    if config['debug'].get('print_config', True):
        print_config(config)
    
    gpu_ids = config['device']['gpu_ids']
    
    # 1. 加载特征提取器
    print("1. 加载4个特征提取器到多GPU...")
    extractors = MultiGPUFeatureExtractors(gpu_ids=gpu_ids)
    print()
    
    # 2. 创建Dataset
    print("2. 创建Dataset（从HDF5读取）...")
    
    # 检查是否使用批量特征提取模式（提升batchsize有效性）
    batch_extract = config['train'].get('batch_extract', False)
    
    dataset = DPRGBOnlineDataset(
        raw_data_root=config['data']['raw_data_root'],
        tasks=config['data']['tasks'],
        horizon=config['data']['horizon'],
        n_obs_steps=config['data']['n_obs_steps'],
        feature_extractors=extractors,
        camera_name=config['data']['camera_name'],
        use_left_arm=config['data']['use_left_arm'],
        use_right_arm=config['data']['use_right_arm'],
        fuse_arms=config['data']['fuse_arms'],
        include_gripper=config['data']['include_gripper'],
        batch_extract=batch_extract,  # 新增参数
    )
    print(f"✓ Dataset: {len(dataset)} samples")
    print(f"✓ Batch extract mode: {'ON (批量提取特征)' if batch_extract else 'OFF (单样本提取)'}\n")
    
    # 3. DataLoader
    # ★★★ 关键修复: num_workers=0 避免多进程复制CUDA模型 ★★★
    # 在线特征提取模式下，多worker会导致每个worker进程都复制一份GPU模型，造成显存浪费
    # 使用主进程单线程 + 批量特征提取，充分利用GPU并行能力
    if batch_extract:
        # 批量提取模式：使用batch_collate_fn
        print("3. 使用批量特征提取DataLoader (num_workers=0)...")
        dataloader = DataLoader(
            dataset,
            batch_size=config['train']['batch_size'],
            shuffle=True,
            num_workers=0,  # ★ 强制为0，避免多进程
            collate_fn=make_batch_collate_fn(extractors),  # 批量提取
            pin_memory=True,
        )
    else:
        # 单样本模式：使用原始collate_fn
        dataloader = DataLoader(
            dataset,
            batch_size=config['train']['batch_size'],
            shuffle=True,
            num_workers=0,  # ★ 强制为0，避免多进程
            collate_fn=collate_fn_online_4,
            pin_memory=True,
        )
    
    # 4. 加载对齐编码器
    print("3. 加载对齐编码器...")
    encoder = RGB2PCAlignedEncoder4Models.from_checkpoint(
        config['encoder']['checkpoint'],
        map_location='cpu',
        freeze=config['encoder']['freeze']
    )
    encoder = encoder.to(f'cuda:{gpu_ids[0]}')
    encoder.eval()
    print(f"✓ Encoder loaded from {config['encoder']['checkpoint']}\n")
    
    # 5. DP Policy Head
    print("4. 创建DP Policy...")
    
    # 计算action维度
    action_dim = 0
    if config['data']['use_left_arm']:
        action_dim += 6
        if config['data']['include_gripper']:
            action_dim += 1
    if config['data']['use_right_arm']:
        action_dim += 6
        if config['data']['include_gripper']:
            action_dim += 1
    
    # 检查是否使用正版DP
    use_official_dp = config['policy'].get('use_official_dp', False)
    
    if use_official_dp and HAS_OFFICIAL_DP:
        print("  使用正版Diffusion Policy")
        obs_dim = config['data']['n_obs_steps'] * 1280  # fuse_dim=1280
        policy = DPRGBPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            horizon=config['data']['horizon'],
            n_obs_steps=config['data']['n_obs_steps'],
            n_action_steps=config['data']['horizon'],
            num_inference_steps=config['policy'].get('num_inference_steps', 100),
        )
        policy = policy.to(f'cuda:{gpu_ids[0]}')
        print(f"✓ Official DP: obs_dim={obs_dim}, action_dim={action_dim}\n")
        use_diffusion_loss = True
    else:
        if use_official_dp:
            print("  [WARNING] 正版DP不可用，回退到SimpleDPHead")
        
        class SimpleDPHead(nn.Module):
            def __init__(self, obs_dim, action_dim, horizon, hidden_dim=512):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(obs_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim // 2, action_dim * horizon),
                )
                self.action_dim = action_dim
                self.horizon = horizon
            
            def forward(self, obs):
                # obs: [B, To, D]
                B = obs.shape[0]
                obs_flat = obs.reshape(B, -1)
                out = self.net(obs_flat)
                return out.reshape(B, self.horizon, self.action_dim)
        
        obs_dim = config['data']['n_obs_steps'] * 1280  # fuse_dim=1280
        hidden_dim = config['policy'].get('hidden_dim', 512)
        
        policy = SimpleDPHead(
            obs_dim, 
            action_dim, 
            config['data']['horizon'],
            hidden_dim
        )
        policy = policy.to(f'cuda:{gpu_ids[0]}')
        print(f"✓ SimpleDPHead: obs_dim={obs_dim}, action_dim={action_dim}, hidden_dim={hidden_dim}\n")
        use_diffusion_loss = False
    
    # 6. 优化器
    lr = float(config['train']['lr'])
    weight_decay = float(config['train'].get('weight_decay', 1e-6))
    
    optimizer = torch.optim.AdamW(
        policy.parameters(), 
        lr=lr,
        weight_decay=weight_decay
    )
    
    # 损失函数
    if use_diffusion_loss:
        criterion = None  # 使用policy.compute_loss()
    else:
        criterion = nn.MSELoss()
    
    # 7. 训练循环
    print("5. 开始训练...")
    print(f"总轮数: {config['train']['epochs']}, "
          f"每轮批次: {len(dataloader)}, "
          f"总步数: {config['train']['epochs'] * len(dataloader)}")
    print(f"{'='*60}\n")
    
    global_step = 0
    best_loss = float('inf')
    
    for epoch in range(config['train']['epochs']):
        policy.train()
        epoch_loss = 0
        
        # 快速开发模式
        if config['debug'].get('fast_dev_run', False):
            print("[DEBUG] Fast dev run mode - only 1 batch")
            dataloader_iter = iter(dataloader)
            batch = next(dataloader_iter)
            batches = [batch]
        else:
            batches = dataloader
        
        # 使用tqdm显示进度
        pbar = tqdm(batches, 
                    desc=f"Epoch {epoch+1}/{config['train']['epochs']}", 
                    ncols=100, leave=True)
        
        for batch_idx, batch in enumerate(pbar):
            obs = batch['obs'].to(f'cuda:{gpu_ids[0]}')
            action_gt = batch['action'].to(f'cuda:{gpu_ids[0]}')
            
            # 前向传播
            obs_encoded = encoder(obs)  # [B, To, 1280]
            
            # 计算损失
            if use_diffusion_loss:
                # 正版DP：使用扩散损失
                loss = policy.compute_loss(obs_encoded, action_gt)
            else:
                # SimpleDPHead：使用MSE损失
                action_pred = policy(obs_encoded)  # [B, Ta, A]
                loss = criterion(action_pred, action_gt)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            global_step += 1
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{epoch_loss/(batch_idx+1):.4f}'
            })
            
            if config['debug'].get('fast_dev_run', False):
                break
        
        avg_loss = epoch_loss / len(pbar)
        
        # Epoch结束统计
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{config['train']['epochs']} 完成")
        print(f"  平均Loss: {avg_loss:.4f}")
        print(f"  全局步数: {global_step}")
        if avg_loss < best_loss:
            best_loss = avg_loss
            print(f"  ✓ 新的最佳Loss!")
        print(f"{'='*60}\n")
        
        # 保存checkpoint
        save_interval = config['output']['save_every_n_epochs']
        is_last_epoch = (epoch + 1) == config['train']['epochs']
        should_save = (epoch + 1) % save_interval == 0 or is_last_epoch
        
        if should_save:
            run_name = config['output'].get('run_name')
            if run_name is None:
                task_str = '_'.join(config['data']['tasks'])
                run_name = f"{task_str}_run"
            
            output_dir = Path(config['output']['dir']) / run_name
            output_dir.mkdir(parents=True, exist_ok=True)
            
            ckpt_path = output_dir / f"ckpt_epoch_{epoch+1:04d}.pt"
            torch.save({
                'epoch': epoch + 1,
                'global_step': global_step,
                'policy': policy.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': avg_loss,
                'best_loss': best_loss,
                'config': config,
            }, ckpt_path)
            print(f"✓ 已保存checkpoint: {ckpt_path}\n")
        
        if config['debug'].get('fast_dev_run', False):
            print("[DEBUG] Fast dev run完成，退出")
            break
    
    print(f"\n{'='*60}")
    print("训练完成！")
    print(f"最佳Loss: {best_loss:.4f}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
