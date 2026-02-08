#!/usr/bin/env python3
"""
离线特征提取脚本 - 为Head离线训练准备数据
流程：
1. 读取Raw Data (HDF5)
2. 提取视觉特征 (4 Models)
3. 通过对齐模块 (RGB2PC Encoder)
4. 保存对齐后的特征到Zarr (obs_aligned: [T, 1280])
5. （可选）保存本体感知 agent_pos 到 Zarr（来自 joint_action/vector）
"""
import torch
import torch.nn as nn
import numpy as np
import zarr
import os
import sys
import yaml
import argparse
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import io
import h5py

# 添加路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from features_common.multi_gpu_extractors import MultiGPUFeatureExtractors
from features_common.alignment.rgb2pc_aligned_encoder_4models import RGB2PCAlignedEncoder4Models

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    if config is None:
        raise ValueError(f"Failed to load config from {config_path}")
    return config

def extract_episode(
    task: str,
    episode_path: Path,
    output_path: Path,
    extractors: MultiGPUFeatureExtractors,
    encoder: RGB2PCAlignedEncoder4Models,
    camera_name: str,
    batch_size: int = 32,
    horizon: int = 4,
    n_obs_steps: int = 4,
    use_token_infer: bool = False,
    student_tokens: int = 64,
    overwrite: bool = False,
    save_tokens: bool = False,
    save_proprio: bool = False,
):
    """处理单个episode"""
    if output_path.exists() and not overwrite:
        # print(f"Skipping {output_path} (exists)")
        return

    agent_pos = None
    
    with h5py.File(episode_path, 'r') as f:
        # 1. 检查数据
        if f'observation/{camera_name}/rgb' not in f:
            print(f"Warning: {camera_name} not found in {episode_path}")
            return
            
        rgb_ds = f[f'observation/{camera_name}/rgb']
        num_frames = rgb_ds.shape[0]
        
        # 读取动作
        # 假设动作在 joint_action
        # 注意：这里我们保存所有帧的动作，训练时的 horizon 切分在 Dataset 中做
        if 'joint_action/left_arm' in f:
            # RoboTwin HDF5 结构
            left = f['joint_action/left_arm'][:]
            right = f['joint_action/right_arm'][:]
            # Gripper
            left_g = f['joint_action/left_gripper'][:] if 'joint_action/left_gripper' in f else np.zeros((num_frames, 1))
            right_g = f['joint_action/right_gripper'][:] if 'joint_action/right_gripper' in f else np.zeros((num_frames, 1))
            
            # 形状调整
            if left_g.ndim == 1: left_g = left_g[:, None]
            if right_g.ndim == 1: right_g = right_g[:, None]
            
            # [T, 14]
            action = np.concatenate([left, left_g, right, right_g], axis=-1)
        elif 'action' in f:
            action = f['action'][:]
        else:
            print(f"Warning: Action not found in {episode_path}")
            return

        # 读取本体感知（agent_pos / joint state）
        # 说明：RoboTwin 数据里常用 joint_action/vector 作为 14 维关节状态
        agent_pos = None
        if save_proprio:
            if 'joint_action/vector' in f:
                agent_pos = f['joint_action/vector'][:]
            elif 'observation/joint_action/vector' in f:
                agent_pos = f['observation/joint_action/vector'][:]
            else:
                print(f"[Warning] agent_pos not found in {episode_path}, skip saving agent_pos")
                agent_pos = None
                save_proprio = False

            if agent_pos is not None:
                # 对齐长度（防止异常帧数）
                if agent_pos.shape[0] != num_frames:
                    min_len = min(agent_pos.shape[0], num_frames)
                    agent_pos = agent_pos[:min_len]
                    num_frames = min_len
                    action = action[:min_len]
                agent_pos = agent_pos.astype(np.float32)

    # 2. 批量提取
    all_aligned_feats = []
    all_token_feats = []
    
    # 再次打开文件以流式读取图像
    f = h5py.File(episode_path, 'r')
    rgb_ds = f[f'observation/{camera_name}/rgb']
    
    for i in range(0, num_frames, batch_size):
        end_i = min(i + batch_size, num_frames)
        
        # 加载图像 batch
        images = []
        for j in range(i, end_i):
            rgb_bytes = rgb_ds[j]
            img = Image.open(io.BytesIO(rgb_bytes)).convert('RGB')
            # HDF5 内JPEG由OpenCV按BGR写入，这里统一转换为RGB
            img_np = np.array(img)
            if img_np.ndim == 3 and img_np.shape[-1] == 3:
                img_np = img_np[:, :, ::-1]
                img = Image.fromarray(img_np, mode='RGB')
            images.append(img)
            
        with torch.no_grad():
            device = next(encoder.parameters()).device

            if use_token_infer:
                tokens_list = extractors.extract_batch_tokens(
                    images,
                    max_tokens=student_tokens,
                    return_torch=True,
                )
                tokens_list = [t.to(device) for t in tokens_list]
                tokens_list = [t.unsqueeze(1) for t in tokens_list]  # [B, 1, K, C_i]
                if save_tokens:
                    aligned_feats, token_feats = encoder(tokens_list, return_tokens=True)
                    aligned_feats = aligned_feats.squeeze(1)
                    token_feats = token_feats.squeeze(1)
                else:
                    aligned_feats = encoder(tokens_list).squeeze(1)  # [B, 1280]
                    token_feats = None
                raw_tensor = None
                raw_feats = None
            else:
                # 提取 4 model 特征 [B, 4, 2048]
                # extract_batch 内部会处理 batch
                raw_feats = extractors.extract_batch(images)
                # raw_feats: [B, 4, 2048] -> [B, 1, 4, 2048]
                raw_tensor = torch.from_numpy(raw_feats).float().to(device).unsqueeze(1)
                # Encoder output: [B, 1, 1280]
                if save_tokens:
                    aligned_feats, token_feats = encoder(raw_tensor, return_tokens=True)
                    aligned_feats = aligned_feats.squeeze(1)
                else:
                    aligned_feats = encoder(raw_tensor).squeeze(1)  # [B, 1280]
                    token_feats = None
            
        all_aligned_feats.append(aligned_feats.cpu().numpy())
        if save_tokens and token_feats is not None:
            all_token_feats.append(token_feats.cpu().numpy())

        # 释放显存，避免长序列处理时显存累积
        del raw_feats, raw_tensor, aligned_feats
        torch.cuda.empty_cache()
    
    f.close()
    
    # 合并
    obs_aligned = np.concatenate(all_aligned_feats, axis=0) # [T, 1280]
    obs_tokens = np.concatenate(all_token_feats, axis=0) if (save_tokens and all_token_feats) else None
    
    # 3. 保存 Zarr
    # 创建目录
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    root = zarr.open(str(output_path), mode='w')
    root.create_dataset('obs_aligned', data=obs_aligned, chunks=(100, 1280), dtype='float32')
    if obs_tokens is not None:
        root.create_dataset(
            'obs_tokens',
            data=obs_tokens,
            chunks=(20, obs_tokens.shape[1], obs_tokens.shape[2]),
            dtype='float32'
        )
    root.create_dataset('action', data=action, chunks=(100, action.shape[1]), dtype='float32')
    if save_proprio and agent_pos is not None:
        root.create_dataset('agent_pos', data=agent_pos, chunks=(100, agent_pos.shape[1]), dtype='float32')
    
    # 保存元数据
    root.attrs['task'] = task
    root.attrs['num_frames'] = num_frames
    root.attrs['has_agent_pos'] = bool(save_proprio and agent_pos is not None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Use train_online_batch_extract.yaml to get paths')
    parser.add_argument('--output_dir', type=str, default=None, help='Override output dir')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing zarr outputs')
    parser.add_argument('--max_episodes', type=int, default=0, help='只处理前 N 个 episode（调试用，0表示不限制）')
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    # 1. 确定路径
    robotwin_data_root = Path(config['data'].get('robotwin_data_root', '/home/gl/RoboTwin/data'))
    task_config = config['data'].get('task_config', 'demo_randomized')
    tasks = config['data']['tasks']
    camera_name = config['data']['camera_name']
    use_proprio = bool(config.get('policy', {}).get('use_proprio', False) or config['data'].get('use_proprio', False))
    
    # 输出目录
    if args.output_dir:
        output_root = Path(args.output_dir)
    else:
        output_root = Path(__file__).parent.parent / "features_dataset" / "aligned_zarr"
    
    output_root.mkdir(parents=True, exist_ok=True)
    print(f"Data Root: {robotwin_data_root}")
    print(f"Output Root: {output_root}")
    print(f"Tasks: {tasks}")
    
    # 2. 加载模型
    gpu_ids = config['device']['gpu_ids']
    
    print("Loading Extractors...")
    extractors = MultiGPUFeatureExtractors(gpu_ids=gpu_ids)
    
    print("Loading Alignment Encoder...")
    encoder_path = config['encoder']['checkpoint']
    enc_ckpt = torch.load(encoder_path, map_location='cpu')
    enc_args = enc_ckpt.get('args', {}) if isinstance(enc_ckpt, dict) else {}
    if not isinstance(enc_args, dict):
        enc_args = vars(enc_args)
    student_pool = str(enc_args.get('student_pool', 'mean'))
    student_tokens = int(enc_args.get('student_tokens', 64))
    use_token_infer = student_pool == 'tokens'
    save_tokens = bool(config.get('encoder', {}).get('save_tokens', False))
    encoder = RGB2PCAlignedEncoder4Models.from_checkpoint(
        encoder_path, map_location='cpu', freeze=True
    )
    encoder = encoder.to(f'cuda:{gpu_ids[0]}').eval()

    print(f"Alignment mode: student_pool={student_pool}, student_tokens={student_tokens}, save_tokens={save_tokens}")
    
    # 3. 处理每个任务
    for task in tasks:
        raw_data_dir = robotwin_data_root / task / task_config / 'data'
        output_task_dir = output_root / task
        
        if not raw_data_dir.exists():
            print(f"Skipping {task}: {raw_data_dir} not found")
            continue
            
        files = sorted(list(raw_data_dir.glob("episode*.hdf5")))
        if int(args.max_episodes) > 0:
            files = files[: int(args.max_episodes)]
        print(f"Processing {task}: {len(files)} episodes")
        
        for fpath in tqdm(files):
            ep_name = fpath.stem
            out_path = output_task_dir / f"{ep_name}.zarr"
            
            try:
                extract_episode(
                    task, fpath, out_path,
                    extractors, encoder, camera_name,
                    batch_size=config['train']['batch_size'],
                    horizon=config['data']['horizon'],
                    n_obs_steps=config['data']['n_obs_steps'],
                    use_token_infer=use_token_infer,
                    student_tokens=student_tokens,
                    overwrite=args.overwrite,
                    save_tokens=save_tokens,
                    save_proprio=use_proprio,
                )
            except Exception as e:
                print(f"Error processing {fpath}: {e}")
                # Optional: raise e

if __name__ == "__main__":
    main()
