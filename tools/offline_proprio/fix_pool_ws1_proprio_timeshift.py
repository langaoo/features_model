#!/usr/bin/env python3
"""
修复 pool_ws1_proprio zarr 数据的时间偏移问题

问题：原始提取 offline_features_pool_ws1_proprio 中 agent_pos[t] == action[t]，
      没有正确做时间偏移。

修复方式：
  - 读取 joint_action/vector 原始数据
  - 重新计算: agent_pos = vector[:-1], action = vector[1:]
  - obs_aligned 也截断到 [:-1]（与 agent_pos 对齐）
  - 保存到新目录或就地覆盖

注意：offline_features_dual_stream_tokens_full_proprio 已经是正确的，不需要修复。
"""

import zarr
import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm
import shutil
import argparse


def fix_episode(zarr_path: Path, raw_hdf5_path: Path, output_path: Path, overwrite: bool = False):
    """修复单个 episode 的时间偏移"""
    if output_path.exists() and not overwrite:
        return
    
    # 读取原始 zarr
    z = zarr.open(str(zarr_path), 'r')
    obs_aligned = z['obs_aligned'][:]  # [T, 1280]
    old_action = z['action'][:]        # [T, 14]
    old_agent_pos = z['agent_pos'][:] if 'agent_pos' in z else None
    
    # 读取原始 HDF5 的 vector 数据来重新做时间偏移
    with h5py.File(str(raw_hdf5_path), 'r') as f:
        if 'joint_action/vector' in f:
            vector_all = f['joint_action/vector'][:]
        elif 'observation/joint_action/vector' in f:
            vector_all = f['observation/joint_action/vector'][:]
        else:
            print(f"  Warning: vector not found in {raw_hdf5_path}, skipping")
            return
    
    # 重新做时间偏移
    # agent_pos[t] = vector[t] (当前状态)
    # action[t] = vector[t+1] (下一步状态 = 要执行的动作)
    new_agent_pos = vector_all[:-1].astype(np.float32)  # [T-1, 14]
    new_action = vector_all[1:].astype(np.float32)       # [T-1, 14]
    new_num_frames = len(new_agent_pos)
    
    # obs_aligned 也截断到 [:-1]（与 agent_pos 对齐）
    new_obs_aligned = obs_aligned[:new_num_frames]
    
    # 验证
    assert new_obs_aligned.shape[0] == new_num_frames, \
        f"obs_aligned shape mismatch: {new_obs_aligned.shape[0]} vs {new_num_frames}"
    
    # 保存
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        shutil.rmtree(str(output_path))
    
    root = zarr.open(str(output_path), mode='w')
    root.create_dataset('obs_aligned', data=new_obs_aligned, 
                       chunks=(100, new_obs_aligned.shape[1]), dtype='float32')
    root.create_dataset('action', data=new_action,
                       chunks=(100, new_action.shape[1]), dtype='float32')
    root.create_dataset('agent_pos', data=new_agent_pos,
                       chunks=(100, new_agent_pos.shape[1]), dtype='float32')
    
    root.attrs['task'] = str(z.attrs.get('task', ''))
    root.attrs['num_frames'] = new_num_frames
    root.attrs['has_agent_pos'] = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, 
                       default='/home/gl/RoboTwin/policy/DP2DP3/features_model/data/offline_features_pool_ws1_proprio')
    parser.add_argument('--output_dir', type=str,
                       default='/home/gl/RoboTwin/policy/DP2DP3/features_model/data/offline_features_pool_ws1_proprio_fixed')
    parser.add_argument('--raw_data_root', type=str,
                       default='/home/gl/RoboTwin/data')
    parser.add_argument('--task_config', type=str, default='demo_clean')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    raw_data_root = Path(args.raw_data_root)
    
    # 遍历所有任务
    for task_dir in sorted(input_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        task_name = task_dir.name
        print(f"\n=== Processing task: {task_name} ===")
        
        raw_data_dir = raw_data_root / task_name / args.task_config / 'data'
        if not raw_data_dir.exists():
            print(f"  Raw data not found: {raw_data_dir}, skipping")
            continue
        
        zarr_files = sorted(task_dir.glob("*.zarr"))
        print(f"  Found {len(zarr_files)} episodes")
        
        for zarr_path in tqdm(zarr_files, desc=task_name):
            ep_name = zarr_path.stem  # e.g., "episode0"
            raw_hdf5 = raw_data_dir / f"{ep_name}.hdf5"
            
            if not raw_hdf5.exists():
                print(f"  Warning: {raw_hdf5} not found, skipping")
                continue
            
            output_path = output_dir / task_name / zarr_path.name
            try:
                fix_episode(zarr_path, raw_hdf5, output_path, overwrite=args.overwrite)
            except Exception as e:
                print(f"  Error fixing {zarr_path}: {e}")
    
    # 验证
    print("\n=== Verification ===")
    for task_dir in sorted(output_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        zarr_files = sorted(task_dir.glob("*.zarr"))
        if not zarr_files:
            continue
        z = zarr.open(str(zarr_files[0]), 'r')
        ap = z['agent_pos'][:]
        act = z['action'][:]
        
        # 验证 agent_pos[t+1] == action[t]
        shift_match = sum(1 for t in range(len(ap)-1) if np.allclose(ap[t+1], act[t]))
        same_match = sum(1 for t in range(len(ap)) if np.allclose(ap[t], act[t]))
        print(f"  {task_dir.name}/{zarr_files[0].name}:")
        print(f"    agent_pos[t+1]==action[t]: {shift_match}/{len(ap)-1}")
        print(f"    agent_pos[t]==action[t]: {same_match}/{len(ap)} (should be LOW)")


if __name__ == "__main__":
    main()
