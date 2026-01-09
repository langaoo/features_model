"""tools/check_action_dims.py

检查所有任务的 action 维度，帮助按 action_dim 分组任务。

用法:
    python tools/check_action_dims.py --traj_root /home/gl/features_model/raw_data
"""

from __future__ import annotations

import argparse
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np


def parse_action_from_joint_path(joint_path_list: list) -> Optional[np.ndarray]:
    """解析 joint_path 获取 action 维度"""
    if not joint_path_list or len(joint_path_list) == 0:
        return None
    
    item = joint_path_list[0]
    
    if isinstance(item, dict):
        # dict 格式: 提取 position 的最后一帧
        if 'position' in item:
            pos = np.array(item['position'])
            if pos.ndim == 2:
                action = pos[-1].flatten()
            else:
                action = pos.flatten()
        else:
            return None
    elif isinstance(item, (list, tuple)):
        # tuple 格式: (joint_angles, gripper)
        joints = np.array(item[0]).flatten() if len(item) > 0 else np.array([])
        gripper = np.array([item[1]]).flatten() if len(item) > 1 else np.array([])
        action = np.concatenate([joints, gripper]) if len(gripper) > 0 else joints
    else:
        # 单个值
        action = np.array([item]).flatten()
    
    return action


def main():
    ap = argparse.ArgumentParser(description="检查任务的 action 维度")
    ap.add_argument("--traj_root", type=str, required=True, help="轨迹数据根目录")
    ap.add_argument("--tasks", nargs="+", help="任务列表（默认自动发现）")
    args = ap.parse_args()
    
    traj_root = Path(args.traj_root)
    
    # 发现任务
    if args.tasks:
        tasks = args.tasks
    else:
        # 自动发现
        tasks = [d.name for d in traj_root.iterdir() if d.is_dir() and not d.name.startswith('_')]
        print(f"自动发现 {len(tasks)} 个任务目录")
    
    # 检查每个任务的 action_dim
    task_info = []
    action_dim_tasks = defaultdict(list)
    
    for task in tasks:
        # 尝试多种路径模式
        patterns = [
            traj_root / task / "_traj_data",
            traj_root / task / "demo_randomized" / "_traj_data",
        ]
        
        traj_dir = None
        for p in patterns:
            if p.exists():
                traj_dir = p
                break
        
        if traj_dir is None:
            print(f"⚠️  {task}: 找不到 _traj_data 目录")
            continue
        
        # 读取第一个 episode
        pkl_files = sorted(traj_dir.glob("episode*.pkl"))
        if not pkl_files:
            print(f"⚠️  {task}: 没有 episode pkl 文件")
            continue
        
        pkl_path = pkl_files[0]
        
        try:
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            
            left = data.get('left_joint_path', [])
            right = data.get('right_joint_path', [])
            
            # 解析 action
            left_action = parse_action_from_joint_path(left) if left else None
            right_action = parse_action_from_joint_path(right) if right else None
            
            left_dim = len(left_action) if left_action is not None else 0
            right_dim = len(right_action) if right_action is not None else 0
            
            info = {
                'task': task,
                'left_dim': left_dim,
                'right_dim': right_dim,
                'left_steps': len(left),
                'right_steps': len(right),
                'pkl_path': pkl_path,
            }
            
            task_info.append(info)
            
            # 分组
            if left_dim > 0:
                action_dim_tasks[f'left_{left_dim}'].append(task)
            if right_dim > 0:
                action_dim_tasks[f'right_{right_dim}'].append(task)
            if left_dim > 0 and right_dim > 0:
                action_dim_tasks[f'dual_{left_dim}+{right_dim}'].append(task)
            
            status = "✅" if (left_dim > 0 or right_dim > 0) else "❌"
            print(f"{status} {task}:")
            print(f"     Left: {info['left_steps']} steps, action_dim={left_dim}")
            print(f"     Right: {info['right_steps']} steps, action_dim={right_dim}")
        
        except Exception as e:
            print(f"❌ {task}: 读取失败 - {e}")
    
    # 输出汇总
    print("\n" + "="*80)
    print("按 action_dim 分组:")
    print("="*80)
    
    for group, tasks in sorted(action_dim_tasks.items()):
        print(f"\n【{group}】 - {len(tasks)} 个任务:")
        for t in tasks:
            print(f"  - {t}")
    
    # 生成配置文件建议
    print("\n" + "="*80)
    print("配置文件建议:")
    print("="*80)
    
    for group, tasks in sorted(action_dim_tasks.items()):
        if not group.startswith('dual'):  # 只输出单臂配置
            arm_type, dim = group.split('_')
            print(f"\n# configs/train_dp_rgb_{group}.yaml")
            print(f"# {len(tasks)} 个任务, {arm_type} arm, action_dim={dim}")
            print("tasks:")
            for t in tasks:
                print(f"  - {t}")
            print(f"\nuse_left_arm: {arm_type == 'left'}")
            print(f"use_right_arm: {arm_type == 'right'}")
            print("fuse_arms: false")
            print(f"\nsave_dir: outputs/dp_rgb_runs/{group}")
            print(f"wandb_run_name: dp_rgb_{group}")


if __name__ == "__main__":
    main()
