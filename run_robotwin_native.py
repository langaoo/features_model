#!/usr/bin/env python3
"""
RoBoTwin原生推理 - 端到端部署
直接使用RoBoTwin官方环境和推理流程
"""
import sys
import os
from pathlib import Path
import numpy as np
import argparse

# 切换到RoBoTwin目录 (必须在导入前完成)
ROBOTWIN_DIR = Path(__file__).parent / "RoBoTwin"
os.chdir(ROBOTWIN_DIR)
sys.path.insert(0, str(ROBOTWIN_DIR))

# 导入RoBoTwin环境
from envs import beat_block_hammer
import yaml

# 返回项目根目录并导入Policy
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from robotwin_policy_adapter import load_robotwin_policy


def load_robotwin_config():
    """加载RoBoTwin环境配置"""
    # Embodiment配置
    embodiment_config_path = ROBOTWIN_DIR / "task_config" / "_embodiment_config.yml"
    with open(embodiment_config_path, "r") as f:
        embodiment_types = yaml.safe_load(f)
    
    embodiment_name = "aloha-agilex"
    robot_file_path = embodiment_types[embodiment_name]["file_path"]
    robot_full_path = ROBOTWIN_DIR / robot_file_path.lstrip("./")
    
    with open(robot_full_path / "config.yml", "r") as f:
        robot_config = yaml.safe_load(f)
    
    # Camera配置
    camera_config_path = ROBOTWIN_DIR / "task_config" / "_camera_config.yml"
    with open(camera_config_path, "r") as f:
        camera_configs = yaml.safe_load(f)
    
    camera_type = "D435"
    camera_config = camera_configs[camera_type]
    
    return {
        'left_robot_file': str(robot_full_path),
        'right_robot_file': str(robot_full_path),
        'left_embodiment_config': robot_config,
        'right_embodiment_config': robot_config,
        'camera_config': camera_config,
    }


def run_robotwin_native_inference(
    rgb2pc_ckpt: str,
    head_ckpt: str,
    task_name: str = 'beat_block_hammer',
    num_episodes: int = 10,
    max_steps: int = 1000,
    gpu_ids: list = [0, 1],
    render: bool = False,
):
    """
    RoBoTwin原生推理
    
    Args:
        rgb2pc_ckpt: RGB2PC编码器checkpoint (绝对路径)
        head_ckpt: DP头checkpoint (绝对路径)
        task_name: 任务名称
        num_episodes: Episodes数量
        max_steps: 每个episode最大步数
        gpu_ids: GPU列表
        render: 是否渲染
    """
    
    print("="*60)
    print("RoBoTwin原生推理")
    print("="*60)
    print(f"任务: {task_name}")
    print(f"Episodes: {num_episodes}")
    print(f"最大步数: {max_steps}")
    print(f"RGB2PC: {rgb2pc_ckpt}")
    print(f"Head: {head_ckpt}")
    print(f"GPU: {gpu_ids}")
    print(f"渲染: {render}")
    print("="*60 + "\n")
    
    # 1. 加载Policy
    policy = load_robotwin_policy(
        rgb2pc_ckpt=rgb2pc_ckpt,
        head_ckpt=head_ckpt,
        gpu_ids=gpu_ids,
        device=f'cuda:{gpu_ids[0]}',
    )
    
    # 2. 加载RoBoTwin配置
    config = load_robotwin_config()
    
    # 3. 运行episodes
    print("="*60)
    print("开始推理")
    print("="*60 + "\n")
    
    success_count = 0
    total_rewards = []
    
    for ep in range(num_episodes):
        print(f"\n{'='*60}")
        print(f"Episode {ep+1}/{num_episodes}")
        print(f"{'='*60}")
        
        # 创建环境
        env = beat_block_hammer.beat_block_hammer()
        
        # Setup环境
        setup_kwargs = {
            'headless': (not render),
            'use_ray_tracing': False,
            'task_name': task_name,
            'now_ep_num': ep,
            'seed': ep,
            'domain_randomization': {
                'random_background': False,
                'cluttered_table': False,
                'clean_background_rate': 1.0,
                'random_head_camera_dis': 0,
                'random_table_height': 0,
                'random_light': False,
                'crazy_random_light_rate': 0,
                'random_embodiment': False,
            },
            'left_robot_file': config['left_robot_file'],
            'right_robot_file': config['right_robot_file'],
            'left_embodiment_config': config['left_embodiment_config'],
            'right_embodiment_config': config['right_embodiment_config'],
            'dual_arm_embodied': True,
            'head_camera_h': config['camera_config']['h'],
            'head_camera_w': config['camera_config']['w'],
        }
        
        env.setup_demo(**setup_kwargs)
        
        # 重置
        obs = env.reset()
        policy.reset()
        
        # 获取RGB观测
        if 'head_camera' in obs['observation']:
            rgb = obs['observation']['head_camera']['rgb']
        else:
            # Fallback: 找第一个相机
            cam_keys = [k for k in obs['observation'].keys() if 'camera' in k]
            if not cam_keys:
                print("  ✗ 无法找到相机观测，跳过episode")
                continue
            rgb = obs['observation'][cam_keys[0]]['rgb']
        
        # Rollout
        done = False
        step = 0
        episode_reward = 0
        
        while not done and step < max_steps:
            # Policy预测
            action = policy.predict(rgb)
            
            # 执行动作 (RoBoTwin原生接口)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            step += 1
            
            # 获取下一帧RGB
            if 'head_camera' in obs['observation']:
                rgb = obs['observation']['head_camera']['rgb']
            else:
                cam_keys = [k for k in obs['observation'].keys() if 'camera' in k]
                if cam_keys:
                    rgb = obs['observation'][cam_keys[0]]['rgb']
            
            # 简单进度显示
            if step % 10 == 0:
                print(f"  Step {step}/{max_steps}, Reward: {episode_reward:.2f}")
        
        # Episode统计
        total_rewards.append(episode_reward)
        
        is_success = info.get('is_success', False) if isinstance(info, dict) else False
        if is_success:
            success_count += 1
            print(f"\n  ✓ Episode {ep+1} 成功!")
        else:
            print(f"\n  ✗ Episode {ep+1} 失败")
        
        print(f"  总步数: {step}, 总奖励: {episode_reward:.2f}")
        
        # 清理环境
        env.close()
    
    # 最终统计
    print("\n" + "="*60)
    print("推理完成!")
    print("="*60)
    print(f"成功率: {success_count}/{num_episodes} ({100*success_count/num_episodes:.1f}%)")
    print(f"平均奖励: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="RoBoTwin原生推理")
    parser.add_argument('--rgb2pc_ckpt', type=str, required=True,
                       help='RGB2PC对齐编码器checkpoint路径')
    parser.add_argument('--head_ckpt', type=str, required=True,
                       help='DP头checkpoint路径')
    parser.add_argument('--task', type=str, default='beat_block_hammer',
                       help='任务名称')
    parser.add_argument('--num_episodes', type=int, default=10,
                       help='Episodes数量')
    parser.add_argument('--max_steps', type=int, default=1000,
                       help='每个episode最大步数')
    parser.add_argument('--gpu_ids', type=int, nargs='+', default=[0, 1],
                       help='GPU列表')
    parser.add_argument('--render', action='store_true',
                       help='是否渲染')
    
    args = parser.parse_args()
    
    # 转换为绝对路径
    rgb2pc_ckpt = str(Path(args.rgb2pc_ckpt).resolve())
    head_ckpt = str(Path(args.head_ckpt).resolve())
    
    run_robotwin_native_inference(
        rgb2pc_ckpt=rgb2pc_ckpt,
        head_ckpt=head_ckpt,
        task_name=args.task,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        gpu_ids=args.gpu_ids,
        render=args.render,
    )


if __name__ == '__main__':
    main()
