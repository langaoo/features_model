"""features_common/dp_rgb_dataset.py

Diffusion Policy 数据集：加载 RGB 蒸馏特征 + 原始轨迹数据。

数据结构：
- RGB 特征：zarr (task/episode.zarr) [T,Hf,Wf,C]
- 轨迹数据：pkl (task/_traj_data/episodeX.pkl) {'left_joint_path': [...], 'right_joint_path': [...]}
- 每个 joint path 是一个包含 (joint_angles, gripper_state) 的列表

Dataset 返回：
- obs: 融合后的 RGB 特征 [To, D]  (To = n_obs_steps)
- action: 关节动作序列 [Ta, A]  (Ta = horizon, A = action_dim)
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import zarr
from torch.utils.data import Dataset

from features_common.zarr_pack import load_zarr_pack


@dataclass
class DPSample:
    """单个训练样本"""
    task: str
    episode: str
    start_idx: int
    
    # RGB observation: fused features [To, D]
    obs: torch.Tensor
    
    # Action sequence [Ta, A]
    action: torch.Tensor


def _parse_joint_path(joint_path_list: list) -> np.ndarray:
    """
    解析 joint_path 列表为 numpy 数组。
    
    joint_path_list 可能有多种格式:
    1. [(q0, gripper0), (q1, gripper1), ...] - tuple 格式
    2. [{'status': ..., 'position': [...], 'velocity': [...]}, ...] - dict 格式
    3. [q0, q1, ...] - 直接数组
    
    返回: [T, action_dim] 数组
    """
    if not joint_path_list:
        return np.zeros((0, 0))
    
    actions = []
    for item in joint_path_list:
        if isinstance(item, dict):
            # dict 格式: 提取 position 的最后一帧（或平均）
            if 'position' in item:
                pos = np.array(item['position'])
                if pos.ndim == 2:
                    # [num_waypoints, dof] -> 取最后一帧
                    action = pos[-1].flatten()
                else:
                    action = pos.flatten()
            else:
                raise ValueError(f"Dict item missing 'position': {item.keys()}")
        elif isinstance(item, (list, tuple)):
            # tuple 格式: (joint_angles, gripper)
            joints = np.array(item[0]).flatten() if len(item) > 0 else np.array([])
            gripper = np.array([item[1]]).flatten() if len(item) > 1 else np.array([])
            action = np.concatenate([joints, gripper]) if len(gripper) > 0 else joints
        else:
            # 单个值
            action = np.array([item]).flatten()
        actions.append(action)
    
    return np.array(actions, dtype=np.float32)  # [T, A]


class DPRGBDataset(Dataset[DPSample]):
    """
    用于 Diffusion Policy 训练的数据集。
    
    参数:
        rgb_zarr_roots: 视觉特征 zarr 根目录列表
        traj_root: 轨迹数据根目录 (包含 task/demo_randomized/_traj_data/)
        tasks: 任务列表
        horizon: 预测时域长度 (action 序列长度)
        n_obs_steps: 观测历史长度
        pad_before: 前补齐长度
        pad_after: 后补齐长度
        use_left_arm: 是否使用左臂
        use_right_arm: 是否使用右臂
        fuse_arms: 是否融合双臂动作 (concat)
    """
    
    def __init__(
        self,
        *,
        rgb_zarr_roots: list[str | Path],
        traj_root: str | Path,
        tasks: list[str],
        horizon: int,
        n_obs_steps: int,
        pad_before: int = 0,
        pad_after: int = 0,
        use_left_arm: bool = True,
        use_right_arm: bool = False,
        fuse_arms: bool = False,
        seed: int = 0,
    ):
        self.rgb_zarr_roots = [Path(x) for x in rgb_zarr_roots]
        self.traj_root = Path(traj_root)
        self.tasks = list(tasks)
        self.horizon = int(horizon)
        self.n_obs_steps = int(n_obs_steps)
        self.pad_before = int(pad_before)
        self.pad_after = int(pad_after)
        self.use_left_arm = bool(use_left_arm)
        self.use_right_arm = bool(use_right_arm)
        self.fuse_arms = bool(fuse_arms)
        
        # 发现所有可用的 (task, episode, length) 三元组
        self.samples = self._discover_samples()
        
        if len(self.samples) == 0:
            raise RuntimeError("No valid samples found. Check rgb_zarr_roots and traj_root.")
        
        # 缓存 (延迟加载)
        self._pack_cache: dict[tuple[int, str, str], Any] = {}
        self._traj_cache: dict[tuple[str, str], dict] = {}
    
    def _discover_samples(self) -> list[tuple[str, str, int, str]]:
        """
        发现所有可用样本。
        
        返回: [(task, episode_zarr, episode_length, episode_pkl), ...]
        """
        samples = []
        
        for task in self.tasks:
            # 查找任务目录
            # 尝试不同可能的路径模式
            traj_task_paths = [
                self.traj_root / task / "_traj_data",  # task/_traj_data/
                self.traj_root / task.split('-')[0] / "demo_randomized" / "_traj_data",  # dump_bin_bigbin/demo_randomized/_traj_data/
            ]
            
            traj_task_dir = None
            for p in traj_task_paths:
                if p.exists():
                    traj_task_dir = p
                    break
            
            if traj_task_dir is None:
                print(f"Warning: trajectory dir not found for task {task}")
                continue
            
            # 列出所有 episode pkl
            pkl_files = sorted(traj_task_dir.glob("episode*.pkl"))
            
            for pkl in pkl_files:
                ep_name_pkl = pkl.stem  # episode0, episode1, ...
                # 转换为 zarr 命名格式: episode0 -> episode_0
                if 'episode' in ep_name_pkl:
                    ep_num = ep_name_pkl.replace('episode', '')
                    ep_name_zarr = f"episode_{ep_num}"
                else:
                    ep_name_zarr = ep_name_pkl
                
                # 检查所有 RGB zarr roots 是否存在该 episode
                has_all = True
                for root in self.rgb_zarr_roots:
                    zarr_path = root / task / f"{ep_name_zarr}.zarr"
                    if not zarr_path.exists():
                        has_all = False
                        break
                
                if not has_all:
                    continue
                
                # 加载轨迹获取长度
                try:
                    with open(pkl, "rb") as f:
                        traj = pickle.load(f)
                    
                    left = traj.get("left_joint_path", [])
                    right = traj.get("right_joint_path", [])
                    
                    if self.use_left_arm and self.use_right_arm and self.fuse_arms:
                        # 双臂融合: 需要长度一致
                        if len(left) != len(right) or len(left) == 0:
                            continue
                        ep_len = len(left)
                    elif self.use_left_arm:
                        ep_len = len(left)
                    elif self.use_right_arm:
                        ep_len = len(right)
                    else:
                        continue
                    
                    if ep_len < self.horizon:
                        continue
                    
                    # 保存时使用 zarr 格式的名称（用于后续加载）
                    samples.append((task, ep_name_zarr, ep_len, ep_name_pkl))
                
                except Exception as e:
                    # print(f"Warning: failed to load {pkl}: {e}")
                    continue
        
        return samples
    
    def __len__(self) -> int:
        # 计算所有可能的滑动窗口数量
        total = 0
        for item in self.samples:
            ep_len = item[2]  # 第3个元素是 ep_len
            # 每个 episode 的可用起始索引数
            total += max(0, ep_len - self.horizon + 1)
        return total
    
    def _get_pack(self, root_i: int, task: str, episode: str):
        key = (int(root_i), task, episode)
        pack = self._pack_cache.get(key)
        if pack is None:
            zp = self.rgb_zarr_roots[root_i] / task / f"{episode}.zarr"
            pack = load_zarr_pack(zp)
            self._pack_cache[key] = pack
        return pack
    
    def _get_traj(self, task: str, episode_zarr: str, episode_pkl: str = None) -> dict:
        # episode_zarr: episode_0 format (for zarr loading)
        # episode_pkl: episode0 format (for pkl loading)
        if episode_pkl is None:
            # Convert zarr format to pkl format: episode_0 -> episode0
            episode_pkl = episode_zarr.replace('_', '')
        
        key = (task, episode_pkl)
        traj = self._traj_cache.get(key)
        if traj is not None:
            return traj
        
        # 查找 pkl 文件（使用 pkl 格式的名称）
        traj_task_paths = [
            self.traj_root / task / "_traj_data" / f"{episode_pkl}.pkl",
            self.traj_root / task.split('-')[0] / "demo_randomized" / "_traj_data" / f"{episode_pkl}.pkl",
        ]
        
        pkl_path = None
        for p in traj_task_paths:
            if p.exists():
                pkl_path = p
                break
        
        if pkl_path is None:
            raise FileNotFoundError(
                f"Trajectory not found: task={task}, episode_zarr={episode_zarr}, episode_pkl={episode_pkl}"
            )
        
        with open(pkl_path, "rb") as f:
            traj = pickle.load(f)
        
        self._traj_cache[key] = traj
        return traj
    
    def __getitem__(self, idx: int) -> DPSample:
        # 将全局 idx 映射到 (task, episode, start_idx)
        cum = 0
        for item in self.samples:
            task = item[0]
            episode_zarr = item[1]
            ep_len = item[2]
            episode_pkl = item[3] if len(item) > 3 else episode_zarr.replace('_', '')
            
            n_starts = max(0, ep_len - self.horizon + 1)
            if idx < cum + n_starts:
                start_idx = idx - cum
                break
            cum += n_starts
        else:
            raise IndexError(f"Index {idx} out of range")
        
        episode = episode_zarr  # 用于 zarr 加载
        
        # 加载 RGB features (从所有 roots)
        # 这里简化：取第一个 root 的 features
        # 在实际使用时，你应该加载你训练好的融合模型来处理多模型 features
        pack = self._get_pack(0, task, episode)
        
        # 采样 observation: [start_idx - pad_before : start_idx + n_obs_steps]
        obs_start = start_idx - self.pad_before
        obs_end = start_idx + self.n_obs_steps
        
        # 处理边界: pad
        obs_features = []
        for t in range(obs_start, obs_end):
            if t < 0 or t >= pack.arr.shape[0]:
                # pad with zeros (or repeat first/last frame)
                if t < 0:
                    t_safe = 0
                else:
                    t_safe = pack.arr.shape[0] - 1
            else:
                t_safe = t
            
            # 获取 window: [T, Hf, Wf, C]
            # 对于每个 window，我们取第一帧作为代表 (简化)
            feat = pack.get_frame(int(t_safe), 0)  # [Hf, Wf, C]
            # Flatten spatial: [Hf*Wf, C] -> mean pool -> [C]
            feat_flat = torch.from_numpy(feat).reshape(-1, feat.shape[-1]).mean(dim=0)  # [C]
            obs_features.append(feat_flat)
        
        obs = torch.stack(obs_features, dim=0)  # [To, C]
        
        # 加载 actions (使用 pkl 格式的名称)
        traj = self._get_traj(task, episode, episode_pkl)
        left = _parse_joint_path(traj.get("left_joint_path", []))
        right = _parse_joint_path(traj.get("right_joint_path", []))
        
        if self.use_left_arm and self.use_right_arm and self.fuse_arms:
            actions = np.concatenate([left, right], axis=-1)  # [T, 2*A]
        elif self.use_left_arm:
            actions = left
        elif self.use_right_arm:
            actions = right
        else:
            raise RuntimeError("No arm selected")
        
        # 提取 action sequence: [start_idx : start_idx + horizon]
        action_seq = actions[start_idx : start_idx + self.horizon]  # [Ta, A]
        
        # 处理边界: pad
        if action_seq.shape[0] < self.horizon:
            pad = np.tile(action_seq[-1:], (self.horizon - action_seq.shape[0], 1))
            action_seq = np.concatenate([action_seq, pad], axis=0)
        
        action = torch.from_numpy(action_seq).float()  # [Ta, A]
        
        return DPSample(
            task=task,
            episode=episode,
            start_idx=start_idx,
            obs=obs,
            action=action,
        )


def collate_fn(samples: list[DPSample]) -> dict[str, torch.Tensor]:
    """
    Collate function for DataLoader.
    
    返回:
        {
            'obs': [B, To, C],
            'action': [B, Ta, A],
        }
    """
    obs = torch.stack([s.obs for s in samples], dim=0)
    action = torch.stack([s.action for s in samples], dim=0)
    
    # keep task/episode metadata on CPU for routing / logging
    tasks = [s.task for s in samples]
    episodes = [s.episode for s in samples]
    start_idx = torch.tensor([int(s.start_idx) for s in samples], dtype=torch.long)

    return {
        'obs': obs,
        'action': action,
        'task': tasks,
        'episode': episodes,
        'start_idx': start_idx,
    }
