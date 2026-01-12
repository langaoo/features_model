"""
在线训练Dataset - 从raw_data的HDF5文件读取图像+轨迹
不再依赖独立的图像目录，所有数据从raw_data统一读取
"""
import torch
import numpy as np
import pickle
import h5py
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass
from PIL import Image
import io


@dataclass
class DP4SampleOnline:
    """在线训练样本"""
    task: str
    episode: str
    start_idx: int
    obs: torch.Tensor  # [To, 4, 2048] 或图像列表（取决于batch_extract模式）
    action: torch.Tensor  # [Ta, A]
    images: List = None  # 用于批量提取模式


class DPRGBOnlineDataset:
    """
    在线训练Dataset - 从raw_data读取所有数据
    
    数据结构:
    raw_data/<task>/demo_randomized/
        ├── data/episode*.hdf5          # 包含图像、轨迹、点云
        └── _traj_data/episode*.pkl     # 规划轨迹（可选）
    
    优先使用HDF5中的joint_action作为动作，pkl中的轨迹为fallback
    """
    
    def __init__(
        self,
        raw_data_root: str | Path,
        tasks: list[str],
        horizon: int,
        n_obs_steps: int,
        feature_extractors,  # MultiGPUFeatureExtractors实例
        camera_name: str = 'head_camera',
        use_left_arm: bool = True,
        use_right_arm: bool = False,
        fuse_arms: bool = False,
        include_gripper: bool = False,
        device: str = 'cuda',
        batch_extract: bool = True,  # 新增：是否批量提取特征
    ):
        self.raw_data_root = Path(raw_data_root)
        self.tasks = tasks
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.feature_extractors = feature_extractors
        self.camera_name = camera_name
        self.use_left_arm = use_left_arm
        self.use_right_arm = use_right_arm
        self.fuse_arms = fuse_arms
        self.include_gripper = include_gripper
        self.device = device
        self.batch_extract = batch_extract  # 新增
        
        # 缓存
        self._hdf5_cache: Dict[Tuple[str, str], h5py.File] = {}
        self._traj_cache: Dict[Tuple[str, str], dict] = {}
        
        # 发现样本
        self.samples: List[Tuple[str, str, int, Path]] = []
        self._discover_samples()
        
        if len(self.samples) == 0:
            raise RuntimeError(f"No valid samples found in {raw_data_root}")
        
        print(f"[Dataset] Loaded {len(self.samples)} episodes, total samples: {len(self)}")
    
    def _discover_samples(self):
        """
        发现所有有效的episode
        
        样本格式: (task, episode_name, ep_len, hdf5_path)
        """
        for task in self.tasks:
            # 提取基础任务名（去除后缀）
            base_task = task.split('-demo_randomized')[0].split('_sapien_head_camera')[0].split('_head_camera')[0]
            
            # 搜索HDF5文件
            search_paths = [
                self.raw_data_root / task / 'demo_randomized' / 'data',
                self.raw_data_root / base_task / 'demo_randomized' / 'data',
                self.raw_data_root / base_task / 'data',
            ]
            
            hdf5_dir = None
            for p in search_paths:
                if p.exists() and p.is_dir():
                    hdf5_dir = p
                    break
            
            if hdf5_dir is None:
                print(f"[Dataset] Warning: no HDF5 data for task={task}")
                continue
            
            # 遍历所有HDF5文件
            hdf5_files = sorted(hdf5_dir.glob('episode*.hdf5'))
            print(f"[Dataset] Found {len(hdf5_files)} HDF5 files for {base_task}")
            
            for hdf5_file in hdf5_files:
                ep_name = hdf5_file.stem  # episode1, episode2, ...
                
                try:
                    # 打开HDF5获取长度
                    with h5py.File(hdf5_file, 'r') as f:
                        # 检查相机是否存在
                        if f'observation/{self.camera_name}' not in f:
                            print(f"[Dataset] Warning: camera '{self.camera_name}' not found in {ep_name}, skipping")
                            continue
                        
                        # 获取帧数
                        ep_len = f[f'observation/{self.camera_name}/rgb'].shape[0]
                        
                        # 检查动作数据
                        if 'joint_action' not in f:
                            print(f"[Dataset] Warning: no joint_action in {ep_name}, skipping")
                            continue
                        
                        # 检查手臂数据
                        left_exists = 'joint_action/left_arm' in f and f['joint_action/left_arm'].shape[0] > 0
                        right_exists = 'joint_action/right_arm' in f and f['joint_action/right_arm'].shape[0] > 0
                        
                        if self.use_left_arm and not left_exists:
                            print(f"[Dataset] Skip {ep_name}: use_left_arm=True but no left_arm data")
                            continue
                        if self.use_right_arm and not right_exists:
                            print(f"[Dataset] Skip {ep_name}: use_right_arm=True but no right_arm data")
                            continue
                        if not left_exists and not right_exists:
                            print(f"[Dataset] Skip {ep_name}: no arm data")
                            continue
                        
                        # 添加样本
                        self.samples.append((task, ep_name, ep_len, hdf5_file))
                
                except Exception as e:
                    print(f"[Dataset] Error loading {hdf5_file}: {e}")
                    continue
    
    def _get_hdf5(self, task: str, episode: str) -> h5py.File:
        """获取并缓存HDF5文件句柄"""
        key = (task, episode)
        if key not in self._hdf5_cache:
            # 从samples中找到对应的路径
            hdf5_path = None
            for t, e, _, p in self.samples:
                if t == task and e == episode:
                    hdf5_path = p
                    break
            
            if hdf5_path is None:
                raise FileNotFoundError(f"HDF5 not found for {task}/{episode}")
            
            self._hdf5_cache[key] = h5py.File(hdf5_path, 'r')
        
        return self._hdf5_cache[key]
    
    def _load_image_from_hdf5(self, task: str, episode: str, frame_idx: int) -> Image.Image:
        """
        从HDF5加载图像
        
        Args:
            task: 任务名
            episode: episode名
            frame_idx: 帧索引
        
        Returns:
            PIL Image (RGB)
        """
        hdf5 = self._get_hdf5(task, episode)
        
        # 读取压缩的RGB数据
        rgb_bytes = hdf5[f'observation/{self.camera_name}/rgb'][frame_idx]
        
        # 解码JPEG
        img = Image.open(io.BytesIO(rgb_bytes))
        
        return img.convert('RGB')
    
    def _extract_features(self, task: str, episode: str, frame_idx: int) -> np.ndarray:
        """
        提取单帧的4模型特征
        
        Returns:
            features: [4, 2048]
        """
        img = self._load_image_from_hdf5(task, episode, frame_idx)
        features = self.feature_extractors(img)  # [4, 2048]
        return features
    
    def _parse_action(self, task: str, episode: str, start_idx: int) -> np.ndarray:
        """
        从HDF5提取动作序列
        
        Returns:
            action: [horizon, A]
        """
        hdf5 = self._get_hdf5(task, episode)
        
        actions = []
        
        # 左臂
        if self.use_left_arm:
            left_pos = hdf5['joint_action/left_arm'][start_idx:start_idx + self.horizon]
            actions.append(left_pos)
            
            if self.include_gripper and 'joint_action/left_gripper' in hdf5:
                left_gripper = hdf5['joint_action/left_gripper'][start_idx:start_idx + self.horizon]
                left_gripper = left_gripper.reshape(-1, 1)
                actions.append(left_gripper)
        
        # 右臂
        if self.use_right_arm:
            right_pos = hdf5['joint_action/right_arm'][start_idx:start_idx + self.horizon]
            if self.fuse_arms:
                actions.append(right_pos)
            else:
                actions = [right_pos]
            
            if self.include_gripper and 'joint_action/right_gripper' in hdf5:
                right_gripper = hdf5['joint_action/right_gripper'][start_idx:start_idx + self.horizon]
                right_gripper = right_gripper.reshape(-1, 1)
                actions.append(right_gripper)
        
        action = np.concatenate(actions, axis=-1)
        
        # Pad if needed
        if len(action) < self.horizon:
            pad_len = self.horizon - len(action)
            action = np.concatenate([action, np.tile(action[-1:], (pad_len, 1))], axis=0)
        
        return action.astype(np.float32)
    
    def __len__(self) -> int:
        total = 0
        for _task, _ep, ep_len, _path in self.samples:
            n_starts = ep_len - self.horizon + 1
            total += max(1, n_starts)
        return total
    
    def __getitem__(self, idx: int) -> DP4SampleOnline:
        # 映射 global idx → (task, episode, start_idx)
        n = int(idx)
        for task, episode, ep_len, _path in self.samples:
            n_starts = max(1, ep_len - self.horizon + 1)
            if n < n_starts:
                start_idx = n
                break
            n -= n_starts
        else:
            raise IndexError(f"Index {idx} out of range")
        
        # 解析动作
        action = self._parse_action(task, episode, start_idx)  # [Ta, A]
        
        if self.batch_extract:
            # 批量提取模式：返回图像列表，在collate_fn中批量提取
            images = []
            for i in range(start_idx, start_idx + self.n_obs_steps):
                if i >= ep_len:
                    i = ep_len - 1
                img = self._load_image_from_hdf5(task, episode, i)
                images.append(img)
            
            return DP4SampleOnline(
                task=task,
                episode=episode,
                start_idx=start_idx,
                obs=None,  # 稍后在collate_fn中填充
                action=torch.from_numpy(action).float(),
                images=images,
            )
        else:
            # 逐样本提取模式（旧方式）
            obs_features = []
            for i in range(start_idx, start_idx + self.n_obs_steps):
                if i >= ep_len:
                    i = ep_len - 1
                
                feat_4 = self._extract_features(task, episode, i)  # [4, 2048]
                obs_features.append(feat_4)
            
            obs = np.stack(obs_features, axis=0)  # [To, 4, 2048]
            
            return DP4SampleOnline(
                task=task,
                episode=episode,
                start_idx=start_idx,
                obs=torch.from_numpy(obs).float(),
                action=torch.from_numpy(action).float(),
                images=None,
            )
    
    def __del__(self):
        """关闭所有HDF5文件"""
        for hdf5 in self._hdf5_cache.values():
            try:
                hdf5.close()
            except:
                pass


def collate_fn_online_4(batch: List[DP4SampleOnline]) -> Dict[str, torch.Tensor]:
    """Collate function - 支持批量特征提取"""
    # 检查是否批量提取模式
    if batch[0].images is not None:
        # 批量提取模式 - 需要在这里批量提取特征
        # 但collate_fn无法访问dataset的extractors
        # 解决方案：返回images，在训练循环中提取
        raise NotImplementedError("Use make_batch_collate_fn to create collate with extractors")
    else:
        # 逐样本提取模式（特征已提取）
        obs = torch.stack([s.obs for s in batch])  # [B, To, 4, 2048]
    
    action = torch.stack([s.action for s in batch])  # [B, Ta, A]
    return {'obs': obs, 'action': action}


class BatchCollateFn:
    """支持批量特征提取的collate函数（可pickle）"""
    def __init__(self, feature_extractors):
        self.feature_extractors = feature_extractors
    
    def __call__(self, batch: List[DP4SampleOnline]) -> Dict[str, torch.Tensor]:
        if batch[0].images is not None:
            # 批量提取模式
            B = len(batch)
            To = len(batch[0].images)
            
            # 收集所有图像 [B*To]
            all_images = []
            for sample in batch:
                all_images.extend(sample.images)
            
            # 批量提取特征 [B*To, 4, 2048]
            all_features = []
            for img in all_images:
                feat = self.feature_extractors(img)  # [4, 2048]
                all_features.append(feat)
            
            all_features = np.stack(all_features, axis=0)  # [B*To, 4, 2048]
            
            # 重组为 [B, To, 4, 2048]
            obs = torch.from_numpy(all_features).float().reshape(B, To, 4, 2048)
        else:
            # 特征已提取
            obs = torch.stack([s.obs for s in batch])
        
        action = torch.stack([s.action for s in batch])
        return {'obs': obs, 'action': action}


def make_batch_collate_fn(feature_extractors):
    """创建支持批量特征提取的collate函数"""
    return BatchCollateFn(feature_extractors)
