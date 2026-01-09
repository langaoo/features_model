"""features_common/dp_rgb_dataset_hdf5_online.py

从 RoBoTwin HDF5 直接读取 RGB + Action，实时提取 4 模型特征（不落盘）。

用途：
- 当你有 RoBoTwin 格式的 HDF5 数据（包含完整 14 维 action）
- 不想先提特征再训练，而是"图像/视频流 → 实时提特征 → 训练 DP head"
- 这是**真正的在线训练**（虽然数据来自文件，但特征是实时生成的）

与 dp_rgb_dataset_4models.py 的区别：
- 旧版：读预先提取的 zarr 特征（离线）
- 本版：读 HDF5 RGB，实时提特征（在线）

注意：
- 需要加载 4 个 backbone（CroCo/VGGT/DINOv3/DA3），显存占用大
- 训练速度会慢（每个 batch 都要跑 4 个 backbone forward）
- 推荐：先用离线 zarr 跑通训练，再考虑用本版做增强/微调
"""

from __future__ import annotations

import h5py
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset

# 需要导入 4 个 backbone（示例：你需要根据实际路径调整）
# from croco.extract_multi_frame_croco_features_unified import load_croco_model, extract_croco_features
# from vggt.extract_features import load_vggt_model, extract_vggt_features
# from dinov3.extract_features import load_dinov3_model, extract_dinov3_features
# from Depth-Anything-3.extract_features import load_da3_model, extract_da3_features


@dataclass
class DP4SampleHDF5:
    task: str
    episode: str
    start_idx: int
    obs: torch.Tensor      # [To, 4, C]  # 实时提取的特征
    action: torch.Tensor   # [Ta, 14]    # 从 HDF5 读的完整 action


class DPRGBHDF5OnlineDataset(Dataset[DP4SampleHDF5]):
    """从 RoBoTwin HDF5 实时提特征并训练 DP head"""

    def __init__(
        self,
        *,
        hdf5_root: str | Path,
        tasks: list[str],
        horizon: int,
        n_obs_steps: int,
        device: str = 'cuda',
        # 4 个 backbone 的模型实例（需要预先加载）
        backbone_models: dict[str, Any] = None,
    ):
        self.hdf5_root = Path(hdf5_root)
        self.tasks = list(tasks)
        self.horizon = int(horizon)
        self.n_obs_steps = int(n_obs_steps)
        self.device = device
        self.backbone_models = backbone_models or {}

        self.samples = self._discover_samples()
        if len(self.samples) == 0:
            raise RuntimeError('No valid HDF5 samples found')

        self._hdf5_cache: dict[tuple[str, str], h5py.File] = {}

    def _discover_samples(self) -> list[tuple[str, str, int]]:
        """扫描 HDF5 文件并记录 (task, episode, ep_len)"""
        samples = []
        for task in self.tasks:
            task_dir = self.hdf5_root / task / 'demo_randomized' / 'data'
            if not task_dir.exists():
                print(f"[DPRGBHDF5OnlineDataset] Warning: {task_dir} not found")
                continue

            for hdf5_path in sorted(task_dir.glob('episode*.hdf5')):
                ep_name = hdf5_path.stem  # episode0
                try:
                    with h5py.File(hdf5_path, 'r') as f:
                        # 读取 action 长度（假设 left_arm 存在）
                        if 'joint_action' not in f or 'left_arm' not in f['joint_action']:
                            continue
                        ep_len = f['joint_action']['left_arm'].shape[0]
                    samples.append((task, ep_name, ep_len))
                except Exception as e:
                    print(f"[DPRGBHDF5OnlineDataset] Error loading {hdf5_path}: {e}")
                    continue

        return samples

    def __len__(self) -> int:
        total = 0
        for _task, _ep, ep_len in self.samples:
            total += max(1, ep_len - self.horizon + 1)
        return total

    def _get_hdf5(self, task: str, episode: str) -> h5py.File:
        """打开并缓存 HDF5 文件"""
        key = (task, episode)
        if key not in self._hdf5_cache:
            hdf5_path = self.hdf5_root / task / 'demo_randomized' / 'data' / f"{episode}.hdf5"
            self._hdf5_cache[key] = h5py.File(hdf5_path, 'r')
        return self._hdf5_cache[key]

    def _decode_rgb(self, rgb_bytes: bytes) -> np.ndarray:
        """解码 HDF5 里的 RGB bytes（JPG/PNG）→ numpy [H,W,3]"""
        img = Image.open(io.BytesIO(rgb_bytes))
        return np.array(img)

    def _extract_features_4models(self, rgb: np.ndarray) -> np.ndarray:
        """
        对单帧 RGB 提取 4 模型特征 → [4, C]
        
        TODO: 你需要实现这里，调用你的 4 个 backbone
        示例伪代码：
        ```
        croco_feat = self.backbone_models['croco'].forward(rgb)  # [1024]
        vggt_feat = self.backbone_models['vggt'].forward(rgb)    # [2048]
        dinov3_feat = self.backbone_models['dinov3'].forward(rgb)  # [768]
        da3_feat = self.backbone_models['da3'].forward(rgb)      # [2048]
        
        # pad 到 max_dim=2048
        features = [
            pad_to_2048(croco_feat, in_dim=1024),
            pad_to_2048(vggt_feat, in_dim=2048),
            pad_to_2048(dinov3_feat, in_dim=768),
            pad_to_2048(da3_feat, in_dim=2048),
        ]
        return np.stack(features, axis=0)  # [4, 2048]
        ```
        """
        # 当前占位：返回随机特征（你需要替换为真实 backbone forward）
        max_dim = 2048
        features = np.random.randn(4, max_dim).astype(np.float32)
        return features

    def __getitem__(self, idx: int) -> DP4SampleHDF5:
        # 映射 global idx → (task, episode, start)
        n = int(idx)
        for task, episode, ep_len in self.samples:
            n_starts = max(1, ep_len - self.horizon + 1)
            if n < n_starts:
                start = n
                break
            n -= n_starts
        else:
            raise IndexError(idx)

        # 打开 HDF5
        f = self._get_hdf5(task, episode)

        # 读取 obs（RGB）：n_obs_steps 帧
        rgb_bytes_arr = f['observation']['head_camera']['rgb'][:]
        frames = []
        for s in range(start, start + self.n_obs_steps):
            if s >= len(rgb_bytes_arr):
                s = len(rgb_bytes_arr) - 1  # padding
            rgb = self._decode_rgb(rgb_bytes_arr[s])
            features_4 = self._extract_features_4models(rgb)  # [4, 2048]
            frames.append(features_4)
        obs = torch.from_numpy(np.stack(frames, axis=0)).to(torch.float32)  # [To, 4, 2048]

        # 读取 action：horizon 步（14 维）
        left_arm = f['joint_action']['left_arm'][start:start + self.horizon]
        left_grip = f['joint_action']['left_gripper'][start:start + self.horizon]
        right_arm = f['joint_action']['right_arm'][start:start + self.horizon]
        right_grip = f['joint_action']['right_gripper'][start:start + self.horizon]

        # 拼成 14 维：[left_arm(6), left_grip(1), right_arm(6), right_grip(1)]
        action = np.concatenate([
            left_arm,
            left_grip[:, None],
            right_arm,
            right_grip[:, None],
        ], axis=-1)  # [horizon, 14]

        # padding if needed
        if action.shape[0] < self.horizon:
            pad = np.repeat(action[-1:], self.horizon - action.shape[0], axis=0)
            action = np.concatenate([action, pad], axis=0)

        action_t = torch.from_numpy(action).to(torch.float32)

        return DP4SampleHDF5(
            task=task,
            episode=episode,
            start_idx=int(start),
            obs=obs,
            action=action_t,
        )

    def __del__(self):
        """关闭所有 HDF5 文件"""
        for f in self._hdf5_cache.values():
            try:
                f.close()
            except:
                pass


def collate_fn_hdf5(batch: list[DP4SampleHDF5]) -> dict:
    obs = torch.stack([b.obs for b in batch], dim=0)
    action = torch.stack([b.action for b in batch], dim=0)
    task = [b.task for b in batch]
    episode = [b.episode for b in batch]
    start_idx = torch.tensor([b.start_idx for b in batch], dtype=torch.long)
    return {
        'obs': obs,
        'action': action,
        'task': task,
        'episode': episode,
        'start_idx': start_idx,
    }
