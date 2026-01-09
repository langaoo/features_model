"""features_common/dp_rgb_dataset_4models.py

单任务/多任务通用：读取 4 个视觉模型的 zarr 特征，并返回 [To, 4, C]。

与 `features_common/dp_rgb_dataset.py` 的区别：
- 原版只读一个 root -> obs: [To, C]
- 本版读四个 root（croco/vggt/dinov3/da3）-> obs: [To, 4, C]

依赖：
- zarr packs 是 `features_common/zarr_pack.py` 生成/读取的格式（per_frame_features: [W,T,Hf,Wf,C]）。

注意：
- 这里仍然是“离线特征”。你想要的“在线跑”最终需要把视频->特征抽取放到推理环节。
  但训练时 encoder 冻结，离线特征训练是最稳且最快的闭环方式。
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from features_common.zarr_pack import load_zarr_pack
from features_common.dp_rgb_dataset import _parse_joint_path


@dataclass
class DP4Sample:
    task: str
    episode: str
    start_idx: int
    obs: torch.Tensor      # [To, 4, C]
    action: torch.Tensor   # [Ta, A]


class DPRGB4ModelDataset(Dataset[DP4Sample]):
    def __init__(
        self,
        *,
        rgb_zarr_roots_4: list[str | Path],
        traj_root: str | Path,
        tasks: list[str],
        horizon: int,
        n_obs_steps: int,
        pad_before: int = 0,
        pad_after: int = 0,
        use_left_arm: bool = True,
        use_right_arm: bool = False,
        fuse_arms: bool = False,
        include_gripper: bool = False,
    ):
        if len(rgb_zarr_roots_4) != 4:
            raise ValueError(f"rgb_zarr_roots_4 must have 4 roots, got {len(rgb_zarr_roots_4)}")
        self.rgb_zarr_roots_4 = [Path(x) for x in rgb_zarr_roots_4]
        self.traj_root = Path(traj_root)
        self.tasks = list(tasks)
        self.horizon = int(horizon)
        self.n_obs_steps = int(n_obs_steps)
        self.pad_before = int(pad_before)
        self.pad_after = int(pad_after)
        self.use_left_arm = bool(use_left_arm)
        self.use_right_arm = bool(use_right_arm)
        self.fuse_arms = bool(fuse_arms)
        self.include_gripper = bool(include_gripper)

        self.samples = self._discover_samples()
        if len(self.samples) == 0:
            raise RuntimeError('No valid samples found for 4-model dataset. Check roots and traj_root.')

        self._pack_cache: dict[tuple[int, str, str], Any] = {}
        self._traj_cache: dict[tuple[str, str], dict] = {}

    def _discover_samples(self) -> list[tuple[str, str, int, str]]:
        samples = []
        for task in self.tasks:
            base_task = task.split('-demo_randomized')[0]
            traj_task_paths = [
                self.traj_root / task / '_traj_data',
                self.traj_root / task.split('-')[0] / 'demo_randomized' / '_traj_data',
                self.traj_root / base_task / 'demo_randomized' / '_traj_data',
            ]
            traj_task_dir = None
            for p in traj_task_paths:
                if p.exists():
                    traj_task_dir = p
                    break
            if traj_task_dir is None:
                print(f"[DPRGB4ModelDataset] Warning: trajectory dir not found for task={task} (base={base_task})")
                continue

            pkl_files = sorted(traj_task_dir.glob('episode*.pkl'))
            for pkl in pkl_files:
                ep_name_pkl = pkl.stem
                if 'episode' in ep_name_pkl:
                    ep_num = ep_name_pkl.replace('episode', '')
                    ep_name_zarr = f"episode_{ep_num}"
                else:
                    ep_name_zarr = ep_name_pkl

                # check all 4 roots exist
                ok = True
                for root in self.rgb_zarr_roots_4:
                    zp = root / task / f"{ep_name_zarr}.zarr"
                    if not zp.exists():
                        ok = False
                        break
                if not ok:
                    # common failure: zarr task folder naming mismatch
                    # print a single succinct line per episode
                    # (keep it lightweight; caller can debug with ls)
                    pass
                    continue

                try:
                    with open(pkl, 'rb') as f:
                        traj = pickle.load(f)
                    left = traj.get('left_joint_path', [])
                    right = traj.get('right_joint_path', [])

                    if self.use_left_arm and self.use_right_arm and self.fuse_arms:
                        # dual-arm: allow slight length mismatch; we'll pad in __getitem__.
                        if len(left) == 0 and len(right) == 0:
                            continue
                        ep_len = max(len(left), len(right))
                    elif self.use_left_arm:
                        ep_len = len(left)
                    elif self.use_right_arm:
                        ep_len = len(right)
                    else:
                        continue

                    # Allow short episodes; we'll pad action window in __getitem__.
                    if ep_len <= 0:
                        continue

                    samples.append((task, ep_name_zarr, ep_len, ep_name_pkl))
                except Exception:
                    continue

        return samples

    def __len__(self) -> int:
        total = 0
        for _task, _ep, ep_len, _pkl in self.samples:
            # For short episodes, still provide at least one sample (start=0).
            total += max(1, ep_len - self.horizon + 1)
        return total

    def _get_pack(self, root_i: int, task: str, episode: str):
        key = (root_i, task, episode)
        pack = self._pack_cache.get(key)
        if pack is None:
            zp = self.rgb_zarr_roots_4[root_i] / task / f"{episode}.zarr"
            pack = load_zarr_pack(zp)
            self._pack_cache[key] = pack
        return pack

    def _get_traj(self, task: str, episode_zarr: str, episode_pkl: str) -> dict:
        key = (task, episode_pkl)
        traj = self._traj_cache.get(key)
        if traj is not None:
            return traj

        traj_task_paths = [
            self.traj_root / task / '_traj_data' / f"{episode_pkl}.pkl",
            self.traj_root / task.split('-')[0] / 'demo_randomized' / '_traj_data' / f"{episode_pkl}.pkl",
            self.traj_root / task.split('-demo_randomized')[0] / 'demo_randomized' / '_traj_data' / f"{episode_pkl}.pkl",
        ]
        pkl_path = None
        for p in traj_task_paths:
            if p.exists():
                pkl_path = p
                break
        if pkl_path is None:
            raise FileNotFoundError(f"traj pkl not found for task={task} episode={episode_pkl}")

        with open(pkl_path, 'rb') as f:
            traj = pickle.load(f)
        self._traj_cache[key] = traj
        return traj

    def __getitem__(self, idx: int) -> DP4Sample:
        # map global idx -> (episode, start)
        n = int(idx)
        for task, ep_zarr, ep_len, ep_pkl in self.samples:
            n_starts = max(1, ep_len - self.horizon + 1)
            if n < n_starts:
                start = n
                break
            n -= n_starts
        else:
            raise IndexError(idx)

        # obs steps use [start : start+n_obs_steps]
        # expected dims per model from alignment ckpt (adapter input dims)
        # 0: croco=1024, 1:vggt=2048, 2:dinov3=768, 3:da3=2048
        expect_dims = (1024, 2048, 768, 2048)
        max_dim = int(max(expect_dims))

        frames = []
        # we index into ZarrPack: [W,T,...], flatten time to step
        packs = [self._get_pack(i, task, ep_zarr) for i in range(4)]
        W, T = packs[0].shape[0], packs[0].shape[1]
        total_steps = W * T
        if start + self.n_obs_steps > total_steps:
            start = max(0, total_steps - self.n_obs_steps)

        for s in range(start, start + self.n_obs_steps):
            wi = s // T
            ti = s % T
            per_model = []
            for mi, pack in enumerate(packs):
                f = pack.get_frame(wi, ti)  # [Hf,Wf,C]
                f = f.reshape(-1, f.shape[-1]).mean(axis=0)  # [C]
                # slice/pad to expected dim (in case extraction config differs)
                ed = int(expect_dims[mi])
                # first slice/pad to model-specific dim
                if f.shape[0] >= ed:
                    f_ed = f[:ed]
                else:
                    f_ed = np.zeros((ed,), dtype=f.dtype)
                    f_ed[: f.shape[0]] = f
                # then pad to max_dim so we can stack across models
                if ed < max_dim:
                    f2 = np.zeros((max_dim,), dtype=f.dtype)
                    f2[:ed] = f_ed
                else:
                    f2 = f_ed
                per_model.append(f2)
            frames.append(np.stack(per_model, axis=0))  # [4,max_dim]

        # NOTE: C here is max per-model dim after pad; we keep as [To,4,Cmax] by padding to 2048.
        obs = torch.from_numpy(np.stack(frames, axis=0)).to(torch.float32)  # [To,4,2048]

        traj = self._get_traj(task, ep_zarr, ep_pkl)
        left = traj.get('left_joint_path', [])
        right = traj.get('right_joint_path', [])

        # Parse joint actions (position only)
        if self.use_left_arm and self.use_right_arm and self.fuse_arms:
            a_l = _parse_joint_path(left)
            a_r = _parse_joint_path(right)
            # pad shorter side by repeating last frame so we can concat safely
            if a_l.shape[0] == 0 and a_r.shape[0] == 0:
                action = np.zeros((0, 0), dtype=np.float32)
            else:
                T = max(a_l.shape[0], a_r.shape[0])
                if a_l.shape[0] == 0:
                    a_l = np.zeros((T, a_r.shape[1]), dtype=np.float32)
                elif a_l.shape[0] < T:
                    pad = np.repeat(a_l[-1:], T - a_l.shape[0], axis=0)
                    a_l = np.concatenate([a_l, pad], axis=0)

                if a_r.shape[0] == 0:
                    a_r = np.zeros((T, a_l.shape[1]), dtype=np.float32)
                elif a_r.shape[0] < T:
                    pad = np.repeat(a_r[-1:], T - a_r.shape[0], axis=0)
                    a_r = np.concatenate([a_r, pad], axis=0)

                action = np.concatenate([a_l, a_r], axis=-1)
        elif self.use_left_arm:
            action = _parse_joint_path(left)
        else:
            action = _parse_joint_path(right)

        # If include_gripper=True, append gripper dimension(s)
        if self.include_gripper:
            # For each arm, append one gripper dim (default: 0.5 = half-open).
            # In a real scenario, you'd infer this from observations or heuristics.
            # action shape: [T, 6] (single arm) or [T, 12] (dual arm)
            if action.ndim == 2 and action.shape[0] > 0:
                T_a = action.shape[0]
                if self.use_left_arm and self.use_right_arm and self.fuse_arms:
                    # dual-arm: append left_gripper + right_gripper
                    grip_l = np.full((T_a, 1), 0.5, dtype=np.float32)
                    grip_r = np.full((T_a, 1), 0.5, dtype=np.float32)
                    action = np.concatenate([action[:, :6], grip_l, action[:, 6:], grip_r], axis=-1)
                elif self.use_left_arm or self.use_right_arm:
                    # single-arm: append one gripper
                    grip = np.full((T_a, 1), 0.5, dtype=np.float32)
                    action = np.concatenate([action, grip], axis=-1)
            else:
                # fallback: action empty or malformed
                pass

        # pick horizon window, pad if needed
        action_win = action[start:start + self.horizon]
        if action_win.shape[0] < self.horizon:
            if action_win.shape[0] == 0:
                # cannot infer action_dim; fall back to zeros
                action_dim = action.shape[1] if action.ndim == 2 else 0
                action_win = np.zeros((self.horizon, action_dim), dtype=np.float32)
            else:
                pad = np.repeat(action_win[-1:], self.horizon - action_win.shape[0], axis=0)
                action_win = np.concatenate([action_win, pad], axis=0)
        action_t = torch.from_numpy(action_win).to(torch.float32)

        return DP4Sample(task=task, episode=ep_zarr, start_idx=int(start), obs=obs, action=action_t)


def collate_fn_4(batch: list[DP4Sample]) -> dict:
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
