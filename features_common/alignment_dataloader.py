"""features_common/alignment_dataloader.py

为 `tools/train_alignment_skeleton.py` 提供 DataLoader 版本的数据供给。

目标
- 把“多任务、多 episode、每 step 随机采样一个 window + 一个 anchor 帧”的逻辑移到 DataLoader worker 中。
- 主进程只做：前向、loss、反传。

输出样本（一个 dict）
- task, episode: str
- step_stem: str  (例如 "step_0123")
- uv: np.ndarray [N,2] float32
- xyz: torch.Tensor [N,3] float32 (CPU)
- pc_feat: torch.Tensor [N,256] float32 (CPU) or None

说明
- 这里不读取视觉特征 pack（因为它们太大，且一旦进 worker，多进程会导致内存爆炸）。
  视觉 pack 仍由主进程按照 LRU 缓存读取。
- 该 dataloader 主要解决：
  1) 点云 step 文件的读取与解析
  2) (task,episode) 的随机采样
  3) window->anchor 的 step 选择

"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import random

import numpy as np
import torch
from torch.utils.data import IterableDataset

from .pointcloud import read_ascii_ply_xyz_rgb


@dataclass(frozen=True)
class PcSample:
    task: str
    episode: str
    step_stem: str
    uv: np.ndarray  # [N,2] float32
    xyz: torch.Tensor  # [N,3] float32 (CPU)
    pc_feat: Optional[torch.Tensor]  # [N,256] float32 (CPU)


def _sorted_steps_in_dir(ep_dir: Path, *, suffix: str) -> list[Path]:
    items = sorted(ep_dir.glob(f"step_*{suffix}"))
    if not items:
        raise FileNotFoundError(f"目录下没有 step_*{suffix}: {ep_dir}")

    def _key(p: Path) -> int:
        try:
            return int(p.stem.split("_")[-1])
        except Exception:
            return 0

    return sorted(items, key=_key)


def _sorted_steps_in_dir_by_stem(ep_dir: Path, *, stem_contains: str) -> list[Path]:
    items = sorted(ep_dir.glob(f"step_*{stem_contains}*"))
    if not items:
        raise FileNotFoundError(f"目录下没有 step_*{stem_contains}*: {ep_dir}")

    def _key(p: Path) -> int:
        name = p.name
        try:
            after = name.split("step_")[-1]
            digits = after.split(".")[0]
            return int(digits)
        except Exception:
            return 0

    return sorted(items, key=_key)


class PcWindowAnchorIterable(IterableDataset):
    """无限流式数据：每次 yield 一个 (task,episode,anchor_step) 的点云样本。"""

    def __init__(
        self,
        *,
        tasks: list[str],
        episodes: int,
        pc_root: str,
        window_size: int,
        stride: int,
        anchor_in_window: str = "middle",
        seed: int = 0,
    ):
        super().__init__()
        self.tasks = list(tasks)
        self.episodes = int(episodes)
        self.pc_root = str(pc_root)
        self.window_size = int(window_size)
        self.stride = int(stride)
        self.anchor_in_window = str(anchor_in_window)
        self.seed = int(seed)

    def _pick_anchor_k(self, rng: random.Random) -> int:
        s = self.window_size
        if self.anchor_in_window == "first":
            return 0
        if self.anchor_in_window == "last":
            return s - 1
        if self.anchor_in_window == "random":
            return rng.randrange(s)
        return s // 2

    def _resolve_episode_dir_and_steps(self, task: str, episode: str):
        ep_dir = Path(self.pc_root) / task / episode
        if not ep_dir.exists():
            raise FileNotFoundError(f"pc episode dir 不存在: {ep_dir}")

        try:
            steps = _sorted_steps_in_dir(ep_dir, suffix=".ply")
            fmt = "ply"
        except FileNotFoundError:
            steps = _sorted_steps_in_dir_by_stem(ep_dir, stem_contains=".ply.ulip_")
            fmt = "ulip_pt"
        return ep_dir, steps, fmt

    def __iter__(self):
        # 每个 worker 都有自己的 RNG
        worker = torch.utils.data.get_worker_info()
        wid = 0 if worker is None else int(worker.id)
        rng = random.Random(self.seed + 9973 * wid)

        while True:
            task = rng.choice(self.tasks)
            ei = rng.randrange(self.episodes)
            episode = f"episode_{ei}"

            _ep_dir, steps, fmt = self._resolve_episode_dir_and_steps(task, episode)
            S = self.window_size
            stride = self.stride

            max_start = len(steps) - (S - 1) * stride - 1
            if max_start < 0:
                # episode 太短，跳过
                continue
            start = rng.randrange(max_start + 1)
            idxs = [start + i * stride for i in range(S)]

            k = self._pick_anchor_k(rng)
            idx_step = idxs[k]
            p = steps[idx_step]

            # 统一 step_stem: step_XXXX
            name = p.name
            if name.startswith("step_"):
                step_stem = "step_" + name.split("step_")[-1].split(".")[0]
            else:
                step_stem = p.stem

            if fmt == "ply":
                pc = read_ascii_ply_xyz_rgb(str(p))
                if pc.uv is None:
                    # 没 uv 就无法对齐
                    continue
                uv = pc.uv.astype(np.float32)
                xyz = torch.from_numpy(pc.xyz.astype(np.float32))
                yield PcSample(task=task, episode=episode, step_stem=step_stem, uv=uv, xyz=xyz, pc_feat=None)

            elif fmt == "ulip_pt":
                obj = torch.load(str(p), map_location="cpu")
                if not isinstance(obj, dict):
                    continue
                uv_t = obj.get("uv", None)
                if uv_t is None:
                    uv_t = obj.get("uvs", None)
                if uv_t is None:
                    continue
                uv = np.asarray(uv_t, dtype=np.float32)

                pc_t = obj.get("pc", None)
                if pc_t is None:
                    continue
                pc_t = torch.as_tensor(pc_t).to(torch.float32)
                if pc_t.ndim != 2 or pc_t.shape[1] < 3:
                    continue
                xyz = pc_t[:, :3].contiguous()

                pc_feat_t = obj.get("pc_feat", None)
                pc_feat = None
                if pc_feat_t is not None:
                    pc_feat_t = torch.as_tensor(pc_feat_t).to(torch.float32)
                    if pc_feat_t.ndim == 2 and pc_feat_t.shape[0] == xyz.shape[0] and pc_feat_t.shape[1] == 256:
                        pc_feat = pc_feat_t.contiguous()

                yield PcSample(task=task, episode=episode, step_stem=step_stem, uv=uv, xyz=xyz, pc_feat=pc_feat)

            else:
                continue
