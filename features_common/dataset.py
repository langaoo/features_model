"""features_common/dataset.py

统一的数据加载器：把任意模型导出的 features_root 变成可迭代 dataset。

目录结构约定
- features_root/
    task_0/
        episode_0.pt
        episode_1.pt
    task_1/
        ...

每个 pt 推荐保存 dict：
- per_frame_features: Tensor[W,T,Hf,Wf,C]
- frame_paths:        List[List[str]]
- meta:               dict
- （可选）features

这个 loader 不做任何模型相关假设，只负责：
- 遍历 task/episode
- 加载 pt 并返回 FeaturePack
- 可选：将 episode 展开成“window 级别样本”（W 条）

"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

import torch  # type: ignore
from torch.utils.data import Dataset, IterableDataset  # type: ignore

from .feature_pack import FeaturePack, infer_task_episode_from_path, load_feature_pack


def list_episode_pts(features_root: str | Path, *, task: str | None = None, episode: str | None = None) -> list[str]:
    root = str(features_root)

    if task is None:
        pts = glob.glob(os.path.join(root, "*", "*.pt"))
    else:
        pts = glob.glob(os.path.join(root, task, "*.pt"))

    pts = sorted(pts)
    if episode is not None:
        pts = [p for p in pts if Path(p).stem == episode]

    return pts


@dataclass
class EpisodeSample:
    task: str
    episode: str
    pack: FeaturePack


@dataclass
class WindowSample:
    task: str
    episode: str
    window_index: int
    per_frame_features: torch.Tensor  # [T,Hf,Wf,C]
    frame_paths: list[str]
    meta: dict[str, Any]
    features: Optional[torch.Tensor] = None


class EpisodeFeaturesDataset(Dataset):
    """按 episode 返回样本（一次返回整个 [W,T,Hf,Wf,C]）。"""

    def __init__(
        self,
        features_root: str | Path,
        *,
        task: str | None = None,
        episode: str | None = None,
        map_location: str = "cpu",
    ):
        self.features_root = str(features_root)
        self.map_location = map_location
        self.paths = list_episode_pts(self.features_root, task=task, episode=episode)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> EpisodeSample:
        pt_path = self.paths[idx]
        task, episode = infer_task_episode_from_path(pt_path)
        pack = load_feature_pack(pt_path, map_location=self.map_location)
        return EpisodeSample(task=task, episode=episode, pack=pack)


class WindowFeaturesDataset(IterableDataset):
    """把每个 episode 展开成 window 样本（每个 window 一条）。

    适合对齐训练：你通常以 window 为训练 sample。
    """

    def __init__(
        self,
        features_root: str | Path,
        *,
        task: str | None = None,
        episode: str | None = None,
        map_location: str = "cpu",
        max_windows_per_episode: int | None = None,
        window_stride: int = 1,
    ):
        self.features_root = str(features_root)
        self.task = task
        self.episode = episode
        self.map_location = map_location
        self.max_windows_per_episode = max_windows_per_episode
        self.window_stride = int(window_stride)

    def __iter__(self) -> Iterator[WindowSample]:
        paths = list_episode_pts(self.features_root, task=self.task, episode=self.episode)
        for pt_path in paths:
            task, episode = infer_task_episode_from_path(pt_path)
            pack = load_feature_pack(pt_path, map_location=self.map_location)
            pf = pack.per_frame_features
            fps = pack.frame_paths

            if pf is None or not torch.is_tensor(pf):
                continue
            if fps is None:
                continue

            w = int(pf.shape[0])
            max_w = w if self.max_windows_per_episode is None else min(w, int(self.max_windows_per_episode))

            for wi in range(0, max_w, self.window_stride):
                yield WindowSample(
                    task=task,
                    episode=episode,
                    window_index=wi,
                    per_frame_features=pf[wi],
                    frame_paths=fps[wi],
                    meta=dict(pack.meta),
                    features=(pack.features[wi] if pack.features is not None and torch.is_tensor(pack.features) else None),
                )
