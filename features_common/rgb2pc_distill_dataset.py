"""features_common/rgb2pc_distill_dataset.py

路线2：RGB-only 蒸馏的数据管线（DataLoader 化）。

目标
- 把“采样 task/episode/step/window + 读取 zarr + 读取 teacher pt + token 采样”
  从训练 loop 里移到 Dataset/worker 中，支持多进程预取。
- 训练端只做：student 向量化前向 + loss/backward/step。

设计约定（contract）
- Dataset 返回一个 sample（不含 batch 维），字段：
  - task, episode: str（调试/日志用）
  - mode: 'step' | 'window'
  - teacher:
    - step: pc_feat_tensor: FloatTensor[Kt,256]（已采样好点）
    - window: teacher_embed: FloatTensor[256]（已聚合好）
  - student_tokens_by_model: List[Tensor[K, C_i]]（每个模型固定 K tokens；dtype float16/float32，device=cpu）

性能策略
- worker 内 lazily 打开 zarr（每个 (root, task, episode) 一个句柄），避免跨进程共享。
- worker 内缓存 frame index（step_stem -> (wi,ti)）加速 strict_pairing。

注意
- 这里不依赖 torch.utils.data.get_worker_info 以外的全局状态。
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import random
from pathlib import Path
from typing import Any, Optional

import torch
import zarr
import numpy as np
from torch.utils.data import Dataset

from features_common.zarr_pack import load_zarr_pack


def _discover_pairs(
    pc_root: Path,
    vis_zarr_roots: list[Path],
    tasks: list[str],
    episodes: int,
) -> list[tuple[str, str]]:
    """返回所有可训练 (task, episode) 对。

    复用训练脚本里同样的策略：teacher + 所有 student roots 同时存在才算可用。

    episodes==0：自动扫描 vis_zarr_roots[0]/task 下的 episode_*.zarr，并取交集。
    episodes>0：按 0..episodes-1 枚举。
    """

    def ep_name(i: int) -> str:
        return f"episode_{i}"

    pairs: list[tuple[str, str]] = []
    for task in tasks:
        # discover episodes
        if int(episodes) == 0:
            cand = []
            base = vis_zarr_roots[0] / task
            if base.exists():
                cand = [p.stem for p in sorted(base.glob("episode_*.zarr"))]
            eps = cand
        else:
            eps = [ep_name(i) for i in range(int(episodes))]

        for ep in eps:
            # teacher episode dir must exist
            # Check for episode dir OR episode.zarr dir
            tdir = pc_root / task / ep
            tzarr = pc_root / task / f"{ep}.zarr"
            
            if not tdir.exists() and not tzarr.exists():
                continue
            # all zarr roots must have the episode
            ok = True
            for r in vis_zarr_roots:
                zp = r / task / f"{ep}.zarr"
                if not zp.exists():
                    ok = False
                    break
            if ok:
                pairs.append((task, ep))

    return pairs


def _step_stem_from_path(p: str) -> str:
    # path ends with step_XXXX.png / step_XXXX.jpg
    name = Path(p).name
    return Path(name).stem


@dataclass
class DistillSample:
    task: str
    episode: str
    sample_unit: str  # 'step' | 'window'

    # teacher
    teacher_points: Optional[torch.Tensor] = None  # [Kt,256], float32
    teacher_embed: Optional[torch.Tensor] = None  # [256], float32

    # student
    tokens_by_model: Optional[list[torch.Tensor]] = None  # list of [K,C_i] on CPU


class RGB2PCDistillDataset(Dataset[DistillSample]):
    def __init__(
        self,
        *,
        pc_root: str | Path,
        vis_zarr_roots: list[str | Path],
        tasks: list[str],
        episodes: int,
        sample_unit: str,
        student_tokens: int,
        teacher_points: int,
        strict_pairing: bool,
        pairing_fallback: str,
        seed: int = 0,
    ):
        self.pc_root = Path(pc_root)
        self.vis_zarr_roots = [Path(x) for x in vis_zarr_roots]
        self.tasks = list(tasks)
        self.episodes = int(episodes)
        self.sample_unit = str(sample_unit)
        self.student_tokens = int(student_tokens)
        self.teacher_points = int(teacher_points)
        self.strict_pairing = bool(strict_pairing)
        self.pairing_fallback = str(pairing_fallback)

        self.rng = random.Random(int(seed))

        self.pairs = _discover_pairs(self.pc_root, self.vis_zarr_roots, self.tasks, self.episodes)
        if len(self.pairs) == 0:
            raise RuntimeError("No available (task,episode) pairs found. Check roots and task list.")

        # worker-local caches (created lazily)
        self._pack_cache: dict[tuple[int, str, str], Any] = {}
        self._frame_index_cache: dict[tuple[int, str, str], dict[str, tuple[int, int]]] = {}

    def __len__(self) -> int:
        # streaming dataset: length is artificial; DataLoader will sample endlessly by cycling
        # Keep it large so that epoch semantics are not relied upon.
        return 10_000_000

    def _get_pack(self, root_i: int, task: str, episode: str):
        key = (int(root_i), task, episode)
        pack = self._pack_cache.get(key)
        if pack is None:
            zp = self.vis_zarr_roots[root_i] / task / f"{episode}.zarr"
            pack = load_zarr_pack(zp)
            self._pack_cache[key] = pack
        return pack

    def _get_frame_index(self, root_i: int, task: str, episode: str) -> dict[str, tuple[int, int]]:
        key = (int(root_i), task, episode)
        idx = self._frame_index_cache.get(key)
        if idx is not None:
            return idx
        pack = self._get_pack(root_i, task, episode)
        # build map: step_stem -> (wi,ti)
        m: dict[str, tuple[int, int]] = {}
        if pack.frame_paths is not None:
            for wi, fps in enumerate(pack.frame_paths):
                for ti, p in enumerate(fps):
                    m[_step_stem_from_path(str(p))] = (int(wi), int(ti))
        self._frame_index_cache[key] = m
        return m

    def _sample_task_episode(self) -> tuple[str, str]:
        return self.rng.choice(self.pairs)

    def _sample_teacher_step_points(self, task: str, episode: str) -> torch.Tensor:
        # choose a random ulip step file (Zarr)
        ep_dir = self.pc_root / task / episode
        # glob for .zarr
        steps = sorted(ep_dir.glob("step_*.ply.ulip_*.zarr"))
        if not steps:
            # fallback to .pt if no zarr found (backward compatibility)
            steps = sorted(ep_dir.glob("step_*.ply.ulip_*.pt"))
            if not steps:
                raise FileNotFoundError(f"no teacher steps (zarr/pt): {ep_dir}")
            
            # .pt path
            pt = self.rng.choice(steps)
            obj = torch.load(str(pt), map_location="cpu", weights_only=False)
            pc_feat = obj.get("pc_feat") if isinstance(obj, dict) else None
            if pc_feat is None:
                raise RuntimeError(f"pc_feat missing in {pt}")
            pc_feat = pc_feat.to(torch.float32)
        else:
            # .zarr path
            zp = self.rng.choice(steps)
            # open zarr array
            arr = zarr.open(str(zp), mode="r")
            # [N, D]
            pc_feat = torch.from_numpy(arr[:]).to(torch.float32)

        n = int(pc_feat.shape[0])
        k = min(int(self.teacher_points), n)
        idx = torch.randint(0, n, (k,), device="cpu")
        return pc_feat[idx]  # [K,D]

    def _teacher_window_embed(self, task: str, episode: str, frame_paths: list[str]) -> torch.Tensor:
        # aggregate 8 steps into one [D]
        z_list: list[torch.Tensor] = []
        ep_dir = self.pc_root / task / episode
        for p in frame_paths:
            stem = _step_stem_from_path(str(p))
            # try zarr first
            cand = sorted(ep_dir.glob(f"{stem}.ply.ulip_*.zarr"))
            if cand:
                arr = zarr.open(str(cand[0]), mode="r")
                pc_feat = torch.from_numpy(arr[:]).to(torch.float32)
                z_list.append(pc_feat.mean(dim=0))
                continue
            
            # fallback to pt
            cand = sorted(ep_dir.glob(f"{stem}.ply.ulip_*.pt"))
            if cand:
                obj = torch.load(str(cand[0]), map_location="cpu", weights_only=False)
                pc_feat = obj.get("pc_feat") if isinstance(obj, dict) else None
                if pc_feat is not None:
                    z_list.append(pc_feat.to(torch.float32).mean(dim=0))

        if not z_list:
            raise RuntimeError(f"teacher window embed empty: {task}/{episode}")
        return torch.stack(z_list, dim=0).mean(dim=0)  # [D]

    def _sample_tokens_fixedK_step(self, pack: Any, *, root_i: int, task: str, episode: str, step_stem: str) -> torch.Tensor:
        # returns [K,C]
        K = int(self.student_tokens)
        if self.strict_pairing:
            idx = self._get_frame_index(root_i, task, episode)
            if step_stem in idx:
                wi, ti = idx[step_stem]
                x = pack.get_frame(int(wi), int(ti))  # [Hf,Wf,C]
            else:
                if self.pairing_fallback == "error":
                    raise KeyError(f"missing step_stem={step_stem} in {task}/{episode}")
                if self.pairing_fallback == "skip":
                    raise RuntimeError("skip")
                # random fallback: sample random frame
                wi = self.rng.randrange(int(pack.arr.shape[0]))
                ti = self.rng.randrange(int(pack.arr.shape[1]))
                x = pack.get_frame(int(wi), int(ti))
        else:
            wi = self.rng.randrange(int(pack.arr.shape[0]))
            ti = self.rng.randrange(int(pack.arr.shape[1]))
            x = pack.get_frame(int(wi), int(ti))

        # zarr pack returns numpy; unify to torch
        if not torch.is_tensor(x):
            x = torch.from_numpy(x)

        # flatten spatial -> tokens
        toks = x.reshape(-1, x.shape[-1])  # [S,C]
        S = int(toks.shape[0])
        if S <= K:
            # pad by repeat
            idx = torch.randint(0, S, (K,), device="cpu")
        else:
            idx = torch.randint(0, S, (K,), device="cpu")
        return toks[idx].contiguous()

    def _sample_tokens_fixedK_window(self, pack: Any, *, wi: int) -> torch.Tensor:
        # returns [K,C]
        K = int(self.student_tokens)
        x = pack.get_window(int(wi))  # [T,Hf,Wf,C]
        if not torch.is_tensor(x):
            x = torch.from_numpy(x)
        T = int(x.shape[0])
        per = max(1, K // max(1, T))
        toks_all: list[torch.Tensor] = []
        for ti in range(T):
            toks = x[ti].reshape(-1, x.shape[-1])
            S = int(toks.shape[0])
            idx = torch.randint(0, S, (per,), device="cpu")
            toks_all.append(toks[idx])
        out = torch.cat(toks_all, dim=0)
        # fix length to K
        if out.shape[0] >= K:
            out = out[:K]
        else:
            pad = out[torch.randint(0, out.shape[0], (K - out.shape[0],), device="cpu")]
            out = torch.cat([out, pad], dim=0)
        return out.contiguous()

    def __getitem__(self, idx: int) -> DistillSample:
        # ignore idx; sample randomly
        task, episode = self._sample_task_episode()

        # load all packs
        packs = [self._get_pack(i, task, episode) for i in range(len(self.vis_zarr_roots))]

        if self.sample_unit == "window":
            W = int(packs[0].arr.shape[0])
            wi = self.rng.randrange(W)
            # teacher window embed
            if packs[0].frame_paths is None:
                raise RuntimeError("frame_paths missing in pack")
            t_embed = self._teacher_window_embed(task, episode, packs[0].frame_paths[wi])
            toks_by_model = [self._sample_tokens_fixedK_window(p, wi=wi) for p in packs]
            return DistillSample(task=task, episode=episode, sample_unit="window", teacher_embed=t_embed, tokens_by_model=toks_by_model)

        # step mode
        # Check for new format: pc_root/task/episode.zarr
        ep_zarr = self.pc_root / task / f"{episode}.zarr"
        if ep_zarr.exists():
            # New format (ULIP episode-level zarr): per_frame_features is per-step embedding.
            # Shape is expected to be [num_steps, D]. We treat it as teacher_embed (global per-step embedding),
            # not teacher_points (point set).
            g = zarr.open(str(ep_zarr), mode="r")
            feats = g["per_frame_features"]
            if feats.ndim != 2:
                raise RuntimeError(f"teacher episode zarr per_frame_features must be 2D [T,D], got shape={feats.shape}")

            num_frames = int(feats.shape[0])
            idx_frame = self.rng.randrange(num_frames)
            stem = f"step_{idx_frame:04d}"

            # Teacher embedding [D]
            t_embed = torch.from_numpy(feats[idx_frame]).to(torch.float32).contiguous()

            # Student tokens (strictly paired by step stem when strict_pairing=True)
            toks_by_model: list[torch.Tensor] = []
            for i, p in enumerate(packs):
                try:
                    toks_by_model.append(
                        self._sample_tokens_fixedK_step(p, root_i=i, task=task, episode=episode, step_stem=stem)
                    )
                except RuntimeError as e:
                    if str(e) == "skip":
                        return self.__getitem__(idx + 1)
                    raise

            return DistillSample(task=task, episode=episode, sample_unit="step", teacher_embed=t_embed, tokens_by_model=toks_by_model)

        # pick a random teacher step stem
        ep_dir = self.pc_root / task / episode
        # try zarr first
        steps = sorted(ep_dir.glob("step_*.ply.ulip_*.zarr"))
        if not steps:
            steps = sorted(ep_dir.glob("step_*.ply.ulip_*.pt"))
        
        if not steps:
            raise FileNotFoundError(f"no teacher steps: {ep_dir}")
        pt = self.rng.choice(steps)
        stem = pt.name.split(".ply")[0]  # step_XXXX

        # teacher points
        t_points = self._sample_teacher_step_points(task, episode)

        # student tokens for that step
        toks_by_model: list[torch.Tensor] = []
        for i, p in enumerate(packs):
            try:
                toks_by_model.append(self._sample_tokens_fixedK_step(p, root_i=i, task=task, episode=episode, step_stem=stem))
            except RuntimeError as e:
                if str(e) == "skip":
                    # resample another item
                    return self.__getitem__(idx + 1)
                raise

        return DistillSample(task=task, episode=episode, sample_unit="step", teacher_points=t_points, tokens_by_model=toks_by_model)
