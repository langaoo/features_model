"""tools/train_dp_rgb_multitask.py

多任务训练入口：
- 任务列表从 raw_data 自动扫描（B 方案）
- 共享 RGB2PC 蒸馏 encoder（可冻结）
- 每个任务一个 policy head（action_dim 根据数据自动推断）

注意：当前 DPRGBDataset 会把 task 目录名直接作为 task 字段。
但它在 _discover_samples 里使用了 task list 并尝试两种 traj 路径模式。

本脚本的策略：
1) 扫描 raw_data 下一级目录作为 base task，例如 shake_bottle/
2) 拼接出 dp_rgb_dataset 能识别的 task 名：
   - 优先探测是否存在 {base}-demo_randomized-20_head_camera 这样的目录结构（在 rgb zarr root 下）
   - 若不存在，则回退到 base 名本身

为了尽快跑通工程链路，这里默认按 action_dim 分组训练：
- 发现每个 task 的 action_dim 后，把相同 action_dim 的 tasks 放到一个训练 run 中。
- 对于 action_dim 不同的任务，会分别训练对应 head checkpoint。

后续如果你确实需要一个 batch 混合不同 action_dim 的任务，需要对 dataset/collate 做 padding+mask。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ensure vendored diffusion_policy package is importable
DP_ROOT = REPO_ROOT / "DP" / "diffusion_policy"
if str(DP_ROOT) not in sys.path:
    sys.path.insert(0, str(DP_ROOT))

from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer

from features_common.dp_rgb_dataset import DPRGBDataset, collate_fn
from features_common.dp_rgb_policy_multitask import MultiTaskDiffusionRGBPolicy, HeadSpec
from features_common.rgb2pc_student_encoder import RGB2PCStudentEncoder


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def scan_tasks_from_raw_data(raw_data_root: Path) -> List[str]:
    # B 方案：扫描 raw_data 下的一级目录
    bases = [p.name for p in raw_data_root.iterdir() if p.is_dir() and not p.name.startswith('.')]
    bases = sorted(bases)
    return bases


def resolve_task_names(*, bases: List[str], rgb_zarr_root: Path) -> List[str]:
    """把 base task 映射成训练用 task 名。

    仓库现有配置使用 {base}-demo_randomized-20_head_camera。
    我们优先用这个，如果在 rgb_zarr_root 下存在对应目录。
    """

    out = []
    for b in bases:
        cand = f"{b}-demo_randomized-20_head_camera"
        if (rgb_zarr_root / cand).exists():
            out.append(cand)
        elif (rgb_zarr_root / b).exists():
            out.append(b)
        else:
            # keep cand as default; dp_rgb_dataset 里 traj_root 也支持 base/demo_randomized/_traj_data
            out.append(cand)
    return out


def create_obs_action_normalizers(dataset: DPRGBDataset, *, num_samples: int = 2000):
    # shared obs normalizer
    n = min(int(num_samples), len(dataset))
    idxs = np.random.choice(len(dataset), n, replace=False)

    obs_all = []
    task_to_actions: Dict[str, List[torch.Tensor]] = defaultdict(list)
    for i in idxs:
        s = dataset[int(i)]
        obs_all.append(s.obs)
        task_to_actions[s.task].append(s.action)

    obs_all_t = torch.stack(obs_all, dim=0)
    obs_norm = SingleFieldLinearNormalizer.create_fit(obs_all_t)

    action_norms = {}
    for task, act_list in task_to_actions.items():
        a = torch.stack(act_list, dim=0)
        action_norms[task] = SingleFieldLinearNormalizer.create_fit(a)

    return obs_norm, action_norms


def train_one_epoch(policy, dataloader, optimizer, device, epoch, *, use_amp=False, log_every=50, use_tqdm=True):
    policy.train()
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    total_loss = 0.0
    n = 0

    it = enumerate(dataloader)
    if use_tqdm and tqdm is not None:
        it = tqdm(it, total=len(dataloader), desc=f"Epoch {epoch}")

    optimizer.zero_grad(set_to_none=True)

    for bi, batch in it:
        # move tensors to device; keep task/episode as python list
        batch = {
            k: (v.to(device) if torch.is_tensor(v) else v)
            for k, v in batch.items()
        }

        if use_amp:
            with torch.cuda.amp.autocast():
                loss = policy.compute_loss(batch)
        else:
            loss = policy.compute_loss(batch)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        optimizer.zero_grad(set_to_none=True)

        total_loss += float(loss.item())
        n += 1

        if (bi + 1) % int(log_every) == 0:
            print(f"  batch {bi+1}/{len(dataloader)} loss={loss.item():.4f} avg={total_loss/max(1,n):.4f}")

    return {"loss": total_loss / max(1, n)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, required=True)
    ap.add_argument('--save_dir', type=str, default='outputs/dp_rgb_runs/multitask')
    args = ap.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    set_seed(int(cfg.get('seed', 0)))

    device = torch.device(str(cfg.get('device', 'cuda')))
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    rgb_zarr_roots = [Path(p) for p in cfg['rgb_zarr_roots']]
    traj_root = Path(cfg['traj_root'])

    # scan tasks from raw_data
    bases = scan_tasks_from_raw_data(traj_root)
    tasks = resolve_task_names(bases=bases, rgb_zarr_root=rgb_zarr_roots[0])

    # allow include/exclude filters
    include = cfg.get('task_include', [])
    exclude = cfg.get('task_exclude', [])
    if include:
        tasks = [t for t in tasks if any(inc in t for inc in include)]
    if exclude:
        tasks = [t for t in tasks if not any(exc in t for exc in exclude)]

    print(f"Discovered {len(tasks)} tasks from raw_data")

    # build dataset
    dataset = DPRGBDataset(
        rgb_zarr_roots=rgb_zarr_roots,
        traj_root=traj_root,
        tasks=tasks,
        horizon=int(cfg['horizon']),
        n_obs_steps=int(cfg['n_obs_steps']),
        pad_before=int(cfg.get('pad_before', 0)),
        pad_after=int(cfg.get('pad_after', 0)),
        use_left_arm=bool(cfg.get('use_left_arm', True)),
        use_right_arm=bool(cfg.get('use_right_arm', False)),
        fuse_arms=bool(cfg.get('fuse_arms', False)),
        seed=int(cfg.get('seed', 0)),
    )

    # infer dims and action_dim per task by sampling
    # since dataset flattens all episodes, we sample a few items and build mapping.
    task_to_action_dim = {}
    for i in range(min(500, len(dataset))):
        s = dataset[i]
        task_to_action_dim.setdefault(s.task, int(s.action.shape[-1]))
        if len(task_to_action_dim) >= len(tasks):
            break

    # group tasks by action_dim
    groups: Dict[int, List[str]] = defaultdict(list)
    for t, ad in task_to_action_dim.items():
        groups[int(ad)].append(t)

    print('Task groups by action_dim:', {k: len(v) for k, v in groups.items()})

    # shared encoder
    ckpt_path = str(cfg['rgb_ckpt_path'])
    encoder = RGB2PCStudentEncoder.from_checkpoint(ckpt_path, freeze=bool(cfg.get('freeze_encoder', True)))

    # run one group per process invocation (for simplicity)
    # choose max group by default
    target_action_dim = int(cfg.get('target_action_dim', 0))
    if target_action_dim == 0:
        target_action_dim = max(groups.keys())

    group_tasks = sorted(groups[target_action_dim])
    print(f"Training group action_dim={target_action_dim} with {len(group_tasks)} tasks")

    # rebuild dataset filtered for this group
    dataset_g = DPRGBDataset(
        rgb_zarr_roots=rgb_zarr_roots,
        traj_root=traj_root,
        tasks=group_tasks,
        horizon=int(cfg['horizon']),
        n_obs_steps=int(cfg['n_obs_steps']),
        pad_before=int(cfg.get('pad_before', 0)),
        pad_after=int(cfg.get('pad_after', 0)),
        use_left_arm=bool(cfg.get('use_left_arm', True)),
        use_right_arm=bool(cfg.get('use_right_arm', False)),
        fuse_arms=bool(cfg.get('fuse_arms', False)),
        seed=int(cfg.get('seed', 0)),
    )

    dl = DataLoader(
        dataset_g,
        batch_size=int(cfg.get('batch_size', 32)),
        shuffle=True,
        num_workers=int(cfg.get('num_workers', 8)),
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=int(cfg.get('num_workers', 8)) > 0,
    )

    # build head specs (one per task)
    # obs_feature_dim equals encoder.spec.fuse_dim
    head_specs = {
        t: HeadSpec(
            action_dim=target_action_dim,
            horizon=int(cfg['horizon']),
            n_obs_steps=int(cfg['n_obs_steps']),
            n_action_steps=int(cfg['n_action_steps']),
            obs_feature_dim=int(encoder.spec.fuse_dim),
            obs_as_global_cond=bool(cfg.get('obs_as_global_cond', True)),
        )
        for t in group_tasks
    }

    policy = MultiTaskDiffusionRGBPolicy(encoder=encoder, head_specs=head_specs).to(device)

    # normalizers
    obs_norm, action_norms = create_obs_action_normalizers(dataset_g, num_samples=int(cfg.get('norm_samples', 2000)))
    policy.set_normalizers(obs_normalizer=obs_norm, action_normalizers=action_norms)

    # optimizer: only heads if encoder frozen
    params = list(policy.heads.parameters())
    if not bool(cfg.get('freeze_encoder', True)):
        params = list(policy.parameters())

    opt = torch.optim.AdamW(
        params,
        lr=float(cfg.get('lr', 1e-4)),
        betas=tuple(cfg.get('betas', [0.95, 0.999])),
        eps=float(cfg.get('eps', 1e-8)),
        weight_decay=float(cfg.get('weight_decay', 1e-6)),
    )

    # train
    epochs = int(cfg.get('epochs', 10))
    for ep in range(epochs):
        m = train_one_epoch(
            policy, dl, opt, device, ep,
            use_amp=bool(cfg.get('amp', False)),
            log_every=int(cfg.get('log_every', 50)),
            use_tqdm=bool(cfg.get('tqdm', True)),
        )
        print(f"epoch {ep+1}/{epochs}: {m}")

        if (ep + 1) % int(cfg.get('save_every', 10)) == 0:
            out = save_dir / f"ckpt_actiondim{target_action_dim}_epoch_{ep+1:04d}.pt"
            torch.save({
                'epoch': ep + 1,
                'encoder': policy.encoder.state_dict(),
                'heads': policy.heads.state_dict(),
                'normalizer': policy.normalizer.state_dict(),
                'cfg': cfg,
                'group_tasks': group_tasks,
                'action_dim': target_action_dim,
            }, out)
            print('saved', out)

    final = save_dir / f"final_actiondim{target_action_dim}.pt"
    torch.save({
        'encoder': policy.encoder.state_dict(),
        'heads': policy.heads.state_dict(),
        'normalizer': policy.normalizer.state_dict(),
        'cfg': cfg,
        'group_tasks': group_tasks,
        'action_dim': target_action_dim,
    }, final)
    print('saved', final)


if __name__ == '__main__':
    main()
