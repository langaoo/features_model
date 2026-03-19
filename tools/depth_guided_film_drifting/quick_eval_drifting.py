#!/usr/bin/env python3
"""
快速评测 Drifting 模型:
1) 加载 checkpoint + 训练数据
2) 对训练数据运行 predict_action，比较预测 vs 真实动作
3) 统计 per-step MSE、轨迹平滑度
4) 检查输出是否在合理范围
"""
import sys, os, argparse, yaml, torch, numpy as np
from pathlib import Path

# 路径设置（与 train_film_drifting.py 一致）
current_file_path = os.path.abspath(__file__)
tools_dir = os.path.dirname(current_file_path)
tools_parent = os.path.dirname(tools_dir)
features_model_dir = os.path.dirname(tools_parent)
sys.path.insert(0, features_model_dir)

DP_OUTER = Path(features_model_dir) / "DP" / "diffusion_policy"
if DP_OUTER.exists():
    sys.path.insert(0, str(DP_OUTER))

from features_common.depth_guided_film_drifting.policy_drifting import DA3FilmDriftingPolicy
from tools.depth_guided_film_drifting.train_film_drifting import CachedTokenDataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, default="best.ckpt", help="ckpt filename")
    parser.add_argument("--n_samples", type=int, default=200, help="num samples to eval")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    gpu_ids = config["device"]["gpu_ids"]
    gpu_id = gpu_ids[0] if isinstance(gpu_ids, list) else int(gpu_ids)
    device = f"cuda:{gpu_id}"

    save_dir = config["checkpoint"]["save_dir"]
    ckpt_path = os.path.join(save_dir, args.ckpt)
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    # 从 ckpt 和 config 恢复策略
    saved_config = ckpt["config"]
    encoder_cfg = ckpt["encoder_cfg"]
    drifting_cfg = ckpt["drifting_cfg"]

    horizon = int(saved_config["data"]["horizon"])
    n_obs_steps = int(saved_config["data"]["n_obs_steps"])
    action_dim = int(saved_config["data"].get("action_dim", 14))
    proprio_dim = int(saved_config["data"].get("proprio_dim", action_dim))
    max_tokens = int(encoder_cfg.get("max_tokens", 196))

    # 构建 fusion_encoder（与训练脚本一致）
    from features_common.depth_guided_film_online.encoder_film_2model import DA3Film2ModelEncoder
    fusion_encoder = DA3Film2ModelEncoder(
        semantic_in_dim=int(encoder_cfg.get("semantic_in_dim", 768)),
        geometric_in_dim=int(encoder_cfg.get("geometric_in_dim", 2048)),
        proj_dim=int(encoder_cfg.get("proj_dim", 256)),
        film_hidden=int(encoder_cfg.get("film_hidden", 256)),
        out_dim=int(encoder_cfg.get("out_dim", 1280)),
        with_pos_enc=encoder_cfg.get("with_pos_enc", True),
        dropout=float(encoder_cfg.get("dropout", 0.1)),
        max_tokens=max_tokens,
    ).to(device)

    policy = DA3FilmDriftingPolicy(
        fusion_encoder=fusion_encoder,
        proprio_dim=proprio_dim,
        action_dim=action_dim,
        horizon=horizon,
        n_obs_steps=n_obs_steps,
        drifting_temp_scale=float(drifting_cfg.get("temp_scale", 0.05)),
        drift_normalize=bool(drifting_cfg.get("drift_normalize", False)),
        drift_norm_mode=drifting_cfg.get("drift_norm_mode", "per_obs"),
        drift_norm_eps=float(drifting_cfg.get("drift_norm_eps", 1e-6)),
        drift_norm_ema_decay=float(drifting_cfg.get("drift_norm_ema_decay", 0.99)),
        drift_target_rms=float(drifting_cfg.get("drift_target_rms", 1.0)),
        drift_use_scale=bool(drifting_cfg.get("drift_use_scale", True)),
    ).to(device)

    policy.normalizer.load_state_dict(ckpt["normalizer"])
    state = ckpt.get("ema_policy") if "ema_policy" in ckpt else ckpt["policy"]
    policy.load_state_dict(state)
    policy.eval()
    print(f"Model loaded. Epoch={ckpt.get('epoch','?')}, Loss={ckpt.get('loss','?'):.6f}")

    # 加载预提取数据
    task_name = saved_config["data"]["tasks"]
    if isinstance(task_name, list): task_name = task_name[0]
    task_config = saved_config["data"].get("task_config", "demo_clean")
    expert_num = int(saved_config.get("checkpoint", {}).get("expert_data_num", 50))

    zarr_base = saved_config["data"].get("zarr_base", "/home/gl/RoboTwin/policy/DP/data")
    zarr_name = f"{task_name}-{task_config}-{expert_num}_multi_cam.zarr"
    zarr_path = os.path.join(zarr_base, zarr_name)

    # precompute tokens (same as training)
    from features_common.depth_guided_film_online.extractors_2model import TwoModelExtractors
    from PIL import Image as PILImage
    import zarr
    zr = zarr.open(zarr_path, mode="r")
    images = np.array(zr["data"]["head_camera"])  # (N, 3, 240, 320) uint8
    states = np.array(zr["data"]["state"])
    actions = np.array(zr["data"]["action"])

    extractors = TwoModelExtractors(gpu_id=gpu_id)
    for m in [extractors.dinov3_model, extractors.da3_model]:
        m.requires_grad_(False)
        m.eval()

    print(f"Precomputing tokens for {len(images)} frames...")
    all_tokens = []
    bs = 32
    for i in range(0, len(images), bs):
        chunk = images[i:i+bs]
        pil_list = [PILImage.fromarray(chunk[j].transpose(1, 2, 0)) for j in range(chunk.shape[0])]
        with torch.no_grad():
            toks = extractors.extract_batch_tokens(pil_list, max_tokens=max_tokens, return_torch=True)
        all_tokens.append((toks[0].half().cpu(), toks[1].half().cpu()))

    dino_tokens = torch.cat([t[0] for t in all_tokens], dim=0)
    da3_tokens = torch.cat([t[1] for t in all_tokens], dim=0)

    # offload backbone
    extractors.dinov3_model.cpu()
    extractors.da3_model.cpu()
    torch.cuda.empty_cache()

    # 创建数据集
    ep_ends = np.array(zr["meta"]["episode_ends"])

    dataset = CachedTokenDataset(
        dino_tokens=dino_tokens,
        da3_tokens=da3_tokens,
        states=states,
        actions=actions,
        episode_ends=ep_ends,
        horizon=horizon,
        n_obs_steps=n_obs_steps,
    )

    # 评估
    n_samples = min(args.n_samples, len(dataset))
    mse_list = []
    step_mse_list = []
    smooth_list = []
    max_abs_list = []

    print(f"\nEvaluating {n_samples} samples...")
    indices = np.random.choice(len(dataset), n_samples, replace=False)

    for idx in indices:
        sample = dataset[idx]
        dino_tok = sample["dino_tokens"].unsqueeze(0).to(device)
        da3_tok = sample["da3_tokens"].unsqueeze(0).to(device)
        agent_pos = sample["agent_pos"].unsqueeze(0).to(device)
        gt_action = sample["action"].numpy()  # [H, D]

        pred_action = policy.predict_action(
            [dino_tok, da3_tok], agent_pos=agent_pos
        ).squeeze(0).cpu().numpy()  # [H, D]

        # Per-sample MSE
        mse = np.mean((pred_action - gt_action) ** 2)
        mse_list.append(mse)

        # Per-step MSE
        per_step = np.mean((pred_action - gt_action) ** 2, axis=1)  # [H]
        step_mse_list.append(per_step)

        # 轨迹平滑度 (consecutive step differences)
        diffs = np.diff(pred_action, axis=0)  # [H-1, D]
        smoothness = np.mean(np.abs(diffs))
        smooth_list.append(smoothness)

        # max abs value
        max_abs_list.append(np.max(np.abs(pred_action)))

    mse_arr = np.array(mse_list)
    step_mse_arr = np.array(step_mse_list)  # [N, H]
    smooth_arr = np.array(smooth_list)

    print(f"\n{'='*60}")
    print(f"评估结果 ({n_samples} samples, ckpt={args.ckpt}):")
    print(f"{'='*60}")
    print(f"  整体 MSE:     mean={mse_arr.mean():.4f}, std={mse_arr.std():.4f}")
    print(f"  整体 RMSE:    {np.sqrt(mse_arr.mean()):.4f}")
    print(f"  Per-step MSE: {' '.join(f'{v:.4f}' for v in step_mse_arr.mean(axis=0))}")
    print(f"  轨迹平滑度:   mean={smooth_arr.mean():.4f}")
    print(f"  最大绝对值:   mean={np.mean(max_abs_list):.3f}, max={np.max(max_abs_list):.3f}")
    print()

    # 检查 MSE 分布
    pcts = [10, 25, 50, 75, 90]
    pct_vals = np.percentile(mse_arr, pcts)
    print(f"  MSE 分位数:")
    for p, v in zip(pcts, pct_vals):
        print(f"    P{p:2d}: {v:.4f}")

    # 对比 GT 轨迹的统计
    gt_smooth = []
    gt_max_abs = []
    for idx in indices:
        sample = dataset[idx]
        gt = sample["action"].numpy()
        diffs = np.diff(gt, axis=0)
        gt_smooth.append(np.mean(np.abs(diffs)))
        gt_max_abs.append(np.max(np.abs(gt)))

    print(f"\n  GT 轨迹平滑度: mean={np.mean(gt_smooth):.4f}")
    print(f"  GT 最大绝对值: mean={np.mean(gt_max_abs):.3f}, max={np.max(gt_max_abs):.3f}")

    # 多次推理一致性测试 (同一输入多次推理)
    print(f"\n{'='*60}")
    print("多次推理一致性测试 (同一输入5次推理):")
    test_idx = indices[0]
    sample = dataset[test_idx]
    dino_tok = sample["dino_tokens"].unsqueeze(0).to(device)
    da3_tok = sample["da3_tokens"].unsqueeze(0).to(device)
    agent_pos = sample["agent_pos"].unsqueeze(0).to(device)

    preds = []
    for _ in range(5):
        p = policy.predict_action([dino_tok, da3_tok], agent_pos=agent_pos)
        preds.append(p.squeeze(0).cpu().numpy())

    preds = np.array(preds)  # [5, H, D]
    var = np.var(preds, axis=0).mean()
    std = np.std(preds, axis=0).mean()
    print(f"  推理方差: {var:.6f}")
    print(f"  推理标准差: {std:.6f}")
    print(f"  (注意: 因为输入包含随机噪声, 每次推理结果会不同)")

    # 打印一个示例轨迹
    print(f"\n{'='*60}")
    print("示例轨迹对比 (前3步, 前7维=左臂关节+夹爪):")
    sample = dataset[indices[0]]
    dino_tok = sample["dino_tokens"].unsqueeze(0).to(device)
    da3_tok = sample["da3_tokens"].unsqueeze(0).to(device)
    agent_pos = sample["agent_pos"].unsqueeze(0).to(device)
    gt = sample["action"].numpy()
    pred = policy.predict_action([dino_tok, da3_tok], agent_pos=agent_pos).squeeze(0).cpu().numpy()

    for step in range(min(3, horizon)):
        print(f"  Step {step}:")
        print(f"    GT:   {gt[step, :7]}")
        print(f"    Pred: {pred[step, :7]}")
        print(f"    Diff: {np.abs(gt[step, :7] - pred[step, :7])}")


if __name__ == "__main__":
    main()
