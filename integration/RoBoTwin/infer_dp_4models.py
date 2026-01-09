"""RoBoTwin/policy/infer_dp_4models.py

在 RoBoTwin 仿真中使用 4-model DP head 推理的最小 wrapper。

功能：
1. 加载 final_head.pt（包含 head + encoder_ckpt + normalizer）
2. 从 RoBoTwin 环境获取最近 To 帧 RGB 观测
3. 提取 4 模型特征 → 对齐 encoder → head → 输出 14 维动作（7+7）
4. 调用 env.take_action(action, action_type='qpos')

使用场景：
- 离线回放：从 zarr 读已有特征
- 在线推理：实时从环境获取 RGB → 提特征（需要 4 个 backbone 都能 streaming）

注意：
- 本脚本默认"离线回放模式"（读 zarr）；若要真在线，需要集成 4 个 backbone 的 forward。
- Action layout: [left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)] = 14
- 如果你训练的 head 输出 12（不含 gripper），需要在此脚本里补 gripper 维度。
"""

import argparse
import sys
from pathlib import Path

# Add features_model to path
FEATURES_ROOT = Path(__file__).resolve().parents[2]  # /home/gl/features_model
sys.path.insert(0, str(FEATURES_ROOT))
sys.path.insert(0, str(FEATURES_ROOT / "DP" / "diffusion_policy"))

import torch
import numpy as np

from features_common.rgb2pc_aligned_encoder_4models import RGB2PCAlignedEncoder4Models
from features_common.dp_rgb_policy_single import HeadSpec, DiffusionRGBHead
from features_common.zarr_pack import load_zarr_pack


def load_inference_model(head_ckpt_path: str, device='cuda'):
    """加载 final_head.pt + encoder"""
    ckpt = torch.load(head_ckpt_path, map_location=device)
    
    # 1. 恢复 head
    spec = HeadSpec(
        action_dim=ckpt['action_dim'],
        horizon=ckpt.get('horizon', 8),
        n_obs_steps=ckpt.get('n_obs_steps', 2),
        obs_feature_dim=ckpt['encoder_spec']['fuse_dim'],
        num_inference_steps=ckpt.get('num_inference_steps', 16),
    )
    head = DiffusionRGBHead(spec).to(device)
    head.load_state_dict(ckpt['head_state'])
    head.eval()
    
    # 2. 恢复冻结 encoder
    encoder_ckpt = ckpt['encoder_ckpt']
    encoder = RGB2PCAlignedEncoder4Models.from_checkpoint(encoder_ckpt, freeze=True).to(device)
    encoder.eval()
    
    # 3. normalizer
    normalizer = ckpt['normalizer']
    
    return head, encoder, normalizer, spec


def load_obs_4models(
    rgb_zarr_roots_4: list[Path],
    task: str,
    episode: str,
    start_frame: int,
    n_obs_steps: int,
    expect_dims=(1024, 2048, 768, 2048),
) -> torch.Tensor:
    """从 4 个 zarr 读取 obs: [1, To, 4, 2048]"""
    max_dim = int(max(expect_dims))
    packs = [load_zarr_pack(root / task / f"{episode}.zarr") for root in rgb_zarr_roots_4]
    
    frames = []
    W, T = packs[0].shape[0], packs[0].shape[1]
    total_steps = W * T
    if start_frame + n_obs_steps > total_steps:
        start_frame = max(0, total_steps - n_obs_steps)
    
    for s in range(start_frame, start_frame + n_obs_steps):
        wi = s // T
        ti = s % T
        per_model = []
        for mi, pack in enumerate(packs):
            f = pack.get_frame(wi, ti)  # [Hf,Wf,C]
            f = f.reshape(-1, f.shape[-1]).mean(axis=0)  # [C]
            ed = int(expect_dims[mi])
            if f.shape[0] >= ed:
                f_ed = f[:ed]
            else:
                f_ed = np.zeros((ed,), dtype=f.dtype)
                f_ed[: f.shape[0]] = f
            if ed < max_dim:
                f2 = np.zeros((max_dim,), dtype=f.dtype)
                f2[:ed] = f_ed
            else:
                f2 = f_ed
            per_model.append(f2)
        frames.append(np.stack(per_model, axis=0))
    
    obs = torch.from_numpy(np.stack(frames, axis=0)).unsqueeze(0).to(torch.float32)  # [1,To,4,2048]
    return obs


def predict_action_14d(
    head,
    encoder,
    normalizer,
    obs_4models: torch.Tensor,
    device='cuda'
) -> np.ndarray:
    """推理：obs_4models [1,To,4,2048] -> action [horizon,14]"""
    with torch.no_grad():
        obs_4models = obs_4models.to(device)
        z = encoder(obs_4models)  # [1,To,fuse_dim]
        out = head.predict_action(z, normalizer['obs'], normalizer['action'])
        action_pred = out['action_pred'].cpu().numpy()[0]  # [horizon,action_dim]
    
    # 如果 action_dim=12（训练时不含 gripper），需要补成 14
    if action_pred.shape[1] == 12:
        # 插入 gripper 维度（默认 0.5）
        left_arm = action_pred[:, :6]
        right_arm = action_pred[:, 6:12]
        left_grip = np.full((action_pred.shape[0], 1), 0.5, dtype=np.float32)
        right_grip = np.full((action_pred.shape[0], 1), 0.5, dtype=np.float32)
        action_pred = np.concatenate([left_arm, left_grip, right_arm, right_grip], axis=-1)
    
    return action_pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--head_ckpt', type=str, required=True, help='Path to final_head.pt')
    parser.add_argument('--rgb_zarr_roots_4', nargs=4, required=True, help='4 zarr roots: croco vggt dinov3 da3')
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--episode', type=str, required=True)
    parser.add_argument('--start_frame', type=int, default=0)
    parser.add_argument('--exec_steps', type=int, default=1, help='Only execute first K steps (receding horizon)')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--robotwin_env_class', type=str, default=None, help='RoBoTwin env class name (e.g., beat_block_hammer)')
    args = parser.parse_args()
    
    # 1. Load model
    print(f"Loading head from {args.head_ckpt} ...")
    head, encoder, normalizer, spec = load_inference_model(args.head_ckpt, device=args.device)
    print(f"Head spec: horizon={spec.horizon}, n_obs_steps={spec.n_obs_steps}, action_dim={spec.action_dim}")
    
    # 2. Load obs
    rgb_zarr_roots_4 = [Path(r) for r in args.rgb_zarr_roots_4]
    print(f"Loading obs from {args.task}/{args.episode} frame {args.start_frame} ...")
    obs_4 = load_obs_4models(rgb_zarr_roots_4, args.task, args.episode, args.start_frame, spec.n_obs_steps)
    
    # 3. Predict action
    action_pred = predict_action_14d(head, encoder, normalizer, obs_4, device=args.device)
    print(f"Predicted action shape: {action_pred.shape}")
    
    # 4. Execute (receding horizon)
    action_exec = action_pred[:args.exec_steps]
    print(f"Executing first {args.exec_steps} steps:")
    print(action_exec)
    
    # 5. (Optional) If you have a RoBoTwin env instance, call env.take_action(action_exec[i], action_type='qpos')
    if args.robotwin_env_class:
        print(f"\n[Integrating with RoBoTwin env: {args.robotwin_env_class}]")
        
        # 示例：集成 RoBoTwin 环境（你需要根据实际任务调整）
        # import sys
        # sys.path.insert(0, str(Path(__file__).parents[1] / 'envs'))
        # from beat_block_hammer import beat_block_hammer
        # 
        # env = beat_block_hammer(
        #     render=True,
        #     need_offscreen_render=True,
        #     eval=True,
        #     eval_video_path='/path/to/save/video.mp4',
        # )
        # 
        # max_steps = 100
        # for step_i in range(max_steps):
        #     # 1. 获取最近 To 帧 RGB obs
        #     obs_4 = load_obs_4models(...)  # 从环境/缓存获取
        #     
        #     # 2. 预测 action
        #     action_pred = predict_action_14d(head, encoder, normalizer, obs_4, device=args.device)
        #     
        #     # 3. 执行 receding horizon（只执行前 K 步）
        #     for k in range(args.exec_steps):
        #         env.take_action(action_pred[k], action_type='qpos')
        #         if env.eval_success:
        #             print(f"Success at step {step_i}")
        #             break
        #     
        #     if env.eval_success:
        #         break
        # 
        # env.close()
        # print(f"Video saved to: {env.eval_video_path}")
        
        print("  [示例代码已注释在脚本里，你需要根据具体任务启用]")
    
    print("\nDone.")


if __name__ == '__main__':
    main()
