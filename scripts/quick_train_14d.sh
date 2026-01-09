#!/bin/bash
# 快速开始：训练 14 维双臂 DP head
# 用法：bash scripts/quick_train_14d.sh

set -e

TASK="beat_block_hammer-demo_randomized-20_head_camera"
ENCODER_CKPT="outputs/train_rgb2pc_runs/run_best_bs32/ckpt_step_0010000.pt"
SAVE_DIR="outputs/dp_rgb_runs/${TASK}_14d"

echo "=========================================="
echo "训练 14 维 DP Head（双臂 + 夹爪）"
echo "=========================================="
echo "任务：${TASK}"
echo "Encoder：${ENCODER_CKPT}"
echo "保存到：${SAVE_DIR}"
echo ""

python tools/train_dp_rgb_single_task_4models.py \
  --task "${TASK}" \
  --encoder_ckpt "${ENCODER_CKPT}" \
  --use_left_arm --use_right_arm --fuse_arms --include_gripper \
  --epochs 50 \
  --batch_size 16 \
  --save_dir "${SAVE_DIR}" \
  --tqdm

echo ""
echo "=========================================="
echo "训练完成！模型已保存到："
echo "${SAVE_DIR}/final_head.pt"
echo "=========================================="
echo ""
echo "验证模型："
python -c "
import torch
ckpt = torch.load('${SAVE_DIR}/final_head.pt', weights_only=False)
print(f'Action 维度: {ckpt[\"action_dim\"]}')
print(f'Obs 维度: {ckpt[\"obs_c\"]}')
print(f'任务: {ckpt[\"config\"][\"task\"]}')
"
