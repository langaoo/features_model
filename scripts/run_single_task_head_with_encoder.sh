#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# 单任务 Head 训练闭环（复用已训练好的 RGB->PC 对齐模块）
#
# 目标：
# - 不需要点云（teacher）即可训练下游任务 head；
# - 输入：/home/gl/RoboTwin/data/<task>/<task_config>/data/episode*.hdf5
# - 输出：
#   1) 离线特征（Zarr）：<OFFLINE_OUT>/<task>/episode*.zarr
#   2) Head ckpt：/home/gl/RoboTwin/policy/DP2DP3/checkpoints/<task>-<ckpt_setting>-<num>-<seed>/<epoch>.ckpt
#
# 典型用法：
#   cd /home/gl/RoboTwin/policy/DP2DP3/features_model
#   export PYTHON_TRAIN=/home/gl/miniconda3/envs/depth3/bin/python
#   bash scripts/run_single_task_head_with_encoder.sh place_a2b_left demo_clean_dp2dp3 50 0 "0,1" \
#     /home/gl/RoboTwin/policy/DP2DP3/features_model/outputs/train_rgb2pc_runs/run_ws1_tokens_full_multitask/ckpt_step_0013000.pt \
#     /media/gl/新加卷/gllll/features_model_ws1/offline_features_place_a2b_left
# ============================================================

TASK_NAME="${1:?task_name}"
TASK_CONFIG="${2:-demo_clean_dp2dp3}"
EXPERT_NUM="${3:-50}"
SEED="${4:-0}"
GPU_IDS="${5:-0,1}"
ENCODER_CKPT="${6:-/home/gl/RoboTwin/policy/DP2DP3/features_model/outputs/train_rgb2pc_runs/run_ws1_tokens_full_multitask/ckpt_final.pt}"
OFFLINE_OUT="${7:-/media/gl/新加卷/gllll/features_model_ws1/offline_features_${TASK_NAME}}"

# ckpt_setting：决定 head 保存目录名，也用于 eval.sh 第3个参数
# 你可以改成更短的名字，但不要和已有实验混淆
CKPT_SETTING="${DP2DP3_HEAD_CKPT_SETTING:-${TASK_CONFIG}_dual_stream_tokens_full_mtenc}"

# 训练超参（可通过 env 覆盖）
HEAD_EPOCHS="${DP2DP3_HEAD_EPOCHS:-600}"
SAVE_EVERY="${DP2DP3_HEAD_SAVE_EVERY:-100}"
BATCH_EXTRACT="${DP2DP3_EXTRACT_BATCH_SIZE:-8}"
NUM_WORKERS="${DP2DP3_EXTRACT_NUM_WORKERS:-8}"
HEAD_BATCH_SIZE="${DP2DP3_HEAD_BATCH_SIZE:-32}"
HEAD_LR="${DP2DP3_HEAD_LR:-1e-4}"
EXTRACT_MAX_EPISODES="${DP2DP3_EXTRACT_MAX_EPISODES:-0}"

# Python
PYTHON_TRAIN="${PYTHON_TRAIN:-/home/gl/miniconda3/envs/depth3/bin/python}"
if ! command -v "${PYTHON_TRAIN}" >/dev/null 2>&1; then
  echo "[ERR] PYTHON_TRAIN not found: ${PYTHON_TRAIN}" >&2
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/logs"
mkdir -p "${LOG_DIR}"

TS="$(date +%Y%m%d_%H%M%S)"
EXTRACT_CFG="${LOG_DIR}/autogen_extract_${TASK_NAME}_${TASK_CONFIG}_${TS}.yaml"
HEAD_CFG="${LOG_DIR}/autogen_head_${TASK_NAME}_${TASK_CONFIG}_${TS}.yaml"

cat > "${EXTRACT_CFG}" <<YAML
data:
  robotwin_data_root: /home/gl/RoboTwin/data
  task_config: ${TASK_CONFIG}
  camera_name: head_camera
  tasks:
    - ${TASK_NAME}
  horizon: 8
  n_obs_steps: 2
  use_left_arm: true
  use_right_arm: true
  fuse_arms: true
  include_gripper: true

encoder:
  checkpoint: ${ENCODER_CKPT}
  freeze: true
  save_tokens: true

train:
  epochs: 1
  batch_size: ${BATCH_EXTRACT}
  batch_extract: true
  lr: 1e-4
  num_workers: ${NUM_WORKERS}
  optimizer: adamw
  weight_decay: 1e-6

policy:
  type: OfficialDP
  use_official_dp: true
  num_inference_steps: 100
  use_tokens: true

device:
  gpu_ids: [${GPU_IDS}]

output:
  dir: _runs/offline_extract
  save_every_n_epochs: 1

checkpoint:
  expert_data_num: ${EXPERT_NUM}
  ckpt_setting: ${CKPT_SETTING}
  seed: ${SEED}
YAML

cat > "${HEAD_CFG}" <<YAML
data:
  features_dataset_dir: ${OFFLINE_OUT}
  horizon: 8
  n_obs_steps: 2
  camera_name: head_camera
  tasks:
    - ${TASK_NAME}
  task_config: ${TASK_CONFIG}
  use_left_arm: true
  use_right_arm: true
  include_gripper: true

encoder:
  checkpoint: ${ENCODER_CKPT}
  save_tokens: true

train:
  epochs: ${HEAD_EPOCHS}
  batch_size: ${HEAD_BATCH_SIZE}
  lr: ${HEAD_LR}
  num_workers: 4
  optimizer: adamw
  weight_decay: 1e-6

policy:
  type: OfficialDP
  use_official_dp: true
  num_inference_steps: 100
  hidden_dim: 512
  use_tokens: true
  token_gate_init: -4.0
  token_dropout: 0.1
  ctx_dropout: 0.1

device:
  gpu_ids: [${GPU_IDS}]

output:
  dir: /home/gl/RoboTwin/policy/DP2DP3/checkpoints
  save_every_n_epochs: ${SAVE_EVERY}

checkpoint:
  expert_data_num: ${EXPERT_NUM}
  ckpt_setting: ${CKPT_SETTING}
  seed: ${SEED}

debug:
  fast_dev_run: false
YAML

echo "[INFO] Extract cfg: ${EXTRACT_CFG}"
echo "[INFO] Head cfg:    ${HEAD_CFG}"
echo "[INFO] Offline out: ${OFFLINE_OUT}"
echo "[INFO] ckpt_setting:${CKPT_SETTING}"

cd "${ROOT_DIR}"

echo "[1/2] Extract offline aligned features..."
"${PYTHON_TRAIN}" -c "import os; print('[INFO] CUDA_VISIBLE_DEVICES=', os.environ.get('CUDA_VISIBLE_DEVICES',''))" || true

EXTRACT_EXTRA_ARGS=()
if [[ "${EXTRACT_MAX_EPISODES}" != "0" ]]; then
  EXTRACT_EXTRA_ARGS+=(--max_episodes "${EXTRACT_MAX_EPISODES}")
fi
"${PYTHON_TRAIN}" tools/offline/extract_offline_features.py \
  --config "${EXTRACT_CFG}" \
  --output_dir "${OFFLINE_OUT}" \
  --overwrite \
  "${EXTRACT_EXTRA_ARGS[@]}"

echo "[2/2] Train head..."
"${PYTHON_TRAIN}" tools/offline/train_offline_head.py --config "${HEAD_CFG}"

echo "[DONE] Head checkpoints at: /home/gl/RoboTwin/policy/DP2DP3/checkpoints/${TASK_NAME}-${CKPT_SETTING}-${EXPERT_NUM}-${SEED}/"
