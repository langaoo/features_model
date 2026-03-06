#!/usr/bin/env bash
set -euo pipefail

# 多任务 ws1 外置特征流水线（可复现，一键跑）
#
# 约束：
# - 训练只用 demo_clean（禁止 demo_randomized 混入）
# - ws1 RGB 特征写到外置盘：/media/gl/新加卷/gllll/features_model_ws1
#
# 说明：
# - 本脚本默认不会“自动收集数据”(HDF5)；只做从已有 RoboTwin data/ 生成 RGB/PC 数据集与特征。
# - 如果你机器上 depth3 / DP3_ULIP 的 python 路径不同，请用环境变量覆盖。
#
# 用法：
#   cd /home/gl/RoboTwin/policy/DP2DP3/features_model
#   bash scripts/run_ws1_multitask_pipeline.sh 2>&1 | tee logs/ws1_multitask_pipeline.log

ROOT="/home/gl/RoboTwin/policy/DP2DP3/features_model"
EXTERNAL_ROOT="${EXTERNAL_ROOT:-/media/gl/新加卷/gllll/features_model_ws1}"

# 训练/特征提取环境（默认 depth3；缺包可把 PYTHON_ULIP 改成 DP3_ULIP）
PYTHON_TRAIN="${PYTHON_TRAIN:-/home/gl/miniconda3/envs/depth3/bin/python}"
PYTHON_ULIP="${PYTHON_ULIP:-${PYTHON_TRAIN}}"

SETTING="demo_clean"
NUM_DEMOS="50"
CAMERA="head_camera"
WINDOW_SIZE="1"
STRIDE="1"

TASKS=(
  "lift_pot"
  "click_bell"
  "move_can_pot"
  "handover_block"
  "pick_diverse_bottles"
  "place_can_basket"
  "rotate_qrcode"
  "move_pillbottle_pad"
)

echo "[WS1-MT] ROOT=${ROOT}"
echo "[WS1-MT] EXTERNAL_ROOT=${EXTERNAL_ROOT}"
echo "[WS1-MT] PYTHON_TRAIN=${PYTHON_TRAIN}"
echo "[WS1-MT] PYTHON_ULIP=${PYTHON_ULIP}"

cd "${ROOT}"

echo "[WS1-MT] Step0: setup external ws1 symlinks"
${PYTHON_TRAIN} scripts/ws1_external_storage.py --external_root "${EXTERNAL_ROOT}"

echo "[WS1-MT] Step1: build RGB/PC datasets from RoboTwin HDF5 (demo_clean only)"
for t in "${TASKS[@]}"; do
  echo "[WS1-MT] process_sapien_pcd: ${t}"
  ${PYTHON_TRAIN} tools/dataset/process_sapien_pcd.py \
    "${t}" "${SETTING}" "${NUM_DEMOS}" \
    --output_root "${ROOT}" \
    --camera "${CAMERA}"
done

echo "[WS1-MT] Step2: extract ws1 RGB features -> rgb_dataset_ws1 (external disk)"
for m in croco vggt dinov3 da3; do
  echo "[WS1-MT] run_extract_features: ${m}"
  ${PYTHON_TRAIN} tools/features/run_extract_features.py \
    --model "${m}" \
    --rgb_root "${ROOT}/rgb_dataset/RGB_ORI" \
    --out_root "${ROOT}/rgb_dataset_ws1" \
    --window_size "${WINDOW_SIZE}" \
    --stride "${STRIDE}" \
    --device cuda
done

echo "[WS1-MT] Step3: extract ULIP point tokens teacher -> external (recommended)"
${PYTHON_ULIP} tools/features/extract_ulip_point_tokens_to_zarr.py \
  --output_dir "${EXTERNAL_ROOT}/ulip_point_tokens_zarr"

echo "[WS1-MT] Step4: train RGB->PC aligner (multitask, tokens_full)"
${PYTHON_TRAIN} tools/alignment/train_rgb2pc_distill.py \
  --config configs/alignment/train_rgb2pc_distill_ws1_tokens_full_multitask.yaml

echo "[WS1-MT] Step5: extract offline aligned features for head (multitask)"
${PYTHON_TRAIN} tools/offline/extract_offline_features.py \
  --config configs/head/train_online_batch_extract_dual_stream_tokens_full_multitask.yaml \
  --output_dir "${EXTERNAL_ROOT}/offline_features_dual_stream_tokens_full_multitask"

echo "[WS1-MT] Step6: train offline head (multitask)"
${PYTHON_TRAIN} tools/offline/train_offline_head.py \
  --config configs/head/train_offline_dual_stream_tokens_full_multitask.yaml

echo "[WS1-MT] Done. Next: use RoboTwin env to eval (see PIPELINE_DETAILED.md)."

