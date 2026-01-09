#!/bin/bash
# scripts/quick_start_dp_rgb.sh
# DP RGB 特征训练快速开始脚本

set -e

echo "=========================================="
echo "DP RGB 特征训练 - 快速开始"
echo "=========================================="

# 检查环境
echo ""
echo "步骤 1: 检查环境..."
if ! python -c "import torch; import yaml; import zarr" 2>/dev/null; then
    echo "❌ 缺少依赖。请安装:"
    echo "  pip install torch pyyaml zarr numcodecs"
    exit 1
fi
echo "✅ 环境检查通过"

# 检查数据
echo ""
echo "步骤 2: 检查 action 维度..."
python tools/check_action_dims.py \
    --traj_root /home/gl/features_model/raw_data

echo ""
echo "=========================================="
read -p "请根据上面的输出，选择你要训练的任务组（按 Enter 继续）..."

# 测试管道
echo ""
echo "步骤 3: 测试训练管道..."
python tools/test_dp_rgb_pipeline.py \
    --config configs/train_dp_rgb_default.yaml

echo ""
echo "=========================================="
echo "✅ 管道测试通过!"
echo ""
echo "接下来你可以："
echo ""
echo "选项 1: 使用默认配置训练（包含多个任务）"
echo "  python tools/train_dp_rgb.py --config configs/train_dp_rgb_default.yaml"
echo ""
echo "选项 2: 单任务测试训练（推荐先做）"
echo "  python tools/train_dp_rgb.py \\"
echo "    --config configs/train_dp_rgb_default.yaml \\"
echo "    --tasks dump_bin_bigbin-demo_randomized-20_head_camera \\"
echo "    --epochs 10 \\"
echo "    --batch_size 8 \\"
echo "    --tqdm"
echo ""
echo "选项 3: 按 action_dim 分组训练"
echo "  1. 根据 check_action_dims.py 的输出创建配置文件"
echo "  2. python tools/train_dp_rgb.py --config configs/train_dp_rgb_arm7.yaml"
echo ""
echo "=========================================="
