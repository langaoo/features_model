#!/bin/bash
# scripts/test_and_train.sh
# 完整测试和训练流程

set -e

echo "=========================================="
echo "DP RGB 训练 - 完整流程"
echo "=========================================="

# 步骤 1: 测试管道（单任务）
echo ""
echo "步骤 1: 测试训练管道..."
echo "使用单个任务进行快速测试..."

python tools/test_dp_rgb_pipeline.py \
    --config configs/train_dp_rgb_left_1.yaml

if [ $? -ne 0 ]; then
    echo "❌ 管道测试失败！请检查配置和数据路径。"
    exit 1
fi

echo ""
echo "✅ 管道测试通过！"
echo ""

# 步骤 2: 小规模训练测试
echo "=========================================="
echo "步骤 2: 小规模训练测试（5 epochs）"
echo "=========================================="

python tools/train_dp_rgb.py \
    --config configs/train_dp_rgb_left_1.yaml \
    --tasks dump_bin_bigbin-demo_randomized-20_head_camera \
    --epochs 5 \
    --batch_size 8 \
    --num_workers 2 \
    --save_dir outputs/dp_rgb_runs/test_run \
    --tqdm

if [ $? -ne 0 ]; then
    echo "❌ 小规模训练失败！"
    exit 1
fi

echo ""
echo "✅ 小规模训练成功！"
echo ""
echo "=========================================="
echo "准备开始完整训练"
echo "=========================================="
echo ""
echo "你现在有两个选项："
echo ""
echo "选项 1: 训练左臂模型 (action_dim=1, 5个任务)"
echo "  python tools/train_dp_rgb.py --config configs/train_dp_rgb_left_1.yaml"
echo ""
echo "选项 2: 训练双臂模型 (action_dim=2, 2个任务)"
echo "  python tools/train_dp_rgb.py --config configs/train_dp_rgb_dual_2.yaml"
echo ""
echo "推荐：先训练左臂模型（数据更多）"
echo ""

read -p "是否现在开始训练左臂模型？(y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "开始训练左臂模型..."
    python tools/train_dp_rgb.py \
        --config configs/train_dp_rgb_left_1.yaml \
        --tqdm
else
    echo ""
    echo "稍后可以手动运行："
    echo "  python tools/train_dp_rgb.py --config configs/train_dp_rgb_left_1.yaml"
fi
