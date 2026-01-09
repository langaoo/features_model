#!/bin/bash
# 安装 pytorch3d 脚本
# 这个脚本会在后台编译安装 pytorch3d（需要 10-20 分钟）

set -e

echo "=========================================="
echo "开始安装 pytorch3d"
echo "=========================================="
echo ""

# 1. 确认环境
echo "步骤 1/4: 检查 PyTorch 环境..."
python -c "import torch; print(f'PyTorch 版本: {torch.__version__}'); print(f'CUDA 可用: {torch.cuda.is_available()}')"
echo ""

# 2. 安装编译依赖
echo "步骤 2/4: 安装编译依赖..."
pip install ninja fvcore iopath --quiet
echo "✓ 编译依赖已安装"
echo ""

# 3. 从源码编译安装 pytorch3d
echo "步骤 3/4: 从源码编译 pytorch3d（这可能需要 10-20 分钟）..."
echo "提示：编译过程会显示大量 C++ 编译信息，这是正常的。"
echo ""
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
echo ""
echo "✓ pytorch3d 编译安装完成"
echo ""

# 4. 验证安装
echo "步骤 4/4: 验证安装..."
python -c "import pytorch3d; import pytorch3d.ops as ops; print('✓ pytorch3d 导入成功'); print(f'版本: {pytorch3d.__version__}')"
echo ""

echo "=========================================="
echo "pytorch3d 安装完成！"
echo "=========================================="
echo ""
echo "下一步："
echo "1. 重新运行 python tools/extract_ulip_features_to_zarr.py"
echo "2. 你应该会看到 '使用 pytorch3d FPS' 而不是 'Warning: pytorch3d not available'"
echo ""
