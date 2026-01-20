"""
DP2DP3 项目路径配置
统一管理所有路径，避免硬编码
"""
import os
from pathlib import Path

# 获取项目根目录
FEATURES_MODEL_ROOT = Path(__file__).parent.resolve()  # /home/gl/RoboTwin/policy/DP2DP3/features_model
POLICY_ROOT = FEATURES_MODEL_ROOT.parent  # /home/gl/RoboTwin/policy/DP2DP3
ROBOTWIN_ROOT = POLICY_ROOT.parent.parent  # /home/gl/RoboTwin

# 数据路径
RGB_DATASET_ROOT = FEATURES_MODEL_ROOT / "rgb_dataset"
RGB_ORI_ROOT = RGB_DATASET_ROOT / "RGB_ORI"
RGB_ROOT = RGB_DATASET_ROOT / "RGB"
PC_DATASET_ROOT = FEATURES_MODEL_ROOT / "pc_dataset"
RAW_DATA_ROOT = FEATURES_MODEL_ROOT / "raw_data"

# 模型权重路径
WEIGHTS_ROOT = FEATURES_MODEL_ROOT / "third_party_weights_archive"
CROCO_CKPT = FEATURES_MODEL_ROOT / "croco" / "pretrained_models" / "CroCo_V2_ViTLarge_BaseDecoder.pth"
VGGT_CKPT = FEATURES_MODEL_ROOT / "vggt" / "weight" / "model.pt"
DINOV3_WEIGHTS = FEATURES_MODEL_ROOT / "dinov3" / "weight" / "B16"
DA3_MODEL_DIR = FEATURES_MODEL_ROOT / "Depth-Anything-3" / "weight"

# 特征路径
FEATURES_CROCO_ROOT = RGB_DATASET_ROOT / "features_croco_v2_encoder_dict_unified_zarr"
FEATURES_VGGT_ROOT = RGB_DATASET_ROOT / "features_vggt_encoder_dict_unified_zarr"
FEATURES_DINOV3_ROOT = RGB_DATASET_ROOT / "features_dinov3_encoder_dict_unified_zarr"
FEATURES_DA3_ROOT = RGB_DATASET_ROOT / "features_da3_encoder_dict_unified_zarr"
FEATURES_ULIP_ROOT = PC_DATASET_ROOT / "PC" / "ULIP_FEAT_PT_POINT"

# 输出路径
OUTPUTS_ROOT = FEATURES_MODEL_ROOT / "outputs"
RGB2PC_RUNS_ROOT = OUTPUTS_ROOT / "train_rgb2pc_runs"
ONLINE_TRAINING_ROOT = FEATURES_MODEL_ROOT / "_runs" / "online_training"

# 标准化 Checkpoint 路径（对齐 DP 策略）
CHECKPOINTS_ROOT = POLICY_ROOT / "checkpoints"

# DP 相关路径
DP_ROOT = FEATURES_MODEL_ROOT / "DP" / "diffusion_policy"

def get_checkpoint_dir(task_name, ckpt_setting, expert_data_num, seed):
    """
    获取标准化的 checkpoint 目录
    格式: checkpoints/{task_name}-{ckpt_setting}-{expert_data_num}-{seed}/
    """
    dir_name = f"{task_name}-{ckpt_setting}-{expert_data_num}-{seed}"
    return CHECKPOINTS_ROOT / dir_name

def get_checkpoint_path(task_name, ckpt_setting, expert_data_num, seed, epoch):
    """
    获取标准化的 checkpoint 文件路径
    格式: checkpoints/{task_name}-{ckpt_setting}-{expert_data_num}-{seed}/{epoch}.ckpt
    """
    ckpt_dir = get_checkpoint_dir(task_name, ckpt_setting, expert_data_num, seed)
    return ckpt_dir / f"{epoch}.ckpt"

def ensure_dir(path):
    """确保目录存在"""
    Path(path).mkdir(parents=True, exist_ok=True)
    return path

# 输出路径信息（用于调试）
if __name__ == "__main__":
    print("=" * 60)
    print("DP2DP3 路径配置")
    print("=" * 60)
    print(f"Features Model Root: {FEATURES_MODEL_ROOT}")
    print(f"Policy Root: {POLICY_ROOT}")
    print(f"RoboTwin Root: {ROBOTWIN_ROOT}")
    print(f"\nDataset Roots:")
    print(f"  RGB: {RGB_ROOT}")
    print(f"  PC: {PC_DATASET_ROOT}")
    print(f"  Raw Data: {RAW_DATA_ROOT}")
    print(f"\nModel Weights:")
    print(f"  CroCo: {CROCO_CKPT}")
    print(f"  VGGT: {VGGT_CKPT}")
    print(f"  DINOv3: {DINOV3_WEIGHTS}")
    print(f"  DA3: {DA3_MODEL_DIR}")
    print(f"\nCheckpoints: {CHECKPOINTS_ROOT}")
    print("=" * 60)
