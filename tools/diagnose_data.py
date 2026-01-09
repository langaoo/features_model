
import torch
import torch.nn.functional as F
import sys
from pathlib import Path
import numpy as np
import zarr

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from features_common.rgb2pc_distill_dataset import RGB2PCDistillDataset
# from configs import train_rgb2pc_distill_default as cfg_dummy # just to find path

def main():
    print("=== Starting Diagnosis ===")
    
    # 1. Setup Dataset (Copy params from config)
    pc_root = "/home/gl/features_model/pc_dataset/PC/ULIP_FEAT_PT_POINT"
    vis_zarr_roots = [
        "/home/gl/features_model/rgb_dataset/features_croco_v2_encoder_dict_unified_zarr",
        "/home/gl/features_model/rgb_dataset/features_vggt_encoder_dict_unified_zarr",
        "/home/gl/features_model/rgb_dataset/features_dinov3_encoder_dict_unified_safe_zarr",
        "/home/gl/features_model/rgb_dataset/features_da3_encoder_dict_unified_zarr"
    ]
    tasks = [
        "dump_bin_bigbin-demo_randomized-20_head_camera",
        "beat_block_hammer-demo_randomized-20_head_camera"
    ]
    
    print(f"Loading dataset with {len(tasks)} tasks...")
    dataset = RGB2PCDistillDataset(
        pc_root=pc_root,
        vis_zarr_roots=vis_zarr_roots,
        tasks=tasks,
        episodes=0,
        sample_unit="step", # Changed to step
        student_tokens=2048,
        teacher_points=2048,
        strict_pairing=True,
        pairing_fallback="error",
        seed=0
    )
    
    print(f"Dataset length (virtual): {len(dataset)}")
    print(f"Found {len(dataset.pairs)} pairs.")
    
    # 2. Sample a few items
    print("\n--- Sampling Batch of 8 ---")
    samples = []
    for i in range(8):
        s = dataset[i]
        samples.append(s)
        # print(f"Sample {i}: {s.task} / {s.episode} | Teacher Embed[:5]: {s.teacher_embed[:5].tolist()}")
        
    # 3. Analyze Teacher Embeddings
    print("\n--- Analyzing Teacher Embeddings ---")
    # For step mode, teacher is [N, 384] points. We need to pool them to check global similarity
    # Or check point-wise similarity?
    # Let's mean pool for now to compare with window mode
    teacher_embeds = torch.stack([s.teacher_points.mean(dim=0) for s in samples]) # [8, 384]
    
    print(f"Teacher Shape: {teacher_embeds.shape}")
    print(f"Teacher Mean: {teacher_embeds.mean():.4f}")
    print(f"Teacher Std:  {teacher_embeds.std():.4f}")
    print(f"Teacher Norms: {teacher_embeds.norm(dim=-1)}")
    
    # Check Self-Similarity
    # te_norm = F.normalize(teacher_embeds, dim=-1)
    
    # Try Centering first
    te_centered = teacher_embeds - teacher_embeds.mean(dim=0, keepdim=True)
    te_norm = F.normalize(te_centered, dim=-1)
    
    sim_matrix = te_norm @ te_norm.t()
    
    diag_mean = sim_matrix.diag().mean().item()
    off_diag_mean = (sim_matrix.sum() - sim_matrix.diag().sum()) / (8*7)
    
    print(f"\nTeacher Self-Similarity Matrix (CENTERED):\n{sim_matrix}")
    print(f"\nAvg Diagonal (Self): {diag_mean:.4f}")
    print(f"Avg Off-Diagonal (Cross): {off_diag_mean:.4f}")
    
    if off_diag_mean > 0.9:
        print("\n[CRITICAL WARNING] Teacher embeddings are almost identical across different samples!")
        print("This explains why Loss is not dropping. The model cannot distinguish between samples.")
    elif off_diag_mean < 0.1:
        print("\n[OK] Teacher embeddings are diverse.")
    else:
        print("\n[INFO] Teacher embeddings have moderate similarity.")

    # 4. Check Student Tokens
    print("\n--- Analyzing Student Tokens (First Model) ---")
    s0_tokens = torch.stack([s.tokens_by_model[0] for s in samples]) # [8, 2048, C]
    print(f"Student Shape: {s0_tokens.shape}")
    print(f"Student Mean: {s0_tokens.mean():.4f}")
    print(f"Student Std:  {s0_tokens.std():.4f}")
    
    if s0_tokens.std() < 1e-6:
        print("\n[CRITICAL WARNING] Student tokens are constant/zero!")

    print("\n=== Diagnosis Complete ===")

if __name__ == "__main__":
    main()
