import zarr
import numpy as np
import os
import torch

def check_features(path):
    print(f"Checking {path}")
    g = zarr.open(path, mode='r')
    feats = g['per_frame_features'][:]
    print(f"Shape: {feats.shape}")
    
    # Calculate similarity
    feats_t = torch.from_numpy(feats)
    feats_t = feats_t / feats_t.norm(dim=-1, keepdim=True)
    sim = feats_t @ feats_t.T
    
    # Mean similarity of off-diagonal elements
    mask = ~torch.eye(sim.shape[0], dtype=torch.bool)
    mean_sim = sim[mask].mean().item()
    print(f"Mean self-similarity: {mean_sim:.4f}")
    print(f"Std of features: {feats.std(axis=0).mean():.4f}")
    
    if mean_sim > 0.9:
        print("WARNING: Features might be collapsed!")
    else:
        print("Features look diverse.")

base_dir = "pc_dataset/ulip_features_zarr"
if os.path.exists(base_dir):
    tasks = os.listdir(base_dir)
    if tasks:
        task = tasks[0]
        eps = os.listdir(os.path.join(base_dir, task))
        if len(eps) >= 2:
            ep1 = eps[0]
            ep2 = eps[1]
            path1 = os.path.join(base_dir, task, ep1)
            path2 = os.path.join(base_dir, task, ep2)
            
            print(f"Comparing {ep1} and {ep2}")
            g1 = zarr.open(path1, mode='r')
            f1 = g1['per_frame_features'][0] # First frame
            
            g2 = zarr.open(path2, mode='r')
            f2 = g2['per_frame_features'][0] # First frame
            
            t1 = torch.from_numpy(f1).unsqueeze(0)
            t2 = torch.from_numpy(f2).unsqueeze(0)
            
            t1 = t1 / t1.norm(dim=-1, keepdim=True)
            t2 = t2 / t2.norm(dim=-1, keepdim=True)
            
            sim = (t1 @ t2.T).item()
            print(f"Similarity between first frames of different episodes: {sim:.4f}")

    else:
        print("No tasks found")
else:
    print("Dataset dir not found")
