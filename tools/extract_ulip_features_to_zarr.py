import sys
import os
import glob
import json
import numpy as np
import torch
import zarr
from tqdm import tqdm
from pathlib import Path

# Try to import pytorch3d for FPS (recommended, same as RoBoTwin)
try:
    import pytorch3d.ops as torch3d_ops
    HAS_PYTORCH3D = True
except ImportError:
    HAS_PYTORCH3D = False
    print("Warning: pytorch3d not available, will use numpy FPS (slower)")

# Setup paths
WORKSPACE_DIR = "/home/gl/features_model"
ULIP_DIR = os.path.join(WORKSPACE_DIR, "ULIP-main/ULIP-main")

# 支持两个数据源（优先使用 PC_ORI，fallback 到 PC_source）
PC_SOURCE_DIR_NEW = os.path.join(WORKSPACE_DIR, "pc_dataset/PC_ORI")
PC_SOURCE_DIR_OLD = os.path.join(WORKSPACE_DIR, "pc_dataset/PC_source")
PC_SOURCE_DIR = PC_SOURCE_DIR_NEW if os.path.exists(PC_SOURCE_DIR_NEW) else PC_SOURCE_DIR_OLD

OUTPUT_ZARR_DIR = os.path.join(WORKSPACE_DIR, "pc_dataset/ulip_features_zarr")

# Add ULIP to sys.path
sys.path.append(ULIP_DIR)

# Change CWD to ULIP_DIR so that relative paths in ULIP_models work
original_cwd = os.getcwd()
os.chdir(ULIP_DIR)

try:
    from models.ULIP_models import ULIP2_WITH_OPENCLIP
    from models.pointbert.point_encoder import PointTransformer_Colored
    from utils.config import cfg_from_yaml_file
except ImportError as e:
    print(f"Error importing ULIP modules: {e}")
    sys.exit(1)

def load_ulip_model(device):
    # Load config
    config_path = os.path.join(ULIP_DIR, 'models/pointbert/ULIP_2_PointBERT_10k_colored_pointclouds.yaml')
    print(f"Loading config from {config_path}")
    config = cfg_from_yaml_file(config_path)
    
    # Instantiate Point Encoder
    # PointTransformer_Colored requires 'args' with 'evaluate_3d'
    class Args:
        evaluate_3d = True
    args = Args()
    
    print("Instantiating PointTransformer_Colored...")
    point_encoder = PointTransformer_Colored(config.model, args=args)
    
    # Instantiate ULIP model
    # pc_feat_dims is 768 for PointBERT (output of transformer)
    print("Instantiating ULIP2_WITH_OPENCLIP...")
    model = ULIP2_WITH_OPENCLIP(point_encoder=point_encoder, open_clip_model=None, pc_feat_dims=768)
    
    # Load checkpoint
    ckpt_path = os.path.join(ULIP_DIR, "pretrain_model/ULIP-2-PointBERT-10k-xyzrgb-pc-vit_g-objaverse_shapenet-pretrained.pt")
    print(f"Loading checkpoint from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    
    state_dict = checkpoint['state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
            
    # We expect missing keys for open_clip_model, so strict=False
    msg = model.load_state_dict(new_state_dict, strict=False)
    print(f"Load status: {msg}")
    
    # Verify that pc_projection was loaded
    # Check if 'pc_projection' is in missing_keys
    if 'pc_projection' in msg.missing_keys:
        print("WARNING: pc_projection was NOT loaded!")
    else:
        print("SUCCESS: pc_projection loaded.")
        
    model.to(device)
    model.eval()
    return model

def read_ply(ply_path):
    """Read a PLY file and return (xyz, rgb).

    Supports:
    - ASCII PLY with at least x y z and optionally rgb
    - Binary little/big endian PLY with common scalar types

    The returned rgb is float32 in [0, 1]. If rgb is missing, returns zeros.
    """

    ply_path = str(ply_path)

    # --- read header as bytes to avoid text decoding issues for binary PLY ---
    with open(ply_path, 'rb') as f:
        header_lines: list[bytes] = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Invalid PLY (unexpected EOF in header): {ply_path}")
            header_lines.append(line)
            if line.strip() == b"end_header":
                break
        data_start = f.tell()

        # Decode header lines safely (header is ASCII by spec)
        header_txt = [ln.decode('ascii', errors='strict').strip() for ln in header_lines]

        if not header_txt or header_txt[0] != "ply":
            raise ValueError(f"Not a PLY file: {ply_path}")

        fmt = None
        vertex_count = None
        in_vertex = False
        properties: list[tuple[str, str]] = []  # (type, name)
        for ln in header_txt:
            if ln.startswith("format "):
                # format ascii 1.0 | format binary_little_endian 1.0 | format binary_big_endian 1.0
                fmt = ln.split()[1]
            elif ln.startswith("element vertex"):
                vertex_count = int(ln.split()[-1])
                in_vertex = True
            elif ln.startswith("element ") and not ln.startswith("element vertex"):
                # we only parse vertex element here
                in_vertex = False
            elif in_vertex and ln.startswith("property "):
                parts = ln.split()
                # property <type> <name>
                # Ignore list properties for now (faces)
                if len(parts) >= 3 and parts[1] != "list":
                    properties.append((parts[1], parts[2]))

        if fmt is None or vertex_count is None:
            raise ValueError(f"Invalid PLY header (missing format/vertex count): {ply_path}")

        prop_names = [n for _, n in properties]
        needed_xyz = all(k in prop_names for k in ("x", "y", "z"))
        if not needed_xyz:
            raise ValueError(f"PLY missing x/y/z properties: {ply_path}. Got properties: {prop_names}")

        # RGB can appear as red/green/blue or r/g/b
        rgb_keys = None
        if all(k in prop_names for k in ("red", "green", "blue")):
            rgb_keys = ("red", "green", "blue")
        elif all(k in prop_names for k in ("r", "g", "b")):
            rgb_keys = ("r", "g", "b")

        if fmt == "ascii":
            # Read rest as text, but only parse required columns based on header property order
            f.seek(data_start)
            # PLY ASCII data is safe to decode line-by-line
            xyz_list = []
            rgb_list = []
            for _ in range(vertex_count):
                line = f.readline()
                if not line:
                    break
                parts = line.decode('ascii', errors='ignore').strip().split()
                if len(parts) < len(properties):
                    continue

                values = {}
                for (typ, name), val in zip(properties, parts):
                    # float/int parsing; for our needs float is fine
                    try:
                        if typ in {"float", "float32", "double", "float64"}:
                            values[name] = float(val)
                        else:
                            # integer types
                            values[name] = float(val)
                    except Exception:
                        values[name] = 0.0

                xyz_list.append([values["x"], values["y"], values["z"]])
                if rgb_keys is not None:
                    rgb_list.append([values[rgb_keys[0]], values[rgb_keys[1]], values[rgb_keys[2]]])

            xyz = np.asarray(xyz_list, dtype=np.float32)
            if rgb_keys is not None and len(rgb_list) == len(xyz_list):
                rgb = np.asarray(rgb_list, dtype=np.float32)
            else:
                rgb = np.zeros((xyz.shape[0], 3), dtype=np.float32)

        elif fmt in {"binary_little_endian", "binary_big_endian"}:
            endian = "<" if fmt == "binary_little_endian" else ">"
            ply_to_np = {
                "char": "i1",
                "int8": "i1",
                "uchar": "u1",
                "uint8": "u1",
                "short": "i2",
                "int16": "i2",
                "ushort": "u2",
                "uint16": "u2",
                "int": "i4",
                "int32": "i4",
                "uint": "u4",
                "uint32": "u4",
                "float": "f4",
                "float32": "f4",
                "double": "f8",
                "float64": "f8",
            }

            dtype_fields = []
            for typ, name in properties:
                if typ == "list":
                    raise ValueError(f"PLY list properties are not supported for vertices: {ply_path}")
                if typ not in ply_to_np:
                    raise ValueError(f"Unsupported PLY property type '{typ}' in {ply_path}")
                dtype_fields.append((name, endian + ply_to_np[typ]))

            dt = np.dtype(dtype_fields)
            f.seek(data_start)
            arr = np.fromfile(f, dtype=dt, count=vertex_count)

            xyz = np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float32)
            if rgb_keys is not None:
                rgb_raw = np.stack([arr[rgb_keys[0]], arr[rgb_keys[1]], arr[rgb_keys[2]]], axis=1)
                rgb = rgb_raw.astype(np.float32)
            else:
                rgb = np.zeros((xyz.shape[0], 3), dtype=np.float32)

        else:
            raise ValueError(f"Unsupported PLY format '{fmt}' in {ply_path}")

    # Normalize RGB to [0, 1] if needed
    if rgb.size > 0:
        mx = float(np.max(rgb)) if rgb.size else 0.0
        if mx > 1.1:
            rgb = rgb / 255.0
        rgb = np.clip(rgb, 0.0, 1.0)

    return xyz, rgb


def _write_json(path: str | os.PathLike, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def normalize_pc(xyz):
    # Center and scale to unit sphere
    centroid = np.mean(xyz, axis=0)
    xyz = xyz - centroid
    m = np.max(np.sqrt(np.sum(xyz ** 2, axis=1)))
    xyz = xyz / (m + 1e-6)
    return xyz

def fps_sample(xyz, n_points=2048, use_cuda=True):
    """
    Farthest Point Sampling (FPS) - 使用 pytorch3d 实现（与 RoBoTwin 一致）
    
    优点：
    - 均匀覆盖整个点云空间
    - 保留物体边缘/关键几何特征
    - 适合机器人操作任务（需要精确几何）
    - 使用 pytorch3d GPU 加速，比纯 numpy 快 10-100 倍
    
    参数：
    - xyz: [N, 3] 点云坐标
    - n_points: 采样点数
    - use_cuda: 是否使用 GPU 加速
    
    返回：
    - idx: [n_points] 采样索引
    """
    N = xyz.shape[0]
    if N <= n_points:
        # 不够就重复采样
        idx = np.arange(N)
        if N < n_points:
            idx = np.concatenate([idx, np.random.choice(N, n_points - N, replace=True)])
        return idx
    
    if HAS_PYTORCH3D:
        # 使用 pytorch3d 的高效 FPS 实现（与 RoBoTwin 一致）
        K = [n_points]
        if use_cuda and torch.cuda.is_available():
            points_tensor = torch.from_numpy(xyz).float().cuda()
            _, indices = torch3d_ops.sample_farthest_points(
                points=points_tensor.unsqueeze(0), K=K
            )
            indices = indices.squeeze(0).cpu().numpy()
        else:
            points_tensor = torch.from_numpy(xyz).float()
            _, indices = torch3d_ops.sample_farthest_points(
                points=points_tensor.unsqueeze(0), K=K
            )
            indices = indices.squeeze(0).numpy()
        return indices
    else:
        # Fallback: 纯 numpy 实现（慢）
        centroids = np.zeros(n_points, dtype=np.int64)
        distance = np.ones(N) * 1e10
        farthest = np.random.randint(0, N)
        
        for i in range(n_points):
            centroids[i] = farthest
            centroid = xyz[farthest, :]
            dist = np.sum((xyz - centroid) ** 2, axis=1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = np.argmax(distance)
        
        return centroids

def sample_pc(xyz, rgb, n_points=10000, method='fps'):
    """
    点云采样
    
    参数：
    - method: 'fps'（推荐）或 'random'
    """
    n = xyz.shape[0]
    
    if method == 'fps':
        idx = fps_sample(xyz, n_points=n_points)
    else:  # random
        if n >= n_points:
            idx = np.random.choice(n, n_points, replace=False)
        else:
            idx = np.random.choice(n, n_points, replace=True)
    
    return xyz[idx], rgb[idx]

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = load_ulip_model(device)
    
    # Find tasks
    tasks = [d for d in os.listdir(PC_SOURCE_DIR) if os.path.isdir(os.path.join(PC_SOURCE_DIR, d))]
    
    for task in tasks:
        print(f"Processing task: {task}")
        task_dir = os.path.join(PC_SOURCE_DIR, task)
        out_task_dir = os.path.join(OUTPUT_ZARR_DIR, task)
        os.makedirs(out_task_dir, exist_ok=True)
        
        episodes = [d for d in os.listdir(task_dir) if d.startswith('episode_')]
        
        for episode in tqdm(episodes, desc=f"Episodes in {task}"):
            episode_dir = os.path.join(task_dir, episode)
            out_episode_zarr = os.path.join(out_task_dir, f"{episode}.zarr")
            
            if os.path.exists(out_episode_zarr):
                 # Check if valid? For now skip if exists to save time if re-running
                 # But user asked to re-implement, so maybe overwrite.
                 # I'll overwrite.
                 pass

            # Find PLY files
            ply_files = sorted(glob.glob(os.path.join(episode_dir, "*.ply")))
            if not ply_files:
                continue
                
            features_list = []
            
            for ply_file in ply_files:
                xyz, rgb = read_ply(ply_file)
                xyz = normalize_pc(xyz)
                # 重要：用 FPS 采样而不是随机采样（保留任务相关几何）
                xyz, rgb = sample_pc(xyz, rgb, n_points=10000, method='fps')
                
                pc = np.concatenate([xyz, rgb], axis=1) # (N, 6)
                pc_tensor = torch.from_numpy(pc).unsqueeze(0).to(device) # (1, N, 6)
                
                with torch.no_grad():
                    # model.encode_pc(pc) returns pc_embed
                    pc_embed = model.encode_pc(pc_tensor)
                    
                features_list.append(pc_embed.cpu().numpy())
                
            # Stack features
            features = np.concatenate(features_list, axis=0) # (T, D)
            
            # Save to Zarr
            store = zarr.DirectoryStore(out_episode_zarr)
            root = zarr.group(store=store, overwrite=True)
            # Chunking: (1, D) allows efficient reading of single frames
            root.create_dataset('per_frame_features', data=features, chunks=(1, features.shape[1]))

            # Sidecar JSONs (CroCo-style)
            zarr_dir = Path(out_episode_zarr)
            frame_paths = [os.path.relpath(p, episode_dir) for p in ply_files]
            meta = {
                "model_name": "ulip",
                "task_name": task,
                "episode_name": episode,
                "episode_dir": os.path.abspath(episode_dir),
                "input_kind": "pointcloud_ply",
                "num_frames": int(features.shape[0]),
                "feature_dim": int(features.shape[1]),
                "save_dtype": str(features.dtype),
            }
            shape = {
                "per_frame_features": list(features.shape),
            }
            _write_json(zarr_dir / "frame_paths.json", frame_paths)
            _write_json(zarr_dir / "meta.json", meta)
            _write_json(zarr_dir / "shape.json", shape)
            
if __name__ == "__main__":
    main()
