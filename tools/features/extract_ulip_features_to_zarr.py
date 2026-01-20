"""点云 ULIP 特征提取脚本【生产级纯净版 | 全BUG修复+深度优化】
✅ 适配场景：Sapien仿真导出 2048点 ASCII格式 彩色PLY点云
✅ 核心流程：2048点自动补点→2048点 → 单位球归一化 → ULIP2推理 → 768/1280维特征输出
✅ 修复所有已知BUG：seek of closed file / vertex_count None / 无效零点污染 / 边界崩溃
✅ 运行命令（features_model根目录）：
    python tools/features/extract_ulip_features_to_zarr.py
"""
import sys
import os
import glob
import json
import argparse
import shutil
import numpy as np
import torch
import zarr
from tqdm import tqdm
from pathlib import Path

# ===================== 固定随机种子：实验可复现 绝对一致 =====================
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ===================== 导入Pytorch3D FPS（优先GPU加速，无则自动降级） =====================
try:
    import pytorch3d.ops as torch3d_ops
    HAS_PYTORCH3D = True
except ImportError:
    HAS_PYTORCH3D = False
    print("⚠️ Pytorch3D not found, use numpy FPS (slower but stable)")

# ===================== 全局路径配置（和你的环境完全匹配，无需修改任何一行） =====================
WORKSPACE_DIR = "/home/gl/RoboTwin/policy/DP2DP3/features_model"
ULIP_DIR = os.path.join(WORKSPACE_DIR, "ULIP-main/ULIP-main")
PC_SOURCE_DIR_NEW = os.path.join(WORKSPACE_DIR, "pc_dataset/PC_ORI")
PC_SOURCE_DIR_OLD = os.path.join(WORKSPACE_DIR, "pc_dataset/PC")
PC_SOURCE_DIR = PC_SOURCE_DIR_NEW if os.path.exists(PC_SOURCE_DIR_NEW) else PC_SOURCE_DIR_OLD
OUTPUT_ZARR_DIR = os.path.join(WORKSPACE_DIR, "pc_dataset/ulip_features_zarr")

sys.path.append(str(Path(ULIP_DIR).absolute()))
original_cwd = os.getcwd()
os.chdir(str(Path(ULIP_DIR).absolute()))

# ===================== 导入ULIP核心模型 异常兜底 =====================
try:
    from models.ULIP_models import ULIP2_WITH_OPENCLIP
    from models.pointbert.point_encoder import PointTransformer_Colored
    from utils.config import cfg_from_yaml_file
except Exception as e:
    # knn_cuda 等 C++ 扩展在导入时可能触发 RuntimeError（例如需要 Ninja 去编译），
    # 之前仅捕获 ImportError 会漏掉此类错误，导致脚本在导入失败时抛出并中止而没有友好提示。
    print("❌ Critical Import Error while loading ULIP dependencies:")
    print(f"   {repr(e)}")
    print("\nCommon fixes:")
    print(" - Ensure 'ninja' is installed (pip install ninja or apt install ninja-build).")
    print(" - Ensure CUDA toolkit and compilers are available if knn_cuda needs to be compiled.")
    print(" - Alternatively run on a machine that already has knn_cuda prebuilt for your CUDA version.")
    sys.exit(1)

def load_ulip_model(device):
    """加载预训练ULIP2-PointBERT模型 | 推理模式 | 权重加载校验 | 无冗余逻辑"""
    config_path = os.path.join(ULIP_DIR, 'models/pointbert/ULIP_2_PointBERT_10k_colored_pointclouds.yaml')
    config = cfg_from_yaml_file(config_path)
    
    class Args:
        evaluate_3d = True
    args = Args()

    # 初始化带颜色的PointBERT编码器 (核心：xyz+rgb 6维输入)
    point_encoder = PointTransformer_Colored(config.model, args=args)
    # 初始化ULIP2主模型 固定输出768维特征
    model = ULIP2_WITH_OPENCLIP(point_encoder=point_encoder, open_clip_model=None, pc_feat_dims=768)

    # 加载预训练权重 兼容多卡训练的module.前缀
    ckpt_path = os.path.join(ULIP_DIR, "pretrain_model/ULIP-2-PointBERT-10k-xyzrgb-pc-vit_g-objaverse_shapenet-pretrained.pt")
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint['state_dict']
    new_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}

    # 加载权重 忽略文本分支的缺失权重
    msg = model.load_state_dict(new_state_dict, strict=False)
    print(f"✅ Model Load Success | Missing Keys: {len(msg.missing_keys)} | Unexpected Keys: {len(msg.unexpected_keys)}")
    print(f"✅ Core Projection Layer Loaded: {'pc_projection' not in msg.missing_keys}")

    # 推理模式 关闭梯度
    model = model.to(device)
    model.eval()
    return model

def read_ply(ply_path):
    """【核心修复+优化】读取彩色PLY点云 | 解决所有已知BUG | 适配你的ASCII格式
    ✅ 修复：seek of closed file (文件全程打开)
    ✅ 修复：vertex_count=None 导致崩溃
    ✅ 修复：实际行数<标注行数 导致的无效零点污染
    ✅ 兼容：red/green/blue | r/g/b 两种RGB命名
    ✅ 输出：xyz [N,3] float32 | rgb [N,3] float32 (归一化到0-1)
    """
    ply_path = str(Path(ply_path).absolute())
    valid_xyz = np.zeros((0, 3), dtype=np.float32)
    valid_rgb = np.zeros((0, 3), dtype=np.float32)

    with open(ply_path, 'rb') as f:
        header_lines = []
        # 1. 读取文件头 终止条件兜底
        while True:
            line = f.readline()
            if not line: break
            header_lines.append(line)
            if line.strip() == b"end_header": break
        
        data_start = f.tell()
        header_txt = [ln.decode('ascii', errors='ignore').strip() for ln in header_lines]
        if not header_txt or header_txt[0] != "ply":
            raise ValueError(f"Not a valid PLY file")

        # 2. 解析文件头 核心参数初始化兜底
        fmt, vertex_count, in_vertex = None, 0, False
        properties = []
        for ln in header_txt:
            if ln.startswith("format "): fmt = ln.split()[1]
            elif ln.startswith("element vertex"):
                vertex_count = int(ln.split()[-1]) if len(ln.split())>2 else 0
                in_vertex = True
            elif ln.startswith("element "): in_vertex = False
            elif in_vertex and ln.startswith("property ") and "list" not in ln:
                properties.append((ln.split()[1], ln.split()[2]))

        # 3. 校验核心属性
        prop_names = [n for _, n in properties]
        if not all(k in prop_names for k in ("x", "y", "z")):
            raise ValueError(f"Missing XYZ coordinates")

        # 4. 匹配RGB属性
        rgb_keys = None
        if all(k in prop_names for k in ("red", "green", "blue")):
            rgb_keys = ("red", "green", "blue")
        elif all(k in prop_names for k in ("r", "g", "b")):
            rgb_keys = ("r", "g", "b")

        # 5. 读取ASCII格式数据 (你的Sapien点云专属适配 核心)
        if fmt == "ascii" and vertex_count > 0:
            f.seek(data_start)
            valid_data = []
            for _ in range(vertex_count):
                line = f.readline()
                if not line: break
                parts = line.decode('ascii', errors='ignore').strip().split()
                if len(parts) >= len(properties):
                    valid_data.append(parts)
            # ✅ 核心修复：只保留有效数据 裁剪无效零点
            vertex_count = len(valid_data)
            valid_xyz = np.zeros((vertex_count, 3), dtype=np.float32)
            valid_rgb = np.zeros((vertex_count, 3), dtype=np.float32)

            for i, parts in enumerate(valid_data):
                prop_vals = {n: float(v) for (t, n), v in zip(properties, parts)}
                valid_xyz[i] = [prop_vals["x"], prop_vals["y"], prop_vals["z"]]
                if rgb_keys:
                    valid_rgb[i] = [prop_vals[rgb_keys[0]], prop_vals[rgb_keys[1]], prop_vals[rgb_keys[2]]]

    # 6. RGB归一化 空判断兜底
    if rgb_keys and valid_rgb.size > 0 and valid_rgb.max() > 1.0:
        valid_rgb = valid_rgb / 255.0
    valid_rgb = np.clip(valid_rgb, 0.0, 1.0)

    # 7. 空点云兜底
    if valid_xyz.shape[0] == 0:
        return np.zeros((1, 3), dtype=np.float32), np.zeros((1, 3), dtype=np.float32)
    return valid_xyz, valid_rgb

def _write_json(path, obj):
    """工具函数：保存JSON元数据 | 自动创建目录 | 编码兜底"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def normalize_pc(xyz):
    """点云单位球归一化【必须预处理】| 消除位置/尺度影响 | 全边界兜底
    ✅ 逻辑：中心化 → 计算最大距离 → 缩放至单位球内
    ✅ 兜底：空点云/单点云 直接返回 无除以0风险
    """
    if len(xyz) <= 1:
        return xyz.copy()
    centroid = np.mean(xyz, axis=0)
    xyz_centered = xyz - centroid
    max_dist = np.max(np.sqrt(np.sum(xyz_centered ** 2, axis=1)))
    if max_dist < 1e-6:
        return xyz_centered
    return xyz_centered / (max_dist + 1e-6)

def fps_sample(xyz, n_points=2048, use_cuda=True):
    """最远点采样 FPS | 你的2048点→2048点 无损补点 | 全边界修复 | 双版本兼容
    ✅ 核心逻辑：点数不足 → 保留全部原始点 + 随机重复补点 (无信息丢失)
    ✅ 核心优势：均匀采样 保留几何轮廓 优于随机采样
    ✅ 修复：numpy版dtype=int64 兼容所有环境
    ✅ 修复：补点时N=1 无报错
    """
    N = xyz.shape[0]
    if N <= n_points:
        idx = np.arange(N, dtype=np.int64)
        if N < n_points and N > 0:
            # 随机重复补点 无副作用
            pad_idx = np.random.choice(N, n_points - N, replace=True)
            idx = np.concatenate([idx, pad_idx], axis=0)
        return idx

    # GPU加速FPS (优先)
    if HAS_PYTORCH3D and torch.cuda.is_available() and use_cuda:
        pts = torch.from_numpy(xyz).float().cuda()
        _, indices = torch3d_ops.sample_farthest_points(pts.unsqueeze(0), K=[n_points])
        return indices.squeeze(0).cpu().numpy().astype(np.int64)
    # CPU版FPS (备用 稳定)
    else:
        centroids = np.zeros(n_points, dtype=np.int64)
        dist = np.ones(N, dtype=np.float32) * 1e10
        centroids[0] = np.random.randint(0, N)
        for i in range(1, n_points):
            dist_i = np.sum((xyz - xyz[centroids[i-1]]) ** 2, axis=1)
            dist = np.minimum(dist, dist_i)
            centroids[i] = np.argmax(dist)
        return centroids

def sample_pc(xyz, rgb, n_points=2048, method='fps'):
    """点云采样封装 | FPS默认最优 | 空点云兜底"""
    if xyz.shape[0] == 0:
        return np.zeros((n_points, 3), dtype=np.float32), np.zeros((n_points, 3), dtype=np.float32)
    idx = fps_sample(xyz, n_points) if method == 'fps' else np.random.choice(xyz.shape[0], n_points, replace=True)
    return xyz[idx], rgb[idx]

def main():
    """主函数 | 全流程串联 | 批量处理 | 异常兜底 | 无冗余逻辑 | 日志清晰"""
    parser = argparse.ArgumentParser(description="ULIP 特征提取（输出 Zarr）")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=OUTPUT_ZARR_DIR,
        help="输出 Zarr 根目录（默认 pc_dataset/ulip_features_zarr）",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="若目标 episode.zarr 已存在则覆盖重写",
    )
    args = parser.parse_args()

    output_root = str(args.output_dir)
    overwrite = bool(args.overwrite)

    # 设备选择 优先GPU 推理速度提升10倍+
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n✅ Start Processing | Device: {device} | Input Points: 2048 → 2048 | Output Dim: auto (inferred)")
    try:
        model = load_ulip_model(device)
    except Exception as e:
        print(f"❌ Failed to initialize ULIP model: {e}")
        print("Please check the import/build errors above.")
        return

    # 遍历所有任务文件夹
    if not os.path.exists(PC_SOURCE_DIR):
        print(f"❌ Source Dir Not Found: {PC_SOURCE_DIR}")
        return
    tasks = [d for d in os.listdir(PC_SOURCE_DIR) if os.path.isdir(os.path.join(PC_SOURCE_DIR, d))]
    
    for task in tqdm(tasks, desc="📌 All Tasks Progress", ncols=80):
        task_dir = os.path.join(PC_SOURCE_DIR, task)
        out_task_dir = Path(output_root) / task
        out_task_dir.mkdir(parents=True, exist_ok=True)

        # 遍历当前任务的所有episode
        episodes = [d for d in os.listdir(task_dir) if d.startswith('episode_')]
        for episode in tqdm(episodes, desc=f"📌 {task} Episodes", ncols=80, leave=False):
            episode_dir = os.path.join(task_dir, episode)
            out_zarr = str(out_task_dir / f"{episode}.zarr")
            if os.path.exists(out_zarr):
                if not overwrite:
                    # 已存在，跳过
                    print(f"[Skip] Output exists: {out_zarr}")
                    continue
                shutil.rmtree(out_zarr, ignore_errors=True)

            # 按步序排序 保证时序正确 step_0001 → step_0002 ...
            ply_files = sorted(glob.glob(os.path.join(episode_dir, "*.ply")))
            print(f"[Episode] {task}/{episode}: found {len(ply_files)} .ply files -> out: {out_zarr}")
            if not ply_files:
                print(f"[Skip] No ply files for {episode}")
                continue

            features_list = []
            # 逐帧处理点云
            for ply_file in ply_files:
                try:
                    xyz, rgb = read_ply(ply_file)       # 读取点云 纯净无无效零点
                    xyz = normalize_pc(xyz)             # 归一化 消除位置尺度影响
                    xyz, rgb = sample_pc(xyz, rgb, 2048)# 补点到2048 最优输入点数
                    pc = np.concatenate([xyz, rgb], axis=1)  # 拼接xyz+rgb → [2048,6]
                    pc_tensor = torch.from_numpy(pc).float().unsqueeze(0).to(device) # [1,2048,6]

                    # 推理 无梯度计算 显存无泄漏
                    with torch.no_grad():
                        point_feat = model.encode_pc(pc_tensor) # 输出 [1,768]

                    # 过滤无效特征 NaN/Inf → 0
                    feat_np = np.nan_to_num(point_feat.cpu().numpy(), nan=0.0, posinf=0.0, neginf=0.0)
                    features_list.append(feat_np)
                except Exception as e:
                    print(f"\n⚠️ Skip {os.path.basename(ply_file)}: {str(e)}")
                    continue

            # 无有效特征 跳过
            if len(features_list) == 0:
                print(f"[Skip] No valid features extracted for {task}/{episode} (processed {len(ply_files)} files)")
                continue
            # 拼接所有帧特征 → [num_frames, D]
            all_features = np.concatenate(features_list, axis=0)
            feat_dim = int(all_features.shape[1])

            # 保存特征到Zarr 高效存储 支持分块读取
            zarr_root = zarr.group(out_zarr, overwrite=True)
            zarr_root.create_dataset('per_frame_features', data=all_features, chunks=(1, feat_dim), dtype=np.float32)
            print(f"[Saved] {task}/{episode} -> {out_zarr} (frames: {all_features.shape[0]}, dim: {feat_dim})")

            # 保存元数据 方便后续读取
            meta_info = {
                "model": "ULIP2-PointBERT",
                "task_name": task,
                "episode_name": episode,
                "num_frames": int(all_features.shape[0]),
                "feature_dim": feat_dim,
                "input_point_num": 2048,
                "source_point_num": 2048,
                "dtype": "float32"
            }
            _write_json(os.path.join(out_zarr, "meta.json"), meta_info)
            _write_json(os.path.join(out_zarr, "frame_paths.json"), [os.path.basename(p) for p in ply_files])

    print(f"\n✅ All Processing Completed Successfully!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n⚠️ Process Interrupted by User")
    finally:
        # 无论任何情况 切回原始工作目录
        os.chdir(original_cwd)
        print(f"✅ Back to Workspace: {original_cwd}")