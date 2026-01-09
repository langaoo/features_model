import os
import glob
import torch
import argparse
from tqdm import tqdm
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

# 设置使用 GPU 1
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def get_dtype():
    # 如果是 Ampere 架构 (如 A100, 3090) 使用 bfloat16，否则使用 float16
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        return torch.bfloat16
    return torch.float16

def load_model(model_path, device):
    """
    加载 VGGT 模型
    """
    print(f"正在加载模型: {model_path}")
    model = VGGT()
    # 加载权重
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print("模型加载完成")
    return model

def get_image_paths(episode_path):
    """
    获取 episode 目录下的所有图片路径，并按文件名排序
    """
    # 假设图片格式为 png 或 jpg
    extensions = ['*.png', '*.jpg', '*.jpeg']
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(episode_path, ext)))
    
    # 按文件名排序，确保时序正确
    image_paths.sort()
    return image_paths

def sliding_window(sequence, window_size, stride):
    """
    生成滑动窗口
    """
    num_frames = len(sequence)
    if num_frames < window_size:
        return []
    
    windows = []
    for i in range(0, num_frames - window_size + 1, stride):
        windows.append(sequence[i : i + window_size])
    return windows

def process_episode(model, episode_path, output_dir, window_size=8, stride=1, device="cuda", dtype=torch.float16):
    """
    处理单个 episode：
    1. 读取所有帧
    2. 生成滑动窗口
    3. 输入 VGGT 提取特征
    4. 保存特征
    """
    image_paths = get_image_paths(episode_path)
    if not image_paths:
        print(f"警告: {episode_path} 中没有找到图片")
        return

    # 生成窗口
    windows = sliding_window(image_paths, window_size, stride)
    if not windows:
        print(f"警告: {episode_path} 帧数 ({len(image_paths)}) 小于窗口大小 ({window_size})")
        return

    episode_features = []
    
    # 遍历每个窗口进行处理
    # 注意：为了避免显存溢出，我们逐个窗口处理。如果显存足够，可以 batch 处理。
    for window_idx, window_paths in enumerate(tqdm(windows, desc=f"Processing {os.path.basename(episode_path)}", leave=False)):
        
        # 加载并预处理图像
        # load_and_preprocess_images 会将图像 resize 到 518x518 (默认) 并归一化
        # 返回形状: [S, 3, H, W]
        images = load_and_preprocess_images(window_paths).to(device)
        
        # 添加 batch 维度 -> [1, S, 3, H, W]
        images = images.unsqueeze(0)
        
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                # VGGT 前向传播
                # aggregated_tokens_list 包含所有层的输出
                # patch_start_idx 指示 patch token 的起始位置 (跳过 register tokens 等)
                aggregated_tokens_list, patch_start_idx = model.aggregator(images)
                
                # 获取第 24 层 (最后一层) 的特征
                # aggregated_tokens_list[-1] 形状: [B, S, Total_Tokens, Embed_Dim]
                last_layer_tokens = aggregated_tokens_list[-1]
                
                # 提取 Patch Tokens (去除特殊 token)
                # 形状: [B, S, N_patches, Embed_Dim]
                patch_tokens = last_layer_tokens[:, :, patch_start_idx:, :]
                B, S, N_patches, C = patch_tokens.shape
                
                # 还原为 2D 空间结构
                # 输入图像形状: [B, S, 3, H, W]
                # Patch Size = 14
                # H_feat = H // 14
                # W_feat = W // 14
                _, _, _, H, W = images.shape
                patch_size = 14
                H_feat = H // patch_size
                W_feat = W // patch_size
                
                feature_map = patch_tokens.view(B, S, H_feat, W_feat, C)
                
                # 转移到 CPU 并保存，减少显存占用
                episode_features.append(feature_map.cpu())

    # 将所有窗口的特征拼接
    # 最终形状: [Num_Windows, B, S, H_feat, W_feat, C] -> [Num_Windows, S, H_feat, W_feat, C] (去除 B=1)
    if episode_features:
        all_features = torch.cat(episode_features, dim=0) # dim=0 是 batch 维度，这里其实是 window 维度
        
        # 构建保存路径
        # 保持原始目录结构: output_dir/task_name/episode_name.pt
        rel_path = os.path.relpath(episode_path, start=os.path.dirname(os.path.dirname(episode_path)))
        # rel_path 可能是 "task_name/episode_name"
        # 但我们需要更稳健的方式，假设 episode_path 是 .../task/episode
        
        task_name = os.path.basename(os.path.dirname(episode_path))
        episode_name = os.path.basename(episode_path)
        
        save_dir = os.path.join(output_dir, task_name)
        os.makedirs(save_dir, exist_ok=True)
        
        save_path = os.path.join(save_dir, f"{episode_name}.pt")
        
        print(f"保存特征到: {save_path}, 形状: {all_features.shape}")
        torch.save(all_features, save_path)

def main():
    parser = argparse.ArgumentParser(description="Extract VGGT features from multi-frame episodes.")
    # 默认路径改为当前workspace（/home/gl/features_model），方便和CroCo/DINOv3统一落到同一份rgb_dataset下
    parser.add_argument("--dataset_root", type=str, default="/home/gl/features_model/rgb_dataset/RGB", help="Path to the dataset root")
    parser.add_argument("--output_root", type=str, default="/home/gl/features_model/rgb_dataset/features_vggt", help="Path to save features")
    parser.add_argument("--model_path", type=str, default="/home/gl/features_model/vggt/weight/model.pt", help="Path to the model weights")
    parser.add_argument("--window_size", type=int, default=8, help="Sliding window size")
    parser.add_argument("--stride", type=int, default=1, help="Sliding window stride")
    args = parser.parse_args()

    # --- 配置参数 ---
    dataset_root = args.dataset_root
    output_root = args.output_root
    model_path = args.model_path
    
    window_size = args.window_size
    stride = args.stride
    
    # --- 核心逻辑说明 ---
    """
    1. 多帧输入支持:
       VGGT 的 Aggregator 模块天然支持多帧输入 (Sequence 维度)。
       它通过 "Alternating Attention" (交替注意力) 机制，在 "Frame Attention" (帧内) 和 "Global Attention" (全局/帧间) 之间交替，
       从而自动融合时序信息。
       因此，我们不需要手动进行特征融合 (如 concat 或 mean)，只需将多帧图像作为一个 batch (S 维度) 输入模型即可。
       
    2. 窗口大小选择:
       窗口大小为 8 帧是合理的。
       - 8 帧通常能覆盖约 0.25~1秒的时间跨度 (取决于 FPS)，足以捕捉短时动作和动态变化。
       - 序列过长会显著增加显存消耗 (Attention 是 O(N^2))。
       - VGGT 训练时通常也使用类似的序列长度。
       
    3. 特征提取:
       我们提取 Aggregator 输出列表的最后一个元素 (第 24 层)。
       该特征已经包含了经过 24 层时空交互后的丰富语义和几何信息。
    """

    device = get_device()
    dtype = get_dtype()
    
    # 加载模型
    model = load_model(model_path, device)
    
    # 遍历数据集
    # 结构: dataset_root/task_name/episode_name/images
    task_dirs = glob.glob(os.path.join(dataset_root, "*"))
    
    for task_dir in task_dirs:
        if not os.path.isdir(task_dir):
            continue
            
        episode_dirs = glob.glob(os.path.join(task_dir, "*"))
        
        for episode_dir in episode_dirs:
            if not os.path.isdir(episode_dir):
                continue
                
            print(f"正在处理 Episode: {episode_dir}")
            process_episode(
                model=model,
                episode_path=episode_dir,
                output_dir=output_root,
                window_size=window_size,
                stride=stride,
                device=device,
                dtype=dtype
            )

if __name__ == "__main__":
    main()
