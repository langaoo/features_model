import os
import glob
import torch
import argparse
from tqdm import tqdm
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from torchvision import transforms

# 设置设备和数据类型
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def get_dtype():
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        return torch.bfloat16
    return torch.float16

def load_dinov3_model(backbone_path, device):
    """加载DINOv3模型（ViT-7B/16）"""
    print(f"加载DINOv3模型: {backbone_path}")
    processor = AutoImageProcessor.from_pretrained(backbone_path)
    model = AutoModel.from_pretrained(
        backbone_path,
        device_map="auto",
        torch_dtype=get_dtype()
    )
    model.eval()
    return model, processor

def preprocess_frames(frame_paths, processor, device):
    """预处理一帧图像（适配DINOv3输入）"""
    frames = [Image.open(path).convert("RGB") for path in frame_paths]
    inputs = processor(images=frames, return_tensors="pt").to(device)
    return inputs

def sliding_window(sequence, window_size, stride):
    """生成滑动窗口（与VGGT代码保持一致）"""
    num_frames = len(sequence)
    if num_frames < window_size:
        return []
    return [sequence[i:i+window_size] for i in range(0, num_frames - window_size + 1, stride)]

def fuse_features(frame_features, mode="attention"):
    """多帧特征融合（支持多种模式）"""
    # frame_features: [window_size, 1, embed_dim]（单帧全局特征）
    frame_features = torch.cat(frame_features, dim=0)  # [window_size, embed_dim]
    
    if mode == "mean":
        # 平均融合（简单高效）
        return torch.mean(frame_features, dim=0, keepdim=True)  # [1, embed_dim]
    
    elif mode == "concat":
        # 通道拼接（维度扩展）
        return frame_features.flatten(0, 1).unsqueeze(0)  # [1, window_size*embed_dim]
    
    elif mode == "attention":
        # 时序注意力融合（动态加权）
        attn_weights = torch.softmax(torch.matmul(frame_features, frame_features.T), dim=1)  # [window_size, window_size]
        weighted_feat = torch.matmul(attn_weights.T, frame_features)  # [window_size, embed_dim]
        return torch.mean(weighted_feat, dim=0, keepdim=True)  # [1, embed_dim]
    
    else:
        raise ValueError(f"不支持的融合模式: {mode}")

def process_episode(model, processor, episode_path, output_dir, window_size=8, stride=1, device="cuda", dtype=torch.float16):
    """处理单个episode，提取多帧融合特征"""
    # 获取帧路径并排序
    image_paths = get_image_paths(episode_path)
    if not image_paths:
        print(f"警告: {episode_path} 无图片")
        return
    
    # 生成滑动窗口
    windows = sliding_window(image_paths, window_size, stride)
    if not windows:
        print(f"警告: {episode_path} 帧数不足")
        return
    
    episode_features = []
    for window_paths in tqdm(windows, desc=f"Processing {os.path.basename(episode_path)}"):
        # 预处理窗口内所有帧
        inputs = preprocess_frames(window_paths, processor, device)
        
        # 提取单帧特征
        frame_features = []
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
            outputs = model(**inputs)
            # 取全局池化特征（[batch_size=window_size, embed_dim]）
            frame_feats = outputs.pooler_output  # [window_size, embed_dim]
            frame_features.append(frame_feats)
        
        # 融合窗口内特征（选择一种模式）
        fused_feat = fuse_features(frame_features, mode="attention")  # [1, embed_dim]
        episode_features.append(fused_feat.cpu())
    
    # 保存特征
    if episode_features:
        all_features = torch.cat(episode_features, dim=0)  # [num_windows, embed_dim]
        task_name = os.path.basename(os.path.dirname(episode_path))
        episode_name = os.path.basename(episode_path)
        save_dir = os.path.join(output_dir, task_name)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{episode_name}.pt")
        torch.save(all_features, save_path)
        print(f"保存特征: {save_path}, 形状: {all_features.shape}")

# 辅助函数（复用VGGT代码逻辑）
def get_image_paths(episode_path):
    extensions = ['*.png', '*.jpg', '*.jpeg']
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(episode_path, ext)))
    image_paths.sort()
    return image_paths

def main():
    parser = argparse.ArgumentParser(description="DINOv3多帧特征提取")
    parser.add_argument("--dataset_root", type=str, default="/home/gl/vggt/rgb_dataset/RGB")
    parser.add_argument("--output_root", type=str, default="/home/gl/vggt/rgb_dataset/dinov3_features")
    parser.add_argument("--model_path", type=str, default="facebook/dinov3-vit7b16-pretrain-lvd1689m")  # 本地权重路径或HF模型名
    parser.add_argument("--window_size", type=int, default=8)
    parser.add_argument("--stride", type=int, default=1)
    args = parser.parse_args()

    device = get_device()
    dtype = get_dtype()
    model, processor = load_dinov3_model(args.model_path, device)

    # 遍历任务和episode（与VGGT代码结构一致）
    task_dirs = glob.glob(os.path.join(args.dataset_root, "*"))
    for task_dir in task_dirs:
        if not os.path.isdir(task_dir):
            continue
        episode_dirs = glob.glob(os.path.join(task_dir, "*"))
        for episode_dir in episode_dirs:
            if not os.path.isdir(episode_dir):
                continue
            print(f"处理Episode: {episode_dir}")
            process_episode(
                model=model,
                processor=processor,
                episode_path=episode_dir,
                output_dir=args.output_root,
                window_size=args.window_size,
                stride=args.stride,
                device=device,
                dtype=dtype
            )

if __name__ == "__main__":
    main()