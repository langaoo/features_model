import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

# 设备配置
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

# 加载模型
model = VGGT()
dir = "/home/gl/vggt/weight/model.pt"
model.load_state_dict(torch.load(dir))
model = model.to(device)
model.eval()  # 推理模式

# 加载并预处理图像（替换为你的图像路径）
image_paths = ["/home/gl/RoboTwin/policy/DP3/rgbpc_dataset/RGB/beat_block_hammer-demo_randomized-50/episode_0/head_camera/step_0000.png",
"/home/gl/RoboTwin/policy/DP3/rgbpc_dataset/RGB/beat_block_hammer-demo_randomized-50/episode_0/head_camera/step_0001.png",
"/home/gl/RoboTwin/policy/DP3/rgbpc_dataset/RGB/beat_block_hammer-demo_randomized-50/episode_0/head_camera/step_0002.png"]
images = load_and_preprocess_images(image_paths).to(device)  # 形状: [S, 3, H, W]，S为图像数量

# 添加batch维度（如果需要）
if len(images.shape) == 4:
    images = images.unsqueeze(0)  # 形状变为 [B, S, 3, H, W]，B=1

# 获取视觉编码特征
with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=dtype):
        # aggregated_tokens_list 为多尺度视觉编码特征，patch_start_idx为patch起始索引（可选）
        aggregated_tokens_list, patch_start_idx = model.aggregator(images)

# 输出特征信息
print("视觉编码特征尺度数量:", len(aggregated_tokens_list))
for i, tokens in enumerate(aggregated_tokens_list):
    print(f"第{i+1}层特征形状:", tokens.shape)  # 形状通常为 [B, S, N_patches, embed_dim]