"""
从原生 HDF5 数据中仅提取 Sapien 的 RGB（不依赖 pointcloud），并保存为 PNG。

用途（对应你在 PIPELINE_DETAILED.md 里提到的 Step6）：
- 当你只想用“已训练好的对齐模块”去做某个新任务的 Head 训练时，
  新任务不需要 pointcloud teacher，因此也不需要再提取/保存 PC_ORI；
- 但你可能仍然希望把 RGB 图像落盘，方便肉眼检查或复用离线四模型特征提取脚本。

示例（在 features_model 根目录运行）：
  python tools/dataset/process_sapien_rgb.py place_a2b_left demo_clean_dp2dp3 50 \\
    --output_root /home/gl/RoboTwin/policy/DP2DP3/features_model \\
    --camera head_camera \\
    --rgb_dirname RGB_TRAIN
"""

from __future__ import annotations

import argparse
import io
import os
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm


DEFAULT_DATA_ROOT = Path("/home/gl/RoboTwin/data")
DEFAULT_CAMERA_NAME = "head_camera"


def find_data_root(provided_root: Optional[str] = None) -> Path:
    if provided_root:
        return Path(provided_root).resolve()
    return DEFAULT_DATA_ROOT.resolve()


def save_rgb(rgb_data, output_path: Path) -> None:
    """保存 RGB 图像为 PNG，并处理 Sapien 数据的 BGR 通道顺序。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if isinstance(rgb_data, (bytes, np.bytes_)):
            img = Image.open(io.BytesIO(bytes(rgb_data))).convert("RGB")
        elif isinstance(rgb_data, np.ndarray):
            img = Image.fromarray(rgb_data.astype(np.uint8))
        else:
            img = Image.fromarray(np.array(rgb_data, dtype=np.uint8))

        img_array = np.array(img)
        if img_array.ndim == 3 and img_array.shape[-1] == 3:
            # HDF5 内 JPEG 常由 OpenCV(BGR) 写入，这里统一转成 RGB
            img_array = img_array[:, :, ::-1]

        Image.fromarray(img_array.astype(np.uint8), mode="RGB").save(str(output_path))
    except Exception as e:
        print(f"❌ 图像保存失败 [{output_path}]: {e}")


def process_sapien_rgb(
    task_name: str,
    tag: str,
    num_episodes: int,
    *,
    output_root: Optional[str] = None,
    camera_name: str = DEFAULT_CAMERA_NAME,
    data_root: Optional[str] = None,
    rgb_dirname: str = "RGB_ORI",
) -> None:
    data_root_path = find_data_root(data_root)
    data_dir = data_root_path / task_name / tag / "data"

    if output_root is None:
        output_root_path = Path(__file__).resolve().parents[2]
    else:
        output_root_path = Path(output_root).resolve()

    output_rgb_dir = (
        output_root_path
        / "rgb_dataset"
        / str(rgb_dirname)
        / f"{task_name}-{tag}-{num_episodes}_sapien_{camera_name}"
    )

    print(f"🔎 数据根: {data_root_path}")
    print(f"📂 数据目录: {data_dir}")
    print(f"📂 RGB输出: {output_rgb_dir}")

    if not data_dir.exists():
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")

    output_rgb_dir.mkdir(parents=True, exist_ok=True)

    total_frames = 0
    for episode_idx in tqdm(range(int(num_episodes)), desc="Processing episodes"):
        hdf5_path = data_dir / f"episode{episode_idx}.hdf5"
        if not hdf5_path.exists():
            print(f"⚠️  Episode {episode_idx} 不存在，跳过")
            continue

        episode_rgb_dir = output_rgb_dir / f"episode_{episode_idx}"
        episode_rgb_dir.mkdir(exist_ok=True)

        rgb_key = f"observation/{camera_name}/rgb"
        with h5py.File(hdf5_path, "r") as f:
            if rgb_key not in f:
                print(f"⚠️  Episode {episode_idx} 无 {camera_name} RGB 数据，跳过")
                continue

            rgb_data = f[rgb_key][:]
            num_frames = int(rgb_data.shape[0])

            for frame_idx in range(num_frames):
                out_png = episode_rgb_dir / f"step_{frame_idx:04d}.png"
                save_rgb(rgb_data[frame_idx], out_png)
                total_frames += 1

    print("=" * 60)
    print("✅ RGB 提取完成")
    print(f"📊 保存帧数: {total_frames}")
    print(f"📂 输出目录: {output_rgb_dir}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="从 HDF5 提取 RGB 并保存为 PNG（不依赖点云）")
    parser.add_argument("task_name", type=str, help="任务名称 (e.g., place_a2b_left)")
    parser.add_argument("tag", type=str, help="数据标签/配置 (e.g., demo_clean_dp2dp3)")
    parser.add_argument("num_episodes", type=int, help="要处理的 episode 数量")
    parser.add_argument("--output_root", type=str, default=None, help="输出根目录（默认: features_model 根）")
    parser.add_argument("--camera", type=str, default=DEFAULT_CAMERA_NAME, help="相机名称（默认: head_camera）")
    parser.add_argument("--data_root", type=str, default=str(DEFAULT_DATA_ROOT), help="数据根目录（默认: /home/gl/RoboTwin/data）")
    parser.add_argument("--rgb_dirname", type=str, default="RGB_ORI", help="rgb_dataset 下的子目录名（默认: RGB_ORI）")
    args = parser.parse_args()

    process_sapien_rgb(
        task_name=str(args.task_name),
        tag=str(args.tag),
        num_episodes=int(args.num_episodes),
        output_root=args.output_root,
        camera_name=str(args.camera),
        data_root=str(args.data_root),
        rgb_dirname=str(args.rgb_dirname),
    )


if __name__ == "__main__":
    main()

