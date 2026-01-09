# 数据路径说明

本项目使用软链接管理数据路径，避免重复存储大文件。

## RGB 数据（rgb_dataset/）

| 路径 | 说明 | 用途 |
|-----|------|------|
| `RGB` | → `/home/gl/RoboTwin/policy/DP3/rgbpc_dataset/RGB` | **处理后的 RGB**（推荐用于训练） |
| `RGB_ORI` | → `/home/gl/RoboTwin/policy/DP3/rgbpc_dataset/RGB_ORI` | **原始 RGB**（用于重新提取特征） |
| `RGB1` | 本地目录 | 旧的备份数据（可删除） |
| `features_*_zarr` | 本地/软链接 | 提取的特征文件（zarr 格式） |

## 点云数据（pc_dataset/）

| 路径 | 说明 | 用途 |
|-----|------|------|
| `PC_source` | → `/home/gl/RoboTwin/policy/DP3/rgbpc_dataset/PC` | **处理后的点云**（推荐） |
| `PC_ORI` | → `/home/gl/RoboTwin/policy/DP3/rgbpc_dataset/PC_ORI` | **原始点云**（用于重新提取） |
| `PC` | 本地目录 | 旧的备份数据（可删除） |
| `ulip_features_zarr/` | 本地目录 | ULIP 提取的特征文件 |

## 轨迹数据（raw_data/）

软链接到外部数据源（具体路径见项目配置）。

---

## 修改数据源

### 提取 RGB 特征时指定新路径

```bash
# 使用原始 RGB（RGB_ORI）
python tools/run_extract_features.py \
  --model croco \
  --rgb_root /home/gl/features_model/rgb_dataset/RGB_ORI \
  --out_root /home/gl/features_model/rgb_dataset \
  --device cuda --all
```

### 提取点云特征时自动使用 PC_ORI

编辑 `tools/extract_ulip_features_to_zarr.py`，脚本会自动优先使用 `PC_ORI`（如果存在），否则 fallback 到 `PC_source`。

---

## 注意事项

1. **不要直接提交数据到 git**：所有数据路径已在 `.gitignore` 中忽略。
2. **软链接在其他机器无效**：部署到新环境时需要重新创建软链接。
3. **外部依赖**：`RoboTwin` 项目需要单独克隆到 `/home/gl/RoboTwin`。
