"""该文件已弃用。

原因：当前工作区并没有任何“更小的 DINO 权重文件”可供离线加载，
这个脚本如果继续保留，会让人误以为可以直接运行。

请使用：
- extract_multi_frame_dinov3_features_local.py（已实现完整 pipeline；如需小模型请提供本地 ckpt 路径，我会把它接入同一个脚本）。
"""

raise SystemExit(
    "该脚本已弃用：请使用 dinov3/extract_multi_frame_dinov3_features_local.py。"
)
