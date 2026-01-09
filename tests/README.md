# tests

这里放的是“格式/对齐工程测试”。

## 运行（示例）

你需要先让四个模型各自导出同一个 task/episode（至少一个）到不同 root，然后：

```bash
FEATURES_CROCO_ROOT=/abs/path/to/features_croco_v2_encoder_dict_unified \
FEATURES_VGGT_ROOT=/abs/path/to/features_vggt_encoder_dict_unified \
FEATURES_DINOV3_ROOT=/abs/path/to/features_dinov3_encoder_dict_unified \
FEATURES_DA3_ROOT=/abs/path/to/features_da3_encoder_dict_unified \
pytest -q
```

如果你只想先做单模型 schema 检查，也可以只设置任意一个 `FEATURES_*_ROOT`。
