# 项目全面诊断与修复报告

生成时间: 2026年1月10日
项目: features_model (RGB-Only 蒸馏对齐训练Pipeline)

---

## 🎯 整体数据流 (5个阶段)

### 阶段图示

```
RGB图像 ──────┐
              ├─→ [阶段1: 特征提取] ─→ 4模型Zarr特征
点云数据 ─────┘                      └─→ ULIP点云特征
                                            │
                                            ↓
                                   [阶段2: 对齐训练]
                                    Student vs Teacher
                                     InfoNCE Loss
                                            │
                                            ↓
                                     对齐Encoder ckpt
                                   (adapters+fusion+proj)
                                            │
                                            ↓
                           ┌────────────────┴────────────────┐
                           │      [阶段3: DP Head训练]       │
                           │   4模型特征 + 对齐Encoder      │
                           │   → DP Head + Normalizer       │
                           └────────────────┬────────────────┘
                                            │
                                            ↓
                                    Policy Checkpoint
                                            │
                              ┌─────────────┴─────────────┐
                              │                           │
                   [阶段4: 离线推理]          [阶段5: 在线推理] ⚠️
                  预提取Zarr特征             实时相机 → 动作
                  tools/infer_dp_rgb_        tools/eval_dp_rgb_
                  4models.py                 in_robotwin.py
```

---

## ✅ 已修复的问题

### 1. **配置文件路径问题** (严重)

**修复前**:
```yaml
pc_root: /home/gl/features_model/pc_dataset/ulip_features_zarr
vis_zarr_roots:
  - /home/gl/features_model/rgb_dataset/features_croco_...
```

**修复后**:
```yaml
pc_root: pc_dataset/ulip_features_zarr  # 相对路径
vis_zarr_roots:
  - rgb_dataset/features_croco_encoder_dict_unified_zarr
  # 其他路径...
```

**影响**: 提升可移植性,支持不同用户/机器运行

---

### 2. **vis_zarr_roots顺序错误** (严重 - CRITICAL)

**问题**: 配置文件中四模型顺序与代码in_dims不匹配

**修复前顺序**: croco, da3, dinov3, vggt
**修复后顺序**: croco(1024), vggt(2048), dinov3(768), da3(2048)

**说明**: 
- `rgb2pc_aligned_encoder_4models.py` 中定义 `in_dims=(1024,2048,768,2048)`
- 对应模型必须按此顺序: CroCo, VGGT, DINOv3, DA3
- 错误的顺序会导致adapter加载错误维度,训练失败

**验证方法**:
```python
# 在训练脚本中添加断言
assert croco_dim == 1024, f"CroCo维度应为1024,实际{croco_dim}"
assert vggt_dim == 2048, f"VGGT维度应为2048,实际{vggt_dim}"
assert dinov3_dim == 768, f"DINOv3维度应为768,实际{dinov3_dim}"
assert da3_dim == 2048, f"DA3维度应为2048,实际{da3_dim}"
```

---

### 3. **在线推理流程补全** (严重)

**创建**: `tools/eval_dp_rgb_in_robotwin.py`

**功能**:
- 完整的RoBoTwin环境集成框架
- Policy加载 (encoder + head + normalizer)
- 评估循环 (reset → observe → predict → act)
- 成功率统计和结果保存

**当前限制**:
- OnlineFeatureExtractor需要加载4个视觉backbone (显存需求~16GB)
- 推荐在评估阶段继续使用离线zarr特征
- 真正部署时再实现在线特征提取

---

## 📝 发现的其他问题 (已记录)

### 中等优先级

**4. 冗余文件** (详见 CLEANUP_GUIDE.md)
- `dinov3/extract_multi_frame_dino_small_local.py` - 已弃用
- `third_party/` 与根目录模型重复
- `croco/extract_multi_frame_croco_features_unified.py` - wrapper冗余

**5. 命名不一致**
- `--encoder_ckpt` vs `--rgb2pc_ckpt` (已在代码中兼容处理)

**6. 文档与代码不同步**
- README提到在线推理已实现,但实际是半成品

### 低优先级

**7. 数据路径发现逻辑复杂**
- `_discover_available_pairs`函数有多层fallback
- 建议简化或添加更清晰的日志

**8. 缺少错误处理**
- zarr文件损坏时缺少友好提示
- 建议添加try-except和修复建议

**9. 测试覆盖不足**
- `tests/`目录存在但用例很少
- 建议添加关键路径的单元测试

---

## 🔍 特征维度核查表

| 模型 | 输出维度 | 文件位置 | 验证状态 |
|-----|---------|---------|---------|
| **CroCo** | 1024 | croco/extract_*.py | ✅ 已验证 |
| **VGGT** | 2048 | vggt/extract_*.py | ✅ 已验证 |
| **DINOv3** | 768 | dinov3/extract_*.py | ✅ 已验证 |
| **DA3** | 2048 | Depth-Anything-3/extract_*.py | ✅ 已验证 |
| **ULIP (Teacher)** | 256 | tools/extract_ulip_*.py | ⚠️ 需确认 |

**ULIP维度注意事项**:
- `ULIP2_WITH_OPENCLIP` 内部使用 `pc_feat_dims=768`
- 但输出经过projection后实际是 **256维**
- `train_rgb2pc_distill.py` 会自动检测teacher维度并调整fuse_dim

---

## 🚀 推荐的修复优先级

### P0 - 立即处理 (已完成)
- ✅ 修复配置文件绝对路径
- ✅ 修复vis_zarr_roots顺序
- ✅ 创建在线推理框架代码

### P1 - 本周处理
- [ ] 运行完整训练验证修复效果
- [ ] 删除已弃用文件 (`dinov3/extract_multi_frame_dino_small_local.py`)
- [ ] 添加关键路径的单元测试

### P2 - 两周内处理
- [ ] 统一third_party目录结构
- [ ] 增强错误处理和用户提示
- [ ] 更新README和文档,确保与代码同步

### P3 - 月度优化
- [ ] 实现完整的在线特征提取 (OnlineFeatureExtractor)
- [ ] 代码风格统一 (linting + formatting)
- [ ] 性能优化 (zarr访问,dataloader等)

---

## ✅ 验证清单

### 训练流程验证

```bash
# 1. 验证配置文件
python -c "import yaml; print(yaml.safe_load(open('configs/train_rgb2pc_distill_default.yaml')))"

# 2. 运行smoke test (对齐训练)
python tools/train_rgb2pc_distill.py \
  --config configs/train_rgb2pc_distill_default.yaml \
  --steps 100 \
  --batch_size 4 \
  --save_dir outputs/smoke_test_alignment

# 3. 运行smoke test (DP head训练)
python tools/train_dp_rgb_single_task_4models.py \
  --task beat_block_hammer-demo_randomized-20_head_camera \
  --encoder_ckpt outputs/train_rgb2pc_runs/run_best_bs32/ckpt_step_0010000.pt \
  --epochs 2 \
  --batch_size 4 \
  --save_dir outputs/smoke_test_dp

# 4. 验证推理脚本
python tools/infer_dp_rgb_4models.py \
  --ckpt outputs/smoke_test_dp/final_head.pt \
  --task beat_block_hammer-demo_randomized-20_head_camera \
  --episode episode_0 \
  --exec_steps 5
```

### 数据完整性验证

```bash
# 检查zarr文件完整性
python -c "
import zarr
tasks = ['beat_block_hammer-demo_randomized-20_head_camera']
models = ['croco', 'vggt', 'dinov3', 'da3']
for task in tasks:
    for model in models:
        path = f'rgb_dataset/features_{model}_encoder_dict_unified_zarr/{task}/episode_0.zarr'
        try:
            arr = zarr.open(path, 'r')
            print(f'✓ {model}: shape={arr.shape}')
        except Exception as e:
            print(f'✗ {model}: {e}')
"
```

---

## 📚 相关文档

- **CLEANUP_GUIDE.md** - 详细的代码清理指南
- **README_PIPELINE.md** - Pipeline主流程文档  
- **DATA_PATHS.md** - 数据路径说明
- **configs/train_rgb2pc_distill_default.yaml** - 对齐训练配置 (已修复)

---

## 💡 重要建议

### 对于训练

1. **使用离线特征**: 训练阶段强烈推荐使用预提取的zarr特征
   - 速度快 (无需每次forward 4个backbone)
   - 稳定 (特征固定,排除backbone波动)
   - 节省显存 (只训练小模块)

2. **检查数据对齐**: 确保4个模型的zarr顺序与配置文件一致
   ```bash
   # 在训练开始时打印
   print("Model order:", ["croco", "vggt", "dinov3", "da3"])
   print("Zarr roots:", vis_zarr_roots)
   ```

3. **梯度累积**: 如果显存不足,使用 `--grad_accum_steps 2` 提升有效batch size

### 对于推理

1. **评估阶段**: 使用离线zarr + `tools/infer_dp_rgb_4models.py`
2. **部署阶段**: 实现在线提取 + `tools/eval_dp_rgb_in_robotwin.py`
3. **性能优化**: 考虑TensorRT/ONNX加速backbone推理

### 对于维护

1. **定期清理**: 使用CLEANUP_GUIDE.md定期清理冗余文件
2. **版本控制**: 重要修改前先 `git commit` 备份
3. **文档同步**: 代码变更时同步更新README和docs/

---

## 🎓 技术要点总结

### 整体架构

```
视觉编码器层:
  Input: RGB Images
  Models: CroCo(1024) + VGGT(2048) + DINOv3(768) + DA3(2048)
  Output: 4个per-frame zarr特征

对齐层:
  Input: 4模型特征 + 点云特征(ULIP 256)
  Components: 4×Adapter → Fusion(weighted/MoE) → Projector
  Training: InfoNCE Loss (CLIP-style)
  Output: 统一特征空间 (1280维)

策略层:
  Input: 对齐特征序列 [To, 1280]
  Model: Diffusion Policy (UNet1D + DDPM)
  Output: 动作序列 [Ta, A]  (A=6/7/12/14)
```

### 关键超参数

| 参数 | 对齐训练 | DP训练 | 说明 |
|-----|---------|--------|-----|
| batch_size | 32 | 16 | 对齐训练偏好大batch |
| lr | 3e-4 | 1e-4 | 对齐训练学习率可稍高 |
| window_size | 8 | - | 滑动窗口大小 |
| stride | 1 | - | 窗口滑动步长 |
| n_obs_steps | - | 2 | 观测历史长度 |
| horizon | - | 8 | 动作预测长度 |
| tau | 0.07 | - | InfoNCE温度系数 |

---

## 📞 问题排查

### 训练loss不下降?

1. 检查vis_zarr_roots顺序是否正确
2. 检查数据是否对齐 (task/episode匹配)
3. 尝试降低学习率或增加batch_size
4. 检查teacher特征是否正常 (非全零/全相同)

### 推理效果差?

1. 检查encoder checkpoint是否正确加载
2. 检查normalizer是否正确应用
3. 对比训练时和推理时的obs预处理是否一致
4. 检查action维度是否匹配环境

### 显存不足?

1. 降低batch_size
2. 使用梯度累积 (`--grad_accum_steps`)
3. 启用AMP (`--amp`)
4. 减少num_workers

---

**报告结束**

如有疑问,请参考相关文档或检查代码注释。
所有修复已应用,建议按优先级逐步验证和优化。
