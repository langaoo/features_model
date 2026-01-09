# 在线（Online）vs 离线（Offline）定义与区别

## TL;DR

| 维度 | **离线（Offline）** | **在线（Online）** |
|-----|-------------------|-------------------|
| **数据来源** | 预先录制/回放 | 实时从环境获取 |
| **闭环** | 无闭环（不执行） | 有闭环（预测→执行→观测→预测...） |
| **是否影响环境** | 否（只读） | 是（执行动作改变环境状态） |
| **典型场景** | 训练、调试、评估 | 真实部署、仿真交互 |

---

## 1. 离线（Offline）

### 定义

**数据/观测来自预先收集的数据集**，不与环境实时交互。

### 特点

- **数据源**：已有的 zarr/pkl/hdf5 文件
- **执行**：不执行动作（或只在回放脚本里"假执行"用于可视化）
- **闭环**：无闭环（预测的动作不会影响下一帧观测）
- **速度**：可以随意快进/慢放/跳帧

### 典型使用场景

1. **离线训练（Offline Training）**：
   - 从 raw_data 读取已有轨迹（obs + action）
   - 训练 DP head 拟合专家动作
   - 例子：`tools/train_dp_rgb_single_task_4models.py`

2. **离线推理（Offline Inference）**：
   - 从 zarr 读取已有观测（obs）
   - 预测动作序列
   - 不执行，只输出/保存/可视化
   - 例子：`tools/infer_dp_rgb_4models.py --task beat_block_hammer --episode episode_0`

3. **离线评估（Offline Evaluation）**：
   - 对比"预测的动作"与"专家动作"的 MSE/cosine similarity
   - 不需要环境

### 优点

- **稳定可复现**：同一数据集每次跑结果一样
- **快速调试**：不需要等仿真/真机运行
- **批量处理**：可以并行处理多个 episode

### 缺点

- **无法发现部署问题**：比如"动作在真实环境里执行会碰撞"、"累积误差导致失败"
- **数据分布固定**：无法测试泛化性（因为观测序列是固定的）

---

## 2. 在线（Online）

### 定义

**实时与环境交互**：预测动作 → 执行 → 观测新状态 → 再预测...

### 特点

- **数据源**：实时从环境/相机获取
- **执行**：每步预测的动作会真实执行
- **闭环**：有闭环反馈（动作影响环境，环境影响下一帧观测）
- **速度**：必须满足实时性（控制频率 1-30Hz）

### 典型使用场景

1. **在线推理（Online Inference）**：
   - 在 RoBoTwin 仿真或真机上部署
   - 每个控制周期：
     1. 获取最近 To 帧 RGB
     2. 提特征 → encoder → head → 输出动作
     3. 执行动作（`env.take_action(...)`）
     4. 环境 step，获取新观测
     5. 回到 1（receding horizon）
   - 例子：`RoBoTwin/policy/infer_dp_4models.py`（需要集成 env）

2. **在线微调（Online Fine-tuning）**：
   - 部署后根据实时反馈（成功/失败）更新策略
   - 例如：DAgger、RL fine-tuning

3. **在线数据收集（Online Data Collection）**：
   - 用当前策略跑真机，收集新轨迹
   - 再用新轨迹更新策略（迭代改进）

### 优点

- **真实反馈**：能发现离线评估看不到的问题（碰撞、抖动、累积误差）
- **闭环纠错**：每步重新预测，能对偏差做出反应

### 缺点

- **慢**：必须等环境真实 step
- **不可复现**：同样的初始状态，由于执行误差/随机性，每次结果可能不同
- **风险**：真机上可能损坏设备

---

## 3. "新数据不落盘直接串联"算在线还是离线？

### 你的问题

> 如果我**实时从相机获取 RGB → 4 个 backbone 提特征 → encoder → head → 输出动作**，但**中间特征不保存到磁盘**，这算在线还是离线？

### 答案：**在线推理（Online Inference）**

**判断标准不是"是否落盘"，而是"是否闭环"**：

- 如果你的流程是：
  1. 相机获取当前帧
  2. 4 backbone 提特征（不保存）
  3. encoder + head 输出动作
  4. **执行动作**
  5. 环境 step，相机获取新帧
  6. 回到 1

→ 这就是**在线推理**，因为有闭环（动作影响环境）。

- "不落盘"只是一个**工程优化**（省磁盘 I/O），不改变"在线"的本质。

### 与"离线推理"的区别

- **离线推理**：从 zarr 读已有特征 → head → 输出动作（不执行）
- **你的方案**：实时提特征（不保存） → head → 执行动作（闭环）

后者是在线。

---

## 4. 混合场景：半在线（Semi-Online）

有时你会遇到"介于在线/离线之间"的场景：

### 4.1 离线回放 + 在线执行

- 从 zarr 读已有观测（离线）
- 预测动作后**真的执行**（在线）
- 但环境状态可能与录制时不同（比如物体位置变了）

→ 这是"伪在线"：观测是离线的，但执行是在线的（会暴露闭环问题）。

### 4.2 在线采集 + 离线训练

- 真机上跑策略，收集轨迹（在线）
- 保存 obs + action
- 再用保存的数据训练新策略（离线）

→ 这是 Imitation Learning 的标准流程（DAgger）。

---

## 5. 你的 Pipeline 在哪个阶段？

### 当前状态（离线）

- **训练**：`tools/train_dp_rgb_single_task_4models.py` 读 zarr + pkl（离线）
- **推理**：`tools/infer_dp_rgb_4models.py` 读 zarr（离线）

### 下一步（在线）

- **RoBoTwin 推理**：`RoBoTwin/policy/infer_dp_4models.py`
  - 方案 A（半在线）：读 zarr obs + 执行动作（观测离线，执行在线）
  - 方案 B（全在线）：实时相机 → 4 backbone → encoder → head → 执行（完全闭环）

你问的"不落盘直接串联"就是**方案 B（全在线）**。

---

## 6. 总结表格

| 场景 | 观测来源 | 动作执行 | 闭环 | 落盘 | 分类 |
|-----|---------|---------|-----|-----|------|
| 训练 DP head | zarr（已有） | 不执行 | 否 | 读盘 | **离线训练** |
| 调试推理输出 | zarr（已有） | 不执行 | 否 | 读盘 | **离线推理** |
| RoBoTwin 回放 | zarr（已有） | 执行 | 是 | 读盘 | **半在线（伪）** |
| **你要做的**：实时提特征 | 相机（实时） | 执行 | 是 | 不落盘 | **在线推理** ✅ |

---

## 7. 实操建议

### 阶段 1：离线验证（当前）

先用离线推理确保 head 能输出合理动作：
```bash
python tools/infer_dp_rgb_4models.py \
  --head_ckpt outputs/dp_rgb_runs/.../final_head.pt \
  --task beat_block_hammer --episode episode_0
```

### 阶段 2：半在线部署（过渡）

在 RoBoTwin 里用已有 zarr obs，但真的执行动作：
```bash
python RoBoTwin/policy/infer_dp_4models.py \
  --head_ckpt outputs/.../final_head.pt \
  --rgb_zarr_roots_4 rgb_dataset/features_croco_... \
  --task beat_block_hammer --episode episode_0 \
  --robotwin_env_class beat_block_hammer
```

这能暴露"执行时的碰撞/抖动"问题。

### 阶段 3：全在线部署（最终目标）

把 4 个 backbone 集成到推理脚本，实时提特征（不落盘）：
```python
# 伪代码
while not done:
    rgb = env.get_camera()  # 实时获取
    features = [croco(rgb), vggt(rgb), dinov3(rgb), da3(rgb)]  # 不保存
    z = encoder(features)
    action = head(z)
    env.take_action(action)
```

这是真正的在线推理。

---

**最终回答你的问题：**

> "新数据不落盘直接串联 4backbone → encoder → head"

= **在线推理（Online Inference）**，因为它实时与环境交互（闭环）。"不落盘"只是工程优化，不改变在线本质。
