"""features_common/depth_guided_film_online_delta/encoder_film_delta.py

FiLM 时序差分 Encoder (DINOv3 + DA3 时序差分调制)
==================================================

动机:
  FiLM v1 中 geo_vec = mean_pool(DA3_tokens[t]) 是"绝对"深度信号.
  在 beat_block_hammer 任务中, 锤子占据画面主区域, DA3 信号被锤子
  几何主导 (锤子是最大、最显著的近景物体). 结果: geo_vec ≈ "锤子深度",
  FiLM 调制变成"锤子几何调制", 对目标木块位置没有帮助.

解决方案: 时序差分 DA3 信号
  - 静止的锤子 (t-1 帧与 t 帧深度相同) → 差分接近 0 → FiLM 调制微弱
  - 目标木块/末端执行器运动 → 差分非零 → FiLM 保留运动信号
  
  delta[t=0] = geo_proj[0]              # 第一帧用绝对特征 (无前一帧可减)
  delta[t>0] = geo_proj[t] - geo_proj[t-1]
  geo_vec    = mean(delta, over K tokens)

设计选择:
  - 差分在 projection 空间计算 (不是原始 2048d 空间), 更稳定
  - 其余架构完全与 FiLM v1 相同 (可直接对比)
  - 继承 DA3Film2ModelEncoder 复用所有权重/层定义
"""

from __future__ import annotations
import torch
import sys
import os

# 添加 features_model/ 到 path 以导入 v1 encoder
_self_dir = os.path.dirname(os.path.abspath(__file__))
_fc_dir = os.path.dirname(_self_dir)           # features_common/
_fm_dir = os.path.dirname(_fc_dir)             # features_model/
if _fm_dir not in sys.path:
    sys.path.insert(0, _fm_dir)

from features_common.depth_guided_film_online.encoder_film_2model import (
    DA3Film2ModelEncoder,
    FiLMLayer,
)


class DA3Film2ModelEncoderDelta(DA3Film2ModelEncoder):
    """FiLM 时序差分 Encoder.

    与 DA3Film2ModelEncoder 唯一区别:
      geo_vec 计算改为时序差分 DA3 投影特征的均值池化.

    所有 __init__ 参数与父类完全相同, 无需任何修改.
    """

    def forward(self, x) -> torch.Tensor:
        """
        x: List[2 个 Tensor]
           - x[0]: DINOv3, [B, To, K_sem, 768]
           - x[1]: DA3,    [B, To, K_geo, 2048]
        Returns: [B, To, out_dim]
        """
        if len(x) != 2:
            raise ValueError(f"Expected 2 models (DINOv3, DA3), got {len(x)}")

        sem_tok = x[self.SEMANTIC_IDX]    # [B, To, K_sem, 768]
        geo_tok = x[self.GEOMETRIC_IDX]  # [B, To, K_geo, 2048]

        if sem_tok.ndim == 3:
            sem_tok = sem_tok.unsqueeze(2)
        if geo_tok.ndim == 3:
            geo_tok = geo_tok.unsqueeze(2)

        B, To = sem_tok.shape[0], sem_tok.shape[1]

        sem_flat = sem_tok.reshape(B * To, sem_tok.shape[2], sem_tok.shape[3])
        geo_flat = geo_tok.reshape(B * To, geo_tok.shape[2], geo_tok.shape[3])

        # Stage 1: 投影
        N_s, K_s, C_s = sem_flat.shape
        q_tokens = self.semantic_proj(
            sem_flat.reshape(N_s * K_s, C_s)
        ).reshape(N_s, K_s, self.proj_dim)
        q_tokens = self._sample_tokens(q_tokens, self.max_tokens)
        K = q_tokens.shape[1]

        N_g, K_g, C_g = geo_flat.shape
        geo_proj = self.geometric_proj(
            geo_flat.reshape(N_g * K_g, C_g)
        ).reshape(N_g, K_g, self.proj_dim)
        geo_proj = self._sample_tokens(geo_proj, self.max_tokens)  # [N_g, K', proj_dim]
        K_geo = geo_proj.shape[1]

        # ★ 时序差分: [B*To, K', proj_dim] → [B, To, K', proj_dim] → delta → 均值
        # 静止前景 (锤子) 的差分 ≈ 0，只有运动目标保留非零信号
        geo_proj_t = geo_proj.reshape(B, To, K_geo, self.proj_dim)
        geo_delta = torch.empty_like(geo_proj_t)
        geo_delta[:, 0] = geo_proj_t[:, 0]              # 第一帧: 绝对特征
        if To > 1:
            geo_delta[:, 1:] = geo_proj_t[:, 1:] - geo_proj_t[:, :-1]   # 差分
        geo_proj_delta = geo_delta.reshape(N_g, K_geo, self.proj_dim)   # [N_g, K', proj_dim]
        geo_vec = geo_proj_delta.mean(dim=1)   # [N_g, proj_dim]

        # 位置编码
        if self.pos_embed is not None:
            pos = self.pos_embed[:, :K, :]
            q_tokens = q_tokens + pos

        # Stage 2: DA3-FiLM 调制 (差分 geo_vec)
        q_tokens = self.film(q_tokens, geo_vec)
        q_tokens = self.post_film_norm(q_tokens)

        # Stage 3: mean_pool → output_proj
        pooled = q_tokens.mean(dim=1)
        z = self.output_proj(pooled)
        z = z.reshape(B, To, self.out_dim)
        return z

    def extra_repr(self):
        base = super().extra_repr()
        return base + ", geo_mode=temporal_delta"
