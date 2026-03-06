"""features_common/depth_guided_dino_online/encoder_dino_only.py

DINO-only 在线 Encoder (消融实验: 去掉 DA3 调制)
================================================

用途: 对比 FiLM (DINOv3+DA3) 和 纯 DINOv3 的性能差异.
如果 DINO-only 在 beat_block_hammer 上 > FiLM (32%), 说明 DA3 调制
在该任务上是主动有害的——DA3 被工具/前景主导, 调制信号引入噪声.

流程:
  sem_tokens = DINOv3_proj(tokens)   # [B*To, K, proj_dim]
  pos_enc (可选)
  output = output_proj(mean_pool(sem_tokens))  # [B, To, out_dim]
"""

from __future__ import annotations
import torch
import torch.nn as nn


class DinoOnlyEncoder(nn.Module):
    """纯 DINOv3 在线 Encoder, 无 DA3 调制.

    Args:
        semantic_in_dim: DINOv3 token 维度 (768)
        proj_dim:        投影维度 (256)
        out_dim:         最终输出维度 (与 FiLM 版对齐: 1280)
        with_pos_enc:    是否加 1D learnable 位置编码
        dropout:         Dropout 概率
        max_tokens:      最大 token 数 (等间隔下采样), 默认 196
    """

    def __init__(
        self,
        semantic_in_dim: int = 768,
        proj_dim: int = 256,
        out_dim: int = 1280,
        with_pos_enc: bool = True,
        dropout: float = 0.1,
        max_tokens: int = 196,
    ):
        super().__init__()
        self.semantic_in_dim = int(semantic_in_dim)
        self.proj_dim = int(proj_dim)
        self.out_dim = int(out_dim)
        self.max_tokens = int(max_tokens)

        self.semantic_proj = nn.Sequential(
            nn.Linear(semantic_in_dim, proj_dim),
            nn.LayerNorm(proj_dim),
        )

        self.pos_embed = (
            nn.Parameter(torch.zeros(1, max_tokens, proj_dim))
            if with_pos_enc
            else None
        )

        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        self.output_proj = nn.Sequential(
            nn.Linear(proj_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    @staticmethod
    def _sample_tokens(tokens: torch.Tensor, max_k: int) -> torch.Tensor:
        K = tokens.shape[1]
        if K <= max_k:
            return tokens
        idx = torch.linspace(0, K - 1, max_k, dtype=torch.long, device=tokens.device)
        return tokens[:, idx, :]

    def forward(self, x) -> torch.Tensor:
        """
        x: Tensor 或 List[Tensor]
           支持两种输入模式:
           (A) 单 Tensor:  [B, To, K, 768]  或  [B, To, 768]
           (B) List[1+个]:  x[0] 作为 DINOv3 tokens, 其余忽略
               (兼容 FiLM 的 List[dino, da3] 输入格式, 方便对比实验)
        Returns:
            [B, To, out_dim]
        """
        if isinstance(x, (list, tuple)):
            sem_tok = x[0]
        else:
            sem_tok = x

        if sem_tok.ndim == 3:
            sem_tok = sem_tok.unsqueeze(2)  # [B, To, 1, C]

        B, To = sem_tok.shape[0], sem_tok.shape[1]
        sem_flat = sem_tok.reshape(B * To, sem_tok.shape[2], sem_tok.shape[3])  # [N, K, 768]

        N, K, C = sem_flat.shape
        q_tokens = self.semantic_proj(
            sem_flat.reshape(N * K, C)
        ).reshape(N, K, self.proj_dim)
        q_tokens = self._sample_tokens(q_tokens, self.max_tokens)
        K = q_tokens.shape[1]

        if self.pos_embed is not None:
            pos = self.pos_embed[:, :K, :]
            q_tokens = q_tokens + pos

        q_tokens = self.dropout(q_tokens)

        pooled = q_tokens.mean(dim=1)          # [N, proj_dim]
        z = self.output_proj(pooled)           # [N, out_dim]
        z = z.reshape(B, To, self.out_dim)

        return z

    def extra_repr(self):
        return (
            f"semantic_in_dim={self.semantic_in_dim}, "
            f"proj_dim={self.proj_dim}, out_dim={self.out_dim}, "
            f"max_tokens={self.max_tokens}"
        )
