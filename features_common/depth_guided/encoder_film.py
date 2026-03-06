"""features_common/depth_guided/encoder_film.py

DA3-FiLM Depth-Guided Fusion Encoder
=====================================

动机
----
原 DepthGuidedFusionEncoder 用 Cross-Attention (~14M参数) 融合语义和几何特征,
在 50 条 demo 数据上容易过拟合, 实测只有 71% (vs 直融 78%).

本方案用 FiLM (Feature-wise Linear Modulation) 替换 Cross-Attention:
  - DA3 几何 mean_pool -> 轻量 MLP -> scale/shift 向量 (参数量 <1.5M)
  - 语义 token 按 scale/shift 逐 token 调制
  - 完全保留语义特征, 不会像 Cross-Attn Value 那样把输出变成几何特征的加权组合

核心公式:
  geo_vec = mean_pool(DA3_tokens)         # [B*To, proj_dim]
  scale, shift = film_mlp(geo_vec)        # 各 [B*To, proj_dim]
  fused_token = sem_token*(1+scale) + shift   # 仍在语义空间

模型顺序约定 (与整个 DP2DP3 项目一致):
  index 0 -> CroCo   (C=1024)
  index 1 -> VGGT    (C=2048)
  index 2 -> DINOv3  (C=768)
  index 3 -> DA3     (C=2048) <- 几何调制源
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLMLayer(nn.Module):
    """FiLM 调制层: scale/shift 向量逐 token 调制.

    Args:
        cond_dim:  条件向量维度 (来自 DA3 几何 mean_pool)
        feat_dim:  被调制特征维度 (语义 tokens)
        hidden:    MLP 隐藏层大小
    """
    def __init__(self, cond_dim: int, feat_dim: int, hidden: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, feat_dim * 2),  # -> [scale, shift]
        )
        # 初始化为 0: 训练初期 scale=0->1+0=1, shift=0, 等同于恒等映射
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, feat: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat: [N, K, D]  语义 tokens
            cond: [N, cond_dim]  几何全局向量
        Returns:
            [N, K, D]  调制后特征
        """
        out = self.mlp(cond)                  # [N, D*2]
        scale, shift = out.chunk(2, dim=-1)   # [N, D] each
        scale = scale.unsqueeze(1)            # [N, 1, D] -> broadcast over K
        shift = shift.unsqueeze(1)
        return feat * (1.0 + scale) + shift


class DA3FilmFusionEncoder(nn.Module):
    """DA3-FiLM Fusion Encoder (轻量深度引导方案).

    三阶段处理:
      Stage 1: 各模型 tokens -> 各自 Linear 投影到 proj_dim
      Stage 2: DA3 mean_pool -> FiLM MLP -> scale/shift -> 调制语义 tokens
      Stage 3: 调制后语义 tokens mean_pool -> output_proj -> out_dim

    Args:
        semantic_in_dims: 语义组各模型输入维度 (CroCo=1024, VGGT=2048, DINOv3=768)
        geometric_in_dim: DA3 输入维度 2048
        proj_dim:         统一投影维度
        film_hidden:      FiLM MLP 隐藏层大小
        out_dim:          最终输出维度 (默认 1280, 与其他实验对齐)
        semantic_fusion:  语义多模型合并策略 ('concat_proj' / 'mean' / 'weighted')
        with_pos_enc:     是否对语义 tokens 加 1D learnable 位置编码
        dropout:          Dropout 概率
        max_tokens:       每模型最大 token 数 (等间隔下采样), 默认 196
    """

    SEMANTIC_IDX = (0, 1, 2)   # CroCo, VGGT, DINOv3
    GEOMETRIC_IDX = 3          # DA3

    def __init__(
        self,
        semantic_in_dims=(1024, 2048, 768),
        geometric_in_dim=2048,
        proj_dim=256,
        film_hidden=256,
        out_dim=1280,
        semantic_fusion='concat_proj',
        with_pos_enc=True,
        dropout=0.1,
        max_tokens=196,
    ):
        super().__init__()
        self.semantic_in_dims = tuple(semantic_in_dims)
        self.geometric_in_dim = int(geometric_in_dim)
        self.proj_dim = int(proj_dim)
        self.out_dim = int(out_dim)
        self.semantic_fusion = semantic_fusion
        self.with_pos_enc = bool(with_pos_enc)
        self.n_semantic = len(semantic_in_dims)
        self.max_tokens = int(max_tokens)

        # Stage 1: 各模型独立投影
        self.semantic_projs = nn.ModuleList([
            nn.Sequential(nn.Linear(int(d), int(proj_dim)), nn.LayerNorm(int(proj_dim)))
            for d in semantic_in_dims
        ])
        self.geometric_proj = nn.Sequential(
            nn.Linear(int(geometric_in_dim), int(proj_dim)), nn.LayerNorm(int(proj_dim))
        )

        # Stage 1b: 语义融合
        if semantic_fusion == 'concat_proj':
            self.semantic_merge = nn.Sequential(
                nn.Linear(int(proj_dim) * self.n_semantic, int(proj_dim) * 2),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(int(proj_dim) * 2, int(proj_dim)),
                nn.LayerNorm(int(proj_dim)),
            )
        elif semantic_fusion == 'weighted':
            self.semantic_weights = nn.Parameter(
                torch.ones(self.n_semantic) / self.n_semantic
            )
            self.semantic_merge = None
        elif semantic_fusion == 'mean':
            self.semantic_merge = None
        else:
            raise ValueError(f"Unknown semantic_fusion: {semantic_fusion}")

        # 可选 1D learnable 位置编码
        if with_pos_enc:
            self.pos_embed = nn.Parameter(
                torch.randn(1, int(max_tokens), int(proj_dim)) * 0.02
            )
        else:
            self.pos_embed = None

        # Stage 2: DA3-FiLM 调制
        self.film = FiLMLayer(
            cond_dim=int(proj_dim),
            feat_dim=int(proj_dim),
            hidden=int(film_hidden),
        )
        self.post_film_norm = nn.LayerNorm(int(proj_dim))

        # Stage 3: pool + 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(int(proj_dim), int(out_dim) * 2),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(out_dim) * 2, int(out_dim)),
            nn.LayerNorm(int(out_dim)),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _sample_tokens(self, tokens: torch.Tensor, max_k: int) -> torch.Tensor:
        """等间隔下采样到 max_k."""
        K = tokens.shape[1]
        if K <= max_k:
            return tokens
        idx = torch.linspace(0, K - 1, max_k, dtype=torch.long, device=tokens.device)
        return tokens[:, idx, :]

    def _project_semantic(self, tokens_list) -> torch.Tensor:
        """
        tokens_list: 3个 [N, K_i, C_i]
        Returns: [N, K, proj_dim]
        """
        projected = []
        for toks, proj in zip(tokens_list, self.semantic_projs):
            N, K_i, C_i = toks.shape
            t = proj(toks.reshape(N * K_i, C_i)).reshape(N, K_i, self.proj_dim)
            t = self._sample_tokens(t, self.max_tokens)
            projected.append(t)

        min_K = min(t.shape[1] for t in projected)
        projected = [t[:, :min_K, :] for t in projected]

        if self.semantic_fusion == 'concat_proj':
            cat = torch.cat(projected, dim=-1)   # [N, K, proj_dim*n]
            N, K, _ = cat.shape
            sem = self.semantic_merge(cat.reshape(N * K, -1)).reshape(N, K, self.proj_dim)
        elif self.semantic_fusion == 'weighted':
            w = torch.softmax(self.semantic_weights, dim=0)
            stacked = torch.stack(projected, dim=0)
            sem = (stacked * w.view(-1, 1, 1, 1)).sum(dim=0)
        else:  # mean
            sem = torch.stack(projected, dim=0).mean(dim=0)

        return sem   # [N, K, proj_dim]

    def forward(self, x) -> torch.Tensor:
        """
        x: List[4个Tensor], 每个 [B, To, K_i, C_i] (token模式)
           或每个 [B, To, C_i] (pool模式, 自动unsqueeze)
        Returns: [B, To, out_dim]
        """
        if len(x) != 4:
            raise ValueError(f"Expected 4 models, got {len(x)}")

        # Pool 模式兼容
        if x[0].ndim == 3:
            x = [t.unsqueeze(2) for t in x]

        B, To = x[0].shape[0], x[0].shape[1]

        sem_list = [x[i] for i in self.SEMANTIC_IDX]
        geo_tok  = x[self.GEOMETRIC_IDX]

        # Flatten B,To -> batch维度
        sem_flat = [t.reshape(B * To, t.shape[2], t.shape[3]) for t in sem_list]
        geo_flat = geo_tok.reshape(B * To, geo_tok.shape[2], geo_tok.shape[3])

        # Stage 1: 投影
        q_tokens = self._project_semantic(sem_flat)   # [B*To, K, proj_dim]
        K = q_tokens.shape[1]

        N_g, Kg, C_g = geo_flat.shape
        geo_proj = self.geometric_proj(
            geo_flat.reshape(N_g * Kg, C_g)
        ).reshape(N_g, Kg, self.proj_dim)
        geo_proj = self._sample_tokens(geo_proj, self.max_tokens)
        geo_vec  = geo_proj.mean(dim=1)               # [B*To, proj_dim]

        # 可选位置编码
        if self.pos_embed is not None:
            pos = self.pos_embed[:, :K, :]            # [1, K, proj_dim]
            q_tokens = q_tokens + pos

        # Stage 2: DA3-FiLM 调制
        q_tokens = self.film(q_tokens, geo_vec)       # [B*To, K, proj_dim]
        q_tokens = self.post_film_norm(q_tokens)

        # Stage 3: mean_pool -> output_proj
        pooled = q_tokens.mean(dim=1)                 # [B*To, proj_dim]
        z = self.output_proj(pooled)                  # [B*To, out_dim]
        z = z.reshape(B, To, self.out_dim)

        return z

    def extra_repr(self):
        return (
            f"semantic_in_dims={self.semantic_in_dims}, proj_dim={self.proj_dim}, "
            f"out_dim={self.out_dim}, semantic_fusion={self.semantic_fusion}, "
            f"with_pos_enc={self.with_pos_enc}, max_tokens={self.max_tokens}"
        )
