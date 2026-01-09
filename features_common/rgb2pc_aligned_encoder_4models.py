"""features_common/rgb2pc_aligned_encoder_4models.py

严格版：4 个视觉模型特征 -> 对齐 encoder -> 融合 -> proj_student

目标：复现 rgb2pc 蒸馏训练时的 student 路径（简化到 per-frame 向量层面）。

输入
    x: FloatTensor[B, To, M, C_in]  (M=4)
输出
    z: FloatTensor[B, To, D]       (D == fuse_dim, 例如 1280)

说明
    - checkpoints: /home/gl/features_model/outputs/train_rgb2pc_runs/.../ckpt_step_*.pt
    - ckpt 内包含：adapters (4 个), fusion (logits 形状 [4]), proj_student
    - DP 训练时我们冻结这个 encoder，仅训练 DP head。

注意
    你当前的 per-frame feature 是 mean pooled 后的向量，因此这里把每帧当作一个 token。
    如果将来希望更接近 token-level 对齐，可把 x 扩展为 [B,To,M,K,C] 并在此处做 token pooling。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from features_common.fusion import WeightedFusion, MoEFusion


def _strip_module_prefix(state_dict: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            out[k[len("module."):]] = v
        else:
            out[k] = v
    return out


@dataclass
class RGB2PCAligned4ModelSpec:
    n_models: int = 4
    in_dims: tuple[int, int, int, int] = (1024, 2048, 768, 2048)
    fuse_dim: int = 1280
    fusion: str = "weighted"


class RGB2PCAlignedEncoder4Models(nn.Module):
    """Load full 4-model student encoder from ckpt."""

    def __init__(self, spec: RGB2PCAligned4ModelSpec, *, moe_hidden: int = 1024):
        super().__init__()
        self.spec = spec

        class _AdapterMLP(nn.Module):
            # match checkpoint structure: <idx>.net.<k>.*
            def __init__(self, in_dim: int, out_dim: int):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(in_dim, out_dim * 2),
                    nn.GELU(),
                    nn.Dropout(0.0),
                    nn.Linear(out_dim * 2, out_dim),
                    nn.LayerNorm(out_dim),
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.net(x)

        if int(spec.n_models) != 4:
            raise ValueError("This encoder currently supports exactly 4 models")
        self.adapters = nn.ModuleList([
            _AdapterMLP(int(spec.in_dims[i]), int(spec.fuse_dim)) for i in range(4)
        ])

        if spec.fusion == "weighted":
            self.fusion = WeightedFusion(num_models=int(spec.n_models))
        elif spec.fusion == "moe":
            self.fusion = MoEFusion(dim=int(spec.fuse_dim), num_models=int(spec.n_models), hidden_dim=int(moe_hidden))
        else:
            raise ValueError(f"Unknown fusion: {spec.fusion}")

        class _ProjMLP(nn.Module):
            # match ckpt: proj_student.net.*
            def __init__(self, dim: int):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(dim, dim * 2),
                    nn.GELU(),
                    nn.Dropout(0.0),
                    nn.Linear(dim * 2, dim),
                    nn.LayerNorm(dim),
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.net(x)

        self.proj_student = _ProjMLP(int(spec.fuse_dim))

    @classmethod
    def from_checkpoint(
        cls,
        ckpt_path: str,
        *,
        map_location: str | torch.device = "cpu",
        freeze: bool = True,
    ) -> "RGB2PCAlignedEncoder4Models":
        ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)
        if not isinstance(ckpt, dict):
            raise TypeError(f"Unexpected ckpt type: {type(ckpt)}")

        args = ckpt.get("args", {})
        if not isinstance(args, dict):
            args = vars(args)

        fuse_dim = int(args.get("fuse_dim", 1280))
        fusion = str(args.get("fusion", "weighted"))
        moe_hidden = int(args.get("moe_hidden", 1024))

        adapters_sd_all = ckpt.get("adapters")
        if adapters_sd_all is None:
            raise KeyError("ckpt missing 'adapters'")
        adapters_sd_all = _strip_module_prefix(adapters_sd_all)

        # infer per-model in_dims from adapter i
        in_dims = []
        for i in range(4):
            w = adapters_sd_all.get(f"{i}.net.0.weight")
            if w is None:
                raise KeyError(f"missing {i}.net.0.weight in ckpt adapters")
            in_dims.append(int(w.shape[1]))
        spec = RGB2PCAligned4ModelSpec(n_models=4, in_dims=tuple(in_dims), fuse_dim=fuse_dim, fusion=fusion)
        model = cls(spec, moe_hidden=moe_hidden)

        # load adapters by slicing per-index prefix
        for i in range(4):
            sub = {k[len(f"{i}."):]: v for k, v in adapters_sd_all.items() if k.startswith(f"{i}.")}
            # keys now like net.0.weight ...
            model.adapters[i].load_state_dict(sub, strict=True)

        # fusion
        fusion_sd = ckpt.get("fusion")
        if fusion_sd is not None:
            fusion_sd = _strip_module_prefix(fusion_sd)
            # should match logits shape [4]
            model.fusion.load_state_dict(fusion_sd, strict=True)

        proj_sd = ckpt.get("proj_student")
        if proj_sd is not None:
            proj_sd = _strip_module_prefix(proj_sd)
            model.proj_student.load_state_dict(proj_sd, strict=True)

        if freeze:
            for p in model.parameters():
                p.requires_grad = False
            model.eval()

        return model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,To,M,C]
        if x.ndim != 4:
            raise ValueError(f"Expected x [B,To,M,C], got {tuple(x.shape)}")
        b, t, m, c = x.shape
        if m != int(self.spec.n_models):
            raise ValueError(f"Model count mismatch: got {m}, expected {self.spec.n_models}")
        # support hetero dims: for simplicity require caller to pad/pack to the right dims per model

        weight_dtype = next(self.parameters()).dtype
        x = x.to(dtype=weight_dtype)

        zs = []
        for mi in range(m):
            ci = int(self.spec.in_dims[mi])
            tok = x[:, :, mi, :ci].reshape(b * t, ci)
            z = self.adapters[mi](tok)  # [B*T,D]
            zs.append(z)

        fused, _w = self.fusion(zs)  # [B*T,D]
        z = self.proj_student(fused)  # [B*T,D]
        return z.reshape(b, t, -1)
