"""features_common/rgb2pc_student_encoder.py

把你训练好的 RGB2PC 蒸馏 checkpoint（tools/train_rgb2pc_distill.py 产物）
封装成一个可复用的 encoder：

- 输入：RGB 融合特征序列  x: FloatTensor[B, To, C_in]
  这里的 C_in 对应 dp_rgb_dataset 里 per-frame mean pooled 后的维度。

- 输出：对齐后的 embedding 序列 z: FloatTensor[B, To, D]  (D == fuse_dim)

注意：蒸馏训练本身是 window/step 级别取 token -> adapter -> fusion -> projector -> z_s[B,D]。
在 DP 这条链路里，我们通常已经做了 token/patch 的预融合（mean pool）得到 per-frame feature 向量。
因此本模块采用一个简化但一致的推理路径：
- 把每个 time-step 的输入向量当作“单 token”
- 走 adapter -> fusion(对单 token 等价于加权求和) -> proj_student

如果你希望更接近蒸馏训练时的 token 级别效果，建议让 dp_rgb_dataset 输出 [B,To,K,C]
并在此处做 token pooling / attention；本文件先把工程骨架跑通。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from features_common.fusion import WeightedFusion, MoEFusion


@dataclass
class RGB2PCStudentSpec:
    n_models: int
    in_dim: int
    fuse_dim: int
    fusion: str


def _strip_module_prefix(state_dict: dict) -> dict:
    out = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            out[k[len("module."):]] = v
        else:
            out[k] = v
    return out


class RGB2PCStudentEncoder(nn.Module):
    """从 rgb2pc_distill checkpoint 构建 student encoder。

    forward 输入是 [B,To,C_in]，输出 [B,To,fuse_dim]。
    """

    def __init__(
        self,
        spec: RGB2PCStudentSpec,
        *,
        moe_hidden: int = 1024,
    ):
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

        # 兼容 tools/train_rgb2pc_distill.py 里的 adapters：
        # checkpoint key 形如 "0.net.0.weight"，对应一个 MLP(in_dim -> fuse_dim)。
        # 这里做成 ModuleList，每个模型一个 MLP；DP 这条链路只用一个“融合后向量”，因此 n_models=1。
        self.adapters = nn.ModuleList([
            _AdapterMLP(int(spec.in_dim), int(spec.fuse_dim))
        ])

        if spec.fusion == "weighted":
            self.fusion = WeightedFusion(num_models=1)
        elif spec.fusion == "moe":
            self.fusion = MoEFusion(dim=int(spec.fuse_dim), num_models=1, hidden_dim=int(moe_hidden))
        else:
            raise ValueError(f"Unknown fusion: {spec.fusion}")

        # proj_student 在蒸馏里是 MLP(fuse_dim -> fuse_dim)
        # 这里严格复现。
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
    ) -> "RGB2PCStudentEncoder":
        ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)
        if not isinstance(ckpt, dict):
            raise TypeError(f"Unexpected ckpt type: {type(ckpt)}")

        args = ckpt.get("args", {})
        if not isinstance(args, dict):
            # 兼容 argparse Namespace 等
            args = vars(args)

        fuse_dim = int(args.get("fuse_dim", 1280))
        fusion = str(args.get("fusion", "weighted"))
        moe_hidden = int(args.get("moe_hidden", 1024))

        # 关键：in_dim 从 adapters 的第一层 weight 形状推断
        # adapters.0.net.0.weight: [hidden, in_dim]
        adapters_sd_all = ckpt.get("adapters")
        if adapters_sd_all is None:
            raise KeyError("ckpt missing 'adapters'")
        adapters_sd_all = _strip_module_prefix(adapters_sd_all)

        # DP 这边只支持单输入向量，因此只加载 adapter[0]
        adapters_sd = {k: v for k, v in adapters_sd_all.items() if k.startswith('0.')}

        first_w = adapters_sd.get("0.net.0.weight")
        if first_w is None:
            # 兜底：找任意一个 *.net.0.weight
            for k, v in adapters_sd_all.items():
                if k.endswith(".net.0.weight"):
                    first_w = v
                    break
        if first_w is None:
            raise KeyError("cannot infer in_dim from adapters state_dict")
        in_dim = int(first_w.shape[1])

        spec = RGB2PCStudentSpec(n_models=1, in_dim=in_dim, fuse_dim=fuse_dim, fusion=fusion)
        model = cls(spec, moe_hidden=moe_hidden)

        # load weights
        # adapters_sd contains keys like "0.net.0.weight" from a ModuleList of MLP(net=Sequential)
        _ = model.adapters.load_state_dict(adapters_sd, strict=True)
        # fusion
        # ckpt 的 fusion 通常是 num_models=4（croco/vggt/dinov3/da3）的 logits；
        # DP 这条链路使用的是“已经融合后的单向量输入”，等价于 num_models=1，
        # 因此不加载 ckpt 的 fusion 权重，避免 shape mismatch。
        # proj
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
        """x: [B,To,C_in] -> [B,To,fuse_dim]"""
        if x.ndim != 3:
            raise ValueError(f"Expected x [B,To,C], got {tuple(x.shape)}")

        b, t, c = x.shape
        if c != int(self.spec.in_dim):
            raise ValueError(f"Input dim mismatch: got {c}, expected {self.spec.in_dim}")

        # treat each step feature as one token, one model
        # keep dtype consistent with weights (AMP / fp16 dataset can cause matmul dtype mismatch)
        weight_dtype = next(self.parameters()).dtype
        tok = x.to(dtype=weight_dtype).reshape(b * t, c)
        y = self.adapters[0](tok)  # [B*T, D]

        # fusion expects list[Tensor[N,D]] and returns (fused, weights)
        fused, _w = self.fusion([y])
        z = fused
        z = self.proj_student(z)  # [B*T, D]
        z = z.reshape(b, t, -1)
        return z
