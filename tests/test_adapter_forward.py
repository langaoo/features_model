"""adapter 最小单元测试：保证 forward 的 shape 正确、能反传梯度。

这不是训练脚本，只是防止后续改 adapter 时把维度搞乱。
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch  # type: ignore

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from features_common.adapters import MultiModelAdapter


def test_adapter_forward_and_backward():
    w, t, hf, wf, c = 2, 8, 14, 14, 1024
    x = torch.randn(w, t, hf, wf, c, dtype=torch.float32, requires_grad=True)

    adapter = MultiModelAdapter(in_dim=c, out_dim=256, use_temporal_pool=True)
    out = adapter(x)

    assert out.per_frame.shape == (w, t, hf, wf, 256)
    assert out.pooled is not None
    assert out.pooled.shape == (w, hf, wf, 256)

    loss = out.pooled.mean() + out.per_frame.mean()
    loss.backward()
    assert x.grad is not None
