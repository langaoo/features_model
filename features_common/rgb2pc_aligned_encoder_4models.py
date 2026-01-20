"""Legacy import shim for RGB2PC aligned encoder.

保留旧路径以兼容历史引用，实际实现位于
`features_common.alignment.rgb2pc_aligned_encoder_4models`。
"""

from features_common.alignment.rgb2pc_aligned_encoder_4models import (  # noqa: F401
    RGB2PCAligned4ModelSpec,
    RGB2PCAlignedEncoder4Models,
)

__all__ = ["RGB2PCAligned4ModelSpec", "RGB2PCAlignedEncoder4Models"]
