"""features_common/depth_guided_film_online/__init__.py

在线 2 模型 DA3-FiLM Fusion
- DINOv3 (768d): 语义
- DA3 (2048d): 几何调制
"""
from .encoder_film_2model import DA3Film2ModelEncoder, FiLMLayer

__all__ = ["DA3Film2ModelEncoder", "FiLMLayer"]
