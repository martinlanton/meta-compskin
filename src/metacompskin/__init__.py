"""Compressed Skinning for Facial Blendshapes.

PyTorch implementation of sparse skinning decomposition for facial
blendshape animation (SIGGRAPH 2024). See the README for usage examples.
"""

from metacompskin.animation_generator import AnimationFrameGenerator
from metacompskin.model_data import BlendshapeModelData, MayaBlendshapeModelData
from metacompskin.model_fit import SkinCompressor

__version__ = "0.1.0"

__all__ = [
    "AnimationFrameGenerator",
    "BlendshapeModelData",
    "MayaBlendshapeModelData",
    "SkinCompressor",
    "__version__",
]
