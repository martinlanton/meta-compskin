"""Compressed Skinning for Facial Blendshapes.

PyTorch implementation of sparse skinning decomposition for facial
blendshape animation (SIGGRAPH 2024). See the README for usage examples.

Note:
    The public classes are re-exported lazily (PEP 562) so that torch-free
    environments — in particular Maya / mayapy running
    :class:`MayaBlendshapeExporter` or :func:`build_skinned_rig` — can import
    the package without having
    PyTorch installed. Torch is only imported when a class that needs it
    (e.g. ``SkinCompressor``) is first accessed.
"""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from metacompskin.animation_generator import AnimationFrameGenerator
    from metacompskin.maya_exporter import MayaBlendshapeExporter
    from metacompskin.maya_pipeline import compress_and_build_rig
    from metacompskin.maya_rig_builder import build_skinned_rig
    from metacompskin.model_data import BlendshapeModelData, MayaBlendshapeModelData
    from metacompskin.model_fit import SkinCompressor

__version__ = "0.1.0"

_EXPORTS = {
    "AnimationFrameGenerator": "metacompskin.animation_generator",
    "BlendshapeModelData": "metacompskin.model_data",
    "MayaBlendshapeExporter": "metacompskin.maya_exporter",
    "MayaBlendshapeModelData": "metacompskin.model_data",
    "SkinCompressor": "metacompskin.model_fit",
    "build_skinned_rig": "metacompskin.maya_rig_builder",
    "compress_and_build_rig": "metacompskin.maya_pipeline",
}

__all__ = [
    "AnimationFrameGenerator",
    "BlendshapeModelData",
    "MayaBlendshapeExporter",
    "MayaBlendshapeModelData",
    "SkinCompressor",
    "__version__",
    "build_skinned_rig",
    "compress_and_build_rig",
]


def __getattr__(name: str) -> Any:
    """Lazily resolve public re-exports on first access (PEP 562)."""
    if name in _EXPORTS:
        return getattr(import_module(_EXPORTS[name]), name)
    raise AttributeError(f"module 'metacompskin' has no attribute {name!r}")


def __dir__() -> list[str]:
    """Include lazy re-exports in dir(metacompskin)."""
    return sorted([*globals(), *_EXPORTS])
