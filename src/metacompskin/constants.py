"""Constants and configuration values for metacompskin.

This module contains model-specific constants and parameters used throughout
the compressed skinning pipeline.
"""

from typing import Dict

# Alpha values for Laplacian regularization
# Lower values lead to sharper results, higher values lead to smoother results.
# Alpha values should be lower for high density meshes and higher for low density meshes.
#
# TODO: Can we procedurally determine the alpha value based on mesh density and sharpness?
# We might be able to use the paper "Line Direction Matters: An Argument For The Use
# Of Principal Directions In 3D Line Drawings"
MODEL_ALPHA_VALUES: Dict[str, float] = {
    "aura": 10.0,
    "jupiter": 10.0,
    "proteus": 50.0,
    "bowen": 50.0,
}

# Default alpha value for unknown models
DEFAULT_ALPHA_VALUE: float = 10.0


def get_alpha_for_model(model_name: str) -> float:
    """Get the alpha value for a specific model.

    Args:
        model_name: Name of the model (e.g., "aura", "jupiter").

    Returns:
        Alpha value for Laplacian regularization.

    Example:
        >>> get_alpha_for_model("aura")
        10.0
        >>> get_alpha_for_model("proteus")
        50.0
        >>> get_alpha_for_model("unknown_model")
        10.0
    """
    return MODEL_ALPHA_VALUES.get(model_name, DEFAULT_ALPHA_VALUE)
