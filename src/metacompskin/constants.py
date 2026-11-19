"""Constants and configuration values for metacompskin.

This module contains model-specific constants and parameters used throughout
the compressed skinning pipeline.
"""

# Alpha values for Laplacian regularization.
# Alpha weighs the Laplacian term against the data term: higher values penalise
# error (or, without residual smoothing, motion) that varies from vertex to vertex
# more strongly; lower values let each vertex fit more freely.
#
# The Laplacian is the uniform one (each vertex minus the mean of its neighbours)
# and is not scaled by edge length. On motion that varies smoothly over the face its
# output shrinks roughly with the square of the edge length, so a denser mesh needs a
# *larger* alpha for the same smoothing; this is why the two dense sample heads use 50
# and the two coarse ones 10. On error that changes from one vertex to the next it
# does not shrink, so one alpha cannot mean the same thing on every mesh.
#
# TODO: Derive alpha from the head (edge length, curvature of the shapes, joint
# spacing) instead of this table; see "Choosing alpha from the head" in
# plan/research_output.md.
MODEL_ALPHA_VALUES: dict[str, float] = {
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
