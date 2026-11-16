r"""Experimental: the Laplacian weight alpha estimated from the head's geometry.

The smoothness weight ``alpha`` is otherwise looked up by model name in
:mod:`metacompskin.constants`, so a new head gets the default. This module
estimates it from the head instead. It is experimental: the rule is calibrated
on few heads, and its constant may change.

The rule, for the uniform Laplacian the solver uses:

.. math::

    \alpha = c \left( \frac{R}{h} \right)^2

where :math:`h` is the median edge length of the rest mesh and :math:`R` the
RMS distance of the moving vertices from their centroid (the size of the part
of the face the shapes move). :math:`R / h` is the moving region's size counted
in edges, so the estimate does not depend on the scene unit, and it grows as the
mesh gets denser over the same face, as the uniform Laplacian needs. A vertex
moves when its largest displacement over the shapes is above 1% of the head's
largest displacement.

:math:`c` was calibrated in the paper's setting (100 anonymous bones, 8
influences, the Laplacian on the reconstruction) by shape-held-out
cross-validation: each solve saw two thirds of the shapes, and the remaining
shapes were fitted with its weights fixed. With :math:`c = 0.02` the estimate
is 10.5 for jupiter, 10.8 for aura, 43.9 for proteus and 44.8 for bowen (the
paper uses 10, 10, 50 and 50), and on each of them it lands where the held-out
error is within 2% of the best alpha's.

The estimate is not used by default. To compress with it, pass it as the
model's alpha:

.. code-block:: python

    model = BlendshapeModelData.from_npz("head.npz")
    alpha = SmoothingAlphaEstimator().estimate(model)
    model = BlendshapeModelData.from_npz("head.npz", alpha=alpha)

Warning:
    Use it on heads in centimetres only for now. The rule is validated there;
    in other units it is not yet. The estimate itself does not depend on the
    unit, but the solver's results still do: the same head in metres compresses
    worse at every alpha, and the best alpha moves with the unit.
"""

import warnings
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from metacompskin.model_data import BlendshapeModelData

CALIBRATED_SCALE = 0.02  # gives 10.5, 10.8, 43.9, 44.8 on jupiter, aura, proteus, bowen
MOVING_SHARE = 0.01  # a vertex moves above this share of the head's largest motion


class ExperimentalWarning(UserWarning):
    """Warns that a feature is experimental and may change without notice."""


@dataclass(frozen=True)
class SmoothingAlphaEstimator:
    r"""Estimates the Laplacian weight alpha from the head's geometry (experimental).

    Warning:
        Experimental, and for heads in centimetres only: in other units it does
        not work yet. The rule is calibrated on few heads, in the paper's setting
        (anonymous bones, smoothing on the reconstruction); its constant may
        change, and it is not used by default. See :mod:`metacompskin.alpha_estimation`.

    Attributes:
        scale: The constant :math:`c` in :math:`\alpha = c (R / h)^2`.
        moving_share: A vertex counts as moving when its largest displacement is
            above this share of the head's largest displacement.

    Example:
        >>> model = BlendshapeModelData.from_npz("jupiter.npz")
        >>> alpha = SmoothingAlphaEstimator().estimate(model)
    """

    scale: float = CALIBRATED_SCALE
    moving_share: float = MOVING_SHARE

    def __post_init__(self) -> None:
        """Warn that the estimator is experimental."""
        warnings.warn(
            "SmoothingAlphaEstimator is experimental and validated for heads in "
            "centimetres only; its rule may change.",
            ExperimentalWarning,
            stacklevel=3,
        )

    def estimate(self, model_data: BlendshapeModelData) -> float:
        """Estimate alpha for a head.

        Args:
            model_data: The head: rest mesh and blendshape deltas, in centimetres.

        Returns:
            The estimated alpha, :math:`c (R / h)^2`.

        Raises:
            ValueError: If no vertex moves in any shape.
        """
        rest_verts = np.asarray(model_data.rest_verts, dtype=np.float64)
        moving = moving_vertices(np.asarray(model_data.deltas), self.moving_share)
        edges_across = moving_region_radius(rest_verts[moving]) / median_edge_length(
            rest_verts, np.asarray(model_data.rest_faces)
        )
        return float(self.scale * edges_across**2)


def moving_vertices(
    deltas: npt.NDArray[np.floating], share: float
) -> npt.NDArray[np.bool_]:
    """Mark the vertices that move, relative to the head's largest motion.

    Args:
        deltas: Blendshape deltas, shape (S, N, 3).
        share: A vertex moves when its largest displacement is above this share
            of the largest displacement of any vertex.

    Returns:
        Boolean mask, shape (N,).

    Raises:
        ValueError: If no vertex moves in any shape.
    """
    motion = np.linalg.norm(np.asarray(deltas, dtype=np.float64), axis=2).max(axis=0)
    largest = motion.max()
    if largest <= 0:
        raise ValueError("Cannot estimate alpha: no vertex moves in any shape.")
    return motion > share * largest


def moving_region_radius(moving_verts: npt.NDArray[np.floating]) -> float:
    """RMS distance of the moving vertices from their centroid.

    Args:
        moving_verts: Rest positions of the moving vertices, shape (M, 3).

    Returns:
        The radius, in the model's unit.
    """
    offsets = moving_verts - moving_verts.mean(axis=0)
    return float(np.sqrt((offsets**2).sum(axis=1).mean()))


def median_edge_length(
    rest_verts: npt.NDArray[np.floating], rest_faces: npt.NDArray[np.integer]
) -> float:
    """Median length of the mesh's polygon edges, each edge counted once.

    Args:
        rest_verts: Rest vertex positions, shape (N, 3).
        rest_faces: Face indices, shape (F, verts_per_face).

    Returns:
        The median edge length, in the model's unit.
    """
    following = np.roll(rest_faces, -1, axis=1)
    edges = np.unique(
        np.sort(np.stack([rest_faces, following], axis=-1).reshape(-1, 2)), axis=0
    )
    lengths = np.linalg.norm(rest_verts[edges[:, 0]] - rest_verts[edges[:, 1]], axis=1)
    return float(np.median(lengths))
