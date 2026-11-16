"""SmoothingAlphaEstimator: alpha from the head's geometry (experimental)."""

import warnings
from pathlib import Path

import numpy as np
import pytest

from metacompskin.alpha_estimation import ExperimentalWarning, SmoothingAlphaEstimator
from metacompskin.model_data import BlendshapeModelData

SOURCE_MODELS = Path(__file__).parent / "test_data" / "source_models"


def grid_head(points_per_side, spacing, offset=0.0):
    """A flat quad grid whose every vertex moves up by one unit in one shape."""
    xs, ys = np.meshgrid(np.arange(points_per_side), np.arange(points_per_side))
    rest = np.stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)], axis=1) * spacing
    rest[:, 0] += offset
    index = np.arange(points_per_side**2).reshape(points_per_side, points_per_side)
    faces = np.stack(
        [index[:-1, :-1], index[:-1, 1:], index[1:, 1:], index[1:, :-1]], axis=-1
    ).reshape(-1, 4)
    deltas = np.zeros((1, len(rest), 3))
    deltas[0, :, 2] = 1.0
    return rest, faces, deltas


def model(rest, faces, deltas):
    return BlendshapeModelData(
        deltas=deltas,
        rest_verts=rest,
        rest_faces=faces,
        inbetween_info={},
        combination_info={},
        model_name="grid",
        alpha=10.0,
    )


def estimate(head):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExperimentalWarning)
        return SmoothingAlphaEstimator().estimate(head)


def test_estimate_does_not_depend_on_the_scene_unit():
    # Arrange
    rest, faces, deltas = grid_head(21, 1.0)
    # Act
    in_centimetres = estimate(model(rest, faces, deltas))
    in_metres = estimate(model(rest * 0.01, faces, deltas * 0.01))
    # Assert
    assert in_metres == pytest.approx(in_centimetres, rel=1e-6)


def test_estimate_quadruples_when_the_edges_halve_over_the_same_area():
    # Arrange
    coarse = model(*grid_head(41, 1.0))
    dense = model(*grid_head(81, 0.5))
    # Act
    ratio = estimate(dense) / estimate(coarse)
    # Assert
    assert ratio == pytest.approx(4.0, rel=0.05)


def test_vertices_that_do_not_move_leave_the_estimate_unchanged():
    # Arrange
    rest, faces, deltas = grid_head(21, 1.0)
    still_rest, still_faces, _ = grid_head(21, 1.0, offset=1000.0)
    with_still_part = model(
        np.concatenate([rest, still_rest]),
        np.concatenate([faces, still_faces + len(rest)]),
        np.concatenate([deltas, np.zeros_like(deltas)], axis=1),
    )
    # Act
    alone = estimate(model(rest, faces, deltas))
    together = estimate(with_still_part)
    # Assert
    assert together == pytest.approx(alone, rel=1e-6)


def test_raises_when_no_vertex_moves():
    # Arrange
    rest, faces, deltas = grid_head(5, 1.0)
    still = model(rest, faces, np.zeros_like(deltas))
    # Act / Assert
    with pytest.raises(ValueError, match="no vertex moves"):
        estimate(still)


def test_warns_that_it_is_experimental():
    # Act / Assert
    with pytest.warns(ExperimentalWarning, match="experimental"):
        SmoothingAlphaEstimator()


@pytest.mark.parametrize(
    ("head", "low", "high"),
    [  # alphas whose shape-held-out error is within 2% of the best alpha's
        ("jupiter", 3.0, 100.0),
        ("aura", 1.0, 55.0),
        ("proteus", 19.0, 300.0),
        ("bowen", 13.0, 300.0),
    ],
)
def test_sample_heads_land_within_two_percent_of_the_best_held_out_error(
    head, low, high
):
    # Arrange
    sample = BlendshapeModelData.from_npz(SOURCE_MODELS / f"{head}.npz")
    # Act
    alpha = estimate(sample)
    # Assert
    assert low <= alpha <= high


def test_is_exported_from_the_package():
    # Arrange
    import metacompskin  # noqa: PLC0415

    # Act / Assert
    assert metacompskin.SmoothingAlphaEstimator is SmoothingAlphaEstimator
