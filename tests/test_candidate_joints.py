"""Tests for the geodesic candidate-joint filter used with user-supplied joints."""

import numpy as np
import pytest
from conftest import GRID, make_grid_model

from metacompskin.model_fit import candidate_joint_mask, geodesic_joint_distances


def _vertex(col: int, row: int) -> int:
    return row * GRID + col


def _grid_faces_without_columns(skip_cols: set[int], rows: range) -> np.ndarray:
    """Grid faces minus the quads in ``skip_cols`` for the given rows (a slit)."""
    faces = []
    for row in range(GRID - 1):
        for col in range(GRID - 1):
            if col in skip_cols and row in rows:
                continue
            i = row * GRID + col
            faces.append([i, i + 1, i + GRID + 1, i + GRID])
    return np.array(faces, dtype=np.int32)


class TestGeodesicJointDistances:
    def test_returns_one_row_per_joint(self):
        model = make_grid_model()
        joints = np.array([[0.0, 0.0, 0.0], [4.0, 4.0, 0.0]])

        distances = geodesic_joint_distances(model.rest_verts, model.rest_faces, joints)

        assert distances.shape == (2, model.n_vertices)

    def test_follows_the_edges_from_a_joint_on_a_vertex(self):
        model = make_grid_model()
        joints = np.array([[0.0, 0.0, 0.0]])

        distances = geodesic_joint_distances(model.rest_verts, model.rest_faces, joints)

        assert distances[0, _vertex(0, 0)] == pytest.approx(0.0)
        assert distances[0, _vertex(4, 0)] == pytest.approx(4.0)
        # Quad diagonals are edges of the graph, so the corner is four diagonals.
        assert distances[0, _vertex(4, 4)] == pytest.approx(4 * np.sqrt(2))

    def test_adds_the_gap_between_a_joint_and_its_nearest_vertex(self):
        model = make_grid_model()
        joints = np.array([[0.0, 0.0, 1.0]])  # one unit above vertex (0, 0)

        distances = geodesic_joint_distances(model.rest_verts, model.rest_faces, joints)

        assert distances[0, _vertex(0, 0)] == pytest.approx(1.0)
        assert distances[0, _vertex(4, 0)] == pytest.approx(5.0)

    def test_walks_around_a_slit_instead_of_across_it(self):
        model = make_grid_model()
        faces = _grid_faces_without_columns({2}, range(0, 3))  # slit x=2..3, y=0..3
        across = np.array([4.0, 0.0, 0.0])  # Euclidean distance 2 from (2, 0)
        same_side = np.array([2.0, 3.0, 0.0])  # Euclidean distance 3 from (2, 0)

        distances = geodesic_joint_distances(
            model.rest_verts, faces, np.stack([across, same_side])
        )

        assert distances[1, _vertex(2, 0)] == pytest.approx(3.0)
        assert distances[0, _vertex(2, 0)] > distances[1, _vertex(2, 0)]

    def test_falls_back_to_spatial_order_for_a_disconnected_shell(self):
        model = make_grid_model()
        faces = _grid_faces_without_columns({2}, range(GRID - 1))  # two shells
        left_joint = np.array([0.0, 0.0, 0.0])
        right_joint = np.array([4.0, 4.0, 0.0])

        distances = geodesic_joint_distances(
            model.rest_verts, faces, np.stack([left_joint, right_joint])
        )

        assert np.isfinite(distances).all()
        # A joint on the vertex's own shell always ranks before one that is not.
        assert distances[1, _vertex(3, 0)] < distances[0, _vertex(3, 0)]


class TestCandidateJointMask:
    def test_keeps_the_nearest_joints_per_vertex(self):
        distances = np.array(
            [
                [0.0, 5.0, 2.0],
                [1.0, 0.0, 9.0],
                [2.0, 1.0, 0.0],
                [3.0, 2.0, 1.0],
            ]
        )

        mask = candidate_joint_mask(distances, 2)

        assert mask.dtype == bool
        assert mask.tolist() == [
            [True, False, False],
            [True, True, False],
            [False, True, True],
            [False, False, True],
        ]

    def test_keeps_every_joint_when_the_count_covers_them_all(self):
        distances = np.random.default_rng(0).random((3, 5))

        mask = candidate_joint_mask(distances, 3)

        assert mask.all()

    def test_rejects_a_count_below_one(self):
        with pytest.raises(ValueError, match="candidates_per_vertex"):
            candidate_joint_mask(np.zeros((3, 5)), 0)
