"""Unit tests for the torch-free math behind the Maya rig builder.

These tests do not need Maya. They cover loading the compressed archive,
slicing it into per-(shape, joint) blocks, recovering the centring offset
from a mesh placed anywhere in a scene, and the per-frame joint world
matrices that a Maya skin cluster needs to reproduce the compressed skinning.
"""

from pathlib import Path

import numpy as np
import pytest

from metacompskin.maya_rig_builder import (
    CompressedSkin,
    frame_joint_matrices,
    joint_bind_matrices,
    rest_to_scene_matrix,
    validate_shape_names,
    weighted_joint_centroids,
)

_N_VERTS = 6
_N_SHAPES = 2
_N_BONES = 3


def _translation(offset) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, 3] = offset
    return matrix


def _make_compressed_skin(seed: int = 0) -> CompressedSkin:
    rng = np.random.default_rng(seed)
    rest = rng.normal(size=(_N_VERTS, 3)).astype(np.float32)
    rest -= rest.mean(axis=0)
    weights = rng.random((_N_VERTS, _N_BONES))
    weights[:, 2] = 0.0  # keep at most two influences per vertex
    weights /= weights.sum(axis=1, keepdims=True)
    return CompressedSkin(
        rest=rest,
        quads=np.array([[0, 1, 2, 3], [2, 3, 4, 5]], dtype=np.int32),
        weights=weights.astype(np.float32),
        rest_xform=np.array([np.eye(3, 4)] * _N_BONES, dtype=np.float32),
        shape_xform=(0.1 * rng.normal(size=(3 * _N_SHAPES, 4 * _N_BONES))).astype(
            np.float32
        ),
    )


def _write_npz(path: Path, skin: CompressedSkin) -> Path:
    np.savez(
        path,
        rest=skin.rest,
        quads=skin.quads,
        weights=skin.weights,
        restXform=skin.rest_xform,
        shapeXform=skin.shape_xform,
    )
    return path


class TestCompressedSkin:
    def test_from_npz_exposes_dimensions_of_the_archive(self, tmp_path):
        path = _write_npz(tmp_path / "skin.npz", _make_compressed_skin())

        skin = CompressedSkin.from_npz(path)

        assert skin.n_vertices == _N_VERTS
        assert skin.n_shapes == _N_SHAPES
        assert skin.n_bones == _N_BONES
        assert skin.max_influences == 2

    def test_from_npz_rejects_archive_missing_a_required_key(self, tmp_path):
        path = tmp_path / "broken.npz"
        np.savez(path, rest=np.zeros((2, 3)))

        with pytest.raises(ValueError, match="shapeXform"):
            CompressedSkin.from_npz(path)

    def test_delta_transforms_slices_shape_xform_into_blocks(self):
        skin = _make_compressed_skin()
        block = skin.shape_xform[3:6, 8:12]  # shape 1, joint 2

        deltas = skin.delta_transforms()

        assert deltas.shape == (_N_SHAPES, _N_BONES, 3, 4)
        np.testing.assert_array_equal(deltas[1, 2], block)

    def test_bind_matrices_promote_rest_xform_to_homogeneous(self):
        skin = _make_compressed_skin()
        skin.rest_xform[1, :, 3] = [1.0, 2.0, 3.0]

        bind = skin.bind_matrices()

        assert bind.shape == (_N_BONES, 4, 4)
        np.testing.assert_array_equal(bind[1], _translation([1.0, 2.0, 3.0]))
        np.testing.assert_array_equal(bind[:, 3], [[0, 0, 0, 1]] * _N_BONES)


class TestRestToSceneMatrix:
    def test_recovers_translation_of_a_mesh_moved_off_the_origin(self):
        skin = _make_compressed_skin()
        offset = np.array([10.0, -2.0, 4.0])
        mesh_points = skin.rest.astype(np.float64) + offset

        matrix = rest_to_scene_matrix(mesh_points, skin.rest, np.eye(4))

        np.testing.assert_allclose(matrix, _translation(offset), atol=1e-6)

    def test_includes_the_mesh_world_matrix(self):
        skin = _make_compressed_skin()
        offset = np.array([1.0, 1.0, 1.0])
        world = np.diag([2.0, 2.0, 2.0, 1.0])
        mesh_points = skin.rest.astype(np.float64) + offset

        matrix = rest_to_scene_matrix(mesh_points, skin.rest, world)

        np.testing.assert_allclose(matrix, world @ _translation(offset), atol=1e-6)

    def test_rejects_a_mesh_with_a_different_vertex_count(self):
        skin = _make_compressed_skin()

        with pytest.raises(ValueError, match="vertex count"):
            rest_to_scene_matrix(skin.rest[:-1], skin.rest, np.eye(4))

    def test_rejects_a_mesh_whose_vertex_order_differs(self):
        skin = _make_compressed_skin()
        reordered = skin.rest[::-1].astype(np.float64)

        with pytest.raises(ValueError, match="vertex order"):
            rest_to_scene_matrix(reordered, skin.rest, np.eye(4))


class TestFrameJointMatrices:
    def test_first_frame_is_the_bind_pose(self):
        skin = _make_compressed_skin()
        bind = skin.bind_matrices()
        bind[0] = _translation([5.0, 0.0, 0.0])

        frames = frame_joint_matrices(
            skin.delta_transforms(), bind, _translation([1.0, 2.0, 3.0])
        )

        assert frames.shape == (_N_SHAPES + 1, _N_BONES, 4, 4)
        np.testing.assert_allclose(frames[0], bind)

    def test_frames_reproduce_reference_skinning_in_scene_space(self):
        skin = _make_compressed_skin()
        bind = skin.bind_matrices()
        bind[1] = _translation([0.0, 3.0, 0.0])
        rest_to_scene = _translation([10.0, -2.0, 4.0])
        rest_to_scene[:3, :3] = np.diag([1.0, 2.0, 1.0])
        rest_homog = np.c_[skin.rest, np.ones(_N_VERTS)]  # (N, 4)
        deltas = skin.delta_transforms()

        frames = frame_joint_matrices(deltas, bind, rest_to_scene)

        for shape in range(_N_SHAPES):
            # Reference: Equation 7 skinning on the centred rest, then to scene.
            per_bone = np.eye(3, 4) + deltas[shape]  # (P, 3, 4)
            centred = np.einsum("ip,pcd,id->ic", skin.weights, per_bone, rest_homog)
            expected = (rest_to_scene @ np.c_[centred, np.ones(_N_VERTS)].T).T[:, :3]
            # Maya: skinned = sum_j w_ij * (bindPreMatrix_j * world_j) * scene point.
            scene_points = (rest_to_scene @ rest_homog.T).T  # (N, 4)
            skin_matrices = frames[shape + 1] @ np.linalg.inv(bind)  # (P, 4, 4)
            actual = np.einsum(
                "ip,pcd,id->ic", skin.weights, skin_matrices, scene_points
            )[:, :3]
            np.testing.assert_allclose(actual, expected, atol=1e-5)


class TestValidateShapeNames:
    def test_generates_placeholder_names_when_none_are_given(self):
        names = validate_shape_names(None, 3)

        assert names == ["shape_000", "shape_001", "shape_002"]

    def test_returns_the_given_names_as_strings(self):
        names = validate_shape_names(np.array(["jawOpen", "smile"]), 2)

        assert names == ["jawOpen", "smile"]

    def test_rejects_a_name_count_that_differs_from_the_archive(self):
        with pytest.raises(ValueError, match="2 shape names"):
            validate_shape_names(["jawOpen", "smile"], 3)


class TestJointPlacement:
    def test_archive_with_only_identity_matrices_has_no_joint_placement(self):
        skin = _make_compressed_skin()

        assert skin.has_joint_placement is False

    def test_archive_with_a_non_identity_matrix_has_joint_placement(self):
        skin = _make_compressed_skin()
        skin.rest_xform[1, :, 3] = [1.0, 2.0, 3.0]

        assert skin.has_joint_placement is True

    def test_weighted_centroids_average_the_vertices_each_joint_influences(self):
        points = np.array([[0, 0, 0], [2, 0, 0], [0, 4, 0], [0, 0, 6]], dtype=float)
        weights = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 0.5], [0.0, 0.5]])

        centroids = weighted_joint_centroids(weights, points)

        np.testing.assert_allclose(centroids, [[1, 0, 0], [0, 2, 3]])

    def test_weighted_centroids_fall_back_to_the_mesh_centre_for_unused_joints(self):
        points = np.array([[0, 0, 0], [2, 0, 0], [0, 4, 0], [0, 0, 6]], dtype=float)
        weights = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])

        centroids = weighted_joint_centroids(weights, points)

        np.testing.assert_allclose(centroids[1], points.mean(axis=0))

    def test_bind_matrices_use_the_provided_joint_matrices_when_present(self):
        skin = _make_compressed_skin()
        skin.rest_xform[:, :, 3] = [[1.0, 2.0, 3.0]] * _N_BONES

        bind = joint_bind_matrices(skin, _translation([10.0, 0.0, 0.0]))

        np.testing.assert_array_equal(bind, skin.bind_matrices())

    def test_bind_matrices_sit_at_scene_space_centroids_without_placement(self):
        skin = _make_compressed_skin()
        rest_to_scene = _translation([10.0, -2.0, 4.0])
        rest_to_scene[:3, :3] = np.diag([1.0, 2.0, 1.0])
        scene_points = skin.rest @ rest_to_scene[:3, :3].T + rest_to_scene[:3, 3]

        bind = joint_bind_matrices(skin, rest_to_scene)

        expected = weighted_joint_centroids(skin.weights, scene_points)
        np.testing.assert_allclose(bind[:, :3, 3], expected, atol=1e-6)
        np.testing.assert_allclose(bind[:, :3, :3], [np.eye(3)] * _N_BONES)
