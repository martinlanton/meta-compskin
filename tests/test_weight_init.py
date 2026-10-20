"""Tests for the initial skinning weights: distance-based bind versus noise."""

import numpy as np
import pytest
from conftest import make_grid_model, write_model_npz

from metacompskin.cli import main
from metacompskin.maya_pipeline import CompressionSettings, compression_command
from metacompskin.model_data import BlendshapeModelData
from metacompskin.model_fit import (
    SkinCompressor,
    distance_weights,
    geodesic_joint_distances,
    static_vertex_mask,
)

_FAST = {"iterations": 200, "total_nnz_B_rt": 60, "max_influences": 2}


def _grid_joints(n_joints: int = 6) -> np.ndarray:
    xs, ys = np.meshgrid(np.arange(5.0), np.arange(5.0))
    points = np.stack([xs.ravel(), ys.ravel(), np.zeros(25)], axis=1)[::4][:n_joints]
    matrices = np.tile(np.eye(4), (n_joints, 1, 1))
    matrices[:, :3, 3] = points
    return matrices


def _with_static_region(model: BlendshapeModelData) -> BlendshapeModelData:
    deltas = model.deltas.copy()
    deltas[:, 22:, :] = 0  # the last three vertices never move
    return BlendshapeModelData(
        model_name=model.model_name,
        deltas=deltas,
        rest_verts=model.rest_verts,
        rest_faces=model.rest_faces,
        inbetween_info=model.inbetween_info,
        combination_info=model.combination_info,
        alpha=0.0,
    )


class TestDistanceWeights:
    distances = np.array(
        [[0.0, 3.0, 1.0], [1.0, 0.0, 2.0], [2.0, 1.0, 0.0], [3.0, 2.0, 3.0]]
    )  # (P=4, N=3)

    def test_each_vertex_sums_to_one(self):
        weights = distance_weights(self.distances, max_influences=2)

        np.testing.assert_allclose(weights.sum(axis=0), 1.0)

    def test_only_the_nearest_joints_get_weight(self):
        weights = distance_weights(self.distances, max_influences=2)

        assert ((weights > 0).sum(axis=0) == 2).all()
        assert weights[3].sum() == 0.0  # never among the two nearest

    def test_the_nearest_joint_gets_the_largest_weight(self):
        weights = distance_weights(self.distances, max_influences=2)

        np.testing.assert_array_equal(
            weights.argmax(axis=0), self.distances.argmin(axis=0)
        )

    def test_weight_falls_off_with_distance(self):
        weights = distance_weights(self.distances, max_influences=3)

        assert weights[0, 0] > weights[1, 0] > weights[2, 0] > 0

    def test_a_budget_covering_every_joint_still_normalises(self):
        weights = distance_weights(self.distances, max_influences=4)

        np.testing.assert_allclose(weights.sum(axis=0), 1.0)
        assert (
            weights[:, 0] > 0
        ).sum() == 3  # the farthest joint is the falloff's zero


class TestStaticVertexMask:
    def test_vertices_that_never_move_are_static(self):
        deltas = np.ones((3, 5, 3))
        deltas[:, 1] = 0.0
        deltas[:, 3] = 1e-9

        static = static_vertex_mask(deltas)

        np.testing.assert_array_equal(static, [False, True, False, True, False])

    def test_nothing_is_static_when_everything_moves(self):
        assert not static_vertex_mask(np.ones((2, 4, 3))).any()

    def test_the_threshold_is_relative_to_the_largest_motion(self):
        deltas = np.zeros((1, 3, 3))
        deltas[0, 0, 0] = 100.0
        deltas[0, 1, 0] = 0.05  # below a tenth of a percent of the largest motion
        deltas[0, 2, 0] = 0.5

        np.testing.assert_array_equal(static_vertex_mask(deltas), [False, True, False])


class TestWeightInitSetting:
    def test_defaults_to_distance_with_joint_matrices(self, grid_model_data):
        compressor = SkinCompressor(
            model_data=grid_model_data, rest_joint_matrices=_grid_joints()
        )

        assert compressor.weight_init == "distance"

    def test_defaults_to_noise_without_joint_matrices(self, grid_model_data):
        compressor = SkinCompressor(model_data=grid_model_data, number_of_bones=10)

        assert compressor.weight_init == "noise"

    def test_noise_can_be_requested_with_joint_matrices(self, grid_model_data):
        compressor = SkinCompressor(
            model_data=grid_model_data,
            rest_joint_matrices=_grid_joints(),
            weight_init="noise",
        )

        assert compressor.weight_init == "noise"

    def test_distance_needs_joint_matrices(self, grid_model_data):
        with pytest.raises(ValueError, match="rest_joint_matrices"):
            SkinCompressor(
                model_data=grid_model_data, number_of_bones=10, weight_init="distance"
            )

    def test_unknown_values_are_rejected(self, grid_model_data):
        with pytest.raises(ValueError, match="weight_init"):
            SkinCompressor(
                model_data=grid_model_data,
                rest_joint_matrices=_grid_joints(),
                weight_init="heat",
            )


class TestDistanceStartRuns:
    def test_static_vertices_land_fully_on_the_root(self, grid_model_data, tmp_path):
        output = tmp_path / "static.npz"
        SkinCompressor(
            model_data=_with_static_region(grid_model_data),
            iterations=(600,),
            rest_joint_matrices=_grid_joints(),
            max_influences=2,
            candidate_joints_per_vertex=3,
            total_nnz_B_rt=100,
        ).run(output)

        weights = np.load(output)["weights"]  # (N, P')
        assert (weights[22:, -1] == 1.0).all()

    def test_the_result_is_valid_and_reproducible(self, grid_model_data, tmp_path):
        def compress(name):
            output = tmp_path / f"{name}.npz"
            SkinCompressor(
                model_data=grid_model_data, rest_joint_matrices=_grid_joints(), **_FAST
            ).run(output)
            return np.load(output)["weights"]

        first, second = compress("a"), compress("b")

        np.testing.assert_array_equal(first, second)
        np.testing.assert_allclose(first.sum(axis=1), 1.0, atol=1e-5)
        assert ((first != 0).sum(axis=1) <= 2).all()

    def test_noise_and_distance_starts_give_different_results(
        self, grid_model_data, tmp_path
    ):
        results = {}
        for init in ("noise", "distance"):
            output = tmp_path / f"{init}.npz"
            SkinCompressor(
                model_data=grid_model_data,
                rest_joint_matrices=_grid_joints(),
                weight_init=init,
                **_FAST,
            ).run(output)
            results[init] = np.load(output)["weights"]

        assert not np.array_equal(results["noise"], results["distance"])

    def test_the_start_follows_the_nearest_joints(self, grid_model_data, tmp_path):
        """With a single short step the output still reflects the start."""
        output = tmp_path / "one_step.npz"
        compressor = SkinCompressor(
            model_data=grid_model_data,
            iterations=1,
            rest_joint_matrices=_grid_joints(),
            max_influences=2,
            total_nnz_B_rt=60,
        )
        compressor.run(output)
        weights = np.load(output)["weights"][:, :-1]  # drop the root column

        distances = geodesic_joint_distances(
            grid_model_data.rest_verts,
            grid_model_data.rest_faces,
            _grid_joints()[:, :3, 3],
        )
        two_nearest = np.sort(distances, axis=0)[:2]
        unambiguous = two_nearest[1] > two_nearest[0] + 0.5  # skip grid ties
        np.testing.assert_array_equal(
            weights.argmax(axis=1)[unambiguous], distances.argmin(axis=0)[unambiguous]
        )
        assert unambiguous.sum() > 10

    def test_a_designated_root_keeps_its_floor(self, grid_model_data, tmp_path):
        output = tmp_path / "designated.npz"
        SkinCompressor(
            model_data=grid_model_data,
            rest_joint_matrices=_grid_joints(),
            root_joint=0,
            **_FAST,
        ).run(output)

        weights = np.load(output)["weights"]
        assert weights.shape[1] == 6
        np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-5)


class TestCandidateDefault:
    def test_default_mask_is_two_wider_than_the_influences(self, grid_model_data):
        compressor = SkinCompressor(
            model_data=grid_model_data,
            rest_joint_matrices=np.tile(np.eye(4), (40, 1, 1)),
        )

        assert compressor.candidate_joints_per_vertex == 10

    def test_single_stage_warm_up_uses_the_default_mask(self, grid_model_data):
        compressor = SkinCompressor(
            model_data=grid_model_data,
            iterations=300,
            rest_joint_matrices=np.tile(np.eye(4), (40, 1, 1)),
            total_nnz_B_rt=100,
        )

        assert [p.max_influences for p in compressor.schedule] == [10, 8]


class TestCliWeightInit:
    def _model(self, tmp_path):
        return write_model_npz(
            tmp_path / "grid.npz", make_grid_model(), rest_joint_matrices=_grid_joints()
        )

    def test_noise_can_be_requested(self, tmp_path):
        output = tmp_path / "noise.npz"

        main(
            [
                str(self._model(tmp_path)),
                str(output),
                "--iterations",
                "200",
                "--total-nnz-b-rt",
                "60",
                "--max-influences",
                "2",
                "--weight-init",
                "noise",
            ]
        )

        assert np.load(output)["weights"].shape == (25, 7)

    def test_unknown_values_are_rejected(self, tmp_path):
        with pytest.raises(SystemExit):
            main(
                [
                    str(self._model(tmp_path)),
                    str(tmp_path / "o.npz"),
                    "--weight-init",
                    "heat",
                ]
            )


class TestPipelineWeightInit:
    def test_the_default_leaves_the_choice_to_the_solver(self, tmp_path):
        command = compression_command(
            "python", tmp_path / "m.npz", tmp_path / "c.npz", CompressionSettings()
        )

        assert "--weight-init" not in command

    def test_noise_is_forwarded(self, tmp_path):
        command = compression_command(
            "python",
            tmp_path / "m.npz",
            tmp_path / "c.npz",
            CompressionSettings(weight_init="noise"),
        )

        assert command[command.index("--weight-init") + 1] == "noise"
