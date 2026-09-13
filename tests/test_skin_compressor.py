"""Tests for SkinCompressor construction: the optimisation settings it accepts."""

import numpy as np
import pytest

from metacompskin.model_fit import (
    SkinCompressor,
    TrainingPhase,
    candidate_joint_mask,
    geodesic_joint_distances,
)


def _identity_joint_matrices(n_bones: int) -> np.ndarray:
    return np.tile(np.eye(4), (n_bones, 1, 1))


def _joint_matrices_at(positions: np.ndarray) -> np.ndarray:
    matrices = np.tile(np.eye(4), (len(positions), 1, 1))
    matrices[:, :3, 3] = positions
    return matrices


def _grid_joint_positions(n_joints: int) -> np.ndarray:
    """Joints on every other vertex of the 5x5 grid, in its (uncentred) space."""
    xs, ys = np.meshgrid(np.arange(5.0), np.arange(5.0))
    points = np.stack([xs.ravel(), ys.ravel(), np.zeros(25)], axis=1)
    return points[::2][:n_joints]


class TestSkinCompressorSettings:
    def test_defaults(self, grid_model_data):
        compressor = SkinCompressor(model_data=grid_model_data)

        assert compressor.number_of_bones == 100
        assert compressor.max_influences == 8
        assert compressor.total_nnz_B_rt == 6000
        assert compressor.init_weight == 1e-3
        assert compressor.power == 2
        assert compressor.seed == 12345
        assert compressor.reconstruction_error is None

    def test_seed_is_taken_from_the_constructor(self, grid_model_data):
        compressor = SkinCompressor(model_data=grid_model_data, seed=7)

        assert compressor.seed == 7

    def test_settings_are_taken_from_the_constructor(self, grid_model_data):
        compressor = SkinCompressor(
            model_data=grid_model_data,
            number_of_bones=12,
            max_influences=4,
            total_nnz_B_rt=500,
            init_weight=1e-2,
            power=12,
        )

        assert compressor.number_of_bones == 12
        assert compressor.max_influences == 4
        assert compressor.total_nnz_B_rt == 500
        assert compressor.init_weight == 1e-2
        assert compressor.power == 12

    def test_number_of_bones_follows_the_rest_joint_matrices(self, grid_model_data):
        compressor = SkinCompressor(
            model_data=grid_model_data,
            rest_joint_matrices=_identity_joint_matrices(7),
        )

        assert compressor.number_of_bones == 7

    def test_number_of_bones_matching_the_rest_joint_matrices_is_accepted(
        self, grid_model_data
    ):
        compressor = SkinCompressor(
            model_data=grid_model_data,
            rest_joint_matrices=_identity_joint_matrices(7),
            number_of_bones=7,
        )

        assert compressor.number_of_bones == 7

    def test_number_of_bones_conflicting_with_rest_joint_matrices_is_rejected(
        self, grid_model_data
    ):
        with pytest.raises(ValueError, match="number_of_bones"):
            SkinCompressor(
                model_data=grid_model_data,
                rest_joint_matrices=_identity_joint_matrices(7),
                number_of_bones=5,
            )

    def test_run_writes_one_weight_column_per_requested_bone(
        self, grid_model_data, tmp_path
    ):
        compressor = SkinCompressor(
            model_data=grid_model_data,
            iterations=300,
            number_of_bones=10,
            total_nnz_B_rt=100,
        )
        output = tmp_path / "compressed.npz"

        compressor.run(output)

        weights = np.load(output)["weights"]
        assert weights.shape == (grid_model_data.n_vertices, 10)

    def test_run_keeps_at_most_max_influences_weights_per_vertex(
        self, grid_model_data, tmp_path
    ):
        compressor = SkinCompressor(
            model_data=grid_model_data,
            iterations=300,
            number_of_bones=10,
            max_influences=3,
            total_nnz_B_rt=100,
        )
        output = tmp_path / "compressed.npz"

        compressor.run(output)

        weights = np.load(output)["weights"]
        assert ((weights != 0).sum(axis=1) <= 3).all()

    def test_same_seed_reproduces_the_output(self, grid_model_data, tmp_path):
        settings = {
            "iterations": 300,
            "number_of_bones": 10,
            "total_nnz_B_rt": 100,
            "seed": 3,
        }

        SkinCompressor(model_data=grid_model_data, **settings).run(tmp_path / "a.npz")
        SkinCompressor(model_data=grid_model_data, **settings).run(tmp_path / "b.npz")

        first, second = np.load(tmp_path / "a.npz"), np.load(tmp_path / "b.npz")
        np.testing.assert_array_equal(first["weights"], second["weights"])
        np.testing.assert_array_equal(first["shapeXform"], second["shapeXform"])

    def test_different_seeds_change_the_output(self, grid_model_data, tmp_path):
        settings = {"iterations": 300, "number_of_bones": 10, "total_nnz_B_rt": 100}

        SkinCompressor(model_data=grid_model_data, seed=3, **settings).run(
            tmp_path / "a.npz"
        )
        SkinCompressor(model_data=grid_model_data, seed=4, **settings).run(
            tmp_path / "b.npz"
        )

        first, second = np.load(tmp_path / "a.npz"), np.load(tmp_path / "b.npz")
        assert not np.array_equal(first["weights"], second["weights"])

    def test_run_records_the_reconstruction_error(self, grid_model_data, tmp_path):
        compressor = SkinCompressor(
            model_data=grid_model_data,
            iterations=300,
            number_of_bones=10,
            total_nnz_B_rt=100,
        )

        compressor.run(tmp_path / "compressed.npz")

        error = compressor.reconstruction_error
        assert error is not None
        assert np.isfinite(error.max_abs) and np.isfinite(error.mean_abs)
        assert error.max_abs >= error.mean_abs > 0


class TestSkinCompressorSchedule:
    def _phases(self, compressor):
        return [(p.iterations, p.max_influences) for p in compressor.schedule]

    def test_default_schedule_is_two_phases_at_the_default_budgets(
        self, grid_model_data
    ):
        compressor = SkinCompressor(model_data=grid_model_data)

        assert compressor.stage_iterations == (10000,)
        assert [
            (p.max_influences, p.total_nnz_B_rt, p.normalize_weights)
            for p in compressor.schedule
        ] == [(8, 6000, False), (8, 6000, True)]
        assert all(p.iterations == 10000 for p in compressor.schedule)

    def test_a_sequence_of_iterations_halves_the_influences_per_stage(
        self, grid_model_data
    ):
        compressor = SkinCompressor(
            model_data=grid_model_data, iterations=(100, 100, 300), max_influences=2
        )

        assert compressor.stage_iterations == (100, 100, 300)
        assert self._phases(compressor) == [(100, 8), (100, 8), (100, 4), (300, 2)]
        assert compressor.max_influences == 2

    def test_an_int_and_a_one_entry_sequence_build_the_same_schedule(
        self, grid_model_data
    ):
        by_int = SkinCompressor(model_data=grid_model_data, iterations=300)
        by_tuple = SkinCompressor(model_data=grid_model_data, iterations=(300,))

        assert by_int.schedule == by_tuple.schedule

    @pytest.mark.parametrize("iterations", [(), 0, (100, 0)])
    def test_invalid_iterations_are_rejected(self, grid_model_data, iterations):
        with pytest.raises(ValueError, match="iterations"):
            SkinCompressor(model_data=grid_model_data, iterations=iterations)

    def test_run_with_staged_iterations_ends_at_max_influences(
        self, grid_model_data, tmp_path
    ):
        compressor = SkinCompressor(
            model_data=grid_model_data,
            iterations=(100, 100, 300),
            number_of_bones=10,
            max_influences=2,
            total_nnz_B_rt=100,
        )

        compressor.run(tmp_path / "compressed.npz")

        weights = np.load(tmp_path / "compressed.npz")["weights"]
        assert ((weights != 0).sum(axis=1) <= 2).all()
        assert (weights >= 0).all()
        np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-5)

    def test_run_executes_a_schedule_assigned_before_it(
        self, grid_model_data, tmp_path
    ):
        compressor = SkinCompressor(
            model_data=grid_model_data,
            iterations=300,
            number_of_bones=10,
            total_nnz_B_rt=100,
        )
        compressor.schedule = (
            TrainingPhase(300, 5, 100, normalize_weights=False),
            TrainingPhase(300, 3, 100, normalize_weights=True),
        )

        compressor.run(tmp_path / "compressed.npz")

        weights = np.load(tmp_path / "compressed.npz")["weights"]
        assert ((weights != 0).sum(axis=1) <= 3).all()

    def test_run_rejects_a_warm_up_not_below_the_bone_count(
        self, grid_model_data, tmp_path
    ):
        compressor = SkinCompressor(
            model_data=grid_model_data,
            iterations=(100,) * 4,
            number_of_bones=10,
            max_influences=2,
            total_nnz_B_rt=100,  # warm-up K = 16
        )

        with pytest.raises(ValueError, match="number_of_bones"):
            compressor.run(tmp_path / "compressed.npz")

    def test_run_rejects_a_coefficient_budget_above_the_model_size(
        self, grid_model_data, tmp_path
    ):
        compressor = SkinCompressor(
            model_data=grid_model_data, number_of_bones=10, total_nnz_B_rt=500
        )

        with pytest.raises(ValueError, match=r"6\*S\*P"):
            compressor.run(tmp_path / "compressed.npz")

    def test_run_rejects_an_empty_schedule(self, grid_model_data, tmp_path):
        compressor = SkinCompressor(
            model_data=grid_model_data, number_of_bones=10, total_nnz_B_rt=100
        )
        compressor.schedule = ()

        with pytest.raises(ValueError, match="at least one"):
            compressor.run(tmp_path / "compressed.npz")


class TestCandidateJoints:
    _FAST = {"iterations": 300, "total_nnz_B_rt": 100}

    def test_default_is_twice_the_influences_with_joint_matrices(self, grid_model_data):
        compressor = SkinCompressor(
            model_data=grid_model_data,
            rest_joint_matrices=_identity_joint_matrices(40),
        )

        assert compressor.candidate_joints_per_vertex == 16

    def test_default_stays_below_the_bone_count(self, grid_model_data):
        compressor = SkinCompressor(
            model_data=grid_model_data,
            rest_joint_matrices=_identity_joint_matrices(10),
        )

        assert compressor.candidate_joints_per_vertex == 9

    def test_default_is_off_when_no_valid_count_exists(self, grid_model_data):
        compressor = SkinCompressor(
            model_data=grid_model_data,
            rest_joint_matrices=_identity_joint_matrices(7),  # fewer than K + 1
        )

        assert compressor.candidate_joints_per_vertex is None

    def test_there_is_no_filter_without_joint_matrices(self, grid_model_data):
        compressor = SkinCompressor(model_data=grid_model_data)

        assert compressor.candidate_joints_per_vertex is None

    def test_a_filter_without_joint_matrices_is_rejected(self, grid_model_data):
        with pytest.raises(ValueError, match="rest_joint_matrices"):
            SkinCompressor(model_data=grid_model_data, candidate_joints_per_vertex=16)

    def test_zero_disables_the_filter(self, grid_model_data):
        compressor = SkinCompressor(
            model_data=grid_model_data,
            rest_joint_matrices=_identity_joint_matrices(40),
            candidate_joints_per_vertex=0,
        )

        assert compressor.candidate_joints_per_vertex is None

    @pytest.mark.parametrize("candidates", [7, 40])
    def test_must_lie_between_the_influences_and_the_bone_count(
        self, grid_model_data, candidates
    ):
        with pytest.raises(ValueError, match="candidate_joints_per_vertex"):
            SkinCompressor(
                model_data=grid_model_data,
                rest_joint_matrices=_identity_joint_matrices(40),
                candidate_joints_per_vertex=candidates,
            )

    def test_caps_the_annealing_ceiling(self, grid_model_data):
        compressor = SkinCompressor(
            model_data=grid_model_data,
            iterations=(100, 100, 300),
            rest_joint_matrices=_identity_joint_matrices(40),
            max_influences=2,
            candidate_joints_per_vertex=6,
        )

        assert [p.max_influences for p in compressor.schedule] == [6, 6, 3, 2]

    def test_run_only_weights_joints_near_each_vertex(self, grid_model_data, tmp_path):
        positions = _grid_joint_positions(10)
        compressor = SkinCompressor(
            model_data=grid_model_data,
            rest_joint_matrices=_joint_matrices_at(positions),
            max_influences=2,
            candidate_joints_per_vertex=3,
            **self._FAST,
        )
        allowed = candidate_joint_mask(
            geodesic_joint_distances(
                grid_model_data.rest_verts, grid_model_data.rest_faces, positions
            ),
            3,
        )

        compressor.run(tmp_path / "compressed.npz")

        weights = np.load(tmp_path / "compressed.npz")["weights"]  # (N, P)
        assert (weights != 0).sum(axis=1).max() <= 2
        assert not ((weights.T != 0) & ~allowed).any()

    def test_run_reports_joints_driving_no_vertex(
        self, grid_model_data, tmp_path, capsys
    ):
        positions = np.vstack([_grid_joint_positions(9), [[50.0, 50.0, 50.0]]])
        compressor = SkinCompressor(
            model_data=grid_model_data,
            rest_joint_matrices=_joint_matrices_at(positions),
            max_influences=2,
            candidate_joints_per_vertex=3,
            **self._FAST,
        )

        compressor.run(tmp_path / "compressed.npz")

        assert "joints driving no vertex: [9]" in capsys.readouterr().out
