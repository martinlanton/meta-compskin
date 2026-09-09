"""Tests for SkinCompressor construction: the optimisation settings it accepts."""

import numpy as np
import pytest

from metacompskin.model_fit import SkinCompressor, TrainingPhase


def _identity_joint_matrices(n_bones: int) -> np.ndarray:
    return np.tile(np.eye(4), (n_bones, 1, 1))


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
