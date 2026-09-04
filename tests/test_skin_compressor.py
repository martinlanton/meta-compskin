"""Tests for SkinCompressor construction: the optimisation settings it accepts."""

import numpy as np
import pytest

from metacompskin.model_fit import SkinCompressor


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
