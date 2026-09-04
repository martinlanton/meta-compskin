"""Tests for the command-line compression entry point used by the Maya pipeline."""

import numpy as np
import pytest
from conftest import make_grid_model, write_model_npz

from metacompskin.cli import main

_FAST = ["--iterations", "300", "--total-nnz-b-rt", "100"]


def test_cli_compresses_a_model_file(grid_model_npz, tmp_path):
    output = tmp_path / "compressed.npz"

    main([str(grid_model_npz), str(output), *_FAST, "--number-of-bones", "10"])

    result = np.load(output)
    assert result["weights"].shape == (25, 10)
    assert result["shapeXform"].shape == (9, 40)


def test_cli_uses_joint_matrices_stored_in_the_model_file(tmp_path):
    matrices = np.tile(np.eye(4), (10, 1, 1))
    matrices[:, :3, 3] = np.arange(30).reshape(10, 3)
    model = write_model_npz(
        tmp_path / "grid.npz", make_grid_model(), rest_joint_matrices=matrices
    )
    output = tmp_path / "compressed.npz"

    main([str(model), str(output), *_FAST])

    result = np.load(output)
    assert result["weights"].shape == (25, 10)
    np.testing.assert_allclose(result["restXform"], matrices[:, :3, :])


def test_cli_can_ignore_stored_joint_matrices(tmp_path):
    model = write_model_npz(
        tmp_path / "grid.npz",
        make_grid_model(),
        rest_joint_matrices=np.tile(np.eye(4), (6, 1, 1)),
    )
    output = tmp_path / "compressed.npz"

    main(
        [
            str(model),
            str(output),
            *_FAST,
            "--number-of-bones",
            "10",
            "--ignore-joint-matrices",
        ]
    )

    assert np.load(output)["weights"].shape == (25, 10)


def test_cli_rejects_a_missing_model_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        main([str(tmp_path / "missing.npz"), str(tmp_path / "out.npz")])


def test_cli_forwards_the_influence_limit(grid_model_npz, tmp_path):
    output = tmp_path / "compressed.npz"

    main(
        [
            str(grid_model_npz),
            str(output),
            *_FAST,
            "--number-of-bones",
            "10",
            "--max-influences",
            "3",
        ]
    )

    weights = np.load(output)["weights"]
    assert ((weights != 0).sum(axis=1) <= 3).all()
