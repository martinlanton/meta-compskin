"""End-to-end compression of the suite's head model, without Maya.

Compresses the head supplied by the ``head_model`` fixture (see conftest.py)
and checks invariants and error bounds rather than exact floats, so it holds
on every platform. test_end_to_end_maya.py turns the same head into a Maya rig.

This module is kept identical in meta-compskin and
meta-compskin_private_tests so the two heads are tested the same way. Set
``COMPSKIN_E2E_ITERATIONS`` for a longer run (default 600 per phase).
"""

import os

import numpy as np
import pytest

from metacompskin.model_fit import SkinCompressor

ITERATIONS = int(os.environ.get("COMPSKIN_E2E_ITERATIONS", "600"))
N_BONES = 40  # the paper's setting for its test rigs (Table 1)
MAX_INFLUENCES = 8  # K, SkinCompressor default
# Fraction of the zero-motion error (predicting no deltas at all) the
# compression must stay under; 600 iterations reach roughly 0.35 on proteus.
MEAN_ERROR_RATIO = 0.5
MAX_ERROR_RATIO = 0.6


@pytest.fixture(scope="module")
def compressed(head_model, tmp_path_factory):
    """Compress the head once and return (compressor, output arrays)."""
    output = tmp_path_factory.mktemp("end_to_end") / "compressed.npz"
    compressor = SkinCompressor(
        model_data=head_model, iterations=ITERATIONS, number_of_bones=N_BONES
    )
    compressor.run(output_location=output)
    return compressor, np.load(output)


def test_output_has_one_weight_column_per_bone_and_keeps_topology(
    compressed, head_model
):
    _, result = compressed

    assert result["weights"].shape == (head_model.n_vertices, N_BONES)
    assert result["shapeXform"].shape == (3 * head_model.n_blendshapes, 4 * N_BONES)
    np.testing.assert_array_equal(result["quads"], head_model.rest_faces)


def test_weights_satisfy_lbs_constraints(compressed):
    weights = compressed[1]["weights"]  # shape: (N, P)

    assert np.isfinite(weights).all()
    assert (weights >= 0).all()
    np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-5)
    assert ((weights != 0).sum(axis=1) <= MAX_INFLUENCES).all()


def test_reconstruction_beats_zero_motion(compressed, head_model):
    error = compressed[0].reconstruction_error
    zero_motion = np.abs(head_model.deltas)  # error of predicting no deltas

    assert error.mean_abs < MEAN_ERROR_RATIO * zero_motion.mean()
    assert error.max_abs < MAX_ERROR_RATIO * zero_motion.max()
