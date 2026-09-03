"""Portable smoke test for the full compression pipeline.

Unlike the platform-specific regression suites (test_default_output*.py), this
test runs on every platform in CI: it compresses a tiny synthetic quad-grid
model in a few seconds and checks structural invariants of the output
(shapes, weight constraints, finiteness) rather than exact float values,
so it is robust to torch/BLAS/hardware differences.
"""

import numpy as np

from metacompskin.animation_generator import AnimationFrameGenerator

_MAX_INFLUENCES = 8  # K, mirrors SkinCompressor.max_influences


def test_output_contains_all_arrays_with_expected_shapes(compressed_model):
    model_data, compressor, output = compressed_model
    result = np.load(output)

    n_verts = model_data.n_vertices
    n_shapes = model_data.n_blendshapes
    n_bones = compressor.number_of_bones

    assert set(result.files) == {"rest", "quads", "weights", "restXform", "shapeXform"}
    assert result["rest"].shape == (n_verts, 3)
    assert result["weights"].shape == (n_verts, n_bones)
    assert result["restXform"].shape == (n_bones, 3, 4)
    assert result["shapeXform"].shape == (3 * n_shapes, 4 * n_bones)
    np.testing.assert_array_equal(result["quads"], model_data.rest_faces)


def test_saved_rest_pose_is_centered_input(compressed_model):
    model_data, _, output = compressed_model
    result = np.load(output)

    expected = model_data.rest_verts - model_data.rest_verts.mean(axis=0)
    np.testing.assert_allclose(result["rest"], expected, atol=1e-6)


def test_weights_satisfy_lbs_constraints(compressed_model):
    _, _, output = compressed_model
    weights = np.load(output)["weights"]  # (N, P)

    assert np.isfinite(weights).all(), "weights contain NaN/inf"
    assert (weights >= 0).all(), "weights must be non-negative"
    np.testing.assert_allclose(
        weights.sum(axis=1), 1.0, atol=1e-5, err_msg="weights must sum to 1 per vertex"
    )
    nonzero_per_vertex = (weights != 0).sum(axis=1)
    assert (nonzero_per_vertex <= _MAX_INFLUENCES).all(), (
        f"at most {_MAX_INFLUENCES} influences allowed per vertex, "
        f"got up to {nonzero_per_vertex.max()}"
    )


def test_shape_transforms_are_finite_and_nontrivial(compressed_model):
    _, _, output = compressed_model
    shape_xform = np.load(output)["shapeXform"]

    assert np.isfinite(shape_xform).all()
    assert np.abs(shape_xform).max() > 0, "optimization produced all-zero transforms"


def test_loss_decreases_during_first_training_phase(compressed_model):
    _, compressor, _ = compressed_model
    # loss_list holds one sample per 200 iterations per phase:
    # [phase1@0, phase1@200, phase2@0, phase2@200]
    assert len(compressor.loss_list) >= 2
    assert compressor.loss_list[1] < compressor.loss_list[0]


def test_animation_generator_produces_finite_vertices(compressed_model):
    model_data, _, output = compressed_model
    generator = AnimationFrameGenerator(
        compressed_data_path=output, model_data=model_data
    )

    one_hot = np.zeros(model_data.n_blendshapes)
    one_hot[0] = 1.0
    vertices = generator.compute_frame_vertices(one_hot)

    assert vertices.shape == (model_data.n_vertices, 3)
    assert np.isfinite(vertices).all()

    rest_only = generator.compute_frame_vertices(np.zeros(model_data.n_blendshapes))
    assert np.abs(vertices - rest_only).max() > 0, (
        "activating a blendshape did not move any vertex"
    )
