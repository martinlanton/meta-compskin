"""Portable smoke test for the full compression pipeline.

Unlike the platform-specific regression suites (test_default_output*.py), this
test runs on every platform in CI: it compresses a tiny synthetic quad-grid
model in a few seconds and checks structural invariants of the output
(shapes, weight constraints, finiteness) rather than exact float values,
so it is robust to torch/BLAS/hardware differences.
"""

import numpy as np
import pytest

from metacompskin.animation_generator import AnimationFrameGenerator
from metacompskin.model_data import BlendshapeModelData
from metacompskin.model_fit import SkinCompressor

_GRID = 5  # 5x5 vertices -> 16 quad faces
_N_SHAPES = 3
_ITERATIONS = 300  # loss is sampled every 200 iterations -> 2 samples per phase
_MAX_INFLUENCES = 8  # K, mirrors SkinCompressor.max_influences


def _make_grid_model() -> BlendshapeModelData:
    """Build a tiny flat quad-grid model with three smooth blendshapes."""
    xs, ys = np.meshgrid(
        np.arange(_GRID, dtype=np.float32),
        np.arange(_GRID, dtype=np.float32),
    )
    n_verts = _GRID * _GRID
    rest_verts = np.stack(
        [xs.ravel(), ys.ravel(), np.zeros(n_verts, dtype=np.float32)], axis=1
    )

    faces = []
    for row in range(_GRID - 1):
        for col in range(_GRID - 1):
            i = row * _GRID + col
            faces.append([i, i + 1, i + _GRID + 1, i + _GRID])
    rest_faces = np.array(faces, dtype=np.int32)

    deltas = np.zeros((_N_SHAPES, n_verts, 3), dtype=np.float32)
    deltas[0, :, 2] = 0.5 * rest_verts[:, 0]  # bend: z grows with x
    deltas[1, :, 2] = np.sin(rest_verts[:, 1])  # ripple along y
    deltas[2, :, 1] = 0.75  # uniform translation in y

    return BlendshapeModelData(
        deltas=deltas,
        rest_verts=rest_verts,
        rest_faces=rest_faces,
        inbetween_info={},
        combination_info={},
        model_name="smoke_grid",
        alpha=10.0,
    )


@pytest.fixture(scope="module")
def compressed_model(tmp_path_factory):
    """Run compression once for the module and return (model, compressor, npz path)."""
    model_data = _make_grid_model()
    compressor = SkinCompressor(model_data=model_data, iterations=_ITERATIONS)
    # B_rt only has 6*S*P elements on this tiny model; the default sparsity
    # budget (6000) exceeds that, which torch.topk rejects.
    max_nnz = int(6 * model_data.n_blendshapes * compressor.number_of_bones * 0.8)
    compressor.total_nnz_B_rt = min(compressor.total_nnz_B_rt, max_nnz)

    output = tmp_path_factory.mktemp("smoke") / "compressed.npz"
    compressor.run(output_location=output)
    return model_data, compressor, output


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
