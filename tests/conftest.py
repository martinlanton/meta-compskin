"""Shared fixtures: a tiny synthetic model and its compressed output.

The quad-grid model compresses in a few seconds on any platform, so it backs
both the portable pipeline smoke test and the Maya rig builder test.
"""

import numpy as np
import pytest

from metacompskin.model_data import BlendshapeModelData
from metacompskin.model_fit import SkinCompressor

GRID = 5  # 5x5 vertices -> 16 quad faces
N_SHAPES = 3
ITERATIONS = 300  # loss is sampled every 200 iterations -> 2 samples per phase


def make_grid_model() -> BlendshapeModelData:
    """Build a tiny flat quad-grid model with three smooth blendshapes."""
    xs, ys = np.meshgrid(
        np.arange(GRID, dtype=np.float32),
        np.arange(GRID, dtype=np.float32),
    )
    n_verts = GRID * GRID
    rest_verts = np.stack(
        [xs.ravel(), ys.ravel(), np.zeros(n_verts, dtype=np.float32)], axis=1
    )

    faces = []
    for row in range(GRID - 1):
        for col in range(GRID - 1):
            i = row * GRID + col
            faces.append([i, i + 1, i + GRID + 1, i + GRID])
    rest_faces = np.array(faces, dtype=np.int32)

    deltas = np.zeros((N_SHAPES, n_verts, 3), dtype=np.float32)
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


@pytest.fixture(scope="session")
def compressed_model(tmp_path_factory):
    """Run compression once per session and return (model, compressor, npz path)."""
    model_data = make_grid_model()
    compressor = SkinCompressor(model_data=model_data, iterations=ITERATIONS)
    # B_rt only has 6*S*P elements on this tiny model; the default sparsity
    # budget (6000) exceeds that, which torch.topk rejects.
    max_nnz = int(6 * model_data.n_blendshapes * compressor.number_of_bones * 0.8)
    compressor.total_nnz_B_rt = min(compressor.total_nnz_B_rt, max_nnz)

    output = tmp_path_factory.mktemp("smoke") / "compressed.npz"
    compressor.run(output_location=output)
    return model_data, compressor, output
