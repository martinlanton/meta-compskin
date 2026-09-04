"""Shared fixtures: a tiny synthetic model and its compressed output.

The quad-grid model compresses in a few seconds on any platform, so it backs
both the portable pipeline smoke test and the Maya rig builder test.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from metacompskin.model_data import BlendshapeModelData
from metacompskin.model_fit import SkinCompressor

GRID = 5  # 5x5 vertices -> 16 quad faces
N_SHAPES = 3
ITERATIONS = 300  # loss is sampled every 200 iterations -> 2 samples per phase
N_BONES = 40  # small P keeps the smoke run fast


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
    # B_rt only has 6*S*P elements on this tiny model; the default sparsity
    # budget (6000) exceeds that, which torch.topk rejects.
    max_nnz = int(6 * model_data.n_blendshapes * N_BONES * 0.8)
    compressor = SkinCompressor(
        model_data=model_data,
        iterations=ITERATIONS,
        number_of_bones=N_BONES,
        total_nnz_B_rt=min(6000, max_nnz),
    )

    output = tmp_path_factory.mktemp("smoke") / "compressed.npz"
    compressor.run(output_location=output)
    return model_data, compressor, output


@pytest.fixture
def grid_model_data():
    """A fresh tiny grid model for constructor-level tests."""
    return make_grid_model()


def write_model_npz(path, model_data: BlendshapeModelData, **extra) -> Path:
    """Write a model in the exporter's NPZ layout (plus any extra arrays)."""
    np.savez(
        path,
        deltas=model_data.deltas,
        rest_verts=model_data.rest_verts,
        rest_faces=model_data.rest_faces,
        inbetween_info=np.array({}, dtype=object),
        combination_info=np.array({}, dtype=object),
        shape_names=np.array([f"grid_shape_{k}" for k in range(N_SHAPES)]),
        **extra,
    )
    return path


@pytest.fixture
def grid_model_npz(tmp_path):
    """The grid model written as the exporter would write it."""
    return write_model_npz(tmp_path / "grid.npz", make_grid_model())


_MAYAPY_LOCATIONS = [
    ("/Applications/Autodesk", "maya*/Maya.app/Contents/bin/mayapy"),
    ("/usr/autodesk", "maya*/bin/mayapy"),
    ("C:/Program Files/Autodesk", "Maya*/bin/mayapy.exe"),
]
MAYA_SCRIPTS = Path(__file__).parent / "maya_scripts"
SRC = Path(__file__).parents[1] / "src"


def find_mayapy() -> str | None:
    """The MAYAPY environment variable, else the newest standard install."""
    if os.environ.get("MAYAPY"):
        return os.environ["MAYAPY"]
    candidates = sorted(
        str(path)
        for base, pattern in _MAYAPY_LOCATIONS
        for path in Path(base).glob(pattern)
    )
    return candidates[-1] if candidates else None


MAYAPY = find_mayapy()
requires_mayapy = pytest.mark.skipif(
    MAYAPY is None, reason="mayapy not found (set MAYAPY to run the Maya tests)"
)


@pytest.fixture(scope="session")
def run_maya_script():
    """Run a script under mayapy and return the JSON report it wrote."""

    def run(script: Path, args: list[str], report_path: Path) -> dict:
        env = {
            **os.environ,
            "PYTHONPATH": os.pathsep.join([str(SRC), os.environ.get("PYTHONPATH", "")]),
        }
        result = subprocess.run(
            [MAYAPY, str(script), *args],
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
        if result.returncode != 0 or not report_path.exists():
            sys.stderr.write(result.stdout)
            sys.stderr.write(result.stderr)
            pytest.fail(f"mayapy exited with {result.returncode}; see captured output")
        return json.loads(report_path.read_text())

    return run
