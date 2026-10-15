"""End-to-end Maya test: the suite's head becomes a full compressed-skin rig.

In a mayapy subprocess, the head supplied by the ``head_model`` fixture (see
conftest.py) is built with a blendShape node, then ``compress_and_build_rig``
exports it, compresses it and builds the skinned duplicate. The scene is saved
to ``output/end_to_end/<model>_rig.mb`` for viewing: the original head, offset
in X, plays each blendshape on the same frame as the rig (frame 0 neutral,
frame k + 1 shape k), so the two can be scrubbed side by side.

The module only knows the head through that fixture, so it runs unchanged on
any model. Skipped when no mayapy is found. Set ``COMPSKIN_E2E_ITERATIONS`` for a longer, cleaner
compression (default 600 per phase).
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import metacompskin
from metacompskin.animation_generator import AnimationFrameGenerator
from metacompskin.model_data import BlendshapeModelData

# Long on CPU for a large head; deselect with -m "not end_to_end".
pytestmark = pytest.mark.end_to_end

ITERATIONS = int(os.environ.get("COMPSKIN_E2E_ITERATIONS", "600"))
N_BONES = 200  # the paper's "Proteus HD" setting (Table 1)
MAX_INFLUENCES = 8  # K, SkinCompressor default
# Non-zero fraction of the 6-DOF transform deltas (6 per shape and bone).
# The paper keeps about 10% (L = 6000 at P = 40, Section 4); 15% here.
NNZ_DENSITY = 0.15
MEAN_ERROR_RATIO = 0.5  # see test_end_to_end_compression.py
MAX_ERROR_RATIO = 0.6
OUTPUT_DIR = Path(__file__).parents[1] / "output" / "end_to_end"
DRIVER = Path(__file__).parent / "maya_scripts" / "end_to_end_rig_in_maya.py"
ATOL = 1e-3  # cm; float32 points read back from Maya


def sparsity_budget(model) -> int:
    """L, the non-zero transform deltas kept: NNZ_DENSITY of all 6 * S * P."""
    return round(NNZ_DENSITY * 6 * model.n_blendshapes * N_BONES)


def run_driver(mayapy: Path, scene_input: Path, report: Path) -> None:
    """Run the driver under mayapy with this checkout's metacompskin on the path."""
    src = Path(metacompskin.__file__).parents[1]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(src), str(DRIVER.parent)])}
    result = subprocess.run(
        [str(mayapy), str(DRIVER), str(scene_input), sys.executable, str(OUTPUT_DIR)]
        + [str(report)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0 or not report.exists():
        sys.stderr.write(result.stdout[-5000:] + result.stderr[-5000:])
        pytest.fail(f"mayapy exited with {result.returncode}; see captured output")


@pytest.fixture(scope="module")
def rig_report(mayapy_path, head_model, head_shape_names, tmp_path_factory):
    """Build the rig once in mayapy and return the driver's report."""
    if mayapy_path is None:
        pytest.skip("mayapy not found")
    workdir = tmp_path_factory.mktemp("end_to_end_maya")
    scene_input = workdir / f"{head_model.model_name}.npz"
    np.savez(
        scene_input,
        rest_verts=head_model.rest_verts,
        faces=head_model.rest_faces,
        deltas=head_model.deltas,
        shape_names=np.array(head_shape_names),
        iterations=ITERATIONS,
        number_of_bones=N_BONES,
        total_nnz_B_rt=sparsity_budget(head_model),
    )
    report = workdir / "report.npz"

    run_driver(mayapy_path, scene_input, report)
    return dict(np.load(report))


def compressed_frames(report) -> np.ndarray:
    """Rest then each shape at full weight, from the compressed file, in cm."""
    model = BlendshapeModelData.from_npz(str(report["model_path"]))
    generator = AnimationFrameGenerator(str(report["compressed_path"]), model)
    coefficients = np.vstack(
        [np.zeros(model.n_blendshapes), np.eye(model.n_blendshapes)]
    )
    centred = np.stack([generator.compute_frame_vertices(c) for c in coefficients])
    return centred + model.rest_verts.mean(axis=0)  # shape: (S + 1, N, 3)


def test_rig_scene_is_saved_for_viewing(rig_report, head_model):
    scene = Path(str(rig_report["scene_path"]))

    assert scene == OUTPUT_DIR / f"{head_model.model_name}_rig.mb"
    assert scene.exists()


def test_rig_has_one_joint_per_bone_within_the_influence_budget(rig_report):
    assert int(rig_report["n_joints"]) == N_BONES
    assert int(rig_report["max_influences"]) <= MAX_INFLUENCES


def test_rig_matches_the_compressed_result_on_every_frame(rig_report):
    expected = compressed_frames(rig_report)

    np.testing.assert_allclose(rig_report["rig_frames"], expected, atol=ATOL)


def test_original_head_plays_each_blendshape_on_its_frame(rig_report, head_model):
    rest = head_model.rest_verts
    expected = np.concatenate([rest[None], rest + head_model.deltas])

    np.testing.assert_allclose(rig_report["source_frames"], expected, atol=ATOL)


def test_frames_are_labelled_with_the_shape_names(rig_report, head_shape_names):
    assert rig_report["current_shape"].tolist() == ["rest", *head_shape_names]


def test_rig_stays_close_to_the_original_head(rig_report, head_model):
    error = np.abs(rig_report["rig_frames"] - rig_report["source_frames"])[1:]
    zero_motion = np.abs(head_model.deltas)

    assert error.mean() < MEAN_ERROR_RATIO * zero_motion.mean()
    assert error.max() < MAX_ERROR_RATIO * zero_motion.max()
