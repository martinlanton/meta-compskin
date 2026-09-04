"""End-to-end test of the one-call pipeline: export, compress, build, in mayapy.

The compression subprocess uses this test run's own interpreter, which has
PyTorch. Skipped when no mayapy is found (set ``MAYAPY`` to choose one).
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from conftest import MAYA_SCRIPTS, make_grid_model, requires_mayapy

from metacompskin.animation_generator import AnimationFrameGenerator
from metacompskin.maya_pipeline import probe_torch
from metacompskin.model_data import BlendshapeModelData

pytestmark = requires_mayapy

_ATOL = 1e-4
_SOURCE_NAME = "head_GEO"


def _scene_space(flat_row_major: list[float], points: np.ndarray) -> np.ndarray:
    world = np.array(flat_row_major).reshape(4, 4).T
    homog = np.concatenate([points, np.ones((*points.shape[:-1], 1))], axis=-1)
    return (homog @ world.T)[..., :3]


@pytest.fixture(scope="module")
def pipeline_report(run_maya_script, tmp_path_factory):
    model = make_grid_model()
    workdir = tmp_path_factory.mktemp("pipeline")
    shape_names = [f"grid_shape_{k}" for k in range(model.n_blendshapes)]
    scene_input = workdir / "scene_input.npz"
    np.savez(
        scene_input,
        rest_verts=model.rest_verts,
        faces=model.rest_faces,
        deltas=model.deltas,
        shape_names=np.array(shape_names),
    )
    output_dir = workdir / "out"
    report_path = workdir / "report.json"

    report = run_maya_script(
        MAYA_SCRIPTS / "run_pipeline_in_maya.py",
        [str(scene_input), sys.executable, str(output_dir), str(report_path)],
        report_path,
    )
    report["shape_names"] = shape_names
    report["output_dir"] = output_dir
    return report


def test_files_land_in_the_output_directory(pipeline_report):
    outcome = pipeline_report["pipeline"]

    assert Path(outcome["model_path"]) == pipeline_report["output_dir"] / "head_GEO.npz"
    assert (
        Path(outcome["compressed_path"])
        == pipeline_report["output_dir"] / "head_GEO_compressed.npz"
    )
    assert Path(outcome["model_path"]).exists()
    assert Path(outcome["compressed_path"]).exists()


def test_compression_honours_the_settings(pipeline_report):
    outcome = pipeline_report["pipeline"]

    weights = np.load(outcome["compressed_path"])["weights"]

    assert weights.shape[1] == 10
    assert outcome["n_joints"] == 10
    assert ((weights != 0).sum(axis=1) <= 3).all()


def test_rig_is_built_on_a_duplicate_and_matches_the_compressed_result(
    pipeline_report,
):
    outcome = pipeline_report["pipeline"]
    model = BlendshapeModelData.from_npz(outcome["model_path"])
    generator = AnimationFrameGenerator(outcome["compressed_path"], model)
    coefficients = np.vstack(
        [np.zeros(model.n_blendshapes), np.eye(model.n_blendshapes)]
    )
    centred = np.stack([generator.compute_frame_vertices(c) for c in coefficients])
    expected = _scene_space(
        outcome["source_world_matrix"], centred + model.rest_verts.mean(axis=0)
    )

    actual = np.array(outcome["frames"])

    assert outcome["rig_mesh"] != _SOURCE_NAME
    assert outcome["source_skin_clusters"] == []
    np.testing.assert_allclose(actual, expected, atol=_ATOL)


def test_frames_are_labelled_with_the_blendshape_target_names(pipeline_report):
    outcome = pipeline_report["pipeline"]

    assert outcome["current_shape"] == ["rest", *pipeline_report["shape_names"]]


def test_source_blendshape_is_restored_after_the_build(pipeline_report):
    outcome = pipeline_report["pipeline"]

    assert outcome["envelope_after"] == 1.0
    assert outcome["weight_after"] == pytest.approx(0.4)


def test_interpreter_is_auto_detected_from_inside_maya(pipeline_report):
    detected = pipeline_report["detected_interpreter"]

    assert detected is not None
    assert not Path(detected).name.startswith("mayapy")
    assert probe_torch(Path(detected)) is not None


def test_mesh_resolution_rules(pipeline_report):
    outcome = pipeline_report["resolution"]

    assert outcome["single_mesh_unselected"].endswith(_SOURCE_NAME)
    assert outcome["shape_name_given"].endswith(_SOURCE_NAME)
    assert "2 meshes" in outcome["two_meshes_unselected"]
    assert "no mesh" in outcome["no_mesh"]
