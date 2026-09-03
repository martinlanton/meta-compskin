"""End-to-end tests of the Maya rig builder, run inside a mayapy subprocess.

Skipped when no mayapy is found. Set the ``MAYAPY`` environment variable to
point at a specific interpreter; otherwise the newest Maya install in the
standard locations is used.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from metacompskin.animation_generator import AnimationFrameGenerator
from metacompskin.maya_rig_builder import weighted_joint_centroids

_MAYAPY_LOCATIONS = [
    ("/Applications/Autodesk", "maya*/Maya.app/Contents/bin/mayapy"),
    ("/usr/autodesk", "maya*/bin/mayapy"),
    ("C:/Program Files/Autodesk", "Maya*/bin/mayapy.exe"),
]
_SCRIPT = Path(__file__).parent / "maya_scripts" / "build_rig_in_maya.py"
_SRC = Path(__file__).parents[1] / "src"
_ATOL = 1e-4
_SOURCE_NAME = "head_GEO"


def _find_mayapy() -> str | None:
    if os.environ.get("MAYAPY"):
        return os.environ["MAYAPY"]
    candidates = sorted(
        str(path)
        for base, pattern in _MAYAPY_LOCATIONS
        for path in Path(base).glob(pattern)
    )
    return candidates[-1] if candidates else None


MAYAPY = _find_mayapy()

pytestmark = pytest.mark.skipif(
    MAYAPY is None, reason="mayapy not found (set MAYAPY to run the Maya tests)"
)


def _reference_frames(model_data, compressed: Path) -> np.ndarray:
    """Centred vertex positions per frame: rest first, then each shape at 1.0."""
    generator = AnimationFrameGenerator(compressed, model_data)
    frames = [generator.compute_frame_vertices(np.zeros(model_data.n_blendshapes))]
    for shape in range(model_data.n_blendshapes):
        one_hot = np.zeros(model_data.n_blendshapes)
        one_hot[shape] = 1.0
        frames.append(generator.compute_frame_vertices(one_hot))
    return np.stack(frames)  # (S + 1, N, 3)


def _custom_rest_xform(n_bones: int) -> np.ndarray:
    """Distinct rigid matrices per joint, shape (P, 3, 4)."""
    rng = np.random.default_rng(7)
    angle = np.deg2rad(10.0)
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    matrices = np.tile(np.eye(3, 4), (n_bones, 1, 1))
    matrices[:, :, :3] = rotation
    matrices[:, :, 3] = rng.normal(size=(n_bones, 3))
    return matrices


def _to_column_matrices(flat_row_major) -> np.ndarray:
    """Maya's flat row-vector matrices (P, 16) to column-vector (P, 4, 4)."""
    return np.array(flat_row_major).reshape(-1, 4, 4).transpose(0, 2, 1)


def _scene_space(world: np.ndarray, points: np.ndarray) -> np.ndarray:
    homog = np.concatenate([points, np.ones((*points.shape[:-1], 1))], axis=-1)
    return (homog @ world.T)[..., :3]


@pytest.fixture(scope="module")
def maya_report(compressed_model, tmp_path_factory):
    model_data, compressor, compressed = compressed_model
    workdir = tmp_path_factory.mktemp("maya")
    shape_names = [f"grid_shape_{k}" for k in range(model_data.n_blendshapes)]

    archive = dict(np.load(compressed))
    custom_rest_xform = _custom_rest_xform(compressor.number_of_bones)
    compressed_custom = workdir / "compressed_custom.npz"
    np.savez(compressed_custom, **{**archive, "restXform": custom_rest_xform})

    scene_input = workdir / "scene_input.npz"
    np.savez(
        scene_input,
        rest_verts=model_data.rest_verts,
        faces=model_data.rest_faces,
        shape_names=np.array(shape_names),
    )
    report_path = workdir / "report.json"

    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(_SRC), os.environ.get("PYTHONPATH", "")]),
    }
    result = subprocess.run(
        [
            MAYAPY,
            str(_SCRIPT),
            str(compressed),
            str(compressed_custom),
            str(scene_input),
            str(report_path),
        ],
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

    report = json.loads(report_path.read_text())
    report["expected_centred_frames"] = _reference_frames(model_data, compressed)
    report["rest_verts"] = model_data.rest_verts
    report["weights"] = archive["weights"]
    report["custom_rest_xform"] = custom_rest_xform
    report["shape_names"] = shape_names
    report["n_bones"] = compressor.number_of_bones
    return report


def test_duplicating_leaves_the_source_mesh_untouched(maya_report):
    scenario = maya_report["duplicate"]

    assert scenario["source_skin_clusters"] == []
    assert scenario["mesh"] != _SOURCE_NAME


def test_one_joint_per_bone_each_under_its_driver_in_a_flat_hierarchy(maya_report):
    scenario = maya_report["duplicate"]

    assert len(scenario["joints"]) == maya_report["n_bones"]
    assert scenario["joint_parents"] == scenario["drivers"]
    assert set(scenario["driver_parents"]) == {scenario["root"]}


@pytest.mark.parametrize("name", ["duplicate", "custom_placement", "selected"])
def test_skinned_mesh_matches_reference_on_every_frame(maya_report, name):
    scenario = maya_report[name]
    world = _to_column_matrices([scenario["source_world_matrix"]])[0]
    scene = maya_report["expected_centred_frames"] + maya_report["rest_verts"].mean(
        axis=0
    )
    expected = _scene_space(world, scene)

    actual = np.array(scenario["frames"])

    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, atol=_ATOL)


def test_joints_sit_at_weighted_vertex_centroids_without_placement(maya_report):
    scenario = maya_report["duplicate"]
    world = _to_column_matrices([scenario["source_world_matrix"]])[0]
    scene_points = _scene_space(world, maya_report["rest_verts"])
    expected = weighted_joint_centroids(maya_report["weights"], scene_points)

    bind = _to_column_matrices(scenario["joint_bind_matrices"])

    np.testing.assert_allclose(bind[:, :3, 3], expected, atol=_ATOL)
    np.testing.assert_allclose(bind[:, :3, :3], [np.eye(3)] * len(bind), atol=_ATOL)


def test_joints_use_the_provided_matrices_when_present(maya_report):
    scenario = maya_report["custom_placement"]
    expected = np.tile(np.eye(4), (maya_report["n_bones"], 1, 1))
    expected[:, :3, :] = maya_report["custom_rest_xform"]

    bind = _to_column_matrices(scenario["joint_bind_matrices"])

    np.testing.assert_allclose(bind, expected, atol=_ATOL)


def test_selected_mesh_is_skinned_in_place(maya_report):
    scenario = maya_report["selected"]

    assert scenario["mesh"] == _SOURCE_NAME
    assert scenario["source_skin_clusters"] == [scenario["skin_cluster"]]


def test_selected_mesh_with_deformers_aborts_before_creating_anything(maya_report):
    scenario = maya_report["selected_with_blendshape"]

    assert scenario["error"] is not None
    assert "head_BS" in scenario["error"]
    assert scenario["nodes_created"] == []
    assert scenario["source_skin_clusters"] == []


def test_root_attribute_names_the_current_shape_on_each_frame(maya_report):
    scenario = maya_report["duplicate"]

    assert scenario["current_shape"] == ["rest", *maya_report["shape_names"]]


def test_timeline_spans_rest_frame_and_every_shape(maya_report):
    scenario = maya_report["duplicate"]

    assert scenario["playback_range"] == [0, len(maya_report["shape_names"])]


def test_skin_cluster_limits_influences_to_the_compressed_budget(maya_report):
    assert maya_report["duplicate"]["max_influences"] <= 8
