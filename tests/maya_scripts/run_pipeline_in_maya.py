"""Runs the one-call pipeline inside mayapy and reports what it produced.

Run by tests/test_maya_pipeline_in_maya.py::

    mayapy run_pipeline_in_maya.py <scene_input.npz> <python_with_torch> \\
        <output_dir> <report.json>

``scene_input.npz`` holds ``rest_verts`` (N, 3), ``faces`` (F, 4), ``deltas``
(S, N, 3) and ``shape_names`` (S,). The rest mesh gets a blendShape node with
one target per shape, then ``compress_and_build_rig`` runs on it. Mesh
resolution rules are probed separately in fresh scenes.
"""

import json
import os
import sys
from pathlib import Path

import maya.standalone
import numpy as np

maya.standalone.initialize()

from maya import cmds  # noqa: E402
from maya_test_helpers import (  # noqa: E402
    SOURCE_NAME,
    create_mesh,
    new_scene_with_source,
    sample_rig,
    skin_clusters_on,
    world_matrix,
)

from metacompskin.maya_pipeline import (  # noqa: E402
    compress_and_build_rig,
    resolve_python_executable,
    resolve_source_mesh,
)

ACTIVE_WEIGHT = 0.4  # shape 0 is left partially on to check it is restored


def run_pipeline(scene, python: str, output_dir: Path) -> dict:
    source = new_scene_with_source(scene)
    targets = [
        create_mesh(scene["rest_verts"] + delta, scene["faces"], str(name))
        for delta, name in zip(scene["deltas"], scene["shape_names"], strict=True)
    ]
    blendshape = cmds.blendShape(*targets, source, name="head_BS")[0]
    cmds.setAttr(f"{blendshape}.weight[0]", ACTIVE_WEIGHT)
    cmds.select(source)

    result = compress_and_build_rig(
        python_executable=python,
        output_dir=output_dir,
        iterations=300,
        number_of_bones=10,
        max_influences=3,
        total_nnz_B_rt=100,
    )
    frames, current_shape = sample_rig(result.rig, len(targets) + 1)
    return {
        "source_mesh": result.source_mesh,
        "model_path": str(result.model_path),
        "compressed_path": str(result.compressed_path),
        "rig_mesh": result.rig.mesh.split("|")[-1],
        "n_joints": len(result.rig.joints),
        "frames": frames,
        "current_shape": current_shape,
        "source_world_matrix": world_matrix(source),
        "source_skin_clusters": skin_clusters_on(source),
        "envelope_after": cmds.getAttr(f"{blendshape}.envelope"),
        "weight_after": cmds.getAttr(f"{blendshape}.weight[0]"),
    }


def probe_resolution(scene) -> dict:
    outcomes = {}

    new_scene_with_source(scene)
    cmds.select(clear=True)
    outcomes["single_mesh_unselected"] = resolve_source_mesh(cmds, None)

    shape = cmds.listRelatives(SOURCE_NAME, shapes=True, fullPath=True)[0]
    outcomes["shape_name_given"] = resolve_source_mesh(cmds, shape)

    create_mesh(scene["rest_verts"], scene["faces"], "other_GEO")
    cmds.select(clear=True)
    outcomes["two_meshes_unselected"] = _error(lambda: resolve_source_mesh(cmds, None))

    cmds.file(new=True, force=True)
    outcomes["no_mesh"] = _error(lambda: resolve_source_mesh(cmds, None))
    return outcomes


def _error(call) -> str | None:
    try:
        call()
    except ValueError as error:
        return str(error)
    return None


def detect_interpreter() -> str | None:
    os.environ.pop("METACOMPSKIN_PYTHON", None)
    try:
        return str(resolve_python_executable(None))
    except ValueError:
        return None


def main(scene_input: Path, python: str, output_dir: Path, report: Path) -> None:
    scene = np.load(scene_input)
    report.write_text(
        json.dumps(
            {
                "pipeline": run_pipeline(scene, python, output_dir),
                "resolution": probe_resolution(scene),
                "detected_interpreter": detect_interpreter(),
            }
        )
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4]))
    maya.standalone.uninitialize()
