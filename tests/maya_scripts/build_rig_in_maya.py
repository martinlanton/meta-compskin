"""Builds compressed-skin rigs inside mayapy and reports what they produced.

Run by tests/test_maya_rig_builder_in_maya.py through a mayapy subprocess::

    mayapy build_rig_in_maya.py <compressed.npz> <compressed_custom.npz> \\
        <scene_input.npz> <report.json>

``scene_input.npz`` holds ``rest_verts`` (N, 3), ``faces`` (F, 4) and
``shape_names`` (S,). ``compressed_custom.npz`` is the same archive with
user-supplied ``restXform`` matrices. Four scenarios run in fresh scenes; the
pytest side computes the expected values and asserts.
"""

import json
import sys
from pathlib import Path

import maya.standalone
import numpy as np

maya.standalone.initialize()

from maya import cmds  # noqa: E402
from maya_test_helpers import (  # noqa: E402
    create_mesh,
    new_scene_with_source,
    sample_rig,
    skin_clusters_on,
    world_matrix,
)

from metacompskin.maya_rig_builder import build_skinned_rig  # noqa: E402


def describe_rig(rig, source: str, n_frames: int) -> dict:
    cmds.currentTime(0)
    bind_matrices = [world_matrix(joint) for joint in rig.joints]
    frames, current_shape = sample_rig(rig, n_frames)
    return {
        "mesh": rig.mesh.split("|")[-1],
        "root": rig.root,
        "joints": rig.joints,
        "drivers": rig.drivers,
        "skin_cluster": rig.skin_cluster,
        "joint_parents": [cmds.listRelatives(j, parent=True)[0] for j in rig.joints],
        "driver_parents": [cmds.listRelatives(d, parent=True)[0] for d in rig.drivers],
        "joint_bind_matrices": bind_matrices,
        "source_world_matrix": world_matrix(source),
        "source_skin_clusters": skin_clusters_on(source),
        "max_influences": cmds.getAttr(f"{rig.skin_cluster}.maxInfluences"),
        "frames": frames,
        "current_shape": current_shape,
        "playback_range": [
            cmds.playbackOptions(q=True, minTime=True),
            cmds.playbackOptions(q=True, maxTime=True),
        ],
    }


def run_duplicate(compressed: Path, scene, names: list[str]) -> dict:
    source = new_scene_with_source(scene)
    rig = build_skinned_rig(compressed, shape_names=names)
    return describe_rig(rig, source, len(names) + 1)


def run_selected(compressed: Path, scene, names: list[str]) -> dict:
    source = new_scene_with_source(scene)
    rig = build_skinned_rig(compressed, shape_names=names, duplicate_mesh=False)
    return describe_rig(rig, source, len(names) + 1)


def run_selected_with_blendshape(compressed: Path, scene, names: list[str]) -> dict:
    source = new_scene_with_source(scene)
    target = create_mesh(
        scene["rest_verts"] + [0.0, 0.0, 1.0], scene["faces"], "smile_GEO"
    )
    cmds.blendShape(target, source, name="head_BS")
    cmds.select(source)
    try:
        build_skinned_rig(compressed, shape_names=names, duplicate_mesh=False)
    except ValueError as error:
        message = str(error)
    else:
        message = None
    return {
        "error": message,
        "nodes_created": cmds.ls("compskin_*"),
        "source_skin_clusters": skin_clusters_on(source),
    }


def main(compressed: Path, compressed_custom: Path, scene_input: Path, report: Path):
    scene = np.load(scene_input)
    names = [str(name) for name in scene["shape_names"]]
    report.write_text(
        json.dumps(
            {
                "duplicate": run_duplicate(compressed, scene, names),
                "custom_placement": run_duplicate(compressed_custom, scene, names),
                "selected": run_selected(compressed, scene, names),
                "selected_with_blendshape": run_selected_with_blendshape(
                    compressed, scene, names
                ),
            }
        )
    )


if __name__ == "__main__":
    main(*(Path(arg) for arg in sys.argv[1:5]))
    maya.standalone.uninitialize()
