"""Builds a full head rig with the one-call pipeline and saves it for viewing.

Run by tests/test_end_to_end_maya.py through a mayapy subprocess::

    mayapy end_to_end_rig_in_maya.py <scene_input.npz> <python_with_torch> \\
        <output_dir> <report.npz>

``scene_input.npz`` holds ``rest_verts`` (N, 3), ``faces`` (F, 4), ``deltas``
(S, N, 3), ``shape_names`` (S,) and the compressor's ``iterations`` and
``number_of_bones``. The head gets a blendShape node with one
target per shape, then ``compress_and_build_rig`` exports, compresses and
builds the skinned duplicate. The original blendShape is then keyed on the
same timeline as the rig (frame 0 neutral, frame k + 1 shape k), so scrubbing
the saved scene shows the original and the compressed head side by side.

This file is kept identical in meta-compskin and meta-compskin_private_tests
so the two suites build their rigs the same way.
"""

import re
import sys
from pathlib import Path

import maya.standalone
import numpy as np

maya.standalone.initialize()

from maya import cmds  # noqa: E402
from maya.api import OpenMaya  # noqa: E402

from metacompskin.maya_pipeline import compress_and_build_rig  # noqa: E402

SOURCE_NAME = "head_GEO"
BLENDSHAPE_NAME = "head_BS"
om = OpenMaya
SIDE_BY_SIDE_GAP = 1.2  # source offset in X, as a multiple of the head width


def create_mesh(rest_verts: np.ndarray, faces: np.ndarray, name: str) -> str:
    points = om.MPointArray([om.MPoint(*map(float, p)) for p in rest_verts])
    counts = om.MIntArray([len(face) for face in faces])
    connects = om.MIntArray([int(i) for face in faces for i in face])
    transform = om.MFnMesh().create(points, counts, connects)
    return cmds.rename(om.MFnDagNode(transform).fullPathName(), name)


def world_points(transform: str) -> np.ndarray:
    selection = om.MSelectionList()
    selection.add(transform)
    dag = selection.getDagPath(0)
    dag.extendToShape()
    points = om.MFnMesh(dag).getPoints(om.MSpace.kWorld)
    return np.array([[p.x, p.y, p.z] for p in points], dtype=np.float32)


def build_source_head(scene) -> tuple[str, str]:
    """The neutral head with a blendShape holding every shape as a target."""
    cmds.file(new=True, force=True)
    source = create_mesh(scene["rest_verts"], scene["faces"], SOURCE_NAME)
    targets = [
        create_mesh(scene["rest_verts"] + delta, scene["faces"], safe_name(name))
        for delta, name in zip(scene["deltas"], scene["shape_names"], strict=True)
    ]
    blendshape = cmds.blendShape(*targets, source, name=BLENDSHAPE_NAME)[0]
    cmds.delete(targets)  # the blendShape keeps the target data
    cmds.select(source)
    return source, blendshape


def safe_name(name) -> str:
    return re.sub(r"\W", "_", str(name))


def key_source_like_the_rig(blendshape: str, n_shapes: int) -> None:
    """Frame 0 neutral, frame k + 1 shape k at full weight, stepped keys."""
    for frame in range(n_shapes + 1):
        for shape in range(n_shapes):
            cmds.setKeyframe(
                f"{blendshape}.weight[{shape}]",
                time=frame,
                value=float(shape == frame - 1),
                outTangentType="step",
            )


def sample(rig, source: str, n_frames: int) -> dict:
    """Per frame: rig and source world points, and the rig's shape label."""
    rig_frames, source_frames, labels = [], [], []
    for frame in range(n_frames):
        cmds.currentTime(frame)
        rig_frames.append(world_points(rig.mesh))
        source_frames.append(world_points(source))
        labels.append(cmds.getAttr(f"{rig.root}.currentShape", asString=True))
    cmds.currentTime(0)
    return {
        "rig_frames": np.stack(rig_frames),  # shape: (S + 1, N, 3)
        "source_frames": np.stack(source_frames),  # shape: (S + 1, N, 3)
        "current_shape": np.array(labels),
    }


def place_side_by_side(source: str, rest_verts: np.ndarray) -> None:
    width = float(np.ptp(rest_verts[:, 0]))
    cmds.setAttr(f"{source}.translateX", SIDE_BY_SIDE_GAP * width)


def main(scene_input: Path, python: str, output_dir: Path, report: Path) -> None:
    scene = np.load(scene_input)
    n_shapes = len(scene["shape_names"])
    source, blendshape = build_source_head(scene)

    result = compress_and_build_rig(
        python_executable=python,
        output_dir=output_dir,
        iterations=int(scene["iterations"]),
        number_of_bones=int(scene["number_of_bones"]),
    )
    key_source_like_the_rig(blendshape, n_shapes)
    frames = sample(result.rig, source, n_shapes + 1)

    place_side_by_side(source, scene["rest_verts"])
    scene_path = Path(output_dir) / f"{Path(scene_input).stem}_rig.mb"
    cmds.file(rename=str(scene_path))
    cmds.file(save=True, type="mayaBinary", force=True)

    np.savez(
        report,
        **frames,
        n_joints=len(result.rig.joints),
        max_influences=cmds.getAttr(f"{result.rig.skin_cluster}.maxInfluences"),
        model_path=str(result.model_path),
        compressed_path=str(result.compressed_path),
        scene_path=str(scene_path),
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4]))
    maya.standalone.uninitialize()
