"""Scene helpers shared by the mayapy test scripts. Import after initialising Maya."""

import numpy as np
from maya import cmds
from maya.api import OpenMaya

om = OpenMaya

# Deliberately non-trivial placement so the scenarios exercise the world matrix.
SOURCE_TRANSLATION = (10.0, 5.0, -3.0)
SOURCE_ROTATION = (0.0, 30.0, 0.0)
SOURCE_NAME = "head_GEO"


def create_mesh(rest_verts: np.ndarray, faces: np.ndarray, name: str) -> str:
    points = om.MPointArray([om.MPoint(*map(float, p)) for p in rest_verts])
    counts = om.MIntArray([len(face) for face in faces])
    connects = om.MIntArray([int(i) for face in faces for i in face])
    transform = om.MFnMesh().create(points, counts, connects)
    return cmds.rename(om.MFnDagNode(transform).fullPathName(), name)


def world_points(transform: str) -> list[list[float]]:
    selection = om.MSelectionList()
    selection.add(transform)
    dag = selection.getDagPath(0)
    dag.extendToShape()
    points = om.MFnMesh(dag).getPoints(om.MSpace.kWorld)
    return [[p.x, p.y, p.z] for p in points]


def world_matrix(node: str) -> list[float]:
    return cmds.xform(node, q=True, ws=True, matrix=True)


def skin_clusters_on(mesh: str) -> list[str]:
    return cmds.ls(cmds.listHistory(mesh), type="skinCluster")


def new_scene_with_source(scene) -> str:
    cmds.file(new=True, force=True)
    source = create_mesh(scene["rest_verts"], scene["faces"], SOURCE_NAME)
    cmds.xform(source, translation=SOURCE_TRANSLATION, rotation=SOURCE_ROTATION)
    cmds.select(source)
    return source


def sample_rig(rig, n_frames: int) -> tuple[list, list]:
    """World points of the rig mesh and the current shape label, per frame."""
    frames = []
    current_shape = []
    for frame in range(n_frames):
        cmds.currentTime(frame)
        frames.append(world_points(rig.mesh))
        current_shape.append(cmds.getAttr(f"{rig.root}.currentShape", asString=True))
    cmds.currentTime(0)
    return frames, current_shape
