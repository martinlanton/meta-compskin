"""Maya-side rig builder for compressed skinning output.

This module runs inside Maya (or ``mayapy``) and turns the NPZ file written by
:meth:`SkinCompressor.run` into a working Maya rig on the selected mesh:

1. The selected mesh is duplicated (the original, and any blendShape node on
   it, is left untouched so the two can be compared side by side), or, with
   ``duplicate_mesh=False``, skinned in place provided it has no deformers.
2. One joint per virtual bone is created under a single root. Joints go to
   the ``restXform`` matrices when the compressor was given joint matrices,
   and otherwise to the weight-averaged position of the vertices they
   influence. Each joint sits under its own driver transform: a Maya joint's
   matrix is scale, rotate, translate only, so it cannot hold the shear that
   the compressed transforms carry (see the user guide, section 3.5). The
   driver, a plain transform, can.
3. The duplicate is bound with a classic linear skin cluster and the
   compressor's weights are applied verbatim.
4. Every driver is keyed so that frame 0 is the neutral pose and frame
   ``k + 1`` shows blendshape ``k`` at full weight, with stepped tangents.

It only requires numpy inside Maya. The maths (centring offset, Equation 7
mixing rule, bind-pose compensation) lives in torch-free functions that are
unit-tested outside Maya; see :doc:`../user_guide/maya_rig_workflow` for the
rigging background.

Example:
    Inside Maya, with the head mesh selected::

        from metacompskin.maya_rig_builder import build_skinned_rig

        rig = build_skinned_rig("D:/exports/head_compressed.npz")
        print(rig.joints)  # ['compskin_joint_000', ...]

    Shape names come from the exporter's model file, if you want the current
    shape shown on the root's ``currentShape`` attribute::

        import numpy as np

        names = list(np.load("D:/exports/head.npz")["shape_names"])
        build_skinned_rig("D:/exports/head_compressed.npz", shape_names=names)
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import numpy.typing as npt

_REQUIRED_KEYS = ("rest", "quads", "weights", "restXform", "shapeXform")

# Relative tolerance (fraction of the mesh extent) when checking that the
# scene mesh matches the compressed rest pose. Loose enough for float32
# round-trips through Maya, tight enough to catch a re-ordered mesh.
_REST_MATCH_TOLERANCE = 1e-4


@dataclass(frozen=True)
class CompressedSkin:
    """In-memory view of the archive written by :meth:`SkinCompressor.run`.

    Attributes:
        rest: Centred rest vertices, shape (N, 3).
        quads: Face indices, shape (F, verts_per_face).
        weights: Skinning weights, shape (N, P). Rows sum to one.
        rest_xform: Joint rest transforms, shape (P, 3, 4).
        shape_xform: Per-shape joint deltas, shape (3S, 4P).
    """

    rest: npt.NDArray[np.floating]
    quads: npt.NDArray[np.integer]
    weights: npt.NDArray[np.floating]
    rest_xform: npt.NDArray[np.floating]
    shape_xform: npt.NDArray[np.floating]

    @classmethod
    def from_npz(cls, path: str | Path) -> "CompressedSkin":
        """Loads a compressed skinning archive.

        Args:
            path: Path to the NPZ file produced by the compressor.

        Returns:
            The loaded archive.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If a required key is missing.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Compressed skin file not found: {path}")
        archive = np.load(path)
        missing = [key for key in _REQUIRED_KEYS if key not in archive.files]
        if missing:
            raise ValueError(
                f"{path} is missing required key(s) {missing}; "
                f"expected {list(_REQUIRED_KEYS)}"
            )
        return cls(
            rest=archive["rest"],
            quads=archive["quads"],
            weights=archive["weights"],
            rest_xform=archive["restXform"],
            shape_xform=archive["shapeXform"],
        )

    @property
    def n_vertices(self) -> int:
        """Number of vertices N."""
        return int(self.rest.shape[0])

    @property
    def n_bones(self) -> int:
        """Number of virtual bones P."""
        return int(self.weights.shape[1])

    @property
    def n_shapes(self) -> int:
        """Number of blendshapes S."""
        return int(self.shape_xform.shape[0] // 3)

    @property
    def max_influences(self) -> int:
        """Largest number of non-zero weights on any single vertex (K)."""
        return int((self.weights != 0).sum(axis=1).max())

    @property
    def has_joint_placement(self) -> bool:
        """Whether ``rest_xform`` holds user-supplied joint matrices.

        The compressor writes identity matrices unless it was given
        ``rest_joint_matrices``, so any non-identity block means the joints
        have a chosen placement.
        """
        return not np.allclose(self.rest_xform, np.eye(3, 4))

    def delta_transforms(self) -> npt.NDArray[np.floating]:
        """Splits ``shape_xform`` into one 3x4 delta block per (shape, joint).

        Returns:
            Delta transforms :math:`N_{k,j}`, shape (S, P, 3, 4).

        References:
            Paper Section 3, Equation 7.
        """
        blocks = self.shape_xform.reshape(self.n_shapes, 3, self.n_bones, 4)
        return blocks.transpose(0, 2, 1, 3)

    def bind_matrices(self) -> npt.NDArray[np.float64]:
        """Promotes ``rest_xform`` to homogeneous 4x4 joint bind matrices.

        Returns:
            Bind matrices, shape (P, 4, 4), float64.
        """
        bind = np.tile(np.eye(4), (self.n_bones, 1, 1))
        bind[:, :3, :] = self.rest_xform
        return bind


def rest_to_scene_matrix(
    mesh_points: npt.NDArray[np.floating],
    rest: npt.NDArray[np.floating],
    mesh_world_matrix: npt.NDArray[np.floating],
) -> npt.NDArray[np.float64]:
    r"""Computes the matrix mapping the compressor's centred space to the scene.

    The compressor subtracts the mean vertex position before solving, so
    ``rest`` sits on the origin while the scene mesh usually does not. This
    recovers that offset from the mesh and folds in the mesh's own world
    matrix, so the result maps centred rest coordinates to world space.

    Args:
        mesh_points: Object-space vertex positions of the scene mesh, shape (N, 3).
        rest: Centred rest vertices from the archive, shape (N, 3).
        mesh_world_matrix: World matrix of the mesh transform, shape (4, 4),
            column-vector convention (translation in the last column).

    Returns:
        Matrix :math:`G \cdot T(c)`, shape (4, 4), where ``G`` is the mesh
        world matrix and ``c`` the centring offset.

    Raises:
        ValueError: If the vertex count differs, or the mesh does not match the
            rest pose up to a translation (typically a different vertex order).
    """
    if mesh_points.shape != rest.shape:
        raise ValueError(
            f"Scene mesh vertex count {mesh_points.shape[0]} does not match the "
            f"compressed rest pose ({rest.shape[0]} vertices)."
        )
    offset = mesh_points.mean(axis=0)
    extent = float(np.ptp(rest, axis=0).max())
    mismatch = np.abs((mesh_points - offset) - rest).max()
    if mismatch > _REST_MATCH_TOLERANCE * max(extent, 1.0):
        raise ValueError(
            "Scene mesh does not match the compressed rest pose (max deviation "
            f"{mismatch:.4g} after centring). Check the vertex order and that "
            "the mesh is the one that was compressed."
        )
    centring = np.eye(4)
    centring[:3, 3] = offset
    return np.asarray(mesh_world_matrix, dtype=np.float64) @ centring


def frame_joint_matrices(
    delta_transforms: npt.NDArray[np.floating],
    bind_matrices: npt.NDArray[np.floating],
    rest_to_scene: npt.NDArray[np.floating],
) -> npt.NDArray[np.float64]:
    r"""Computes the joint world matrices for the neutral frame and every shape.

    For shape ``k`` the mixing rule (Equation 7) with a one-hot weight gives
    :math:`M_j = I + N_{k,j}`. A Maya skin cluster applies
    ``worldMatrix[j] · bindPreMatrix[j]`` to world-space points, so the joint
    must be placed at

    .. math::

        W_j = R \, M_j \, R^{-1} \, B_j

    where ``R`` is the rest-to-scene matrix and ``B_j`` the joint's bind matrix.

    Args:
        delta_transforms: Delta transforms :math:`N_{k,j}`, shape (S, P, 3, 4).
        bind_matrices: Joint bind matrices, shape (P, 4, 4).
        rest_to_scene: Matrix from centred rest space to world space, shape (4, 4).

    Returns:
        Joint world matrices, shape (S + 1, P, 4, 4), column-vector
        convention. Index 0 is the bind pose; index ``k + 1`` is shape ``k``.

    References:
        Paper Section 3, Equation 7; user guide section 3.6 (rest pose and
        coordinate space).
    """
    n_shapes, n_bones = delta_transforms.shape[:2]
    mixed = np.tile(np.eye(4), (n_shapes + 1, n_bones, 1, 1))
    mixed[1:, :, :3, :] += delta_transforms
    scene_to_rest = np.linalg.inv(rest_to_scene)
    return rest_to_scene @ mixed @ scene_to_rest @ bind_matrices


def weighted_joint_centroids(
    weights: npt.NDArray[np.floating], points: npt.NDArray[np.floating]
) -> npt.NDArray[np.float64]:
    """Computes, per joint, the weight-averaged position of its vertices.

    Joints with no influence at all (an all-zero weight column) are placed at
    the mean of all points so they still get a sensible display position.

    Args:
        weights: Skinning weights, shape (N, P).
        points: Vertex positions, shape (N, 3).

    Returns:
        Centroid per joint, shape (P, 3), float64.
    """
    weights = np.asarray(weights, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)
    totals = weights.sum(axis=0)  # (P,)
    used = totals > 0
    centroids = np.tile(points.mean(axis=0), (weights.shape[1], 1))
    centroids[used] = (weights[:, used].T @ points) / totals[used, None]
    return centroids


def joint_bind_matrices(
    skin: CompressedSkin, rest_to_scene: npt.NDArray[np.floating]
) -> npt.NDArray[np.float64]:
    """Chooses where the joints sit in the neutral pose.

    If the archive carries user-supplied joint matrices they are used as-is.
    Otherwise each joint is placed, with identity orientation, at the
    weight-averaged scene-space position of the vertices it influences, so it
    visually sits over the region it drives (user guide section 3.1).

    Args:
        skin: The compressed archive.
        rest_to_scene: Matrix from centred rest space to world space, shape (4, 4).

    Returns:
        Joint bind matrices, shape (P, 4, 4), column-vector convention.
    """
    if skin.has_joint_placement:
        return skin.bind_matrices()
    rest_homog = np.c_[skin.rest, np.ones(skin.n_vertices)]  # (N, 4)
    scene_points = (rest_homog @ np.asarray(rest_to_scene).T)[:, :3]
    bind = np.tile(np.eye(4), (skin.n_bones, 1, 1))
    bind[:, :3, 3] = weighted_joint_centroids(skin.weights, scene_points)
    return bind


def default_shape_names(n_shapes: int) -> list[str]:
    """Builds placeholder shape names when the archive carries none.

    Args:
        n_shapes: Number of blendshapes S.

    Returns:
        Names ``shape_000`` ... ``shape_{S-1:03d}``.
    """
    return [f"shape_{index:03d}" for index in range(n_shapes)]


def validate_shape_names(names: Sequence[str] | None, n_shapes: int) -> list[str]:
    """Checks user-supplied shape names against the archive, or generates them.

    Args:
        names: Ordered blendshape names, or None to use placeholders.
        n_shapes: Number of blendshapes S in the archive.

    Returns:
        A list of exactly ``n_shapes`` names.

    Raises:
        ValueError: If ``names`` is given but its length differs from ``n_shapes``.
    """
    if names is None:
        return default_shape_names(n_shapes)
    if len(names) != n_shapes:
        raise ValueError(
            f"Got {len(names)} shape names but the archive holds {n_shapes} shapes."
        )
    return [str(name) for name in names]


@dataclass(frozen=True)
class MayaSkinRig:
    """Names of the nodes created by :func:`build_skinned_rig`.

    Attributes:
        mesh: Transform of the skinned mesh (the duplicate, or the source
            mesh itself when ``duplicate_mesh`` was False).
        root: Root joint that parents every driver.
        joints: Virtual joint names, in the order of the weight columns.
        drivers: Animated transform above each joint, same order as ``joints``.
        skin_cluster: The skinCluster node bound to ``mesh``.
        shape_names: Blendshape names, in frame order (frame ``k + 1``).
    """

    mesh: str
    root: str
    joints: list[str]
    drivers: list[str]
    skin_cluster: str
    shape_names: list[str]


def build_skinned_rig(
    compressed_path: str | Path,
    mesh: str | None = None,
    shape_names: Sequence[str] | None = None,
    duplicate_mesh: bool = True,
    name: str = "compskin",
) -> MayaSkinRig:
    """Builds joints, skin cluster and per-shape animation from a compressed file.

    The mesh (selected, or named) must be the neutral pose of the mesh that was
    compressed, with the same vertex order. By default it is duplicated and the
    duplicate is skinned; the original and any deformer on it are left
    untouched. With ``duplicate_mesh=False`` the mesh itself is skinned, which
    is only allowed when it carries no deformer (blendShape, skinCluster or any
    other): otherwise the face would be deformed twice, so the call aborts
    before creating anything.

    Joints are placed at the ``restXform`` matrices when the compressor was
    given joint matrices. Otherwise each joint sits, with identity orientation,
    at the weight-averaged position of the vertices it influences.

    Frame 0 of the resulting animation is the neutral pose and frame ``k + 1``
    shows blendshape ``k`` at full weight. Keys use stepped tangents so no two
    shapes blend between frames. The root joint gets a keyed ``currentShape``
    enum attribute naming the shape shown on each frame.

    The animation is keyed on a driver transform above each joint, not on the
    joint itself, because a Maya joint drops the shear component of the
    compressed transforms. The joints stay at the identity below their drivers
    and are the skin cluster's influences.

    Args:
        compressed_path: NPZ file written by :meth:`SkinCompressor.run`.
        mesh: Transform or mesh shape to skin. Defaults to the current selection.
        shape_names: Ordered blendshape names (for the ``currentShape``
            attribute). Defaults to ``shape_000``, ``shape_001``, ...
        duplicate_mesh: Skin a duplicate of the mesh (True, default) or the
            mesh itself (False; it must have no deformers).
        name: Prefix for every node created.

    Returns:
        The names of the created nodes.

    Raises:
        RuntimeError: If not running inside Maya or mayapy.
        ValueError: If the selection is not a single mesh, the mesh does not
            match the compressed rest pose, ``shape_names`` has the wrong
            length, or ``duplicate_mesh`` is False and the mesh already has
            deformers.

    References:
        Paper Section 3, Equation 7; user guide "Maya rig workflow" section 4.
    """
    api = _import_maya()
    cmds = api.cmds
    skin = CompressedSkin.from_npz(compressed_path)
    names = validate_shape_names(shape_names, skin.n_shapes)

    source = _resolve_mesh_transform(api, mesh)
    if not duplicate_mesh:
        _ensure_no_deformers(api, source)
    rest_to_scene = rest_to_scene_matrix(
        _object_space_points(api, source), skin.rest, _world_matrix(api, source)
    )
    target = _duplicate_mesh(api, source, name) if duplicate_mesh else source

    bind = joint_bind_matrices(skin, rest_to_scene)
    root = cmds.createNode("joint", name=f"{name}_root")
    drivers, joints = _create_joints(api, root, bind, name)
    skin_cluster = _bind(api, target, joints, skin, name)

    frames = frame_joint_matrices(skin.delta_transforms(), bind, rest_to_scene)
    _key_driver_frames(api, drivers, frames)
    _add_current_shape_attribute(api, root, names)
    cmds.playbackOptions(
        animationStartTime=0,
        minTime=0,
        maxTime=skin.n_shapes,
        animationEndTime=skin.n_shapes,
    )
    cmds.currentTime(0)
    cmds.select(target)

    return MayaSkinRig(
        mesh=target,
        root=root,
        joints=joints,
        drivers=drivers,
        skin_cluster=skin_cluster,
        shape_names=names,
    )


class _MayaApi(NamedTuple):
    """The Maya modules the builder needs, imported lazily."""

    cmds: Any
    om: Any
    oma: Any


def _import_maya() -> _MayaApi:
    """Imports maya.cmds and the OpenMaya 2.0 API modules.

    Returns:
        The imported modules bundled as a :class:`_MayaApi`.

    Raises:
        RuntimeError: If the Maya modules are not importable.
    """
    try:
        from maya import cmds  # noqa: PLC0415
        from maya.api import OpenMaya, OpenMayaAnim  # noqa: PLC0415
    except ImportError as e:
        raise RuntimeError(
            "Maya modules are not importable. build_skinned_rig must run inside "
            "Maya or mayapy."
        ) from e
    return _MayaApi(cmds, OpenMaya, OpenMayaAnim)


def _resolve_mesh_transform(api: _MayaApi, mesh: str | None) -> str:
    """Returns the transform of the mesh to skin, from a name or the selection.

    Args:
        api: The Maya modules.
        mesh: Transform or mesh shape name, or None for the current selection.

    Returns:
        Full path of the mesh transform.

    Raises:
        ValueError: If nothing (or more than one node) is selected, the node
            does not exist, or it carries no polygon mesh.
    """
    cmds = api.cmds
    if mesh is None:
        selected = cmds.ls(selection=True, long=True)
        if len(selected) != 1:
            raise ValueError(
                f"Select exactly one mesh before building the rig (got {len(selected)})."
            )
        mesh = selected[0]
    elif not cmds.objExists(mesh):
        raise ValueError(f"Mesh does not exist in the scene: '{mesh}'")

    if cmds.nodeType(mesh) == "mesh":
        return cmds.listRelatives(mesh, parent=True, fullPath=True)[0]
    shapes = cmds.listRelatives(mesh, shapes=True, noIntermediate=True, type="mesh")
    if not shapes:
        raise ValueError(f"'{mesh}' is not a polygon mesh.")
    return cmds.ls(mesh, long=True)[0]


def _ensure_no_deformers(api: _MayaApi, transform: str) -> None:
    """Refuses a mesh that already has deformers in its history.

    Args:
        api: The Maya modules.
        transform: Mesh transform name.

    Raises:
        ValueError: If any deformer (``geometryFilter`` node) drives the mesh.
    """
    cmds = api.cmds
    deformers = cmds.ls(cmds.listHistory(transform), type="geometryFilter")
    if deformers:
        raise ValueError(
            f"'{transform}' already has deformers {deformers}. Skinning it in "
            "place would deform it twice; remove them first, or call with "
            "duplicate_mesh=True to skin a clean copy instead."
        )


def _duplicate_mesh(api: _MayaApi, source: str, name: str) -> str:
    """Duplicates a mesh without its construction history.

    Args:
        api: The Maya modules.
        source: Mesh transform to copy.
        name: Node name prefix.

    Returns:
        The duplicate's transform name.
    """
    cmds = api.cmds
    duplicate = cmds.duplicate(source, name=f"{name}_skinned")[0]
    cmds.delete(duplicate, constructionHistory=True)
    return duplicate


def _object_space_points(api: _MayaApi, transform: str) -> npt.NDArray[np.float64]:
    """Reads the object-space vertex positions of a mesh transform.

    Args:
        api: The Maya modules.
        transform: Mesh transform name.

    Returns:
        Vertex positions, shape (N, 3), float64.
    """
    om = api.om
    selection = om.MSelectionList()
    selection.add(transform)
    dag = selection.getDagPath(0)
    dag.extendToShape()
    points = om.MFnMesh(dag).getPoints(om.MSpace.kObject)
    return np.array([[p.x, p.y, p.z] for p in points], dtype=np.float64)


def _world_matrix(api: _MayaApi, transform: str) -> npt.NDArray[np.float64]:
    """Reads a transform's world matrix in column-vector convention.

    Args:
        api: The Maya modules.
        transform: Transform name.

    Returns:
        World matrix, shape (4, 4), translation in the last column.
    """
    flat = api.cmds.xform(transform, query=True, worldSpace=True, matrix=True)
    return np.array(flat, dtype=np.float64).reshape(4, 4).T


def _to_maya_matrix(matrix: npt.NDArray[np.floating]) -> list[float]:
    """Flattens a column-vector 4x4 matrix into Maya's row-vector layout.

    Args:
        matrix: Matrix in column-vector convention, shape (4, 4).

    Returns:
        16 floats, row-major, translation in the last row.
    """
    return [float(value) for value in np.asarray(matrix).T.ravel()]


def _create_joints(
    api: _MayaApi, root: str, bind_matrices: npt.NDArray[np.floating], name: str
) -> tuple[list[str], list[str]]:
    """Creates a driver transform and a joint per bone under ``root``.

    The driver is placed at the bone's bind matrix; the joint below it stays
    at the identity so its world matrix is the driver's.

    Args:
        api: The Maya modules.
        root: Parent for every driver. Must sit at the identity.
        bind_matrices: Joint world matrices, shape (P, 4, 4).
        name: Node name prefix.

    Returns:
        Tuple of (driver names, joint names), both in bone order.
    """
    cmds = api.cmds
    drivers = []
    joints = []
    for index, bind in enumerate(bind_matrices):
        driver = cmds.createNode(
            "transform", name=f"{name}_joint_{index:03d}_driver", parent=root
        )
        _set_local_matrix(api, _transform_fn(api, driver), bind)
        joint = cmds.createNode(
            "joint", name=f"{name}_joint_{index:03d}", parent=driver
        )
        cmds.setAttr(f"{joint}.segmentScaleCompensate", 0)
        drivers.append(driver)
        joints.append(joint)
    return drivers, joints


def _transform_fn(api: _MayaApi, node: str) -> Any:
    """Attaches an ``MFnTransform`` function set to a transform or joint.

    Args:
        api: The Maya modules.
        node: Transform or joint name.

    Returns:
        An ``MFnTransform`` bound to the node.
    """
    selection = api.om.MSelectionList()
    selection.add(node)
    return api.om.MFnTransform(selection.getDagPath(0))


def _set_local_matrix(
    api: _MayaApi, fn_transform: Any, matrix: npt.NDArray[np.floating]
) -> None:
    """Sets a node's local transform from a full affine matrix, shear included.

    ``xform -matrix`` discards shear, which the compressed transforms carry
    (user guide section 3.5), so the matrix is decomposed through
    ``MTransformationMatrix`` instead. The root created by
    :func:`build_skinned_rig` sits at the identity, so for the drivers the
    local matrix equals the world matrix.

    Args:
        api: The Maya modules.
        fn_transform: ``MFnTransform`` bound to the node.
        matrix: Column-vector 4x4 matrix, shape (4, 4).
    """
    om = api.om
    fn_transform.setTransformation(
        om.MTransformationMatrix(om.MMatrix(_to_maya_matrix(matrix)))
    )


def _bind(
    api: _MayaApi,
    mesh: str,
    joints: list[str],
    skin: CompressedSkin,
    name: str,
) -> str:
    """Creates a linear skin cluster and applies the compressed weights verbatim.

    Args:
        api: The Maya modules.
        mesh: Transform of the mesh to bind.
        joints: Joint names, in weight-column order.
        skin: The compressed archive.
        name: Node name prefix.

    Returns:
        The skinCluster node name.
    """
    skin_cluster = api.cmds.skinCluster(
        joints,
        mesh,
        name=f"{name}_skinCluster",
        toSelectedBones=True,
        skinMethod=0,  # classic linear
        normalizeWeights=0,  # none: the compressed weights are used as-is
        maximumInfluences=skin.max_influences,
        obeyMaxInfluences=True,
    )[0]
    _set_skin_weights(api, skin_cluster, mesh, joints, skin.weights)
    return skin_cluster


def _set_skin_weights(
    api: _MayaApi,
    skin_cluster: str,
    mesh: str,
    joints: list[str],
    weights: npt.NDArray[np.floating],
) -> None:
    """Writes a dense (N, P) weight map onto a skinCluster through the API.

    Args:
        api: The Maya modules.
        skin_cluster: skinCluster node name.
        mesh: Transform of the bound mesh.
        joints: Joint names, in weight-column order.
        weights: Skinning weights, shape (N, P).
    """
    om = api.om
    selection = om.MSelectionList()
    selection.add(skin_cluster)
    selection.add(mesh)
    for joint in joints:
        selection.add(joint)
    fn_skin = api.oma.MFnSkinCluster(selection.getDependNode(0))
    mesh_dag = selection.getDagPath(1)
    mesh_dag.extendToShape()

    influence_indices = om.MIntArray(
        [
            fn_skin.indexForInfluenceObject(selection.getDagPath(2 + column))
            for column in range(len(joints))
        ]
    )
    components = om.MFnSingleIndexedComponent()
    vertex_component = components.create(om.MFn.kMeshVertComponent)
    components.setCompleteData(weights.shape[0])

    fn_skin.setWeights(
        mesh_dag,
        vertex_component,
        influence_indices,
        om.MDoubleArray(weights.astype(np.float64).ravel().tolist()),
        False,  # normalize
    )


def _key_driver_frames(
    api: _MayaApi, drivers: list[str], frames: npt.NDArray[np.floating]
) -> None:
    """Keys every driver's transform channels on every frame, stepped.

    Args:
        api: The Maya modules.
        drivers: Driver transform names, in bone order.
        frames: Joint world matrices, shape (S + 1, P, 4, 4).
    """
    cmds = api.cmds
    channels = ["translate", "rotate", "scale", "shear"]
    transforms = [_transform_fn(api, driver) for driver in drivers]
    for frame, joint_matrices in enumerate(frames):
        for driver, fn_transform, matrix in zip(
            drivers, transforms, joint_matrices, strict=True
        ):
            _set_local_matrix(api, fn_transform, matrix)
            cmds.setKeyframe(
                driver,
                attribute=channels,
                time=frame,
                inTangentType="linear",
                outTangentType="step",
            )


def _add_current_shape_attribute(
    api: _MayaApi, root: str, shape_names: list[str]
) -> None:
    """Adds a keyed enum attribute on the root naming the shape on each frame.

    Args:
        api: The Maya modules.
        root: Root joint name.
        shape_names: Blendshape names, in frame order.
    """
    cmds = api.cmds
    labels = ["rest", *(_enum_safe(name) for name in shape_names)]
    cmds.addAttr(
        root,
        longName="currentShape",
        attributeType="enum",
        enumName=":".join(labels),
        keyable=True,
    )
    for frame in range(len(labels)):
        cmds.setKeyframe(
            root,
            attribute="currentShape",
            time=frame,
            value=frame,
            inTangentType="linear",
            outTangentType="step",
        )


def _enum_safe(name: str) -> str:
    """Replaces characters that Maya enum field names cannot contain.

    Args:
        name: A blendshape name.

    Returns:
        The name with ``:`` and ``=`` replaced by underscores.
    """
    return name.replace(":", "_").replace("=", "_")
