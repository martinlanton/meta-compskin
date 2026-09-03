"""Maya-side exporter for blendshape model data.

This module runs inside Maya (or mayapy) and writes the NPZ file consumed by
:meth:`BlendshapeModelData.from_npz`, so that compression can run outside Maya
in an environment where PyTorch is available. It only requires numpy, which
ships with recent Maya versions.

Two capture modes are supported:

1. **BlendShape node mode** (default): the exporter finds the ``blendShape``
   deformer on the rest mesh (or uses the one you name), toggles each target
   weight to 1 in turn, and records the resulting deltas.
2. **Target mesh mode**: you provide a list of separate target meshes in the
   scene; deltas are computed as ``target_points - rest_points``.

Example:
    Inside Maya / mayapy::

        from metacompskin.maya_exporter import MayaBlendshapeExporter

        # BlendShape node mode (auto-discovers the deformer):
        exporter = MayaBlendshapeExporter("head_GEO")
        exporter.export("D:/exports/head.npz")

        # Target mesh mode, including skeleton joints:
        exporter = MayaBlendshapeExporter(
            "head_GEO",
            target_meshes=["smile_GEO", "frown_GEO"],
            joints=["jaw_JNT", "cheek_L_JNT"],
        )
        exporter.export("D:/exports/head.npz")

    Then outside Maya::

        from metacompskin.model_data import BlendshapeModelData
        from metacompskin.model_fit import SkinCompressor

        model_data = BlendshapeModelData.from_npz("D:/exports/head.npz")
        SkinCompressor(model_data=model_data).run("D:/exports/head_compressed.npz")
"""

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from metacompskin.model_data import BlendshapeModelData


def _import_maya() -> tuple[Any, Any]:
    """Import maya.cmds and the OpenMaya 2.0 API.

    Returns:
        Tuple of (cmds, OpenMaya) modules.

    Raises:
        RuntimeError: If the Maya modules are not importable (i.e. not running
            inside Maya or mayapy).
    """
    try:
        from maya import cmds  # noqa: PLC0415
        from maya.api import OpenMaya  # noqa: PLC0415
    except ImportError as e:
        raise RuntimeError(
            "Maya modules are not importable. MayaBlendshapeExporter must run "
            "inside Maya or mayapy."
        ) from e
    return cmds, OpenMaya


class MayaBlendshapeExporter:
    """Exports blendshape model data from the current Maya scene to an NPZ file.

    The written NPZ contains every key required by
    :meth:`BlendshapeModelData.from_npz` (``deltas``, ``rest_verts``,
    ``rest_faces``, ``inbetween_info``, ``combination_info``) plus:

    - ``shape_names``: target names, aligned with the first axis of ``deltas``.
    - ``rest_joint_matrices``: world-space 4x4 joint matrices (only when
      ``joints`` is provided). Suitable for ``SkinCompressor``'s
      ``rest_joint_matrices`` argument; translation is stored in the last
      column (the matrices are transposed from Maya's row-vector convention).

    Points are read in object space, so targets that have been translated
    aside for layout in the scene still produce correct deltas.

    Attributes:
        rest_mesh: Name of the neutral/rest mesh (transform or shape).
        blendshape_node: Explicit blendShape node name, or None to
            auto-discover it from the rest mesh's history.
        target_meshes: Optional list of separate target mesh names. Mutually
            exclusive with blendshape_node.
        joints: Optional list of joint (or transform) names whose world
            matrices are exported alongside the model data.

    Note:
        Inbetween targets are not exported individually: each blendShape
        target is captured at full weight (1.0), and ``inbetween_info`` /
        ``combination_info`` are written as empty dicts.
    """

    def __init__(
        self,
        rest_mesh: str,
        blendshape_node: str | None = None,
        target_meshes: list[str] | None = None,
        joints: list[str] | None = None,
    ):
        """Initializes the exporter and validates the rest mesh.

        Args:
            rest_mesh: Name of the neutral mesh (transform or shape node).
            blendshape_node: Optional explicit blendShape node. If None and
                target_meshes is also None, the node is auto-discovered from
                the rest mesh's deformation history at export time.
            target_meshes: Optional list of target mesh names to diff against
                the rest mesh instead of using a blendShape node.
            joints: Optional list of joint names whose world-space rest
                matrices are exported as ``rest_joint_matrices``.

        Raises:
            ValueError: If both blendshape_node and target_meshes are given,
                or if the rest mesh does not exist / has no mesh shape.
            RuntimeError: If not running inside Maya or mayapy.
        """
        self._cmds, self._om = _import_maya()

        if blendshape_node is not None and target_meshes is not None:
            raise ValueError(
                "Provide either blendshape_node or target_meshes, not both."
            )

        self.rest_mesh = rest_mesh
        self.blendshape_node = blendshape_node
        self.target_meshes = list(target_meshes) if target_meshes else None
        self.joints = list(joints) if joints else None
        self._rest_shape = self._resolve_mesh_shape(rest_mesh)

    def export(self, output_path: str | Path) -> Path:
        """Captures the model data and writes it to an NPZ file.

        Args:
            output_path: Destination file path. A ``.npz`` suffix is appended
                if missing; parent directories are created as needed.

        Returns:
            Path of the written NPZ file.

        Raises:
            ValueError: If captured arrays fail model-data validation (e.g.
                inconsistent topology between rest mesh and targets).
            RuntimeError: If blendShape weights cannot be controlled (locked
                or incoming connections), or no blendShape node is found.
        """
        if self.target_meshes is not None:
            rest_verts, shape_names, deltas = self._capture_from_target_meshes(
                self.target_meshes
            )
        else:
            rest_verts, shape_names, deltas = self._capture_from_blendshape()

        rest_faces = self._mesh_faces(self._rest_shape)

        # Reuse the loader-side validation so the file is guaranteed to
        # round-trip through BlendshapeModelData.from_npz.
        BlendshapeModelData._validate_arrays(deltas, rest_verts, rest_faces, {}, {})

        data: dict[str, Any] = {
            "deltas": deltas,
            "rest_verts": rest_verts,
            "rest_faces": rest_faces,
            "inbetween_info": np.array({}, dtype=object),
            "combination_info": np.array({}, dtype=object),
            "shape_names": np.array(shape_names),
        }
        if self.joints:
            data["rest_joint_matrices"] = self._joint_matrices(self.joints)

        output_path = Path(output_path)
        if output_path.suffix != ".npz":
            output_path = output_path.with_name(output_path.name + ".npz")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(str(output_path), **data)
        return output_path

    def _resolve_mesh_shape(self, node: str) -> str:
        """Resolves a transform or shape name to a non-intermediate mesh shape.

        Args:
            node: Transform or mesh shape node name.

        Returns:
            Name of the mesh shape node.

        Raises:
            ValueError: If the node does not exist or has no mesh shape.
        """
        if not self._cmds.objExists(node):
            raise ValueError(f"Node does not exist in the scene: '{node}'")

        if self._cmds.nodeType(node) == "mesh":
            return node

        shapes = (
            self._cmds.listRelatives(
                node, shapes=True, noIntermediate=True, type="mesh", fullPath=True
            )
            or []
        )
        if not shapes:
            raise ValueError(f"No mesh shape found under node: '{node}'")
        return shapes[0]

    def _mesh_points(self, shape: str) -> npt.NDArray[np.float32]:
        """Reads object-space vertex positions via the OpenMaya API.

        Args:
            shape: Mesh shape node name.

        Returns:
            Vertex positions, shape (N, 3), float32.
        """
        sel = self._om.MSelectionList()
        sel.add(shape)
        fn_mesh = self._om.MFnMesh(sel.getDagPath(0))
        points = fn_mesh.getPoints(self._om.MSpace.kObject)
        return np.array([[p.x, p.y, p.z] for p in points], dtype=np.float32)

    def _mesh_faces(self, shape: str) -> npt.NDArray[np.int32]:
        """Reads face connectivity via the OpenMaya API.

        Args:
            shape: Mesh shape node name.

        Returns:
            Face indices, shape (F, verts_per_face), int32.

        Raises:
            ValueError: If the mesh mixes face sizes or uses faces that are
                neither triangles nor quads.
        """
        sel = self._om.MSelectionList()
        sel.add(shape)
        fn_mesh = self._om.MFnMesh(sel.getDagPath(0))
        counts, connects = fn_mesh.getVertices()

        unique_counts = set(counts)
        if len(unique_counts) > 1:
            raise ValueError(
                f"Mesh '{shape}' mixes face sizes {sorted(unique_counts)}. "
                "All faces must be triangles or all quads; triangulate or "
                "quadrangulate the mesh before exporting."
            )
        verts_per_face = unique_counts.pop()
        if verts_per_face not in (3, 4):
            raise ValueError(
                f"Mesh '{shape}' has {verts_per_face}-sided faces; only "
                "triangles (3) and quads (4) are supported."
            )
        return np.array(connects, dtype=np.int32).reshape(-1, verts_per_face)

    def _discover_blendshape(self) -> str:
        """Finds the blendShape node deforming the rest mesh.

        Returns:
            Name of the blendShape node.

        Raises:
            RuntimeError: If no blendShape node is found, or several are and
                none was specified explicitly.
        """
        history = self._cmds.listHistory(self._rest_shape, pruneDagObjects=True) or []
        blendshapes = self._cmds.ls(history, type="blendShape")
        if not blendshapes:
            raise RuntimeError(
                f"No blendShape node found in the history of '{self.rest_mesh}'. "
                "Pass blendshape_node or target_meshes explicitly."
            )
        if len(blendshapes) > 1:
            raise RuntimeError(
                f"Multiple blendShape nodes found on '{self.rest_mesh}': "
                f"{blendshapes}. Pass blendshape_node explicitly."
            )
        return blendshapes[0]

    def _capture_from_blendshape(
        self,
    ) -> tuple[npt.NDArray[np.float32], list[str], npt.NDArray[np.float32]]:
        """Captures rest points and per-target deltas from a blendShape node.

        Each target weight is set to 1 with all others at 0; the rest pose is
        captured with all weights at 0. Original weights and envelope are
        restored afterwards, even on failure.

        Returns:
            Tuple of (rest_verts (N, 3), shape_names, deltas (S, N, 3)).

        Raises:
            RuntimeError: If the node has no targets, or weights cannot be set
                because they are locked or have incoming connections.
        """
        cmds = self._cmds
        blendshape = self.blendshape_node or self._discover_blendshape()

        indices = cmds.getAttr(f"{blendshape}.weight", multiIndices=True) or []
        if not indices:
            raise RuntimeError(f"blendShape node '{blendshape}' has no targets.")

        blocked = [
            i
            for i in indices
            if not cmds.getAttr(f"{blendshape}.weight[{i}]", settable=True)
        ]
        if blocked:
            raise RuntimeError(
                f"Cannot control weights {blocked} on '{blendshape}': they are "
                "locked or have incoming connections. Disconnect the rig "
                "inputs (or export from an unrigged scene) and retry."
            )

        alias_by_index = self._weight_aliases(blendshape)
        original_weights = {
            i: cmds.getAttr(f"{blendshape}.weight[{i}]") for i in indices
        }
        original_envelope = cmds.getAttr(f"{blendshape}.envelope")

        try:
            cmds.setAttr(f"{blendshape}.envelope", 1.0)
            for i in indices:
                cmds.setAttr(f"{blendshape}.weight[{i}]", 0.0)
            rest_verts = self._mesh_points(self._rest_shape)

            shape_names = []
            deltas = np.zeros((len(indices), rest_verts.shape[0], 3), dtype=np.float32)
            for row, i in enumerate(indices):
                cmds.setAttr(f"{blendshape}.weight[{i}]", 1.0)
                deltas[row] = self._mesh_points(self._rest_shape) - rest_verts
                cmds.setAttr(f"{blendshape}.weight[{i}]", 0.0)
                shape_names.append(alias_by_index.get(i, f"target_{i}"))
        finally:
            for i, weight in original_weights.items():
                cmds.setAttr(f"{blendshape}.weight[{i}]", weight)
            cmds.setAttr(f"{blendshape}.envelope", original_envelope)

        return rest_verts, shape_names, deltas

    def _weight_aliases(self, blendshape: str) -> dict[int, str]:
        """Maps blendShape weight indices to their target alias names.

        Args:
            blendshape: blendShape node name.

        Returns:
            Dict mapping weight index to alias (e.g. {0: "smile"}).
        """
        raw = self._cmds.aliasAttr(blendshape, query=True) or []
        alias_by_index: dict[int, str] = {}
        # aliasAttr returns a flat [alias, "weight[i]", alias, "weight[j]", ...]
        for alias, attr in zip(raw[0::2], raw[1::2], strict=False):
            if attr.startswith("weight["):
                alias_by_index[int(attr[len("weight[") : -1])] = alias
        return alias_by_index

    def _capture_from_target_meshes(
        self,
        target_meshes: list[str],
    ) -> tuple[npt.NDArray[np.float32], list[str], npt.NDArray[np.float32]]:
        """Captures rest points and deltas by diffing separate target meshes.

        Args:
            target_meshes: Names of the target meshes to diff against the
                rest mesh.

        Returns:
            Tuple of (rest_verts (N, 3), shape_names, deltas (S, N, 3)).

        Raises:
            ValueError: If a target mesh does not exist or its vertex count
                differs from the rest mesh.
        """
        rest_verts = self._mesh_points(self._rest_shape)

        shape_names = []
        deltas = np.zeros(
            (len(target_meshes), rest_verts.shape[0], 3), dtype=np.float32
        )
        for row, target in enumerate(target_meshes):
            target_shape = self._resolve_mesh_shape(target)
            target_verts = self._mesh_points(target_shape)
            if target_verts.shape[0] != rest_verts.shape[0]:
                raise ValueError(
                    f"Topology mismatch: target '{target}' has "
                    f"{target_verts.shape[0]} vertices, rest mesh "
                    f"'{self.rest_mesh}' has {rest_verts.shape[0]}. All targets "
                    "must share the rest mesh topology."
                )
            deltas[row] = target_verts - rest_verts
            shape_names.append(target.split("|")[-1].split(":")[-1])

        return rest_verts, shape_names, deltas

    def _joint_matrices(self, joints: list[str]) -> npt.NDArray[np.float64]:
        """Reads world-space rest matrices for the given joints.

        Maya's ``xform -matrix`` uses the row-vector convention (translation
        in the last row); the matrices are transposed so translation sits in
        the last column, matching ``SkinCompressor``'s expectation of standard
        homogeneous transforms.

        Args:
            joints: Joint (or transform) node names.

        Returns:
            World-space matrices, shape (P, 4, 4), float64.

        Raises:
            ValueError: If a joint does not exist.
        """
        matrices = np.zeros((len(joints), 4, 4), dtype=np.float64)
        for row, joint in enumerate(joints):
            if not self._cmds.objExists(joint):
                raise ValueError(f"Joint does not exist in the scene: '{joint}'")
            flat = self._cmds.xform(joint, query=True, worldSpace=True, matrix=True)
            matrices[row] = np.array(flat, dtype=np.float64).reshape(4, 4).T
        return matrices
