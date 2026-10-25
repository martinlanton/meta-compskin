"""Export source blendshape models' neutral poses to Maya scenes.

Writes one Maya binary scene per model holding only neutral-pose geometry: the
head mesh, the mouth interior, both eyes, and locators at whichever rig pivots
the model records. No blendshape targets, deformers or animation are created.
The scenes exist so joints can be authored by hand and fed back into local
tests.

Geometry is written in the model's own local coordinates, the same frame the
solver works in, so a joint placed in the scene lands where the solver expects
it. The model's ``rest_center`` (its original world position) would move the
mesh away from the origin, so it is stored as an attribute on the top group
rather than applied to the transforms.

Run under mayapy, not a standard Python interpreter::

    "C:/Program Files/Autodesk/Maya2025/bin/mayapy.exe" \
        scripts/export_source_model_to_maya.py

References:
    Model names and their alpha values: metacompskin.constants.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

DEFAULT_MODELS = ("aura", "bowen", "jupiter", "proteus")

# Sub-meshes beyond the head, as (npz key prefix, node suffix).
SUB_MESHES = (("mouth", "mouth"), ("l_eye", "l_eye"), ("r_eye", "r_eye"))

# Pivot keys to turn into locators. Some models store these as None.
PIVOT_KEYS = ("head_pivot", "neck_pivot", "l_eye_pivot", "r_eye_pivot")


def init_maya() -> tuple[Any, Any]:
    """Start Maya in standalone mode and import its APIs.

    Returns:
        Tuple of (maya.cmds module, maya.api.OpenMaya module).

    Raises:
        SystemExit: If Maya's Python modules are unavailable, which means the
            script is running under a standard interpreter instead of mayapy.
    """
    try:
        import maya.standalone

        maya.standalone.initialize(name="python")
        from maya import cmds
        import maya.api.OpenMaya as om
    except ImportError as exc:  # pragma: no cover - requires mayapy
        raise SystemExit(
            f"Could not import Maya ({exc}). Run this script with mayapy, for "
            'example "C:/Program Files/Autodesk/Maya2025/bin/mayapy.exe".'
        ) from exc
    return cmds, om


def create_mesh(
    om: Any,
    cmds: Any,
    name: str,
    vertices: npt.NDArray[np.floating],  # shape: (N, 3)
    faces: npt.NDArray[np.integer],  # shape: (F, verts_per_face)
) -> str:
    """Build a polygon mesh from vertex positions and face connectivity.

    Args:
        om: The maya.api.OpenMaya module.
        cmds: The maya.cmds module.
        name: Name for the mesh's transform node.
        vertices: Vertex positions in local space, shape (N, 3).
        faces: Face connectivity, shape (F, verts_per_face), 3 or 4 sided.

    Returns:
        Name of the created transform node.

    Raises:
        ValueError: If the faces are neither triangles nor quads.
    """
    verts_per_face = int(faces.shape[1])
    if verts_per_face not in (3, 4):
        raise ValueError(
            f"{name}: faces must be triangles (3) or quads (4), got {verts_per_face}"
        )

    points = [om.MPoint(float(x), float(y), float(z)) for x, y, z in vertices]
    counts = om.MIntArray([verts_per_face] * int(faces.shape[0]))
    connects = om.MIntArray([int(i) for i in faces.ravel()])

    transform = om.MFnMesh().create(points, counts, connects)
    dag = om.MFnDagNode(transform)
    dag.setName(name)
    transform_name = dag.name()

    # New meshes carry no shading assignment and draw black in the viewport.
    shape = cmds.listRelatives(transform_name, shapes=True, fullPath=True)[0]
    cmds.sets(shape, edit=True, forceElement="initialShadingGroup")
    return transform_name


def pivot_position(
    data: Any, key: str
) -> npt.NDArray[np.floating] | None:  # shape: (3,) or None
    """Read a pivot position, tolerating models that do not record it.

    Args:
        data: An open NpzFile for the model.
        key: Pivot key to read, for example "head_pivot".

    Returns:
        The pivot position as shape (3,), or None when the model stores no
        pivot under that key.
    """
    if key not in data.files:
        return None
    raw = data[key]
    if raw.dtype == object:
        value = raw.item()
        if value is None:
            return None
        raw = np.asarray(value)
    flat = np.asarray(raw, dtype=np.float64).ravel()
    return flat if flat.size == 3 else None


def export_model(
    cmds: Any, om: Any, model: str, npz_path: Path, output_path: Path
) -> Path:
    """Build and save one model's neutral-pose scene.

    Args:
        cmds: The maya.cmds module.
        om: The maya.api.OpenMaya module.
        model: Model name, used to prefix every node.
        npz_path: Source .npz holding the model's rest geometry.
        output_path: Destination .mb path.

    Returns:
        The path the scene was written to.
    """
    data = np.load(npz_path, allow_pickle=True)

    cmds.file(new=True, force=True)
    cmds.currentUnit(linear="cm")

    members: list[str] = [
        create_mesh(
            om, cmds, f"{model}_head", data["rest_verts"], data["rest_faces"]
        )
    ]
    for key_prefix, suffix in SUB_MESHES:
        verts_key = f"{key_prefix}_rest_verts"
        faces_key = f"{key_prefix}_rest_faces"
        if verts_key not in data.files or faces_key not in data.files:
            continue
        members.append(
            create_mesh(
                om, cmds, f"{model}_{suffix}", data[verts_key], data[faces_key]
            )
        )

    skipped: list[str] = []
    for key in PIVOT_KEYS:
        position = pivot_position(data, key)
        if position is None:
            skipped.append(key)
            continue
        locator = cmds.spaceLocator(name=f"{model}_{key}")[0]
        cmds.setAttr(f"{locator}.translate", *position.tolist())
        members.append(locator)

    group = cmds.group(members, name=model)

    rest_center = pivot_position(data, "rest_center")
    if rest_center is not None:
        cmds.addAttr(group, longName="restCenter", dataType="double3")
        cmds.setAttr(f"{group}.restCenter", *rest_center.tolist(), type="double3")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmds.file(rename=str(output_path))
    saved = cmds.file(save=True, type="mayaBinary", force=True)

    n_verts = int(data["rest_verts"].shape[0])
    n_faces = int(data["rest_faces"].shape[0])
    note = f" (no {', '.join(skipped)})" if skipped else ""
    print(f"  {model}: head {n_verts} verts / {n_faces} faces{note} -> {saved}")
    return Path(saved)


def main() -> int:
    """Export each requested model's neutral pose to a Maya scene.

    Returns:
        Process exit code: 0 on success.
    """
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models", nargs="+", default=list(DEFAULT_MODELS), help="Model names."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=repo_root / "tests" / "test_data" / "source_models",
        help="Directory holding the <model>.npz files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "tests" / "test_data" / "source_models" / "maya",
        help="Directory to write the <model>.mb scenes into.",
    )
    args = parser.parse_args()

    cmds, om = init_maya()
    print(f"Writing neutral-pose scenes to {args.output_dir}")
    for model in args.models:
        npz_path = args.source_dir / f"{model}.npz"
        if not npz_path.is_file():
            raise SystemExit(f"Source model not found: {npz_path}")
        export_model(cmds, om, model, npz_path, args.output_dir / f"{model}.mb")
    return 0


if __name__ == "__main__":
    sys.exit(main())
