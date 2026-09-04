"""Example: Exporting blendshape model data from Autodesk Maya.

``MayaBlendshapeExporter`` runs INSIDE Maya (script editor) or mayapy and
writes an NPZ file that ``BlendshapeModelData.from_npz`` can load anywhere,
so the heavy compression step can run outside Maya in an environment with
PyTorch (and optionally CUDA) available.

It only needs numpy inside Maya - importing ``metacompskin`` in Maya does NOT
require PyTorch.

Typical workflow:
    1. In Maya: run one of the export examples below -> ``head.npz``
    2. Outside Maya: run ``compress_exported_data()`` -> ``head_compressed.npz``

Note:
    Imports are done inside each function on purpose: the export functions
    only work inside Maya, while ``compress_exported_data`` needs PyTorch,
    and this module must be importable in both environments.
"""


def export_from_blendshape_node():
    """Export directly from a mesh that has a blendShape deformer.

    This is the simplest mode: the exporter auto-discovers the blendShape
    node in the mesh's history, toggles each target weight to 1.0 in turn
    (with every other weight at 0) and records the deltas. The original
    weights and envelope are restored afterwards, even if the export fails.

    Run this inside Maya or mayapy, with the scene open.
    """
    from metacompskin.maya_exporter import MayaBlendshapeExporter  # noqa: PLC0415

    exporter = MayaBlendshapeExporter("head_GEO")
    output = exporter.export("exports/head.npz")
    print(f"Exported to {output}")

    # If the mesh has several blendShape nodes, name the one to use:
    exporter = MayaBlendshapeExporter("head_GEO", blendshape_node="shapes_BS")
    exporter.export("exports/head.npz")


def export_from_target_meshes():
    """Export by diffing separate target meshes against the rest mesh.

    Use this when the blendshape targets exist as individual meshes in the
    scene (e.g. a modeling scene where shapes are laid out in a grid).
    Points are read in OBJECT space, so targets that were translated aside
    for layout still produce correct deltas.

    Every target must share the rest mesh's topology (same vertex count);
    the exporter raises a descriptive error otherwise.
    """
    from metacompskin.maya_exporter import MayaBlendshapeExporter  # noqa: PLC0415

    exporter = MayaBlendshapeExporter(
        "head_GEO",
        target_meshes=["smile_GEO", "frown_GEO", "jawOpen_GEO"],
    )
    output = exporter.export("exports/head.npz")
    print(f"Exported to {output}")


def export_with_joint_matrices():
    """Export model data together with skeleton joint rest matrices.

    When ``joints`` is provided, the world-space 4x4 matrix of each joint is
    written to the NPZ under the extra key ``rest_joint_matrices``, ready to
    pass to ``SkinCompressor(rest_joint_matrices=...)``. The matrices are
    stored with translation in the last column (standard homogeneous
    convention), i.e. transposed from Maya's row-vector convention.

    Note:
        SkinCompressor requires at least max_influences + 1 bones
        (9 with default settings), so export at least 9 joints if you plan
        to compress with them.
    """
    from metacompskin.maya_exporter import MayaBlendshapeExporter  # noqa: PLC0415

    joints = [
        "jaw_JNT",
        "cheek_L_JNT",
        "cheek_R_JNT",
        "brow_L_JNT",
        "brow_R_JNT",
        "lip_upper_JNT",
        "lip_lower_JNT",
        "eye_L_JNT",
        "eye_R_JNT",
    ]
    exporter = MayaBlendshapeExporter("head_GEO", joints=joints)
    output = exporter.export("exports/head.npz")
    print(f"Exported to {output}")


def compress_exported_data():
    """Load an exported NPZ and run compression - OUTSIDE Maya.

    This part requires PyTorch and runs in a regular Python environment
    (ideally with CUDA for ~60x faster optimization).
    """
    import numpy as np  # noqa: PLC0415

    from metacompskin.model_data import BlendshapeModelData  # noqa: PLC0415
    from metacompskin.model_fit import SkinCompressor  # noqa: PLC0415

    model_data = BlendshapeModelData.from_npz("exports/head.npz")
    print(f"Loaded: {model_data}")

    # The exporter also saves the target names for reference:
    raw = np.load("exports/head.npz", allow_pickle=True)
    print(f"Shapes: {list(raw['shape_names'])}")

    # Plain compression (100 virtual bones at the origin):
    compressor = SkinCompressor(model_data=model_data, iterations=10000)
    compressor.run(output_location="exports/head_compressed.npz")

    # Or, if joints were exported, compress against the real skeleton:
    if "rest_joint_matrices" in raw:
        compressor = SkinCompressor(
            model_data=model_data,
            iterations=10000,
            rest_joint_matrices=raw["rest_joint_matrices"],
        )
        compressor.run(output_location="exports/head_compressed_joints.npz")


if __name__ == "__main__":
    print("Example: exporting blendshape model data from Maya")
    print("=" * 60)
    print("\nInside Maya / mayapy, run one of:")
    print("  - export_from_blendshape_node()")
    print("  - export_from_target_meshes()")
    print("  - export_with_joint_matrices()")
    print("\nOutside Maya (with PyTorch installed), run:")
    print("  - compress_exported_data()")
    print("\nUpdate the node names and file paths in the code to match")
    print("your scene before running.")
