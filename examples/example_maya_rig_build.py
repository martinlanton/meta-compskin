"""Example: Building the compressed result back into Autodesk Maya.

``build_skinned_rig`` runs INSIDE Maya (script editor) or mayapy. It reads the
NPZ written by ``SkinCompressor.run`` and, on the selected neutral head:

1. duplicates the mesh (the original and its blendShape node stay untouched;
   pass ``duplicate_mesh=False`` to skin a deformer-free mesh in place),
2. creates one joint per virtual bone under a single root, at the compressor's
   ``restXform`` matrices if you supplied some, else over the region it drives,
3. binds the duplicate with the compressor's skin weights, and
4. keys one blendshape per frame: frame 0 is the neutral pose, frame k + 1 is
   blendshape k at 100%, with stepped tangents.

It only needs numpy inside Maya - importing ``metacompskin`` in Maya does NOT
require PyTorch.

Usage:
    1. Open the scene that holds the neutral head that was exported.
    2. Select the head mesh.
    3. Run ``build_rig_on_selection()`` in the script editor.
    4. Scrub the timeline; the root joint's ``currentShape`` attribute names the
       shape shown on each frame.
"""

import numpy as np

from metacompskin.maya_rig_builder import build_skinned_rig


def build_rig_on_selection(
    compressed_path: str = "exports/head_compressed.npz",
    model_path: str = "exports/head.npz",
):
    """Build the skinned rig on the selected mesh, labelling frames by shape name.

    Args:
        compressed_path: NPZ file written by ``SkinCompressor.run``.
        model_path: NPZ file written by ``MayaBlendshapeExporter``; only its
            ``shape_names`` array is used, to label the frames. Pass an empty
            string to use ``shape_000``-style placeholders instead.
    """
    shape_names = list(np.load(model_path)["shape_names"]) if model_path else None

    rig = build_skinned_rig(compressed_path, shape_names=shape_names)

    print(f"Skinned duplicate: {rig.mesh}")
    print(f"Root joint:        {rig.root}")
    print(f"Joints:            {len(rig.joints)} (first: {rig.joints[0]})")
    print(f"Skin cluster:      {rig.skin_cluster}")
    print(f"Frames:            0 = rest, 1..{len(rig.shape_names)} = one shape each")
    return rig


if __name__ == "__main__":
    build_rig_on_selection()
