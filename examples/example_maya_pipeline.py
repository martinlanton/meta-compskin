"""Example: Export, compress and rebuild a head from a single call inside Maya.

``compress_and_build_rig`` runs INSIDE Maya (script editor) or mayapy. With the
head selected (or nothing selected when it is the only mesh in the scene) it:

1. exports the blendShape targets with ``MayaBlendshapeExporter``,
2. runs ``python -m metacompskin`` in a subprocess with a PyTorch interpreter
   (CUDA is used automatically when that interpreter sees a GPU), and
3. builds joints, skin cluster and a one-shape-per-frame animation on a
   duplicate of the same mesh with ``build_skinned_rig``.

Maya's own Python has no PyTorch, so tell the pipeline which interpreter to
use: either set the ``METACOMPSKIN_PYTHON`` environment variable before
starting Maya, or pass ``python_executable`` as below. With neither, the
pipeline searches the usual environment locations for a Python that imports
torch, preferring one with CUDA.

The call blocks until the compression finishes and streams the solver's
progress to the script editor.
"""

from metacompskin.maya_pipeline import compress_and_build_rig


def compress_selected_head(
    python_executable: str = "D:/envs/compskin/python.exe",
    iterations: int = 10000,
):
    """Run the whole pipeline on the selected head and report what was made.

    Args:
        python_executable: Interpreter with PyTorch that runs the compression.
        iterations: Compressor iterations per phase; 600 gives a quick preview.
    """
    result = compress_and_build_rig(
        python_executable=python_executable, iterations=iterations
    )

    print(f"Exported model:    {result.model_path}")
    print(f"Compressed result: {result.compressed_path}")
    print(f"Skinned duplicate: {result.rig.mesh}")
    print(f"Joints:            {len(result.rig.joints)} under {result.rig.root}")
    print(
        f"Frames:            0 = rest, 1..{len(result.rig.shape_names)} = one shape each"
    )
    return result


if __name__ == "__main__":
    compress_selected_head()
