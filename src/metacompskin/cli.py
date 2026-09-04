r"""Command-line compression entry point.

Runs :class:`SkinCompressor` on a model file written by
:class:`MayaBlendshapeExporter` (or any file :meth:`BlendshapeModelData.from_npz`
accepts) and writes the compressed archive. This is what the Maya pipeline
launches in a subprocess, but it works on its own too::

    python -m metacompskin exports/head.npz exports/head_compressed.npz \\
        --iterations 10000 --number-of-bones 100

If the model file carries ``rest_joint_matrices`` (the exporter writes them
when given ``joints``) they are used unless ``--ignore-joint-matrices`` is
passed. CUDA is used automatically when this interpreter's torch sees a GPU.
"""

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch

from metacompskin.model_data import BlendshapeModelData
from metacompskin.model_fit import SkinCompressor


def build_parser() -> argparse.ArgumentParser:
    """Builds the argument parser for ``python -m metacompskin``.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(
        prog="python -m metacompskin",
        description="Compress a blendshape model into sparse linear blend skinning.",
    )
    parser.add_argument("model", help="Input model NPZ (from MayaBlendshapeExporter).")
    parser.add_argument("output", help="Compressed NPZ to write.")
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--number-of-bones", type=int, default=None, help="P")
    parser.add_argument("--max-influences", type=int, default=None, help="K")
    parser.add_argument("--total-nnz-b-rt", type=int, default=None, help="L")
    parser.add_argument("--init-weight", type=float, default=None)
    parser.add_argument("--power", type=int, default=None, help="p")
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument(
        "--ignore-joint-matrices",
        action="store_true",
        help="Do not use rest_joint_matrices stored in the model file.",
    )
    return parser


def main(argv: list[str] | None = None) -> Path:
    """Compresses a model file from command-line arguments.

    Args:
        argv: Arguments without the program name; None reads ``sys.argv``.

    Returns:
        Path of the compressed archive written.

    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    args = build_parser().parse_args(argv)
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model_data = BlendshapeModelData.from_npz(str(model_path), alpha=args.alpha)
    settings = _compressor_settings(args)
    if not args.ignore_joint_matrices:
        settings["rest_joint_matrices"] = _stored_joint_matrices(model_path)

    print(f"Compressing on {'cuda' if torch.cuda.is_available() else 'cpu'}")
    compressor = SkinCompressor(model_data=model_data, **settings)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    compressor.run(output)
    print(f"Wrote {output}")
    return output


def _compressor_settings(args: argparse.Namespace) -> dict[str, Any]:
    """Collects the SkinCompressor keyword arguments that were given.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Keyword arguments; options left at None are omitted so the
        compressor's own defaults apply.
    """
    settings: dict[str, Any] = {"iterations": args.iterations}
    for option in (
        "number_of_bones",
        "max_influences",
        "total_nnz_b_rt",
        "init_weight",
        "power",
    ):
        value = getattr(args, option)
        if value is not None:
            settings[option.replace("nnz_b_rt", "nnz_B_rt")] = value
    return settings


def _stored_joint_matrices(model_path: Path) -> np.ndarray | None:
    """Reads ``rest_joint_matrices`` from a model file, if present.

    Args:
        model_path: Model NPZ path.

    Returns:
        Matrices of shape (P, 4, 4), or None when the file has none.
    """
    with np.load(model_path, allow_pickle=True) as archive:
        if "rest_joint_matrices" not in archive.files:
            return None
        return archive["rest_joint_matrices"]


if __name__ == "__main__":
    main()
