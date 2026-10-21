r"""Command-line compression entry point.

Runs :class:`SkinCompressor` on a model file written by
:class:`MayaBlendshapeExporter` (or any file :meth:`BlendshapeModelData.from_npz`
accepts) and writes the compressed archive. This is what the Maya pipeline
launches in a subprocess, but it works on its own too::

    python -m metacompskin exports/head.npz exports/head_compressed.npz \\
        --iterations 10000 --number-of-bones 100 --seed 7

``--iterations`` also takes a comma-separated list, one step count per stage,
to anneal the influence budget (see :func:`metacompskin.model_fit.build_training_schedule`)::

    python -m metacompskin exports/head.npz exports/head_compressed.npz \\
        --iterations 5000,5000,10000

If the model file carries ``rest_joint_matrices`` (the exporter writes them
when given ``joints``) they are used unless ``--ignore-joint-matrices`` is
passed; each joint then only drives the vertices nearest to it along the mesh,
``--candidate-joints-per-vertex`` sets how many joints a vertex may use (0 turns
the filter off). With joints the weights start from each vertex's distance to
them (``--weight-init noise`` starts from random values instead). If the file
also carries ``joint_parents`` (the exporter writes them with the joints),
``--joint-hierarchy`` solves the deltas through that hierarchy, so a child joint
follows its parent; by default every joint is solved on its own.
``--distance-penalty`` pulls each joint's skin toward the joint,
``--smoothing residual`` smooths the error instead of the result, and
``--still-vertices static`` (or a distance) pins skin that does not move to
the root and leaves it out of the solve. Pass ``--warm-up false`` to disable the unnormalised warm-up
phase and run only normalised training (useful for research). CUDA is used
automatically when this interpreter's torch sees a GPU.
"""

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch

from metacompskin.model_data import BlendshapeModelData
from metacompskin.model_fit import SkinCompressor


def _int_list(text: str) -> tuple[int, ...]:
    """Parses ``"5000,5000,10000"`` into ``(5000, 5000, 10000)`` for argparse.

    A single integer with no comma is a one-element tuple, e.g. ``"300"`` ->
    ``(300,)``, so ``--iterations 300`` keeps working unchanged.

    Args:
        text: The raw ``--iterations`` value.

    Returns:
        One step count per stage.

    Raises:
        argparse.ArgumentTypeError: If any item is not an integer.
    """
    try:
        return tuple(int(item) for item in text.split(","))
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"expected comma-separated integers, got {text!r}"
        ) from e


def _still_vertices_arg(text: str) -> str | float:
    """Converts a --still-vertices argument: "static" or a distance.

    Args:
        text: The raw --still-vertices value.

    Returns:
        "static", or the distance in model units as a float.

    Raises:
        argparse.ArgumentTypeError: If the value is neither.
    """
    if text == "static":
        return text
    try:
        return float(text)
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"--still-vertices must be 'static' or a distance, got {text!r}"
        ) from e


def _root_joint_arg(text: str) -> int | str | None:
    """Converts a root_joint CLI argument to its resolved form.

    Args:
        text: The raw --root-joint value.

    Returns:
        None if 'auto', a string if 'new' or 'none', or an int for digit strings.

    Raises:
        argparse.ArgumentTypeError: If the value is unrecognized.
    """
    if text == "auto":
        return None
    if text in ("new", "none"):
        return text
    try:
        return int(text)
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"--root-joint must be 'auto', 'new', 'none', or an integer, got {text!r}"
        ) from e


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
    parser.add_argument(
        "--iterations",
        type=_int_list,
        default=(10000,),
        help="steps per stage, e.g. 5000,5000,10000 (a single number is one stage)",
    )
    parser.add_argument("--number-of-bones", type=int, default=None, help="P")
    parser.add_argument("--max-influences", type=int, default=None, help="K")
    parser.add_argument("--total-nnz-b-rt", type=int, default=None, help="L")
    parser.add_argument("--init-weight", type=float, default=None)
    parser.add_argument("--power", type=int, default=None, help="p")
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--candidate-joints-per-vertex",
        type=int,
        default=None,
        help="M, nearest joints a vertex may be weighted to (needs joint "
        "matrices; 0 turns the filter off)",
    )
    parser.add_argument(
        "--ignore-joint-matrices",
        action="store_true",
        help="Do not use rest_joint_matrices stored in the model file.",
    )
    parser.add_argument(
        "--joint-hierarchy",
        action="store_true",
        help="Solve through the joint_parents stored in the model file, so a child "
        "joint follows its parent. Off by default: every joint is solved on its own.",
    )
    parser.add_argument(
        "--weight-init",
        choices=["noise", "distance"],
        default=None,
        help="Where the weights start: 'distance' (from each vertex's distance to the "
        "joints; the default with joints) or 'noise' (random; the default without).",
    )
    parser.add_argument(
        "--distance-penalty",
        type=float,
        default=None,
        help="Strength of the penalty on weight far from each vertex's nearest joint "
        "(needs joints; default 0, off).",
    )
    parser.add_argument(
        "--smoothing",
        choices=["reconstruction", "residual"],
        default=None,
        help="What alpha smooths: the skinned result (default) or the error.",
    )
    parser.add_argument(
        "--still-vertices",
        type=_still_vertices_arg,
        default=None,
        help="Pin to the root and leave out of the solve: 'static' (no motion) or a "
        "distance in model units (largest motion below it).",
    )
    parser.add_argument(
        "--deltas-about-joints",
        action="store_true",
        help="With a joint hierarchy, solve each joint's own delta about the joint itself.",
    )
    parser.add_argument(
        "--root-joint",
        type=_root_joint_arg,
        default=None,
        help="Root influence designation: 'auto' (default based on joint matrices), "
        "'none' (no root), 'new' (synthetic root), or an index [0, P)",
    )
    parser.add_argument(
        "--warm-up",
        type=lambda x: x.lower() in ("true", "on", "1", "yes"),
        default=True,
        help="Enable warm-up phase (true/false, default true)",
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
        if settings["rest_joint_matrices"] is not None and args.joint_hierarchy:
            parents = _stored_joint_parents(model_path)
            if parents is None:
                raise ValueError(
                    "--joint-hierarchy needs joint_parents in the model file; "
                    "export it again with the current exporter"
                )
            settings["joint_parents"] = parents
            if args.deltas_about_joints:
                settings["deltas_about_joints"] = True

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
        "seed",
        "candidate_joints_per_vertex",
        "root_joint",
        "weight_init",
        "distance_penalty",
        "smoothing",
        "still_vertices",
    ):
        value = getattr(args, option, None)
        if value is not None:
            settings[option.replace("nnz_b_rt", "nnz_B_rt")] = value

    # Handle warm_up -> warm_up_enabled mapping
    if hasattr(args, "warm_up"):
        settings["warm_up_enabled"] = args.warm_up

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


def _stored_joint_parents(model_path: Path) -> np.ndarray | None:
    """Reads ``joint_parents`` from a model file, if present.

    Args:
        model_path: Model NPZ path.

    Returns:
        Parent index per joint, shape (P,), or None when the file has none.
    """
    with np.load(model_path, allow_pickle=True) as archive:
        if "joint_parents" not in archive.files:
            return None
        return archive["joint_parents"]


if __name__ == "__main__":
    main()
