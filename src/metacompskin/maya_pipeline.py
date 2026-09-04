"""One-call pipeline inside Maya: export, compress in a subprocess, build the rig.

:func:`compress_and_build_rig` chains the three Maya-facing pieces of the
package in the current Maya session:

1. :class:`MayaBlendshapeExporter` writes the blendshape model of the selected
   mesh (or the only mesh in the scene) to ``<output_dir>/<mesh>.npz``.
2. ``python -m metacompskin`` runs in a subprocess with an interpreter that has
   PyTorch, so compression uses CUDA whenever that interpreter sees a GPU. Its
   output is streamed to the script editor; the call blocks until it is done.
3. :func:`build_skinned_rig` builds joints, skin cluster and the one-shape-per-
   frame animation on a duplicate of the same mesh.

Maya's own interpreter has no PyTorch, so the compression interpreter is given
explicitly, through the ``METACOMPSKIN_PYTHON`` environment variable, or found
by probing likely interpreters (project virtualenvs, ``python`` on PATH, conda,
pyenv and virtualenvwrapper environments) for one that imports torch, with
CUDA-capable ones preferred. The package directory used inside Maya is added to
that subprocess's ``PYTHONPATH``, so ``metacompskin`` need not be installed in
the other environment.

Example:
    Inside Maya, with the head selected::

        from metacompskin.maya_pipeline import compress_and_build_rig

        result = compress_and_build_rig(
            python_executable="D:/envs/compskin/python.exe", iterations=10000
        )
        print(result.compressed_path, result.rig.joints)
"""

import contextlib
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import metacompskin
from metacompskin.maya_exporter import MayaBlendshapeExporter
from metacompskin.maya_rig_builder import MayaSkinRig, build_skinned_rig

ENV_PYTHON = "METACOMPSKIN_PYTHON"
_LOG_TAIL_LINES = 30
_PROBE_SCRIPT = "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')"
_PROBE_TIMEOUT_SECONDS = 120.0
_ENVIRONMENT_ROOTS = (
    "miniconda3/envs",
    "anaconda3/envs",
    "miniforge3/envs",
    ".conda/envs",
    ".virtualenvs",
    ".pyenv/versions",
)


@dataclass(frozen=True)
class TorchProbe:
    """What probing an interpreter for PyTorch found.

    Attributes:
        cuda: Whether torch reports a usable CUDA device there.
    """

    cuda: bool


@dataclass(frozen=True)
class CompressionSettings:
    """Compressor settings forwarded to the subprocess.

    Used by :func:`compression_command` and :func:`run_compression`;
    :func:`compress_and_build_rig` takes the same values as plain arguments.
    Fields left at None are not passed, so the compressor's own defaults apply
    (see :class:`SkinCompressor`).

    Attributes:
        iterations: Optimisation iterations per phase.
        number_of_bones: Number of virtual bones P.
        max_influences: Maximum non-zero weights per vertex K.
        total_nnz_B_rt: Sparsity budget L.
        init_weight: Scale of the random initial deltas.
        power: Exponent p of the error norm.
        alpha: Laplacian smoothness weight.
        use_joint_matrices: Use the ``rest_joint_matrices`` the exporter wrote
            (when ``joints`` were given), so P and the joint placement follow them.
    """

    iterations: int = 10000
    number_of_bones: int | None = None
    max_influences: int | None = None
    total_nnz_B_rt: int | None = None  # noqa: N815 (matches SkinCompressor)
    init_weight: float | None = None
    power: int | None = None
    alpha: float | None = None
    use_joint_matrices: bool = True


@dataclass(frozen=True)
class PipelineResult:
    """What :func:`compress_and_build_rig` produced.

    Attributes:
        source_mesh: Transform of the mesh that was exported.
        model_path: Exported model NPZ.
        compressed_path: Compressed NPZ.
        rig: The nodes built in the scene.
    """

    source_mesh: str
    model_path: Path
    compressed_path: Path
    rig: MayaSkinRig


def compress_and_build_rig(  # noqa: PLR0913
    python_executable: str | Path | None = None,
    output_dir: str | Path | None = None,
    mesh: str | None = None,
    joints: list[str] | None = None,
    iterations: int = 10000,
    number_of_bones: int | None = None,
    max_influences: int | None = None,
    total_nnz_B_rt: int | None = None,
    init_weight: float | None = None,
    power: int | None = None,
    alpha: float | None = None,
    use_joint_matrices: bool = True,
    name: str = "compskin",
) -> PipelineResult:
    """Exports the mesh, compresses it in a subprocess and builds the rig.

    The mesh must carry a single blendShape node whose weights are free to be
    driven (the exporter's blendShape-node mode). It is left untouched; the
    rig is built on a duplicate, with the blendShape temporarily disabled so
    the duplicate is taken in the neutral pose.

    Compressor settings left at None take the compressor's own defaults (see
    :class:`SkinCompressor`).

    Args:
        python_executable: Interpreter with PyTorch for the compression. Falls
            back to the ``METACOMPSKIN_PYTHON`` environment variable, then to
            auto-detection (see :func:`detect_python_executable`).
        output_dir: Where the model and compressed NPZ files go. Defaults to a
            ``compskin`` folder beside the saved scene, or a temp directory when
            the scene is unsaved.
        mesh: Mesh to process. Defaults to the selection, or to the only mesh in
            the scene when nothing is selected.
        joints: Optional joint names whose rest matrices are exported and used
            for the compression (see :class:`MayaBlendshapeExporter`).
        iterations: Optimisation iterations per phase (default 10000).
        number_of_bones: Number of virtual bones P (default 100).
        max_influences: Maximum non-zero weights per vertex K (default 8).
        total_nnz_B_rt: Sparsity budget L (default 6000).
        init_weight: Scale of the random initial deltas (default 1e-3).
        power: Exponent p of the error norm (default 2).
        alpha: Laplacian smoothness weight (default from the model name).
        use_joint_matrices: Use the rest matrices exported for ``joints`` so P
            and the joint placement follow them (default True).
        name: Prefix for every node the rig builder creates.

    Returns:
        The file paths and the built rig.

    Raises:
        ValueError: If no interpreter is configured, or the mesh cannot be
            determined unambiguously.
        FileNotFoundError: If the interpreter does not exist.
        RuntimeError: If the compression subprocess fails.
    """
    cmds = _import_cmds()
    python = resolve_python_executable(python_executable)
    settings = CompressionSettings(
        iterations=iterations,
        number_of_bones=number_of_bones,
        max_influences=max_influences,
        total_nnz_B_rt=total_nnz_B_rt,
        init_weight=init_weight,
        power=power,
        alpha=alpha,
        use_joint_matrices=use_joint_matrices,
    )
    source = resolve_source_mesh(cmds, mesh)
    directory = (
        Path(output_dir)
        if output_dir
        else default_output_dir(cmds.file(query=True, sceneName=True))
    )
    directory.mkdir(parents=True, exist_ok=True)
    stem = source.split("|")[-1].split(":")[-1]

    model_path = MayaBlendshapeExporter(source, joints=joints).export(
        directory / f"{stem}.npz"
    )
    compressed_path = directory / f"{stem}_compressed.npz"
    run_compression(python, model_path, compressed_path, settings)

    with np.load(model_path, allow_pickle=True) as archive:
        shape_names = [str(shape) for shape in archive["shape_names"]]
    with _blendshapes_disabled(cmds, source):
        rig = build_skinned_rig(
            compressed_path, mesh=source, shape_names=shape_names, name=name
        )
    return PipelineResult(
        source_mesh=source,
        model_path=model_path,
        compressed_path=compressed_path,
        rig=rig,
    )


def resolve_python_executable(python_executable: str | Path | None) -> Path:
    """Finds the interpreter that runs the compression.

    The explicit argument wins, then the ``METACOMPSKIN_PYTHON`` environment
    variable, then auto-detection among likely interpreters on this machine.

    Args:
        python_executable: Explicit interpreter path, or None.

    Returns:
        The interpreter path.

    Raises:
        ValueError: If nothing is configured and no interpreter with PyTorch
            can be found.
        FileNotFoundError: If the configured interpreter does not exist.
    """
    candidate = python_executable or os.environ.get(ENV_PYTHON)
    if candidate:
        python = Path(candidate)
        if not python.exists():
            raise FileNotFoundError(f"Python interpreter not found: {python}")
        return python

    print("No compression interpreter configured; looking for one with PyTorch...")
    detected = detect_python_executable()
    if detected is None:
        raise ValueError(
            "No Python interpreter with PyTorch found. Pass python_executable or "
            f"set the {ENV_PYTHON} environment variable to one that has torch "
            "installed."
        )
    print(f"Using {detected} (set {ENV_PYTHON} to skip this search next time).")
    return detected


def detect_python_executable(
    candidates: list[Path] | None = None,
    probe: Callable[[Path], TorchProbe | None] | None = None,
) -> Path | None:
    """Picks an interpreter that has PyTorch, preferring one that sees a GPU.

    Args:
        candidates: Interpreters to try, in order of preference. Defaults to
            :func:`candidate_interpreters`.
        probe: Function that reports whether an interpreter has torch and
            CUDA. Defaults to :func:`probe_torch`.

    Returns:
        The first CUDA-capable candidate, else the first candidate with torch,
        else None.
    """
    probe = probe or probe_torch
    fallback: Path | None = None
    for candidate in candidate_interpreters() if candidates is None else candidates:
        found = probe(candidate)
        if found is None:
            continue
        if found.cuda:
            return candidate
        fallback = fallback or candidate
    return fallback


def probe_torch(python: Path) -> TorchProbe | None:
    """Checks whether an interpreter can import torch, and whether it has CUDA.

    Args:
        python: Interpreter to run.

    Returns:
        The probe result, or None if the interpreter cannot run or lacks torch.
    """
    try:
        result = subprocess.run(  # noqa: S603
            [str(python), "-c", _PROBE_SCRIPT],
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return TorchProbe(cuda=result.stdout.strip().endswith("cuda"))


def candidate_interpreters() -> list[Path]:
    """Lists interpreters worth probing, most likely first.

    Order: virtualenvs beside this package (``.venv``, ``venv``), ``python3``
    and ``python`` on PATH, then every environment under the usual conda,
    virtualenvwrapper and pyenv directories in the home folder. Maya's own
    interpreters and duplicates are dropped; only existing files are returned.

    Returns:
        Candidate interpreter paths.
    """
    root = _package_root()
    venv_dirs = [
        root / ".venv",
        root.parent / ".venv",
        root / "venv",
        root.parent / "venv",
    ]
    found: list[Path] = [
        interpreter for env in venv_dirs for interpreter in _interpreters_in(env)
    ]
    found += [
        Path(path) for name in ("python3", "python") if (path := shutil.which(name))
    ]
    for env_root in _ENVIRONMENT_ROOTS:
        for env in sorted((Path.home() / env_root).glob("*")):
            found += _interpreters_in(env)

    unique: list[Path] = []
    seen: set[Path] = set()
    for interpreter in found:
        key = interpreter.resolve()
        if key in seen or not interpreter.is_file() or key.name.startswith("mayapy"):
            continue
        seen.add(key)
        unique.append(interpreter)
    return unique


def _interpreters_in(environment: Path) -> list[Path]:
    """Possible interpreter locations inside a virtualenv or conda environment.

    Args:
        environment: Environment root directory.

    Returns:
        Paths to check (they may not exist).
    """
    if sys.platform == "win32":
        return [environment / "Scripts" / "python.exe", environment / "python.exe"]
    return [environment / "bin" / "python"]


def _package_root() -> Path:
    """The directory that holds the ``metacompskin`` package.

    Returns:
        Parent of the package directory (``src`` in a source checkout).
    """
    return Path(metacompskin.__file__).resolve().parents[1]


def default_output_dir(scene_path: str) -> Path:
    """Chooses where the pipeline's files go when no directory is given.

    Args:
        scene_path: The current scene file, or an empty string when unsaved.

    Returns:
        ``<scene folder>/compskin`` for a saved scene, otherwise a fresh
        temporary directory.
    """
    if scene_path:
        return Path(scene_path).parent / "compskin"
    return Path(tempfile.mkdtemp(prefix="compskin_"))


def compression_command(
    python: Path, model_path: Path, compressed_path: Path, settings: CompressionSettings
) -> list[str]:
    """Builds the ``python -m metacompskin`` command line.

    Args:
        python: Interpreter with PyTorch.
        model_path: Exported model NPZ.
        compressed_path: Compressed NPZ to write.
        settings: Compressor settings; None fields are omitted.

    Returns:
        The argument list for ``subprocess``.
    """
    command = [
        str(python),
        "-m",
        "metacompskin",
        str(model_path),
        str(compressed_path),
        "--iterations",
        str(settings.iterations),
    ]
    options = {
        "--number-of-bones": settings.number_of_bones,
        "--max-influences": settings.max_influences,
        "--total-nnz-b-rt": settings.total_nnz_B_rt,
        "--init-weight": settings.init_weight,
        "--power": settings.power,
        "--alpha": settings.alpha,
    }
    for flag, value in options.items():
        if value is not None:
            command += [flag, str(value)]
    if not settings.use_joint_matrices:
        command.append("--ignore-joint-matrices")
    return command


def run_compression(
    python: Path, model_path: Path, compressed_path: Path, settings: CompressionSettings
) -> None:
    """Runs the compression CLI in a subprocess, streaming its output.

    Args:
        python: Interpreter with PyTorch.
        model_path: Exported model NPZ.
        compressed_path: Compressed NPZ to write.
        settings: Compressor settings.

    Raises:
        RuntimeError: If the subprocess exits with an error or writes no file.
    """
    command = compression_command(python, model_path, compressed_path, settings)
    print("Running: " + " ".join(command))
    process = subprocess.Popen(  # noqa: S603
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=_subprocess_environment(),
    )
    tail: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
        tail = [*tail, line][-_LOG_TAIL_LINES:]
    if process.wait() != 0:
        raise RuntimeError(
            f"Compression failed (exit code {process.returncode}). Last output:\n"
            + "".join(tail)
        )
    if not compressed_path.exists():
        raise RuntimeError(f"Compression finished but wrote no file: {compressed_path}")


def _subprocess_environment() -> dict[str, str]:
    """Environment for the compression subprocess.

    The directory that holds this package is prepended to ``PYTHONPATH`` so
    the subprocess imports the same ``metacompskin`` that runs inside Maya.

    Returns:
        A copy of the current environment with ``PYTHONPATH`` extended.
    """
    package_root = str(_package_root())
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [package_root, existing]))
    return env


def resolve_source_mesh(cmds: Any, mesh: str | None) -> str:
    """Determines the mesh to process.

    Args:
        cmds: The maya.cmds module.
        mesh: Explicit transform or mesh shape name, or None.

    Returns:
        Full path of the mesh transform: the named one, else the single
        selected one, else the only mesh in the scene.

    Raises:
        ValueError: If the choice is ambiguous or no mesh is found.
    """
    if mesh is not None:
        return _mesh_transform(cmds, mesh)
    selected = cmds.ls(selection=True, long=True)
    if len(selected) > 1:
        raise ValueError(f"Select a single mesh (got {len(selected)} nodes).")
    if selected:
        return _mesh_transform(cmds, selected[0])
    shapes = cmds.ls(type="mesh", long=True, noIntermediate=True) or []
    transforms = sorted(
        {cmds.listRelatives(shape, parent=True, fullPath=True)[0] for shape in shapes}
    )
    if len(transforms) == 1:
        return transforms[0]
    if not transforms:
        raise ValueError("No mesh selected and the scene contains no mesh.")
    raise ValueError(
        f"No mesh selected and the scene contains {len(transforms)} meshes; "
        "select the one to process."
    )


def _mesh_transform(cmds: Any, node: str) -> str:
    """Resolves a transform or mesh shape to its transform's full path.

    Args:
        cmds: The maya.cmds module.
        node: Transform or mesh shape name.

    Returns:
        Full path of the transform.

    Raises:
        ValueError: If the node does not exist or is not a polygon mesh.
    """
    if not cmds.objExists(node):
        raise ValueError(f"Node does not exist in the scene: '{node}'")
    if cmds.nodeType(node) == "mesh":
        return cmds.listRelatives(node, parent=True, fullPath=True)[0]
    if not cmds.listRelatives(node, shapes=True, noIntermediate=True, type="mesh"):
        raise ValueError(f"'{node}' is not a polygon mesh.")
    return cmds.ls(node, long=True)[0]


@contextlib.contextmanager
def _blendshapes_disabled(cmds: Any, mesh: str) -> Iterator[None]:
    """Temporarily sets every blendShape envelope on ``mesh`` to zero.

    Args:
        cmds: The maya.cmds module.
        mesh: Mesh transform.

    Yields:
        Nothing; the envelopes are restored afterwards, even on error.
    """
    blendshapes = cmds.ls(
        cmds.listHistory(mesh, pruneDagObjects=True), type="blendShape"
    )
    envelopes = {node: cmds.getAttr(f"{node}.envelope") for node in blendshapes}
    try:
        for node in blendshapes:
            cmds.setAttr(f"{node}.envelope", 0.0)
        yield
    finally:
        for node, envelope in envelopes.items():
            cmds.setAttr(f"{node}.envelope", envelope)


def _import_cmds() -> Any:
    """Imports maya.cmds.

    Returns:
        The maya.cmds module.

    Raises:
        RuntimeError: If not running inside Maya or mayapy.
    """
    try:
        from maya import cmds  # noqa: PLC0415
    except ImportError as e:
        raise RuntimeError(
            "maya.cmds is not importable. compress_and_build_rig must run inside "
            "Maya or mayapy."
        ) from e
    return cmds
