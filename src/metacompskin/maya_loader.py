"""Maya-based geometry loading utilities.

This module provides utilities for loading mesh geometry using Maya's API,
supporting both direct execution (when running inside Maya) and external
execution (via mayapy subprocess).

The primary use case is loading blendshape geometry from OBJ files, but this
module is designed to support .ma and .mb files in the future.

Example:
    >>> from pathlib import Path
    >>> from metacompskin.maya_loader import load_obj_with_maya
    >>>
    >>> # Direct mode (running in Maya/mayapy)
    >>> verts, faces = load_obj_with_maya(Path("HEAD.obj"))
    >>>
    >>> # Subprocess mode (running in standard Python)
    >>> mayapy = Path("C:/Program Files/Autodesk/Maya2024/bin/mayapy.exe")
    >>> verts, faces = load_obj_with_maya(Path("HEAD.obj"), mayapy)
"""

import contextlib
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt


def load_obj_with_maya(
    obj_path: Path, maya_interpreter_path: Optional[Path] = None
) -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.integer]]:
    """Load OBJ file using Maya API.

    This function loads mesh geometry from an OBJ file using Maya's native
    file import functionality. It can operate in two modes:

    1. **Direct mode** (maya_interpreter_path=None): Imports maya.cmds directly.
       Use when running inside Maya or with mayapy.

    2. **Subprocess mode** (maya_interpreter_path provided): Spawns mayapy
       as a subprocess to load the file. Use when running in standard Python.

    Args:
        obj_path: Path to OBJ file to load.
        maya_interpreter_path: Optional path to mayapy executable
            (e.g., "C:/Program Files/Autodesk/Maya2024/bin/mayapy.exe").
            If None, assumes running inside Maya and imports maya.cmds directly.

    Returns:
        Tuple of (vertices, faces):
            - vertices: (N, 3) array of vertex positions (float64)
            - faces: (F, verts_per_face) array of face indices (int32)

    Raises:
        FileNotFoundError: If obj_path or maya_interpreter_path doesn't exist.
        RuntimeError: If Maya import fails or geometry extraction fails.

    Example:
        >>> # Running inside Maya or with mayapy
        >>> verts, faces = load_obj_with_maya(Path("mesh.obj"))
        >>> print(verts.shape)  # (7306, 3)

        >>> # Running in standard Python with mayapy path
        >>> mayapy = Path("C:/Program Files/Autodesk/Maya2024/bin/mayapy.exe")
        >>> verts, faces = load_obj_with_maya(Path("mesh.obj"), mayapy)
    """
    if not obj_path.exists():
        raise FileNotFoundError(f"OBJ file not found: {obj_path}")

    if maya_interpreter_path is None:
        # Direct mode: running inside Maya
        return _load_obj_direct(obj_path)
    # Subprocess mode: spawn mayapy
    return _load_obj_subprocess(obj_path, maya_interpreter_path)


def _load_obj_direct(obj_path: Path) -> Tuple[npt.NDArray, npt.NDArray]:
    """Load OBJ directly using maya.cmds (running inside Maya).

    This function is called when maya_interpreter_path is None,
    indicating we're already in a Maya Python environment.

    Args:
        obj_path: Path to OBJ file.

    Returns:
        Tuple of (vertices, faces) as numpy arrays.

    Raises:
        RuntimeError: If Maya import fails or mesh loading fails.
    """
    try:
        import maya.standalone  # noqa: PLC0415
        from maya import cmds  # noqa: PLC0415
    except ImportError as e:
        raise RuntimeError(
            "Failed to import maya.cmds. Are you running inside Maya or mayapy? "
            "If using standard Python, provide maya_interpreter_path parameter."
        ) from e

    # Initialize Maya standalone if needed (no-op if already initialized)
    # Already initialized (running in Maya UI or already standalone)
    with contextlib.suppress(RuntimeError):
        maya.standalone.initialize()

    # Create new scene
    cmds.file(new=True, force=True)

    # Import OBJ file
    cmds.file(str(obj_path), i=True, type="OBJ", options="mo=0")

    # Get mesh node (assumes single mesh in OBJ)
    meshes = cmds.ls(type="mesh")
    if not meshes:
        raise RuntimeError(f"No mesh found in {obj_path}")

    mesh_shape = meshes[0]

    # Get vertex positions
    vertex_count = cmds.polyEvaluate(mesh_shape, vertex=True)
    vertices = []
    for i in range(vertex_count):
        pos = cmds.xform(f"{mesh_shape}.vtx[{i}]", q=True, ws=True, t=True)
        vertices.append(pos)

    vertices = np.array(vertices, dtype=np.float64)

    # Get face connectivity
    face_count = cmds.polyEvaluate(mesh_shape, face=True)
    faces = []
    for i in range(face_count):
        # Get vertex indices for this face
        face_verts = cmds.polyInfo(f"{mesh_shape}.f[{i}]", faceToVertex=True)[0]
        # Parse the string output (format: "FACE 0: 1 2 3 4\n")
        indices = [int(x) for x in face_verts.split(":")[1].split()]
        faces.append(indices)

    faces = np.array(faces, dtype=np.int32)

    return vertices, faces


def _validate_maya_interpreter(maya_interpreter_path: Path) -> None:
    """Validate that Maya interpreter exists.

    Args:
        maya_interpreter_path: Path to mayapy executable.

    Raises:
        FileNotFoundError: If mayapy doesn't exist.
    """
    if not maya_interpreter_path.exists():
        raise FileNotFoundError(f"Maya interpreter not found: {maya_interpreter_path}")


def _generate_maya_loading_script(obj_path: Path) -> str:
    """Generate Maya Python script for loading OBJ file.

    This generates a standalone Python script that:
    - Initializes Maya in standalone mode
    - Imports the OBJ file
    - Extracts vertex and face data
    - Outputs as JSON

    Args:
        obj_path: Path to OBJ file to load.

    Returns:
        String containing complete Maya Python script.

    Note:
        Uses raw strings and absolute paths for Windows compatibility.
    """
    return f"""
import sys
import json
from maya import cmds
import maya.standalone

try:
    maya.standalone.initialize()

    # Import OBJ
    cmds.file(new=True, force=True)
    cmds.file(r"{obj_path}", i=True, type="OBJ", options="mo=0")

    # Get mesh
    meshes = cmds.ls(type="mesh")
    if not meshes:
        print(json.dumps({{"error": "No mesh found in file"}}))
        sys.exit(1)

    mesh_shape = meshes[0]

    # Get vertices
    vertex_count = cmds.polyEvaluate(mesh_shape, vertex=True)
    vertices = []
    for i in range(vertex_count):
        pos = cmds.xform(f"{{mesh_shape}}.vtx[{{i}}]", q=True, ws=True, t=True)
        vertices.append(pos)

    # Get faces
    face_count = cmds.polyEvaluate(mesh_shape, face=True)
    faces = []
    for i in range(face_count):
        face_verts = cmds.polyInfo(f"{{mesh_shape}}.f[{{i}}]", faceToVertex=True)[0]
        indices = [int(x) for x in face_verts.split(':')[1].split()]
        faces.append(indices)

    # Output as JSON
    result = {{
        "vertices": vertices,
        "faces": faces
    }}
    print(json.dumps(result))

    maya.standalone.uninitialize()

except Exception as e:
    print(json.dumps({{"error": str(e)}}))
    sys.exit(1)
"""


def _write_temp_script(script_content: str) -> Path:
    """Write Maya script to temporary file.

    Args:
        script_content: Python script content to write.

    Returns:
        Path to created temporary file.

    Note:
        Caller is responsible for cleanup.
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        script_path = Path(f.name)
        f.write(script_content)
    return script_path


def _execute_mayapy_script(maya_interpreter_path: Path, script_path: Path) -> str:
    """Execute Maya Python script and return output.

    Args:
        maya_interpreter_path: Path to mayapy executable.
        script_path: Path to script file to execute.

    Returns:
        Standard output from script execution.

    Raises:
        RuntimeError: If execution fails.
        subprocess.TimeoutExpired: If execution exceeds 60 seconds.
    """
    result = subprocess.run(
        [str(maya_interpreter_path), str(script_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"mayapy execution failed with code {result.returncode}:\n"
            f"STDOUT: {result.stdout}\n"
            f"STDERR: {result.stderr}"
        )

    return result.stdout


def _parse_geometry_data(stdout: str) -> Tuple[npt.NDArray, npt.NDArray]:
    """Parse geometry data from Maya script output.

    Args:
        stdout: Standard output containing JSON geometry data.

    Returns:
        Tuple of (vertices, faces) as numpy arrays.

    Raises:
        RuntimeError: If JSON parsing fails or data is invalid.
    """
    # Parse JSON output (last line should be JSON)
    output_lines = stdout.strip().split("\n")
    json_output = output_lines[-1]

    try:
        data = json.loads(json_output)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse Maya output as JSON: {e}") from e

    if "error" in data:
        raise RuntimeError(f"Maya loading error: {data['error']}")

    vertices = np.array(data["vertices"], dtype=np.float64)
    faces = np.array(data["faces"], dtype=np.int32)

    return vertices, faces


def _load_obj_subprocess(
    obj_path: Path, maya_interpreter_path: Path
) -> Tuple[npt.NDArray, npt.NDArray]:
    """Load OBJ via mayapy subprocess (refactored for clarity).

    This function spawns mayapy as a subprocess and uses it to load
    the OBJ file and extract geometry data. The implementation is split
    into focused helper functions for better maintainability and testability.

    Args:
        obj_path: Path to OBJ file.
        maya_interpreter_path: Path to mayapy executable.

    Returns:
        Tuple of (vertices, faces) as numpy arrays.

    Raises:
        FileNotFoundError: If mayapy executable doesn't exist.
        RuntimeError: If mayapy execution fails.
        subprocess.TimeoutExpired: If mayapy takes longer than 60 seconds.
    """
    _validate_maya_interpreter(maya_interpreter_path)
    script_content = _generate_maya_loading_script(obj_path)
    script_path = _write_temp_script(script_content)

    try:
        stdout = _execute_mayapy_script(maya_interpreter_path, script_path)
        return _parse_geometry_data(stdout)
    finally:
        script_path.unlink(missing_ok=True)
