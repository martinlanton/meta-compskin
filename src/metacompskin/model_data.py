"""Blendshape model data container.

This module provides the BlendshapeModelData dataclass for loading and storing
facial blendshape model data from NPZ files. The class encapsulates all model-specific
data including deltas, rest pose geometry, and rig logic information.

Example:
    >>> from metacompskin.data import BlendshapeModelData
    >>> model_data = BlendshapeModelData.from_npz("path/to/aura.npz")
    >>> print(f"Model: {model_data.model_name}, vertices: {model_data.n_vertices}")
    Model: aura, vertices: 7306
    >>> print(f"Alpha: {model_data.alpha}")
    Alpha: 10.0
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import numpy.typing as npt

from metacompskin.constants import get_alpha_for_model

# Validation constants for array shapes
_SPATIAL_DIMS = 3  # X, Y, Z coordinates
_DELTAS_NDIM = 3  # Deltas are 3D: (n_blendshapes, n_vertices, spatial_dims)
_VERTS_NDIM = 2  # Vertices are 2D: (n_vertices, spatial_dims)
_FACES_NDIM = 2  # Faces are 2D: (n_faces, vertices_per_face)
_QUAD_VERTS = 4  # Number of vertices per quad face


@dataclass(frozen=True)
class BlendshapeModelData:
    """Immutable container for blendshape model data loaded from NPZ files.

    This dataclass encapsulates all mesh and rig data needed for compressed skinning
    optimization. It loads data from NPZ archives and validates shape consistency.

    The model data follows the delta-blendshape formulation from Equation 1 of the paper:
        v̂₀,ᵢ + Σ(k=1 to S) cₖ(v̂ₖ,ᵢ - v̂₀,ᵢ)

    where deltas contain (v̂ₖ,ᵢ - v̂₀,ᵢ) for all shapes k and vertices i.

    Attributes:
        deltas: Blendshape delta vectors, shape (n_blendshapes, n_vertices, 3).
            Contains vertex displacements from rest pose for each blendshape.
        rest_verts: Rest pose vertex positions, shape (n_vertices, 3).
            The neutral facial expression (all blend weights = 0).
        rest_faces: Mesh face indices, shape (n_faces, 4) for quad meshes.
            Defines mesh topology/connectivity.
        inbetween_info: Dictionary containing inbetween shape definitions.
            Used by rig logic for interpolated shapes between extremes.
        combination_info: Dictionary containing corrective shape definitions.
            Used by rig logic for fixing artifacts from linear blending.
        model_name: Name extracted from NPZ filename (e.g., "aura", "jupiter").
            Used for model-specific parameter lookup.
        alpha: Laplacian regularization parameter for this model.
            Lower values (e.g., 10) for high-density meshes, higher values (e.g., 50)
            for low-density meshes. Controls smoothness of optimization results.

    Example:
        >>> # Load from NPZ file
        >>> model = BlendshapeModelData.from_npz("data/source_models/aura.npz")
        >>> print(model)
        BlendshapeModelData(model_name='aura', n_blendshapes=230, n_vertices=7306)
        >>> print(f"Faces: {model.rest_faces.shape}")
        Faces: (14408, 4)
        >>> print(f"Alpha: {model.alpha}")
        Alpha: 10.0

        >>> # Or create from arrays directly
        >>> deltas = np.random.randn(230, 7306, 3)
        >>> rest_verts = np.random.randn(7306, 3)
        >>> rest_faces = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])  # Quads
        >>> model = BlendshapeModelData(
        ...     deltas=deltas,
        ...     rest_verts=rest_verts,
        ...     rest_faces=rest_faces,
        ...     inbetween_info={},
        ...     combination_info={},
        ...     model_name="custom",
        ...     alpha=15.0,
        ... )

    References:
        Section 2.1 of "Compressed Skinning for Facial Blendshapes" for blendshape
        model formulation and delta notation.
    """

    deltas: npt.NDArray[np.floating]
    rest_verts: npt.NDArray[np.floating]
    rest_faces: npt.NDArray[np.integer]
    inbetween_info: Dict[str, Any]
    combination_info: Dict[str, Any]
    model_name: str
    alpha: float

    @classmethod
    def from_npz(
        cls, npz_file_path: str, alpha: Optional[float] = None
    ) -> "BlendshapeModelData":
        """Create BlendshapeModelData instance from an NPZ file.

        Loads all required arrays from the NPZ archive, extracts the model name
        from the filename, and validates data consistency before creating the
        immutable instance. The alpha value for Laplacian regularization is
        automatically determined based on the model name, but can be overridden.

        Args:
            npz_file_path: Path to NPZ file containing model data.
                Expected keys: 'deltas', 'rest_verts', 'rest_faces',
                'inbetween_info', 'combination_info'.
            alpha: Optional override for Laplacian regularization parameter.
                If None, uses value from constants.MODEL_ALPHA_VALUES for the model.

        Returns:
            BlendshapeModelData instance with data loaded from the NPZ file.

        Raises:
            FileNotFoundError: If NPZ file does not exist.
            KeyError: If required keys are missing from NPZ file.
            ValueError: If array shapes are inconsistent or validation fails.

        Example:
            >>> # Alpha automatically determined from model name
            >>> model = BlendshapeModelData.from_npz("data/source_models/aura.npz")
            >>> print(model.alpha)  # 10.0 (from constants)

            >>> # Override alpha manually
            >>> model = BlendshapeModelData.from_npz("data/source_models/aura.npz", alpha=25.0)
            >>> print(model.alpha)  # 25.0
        """
        path = Path(npz_file_path)

        if not path.exists():
            raise FileNotFoundError(f"NPZ file not found: {npz_file_path}")

        # Extract model name from filename (e.g., "aura.npz" -> "aura")
        model_name = path.stem

        # Get alpha value: use override if provided, otherwise lookup from constants
        if alpha is None:
            alpha = get_alpha_for_model(model_name)

        # Load NPZ file
        npz_data = np.load(str(path), allow_pickle=True)

        # Extract required fields
        required_keys = [
            "deltas",
            "rest_verts",
            "rest_faces",
            "inbetween_info",
            "combination_info",
        ]
        missing_keys = [key for key in required_keys if key not in npz_data.files]
        if missing_keys:
            raise KeyError(f"Missing required keys in NPZ file: {missing_keys}")

        deltas = npz_data["deltas"]
        rest_verts = npz_data["rest_verts"]
        rest_faces = npz_data["rest_faces"]
        inbetween_info = npz_data["inbetween_info"].item()
        combination_info = npz_data["combination_info"].item()

        # VALIDATE BEFORE creating the frozen immutable instance
        cls._validate_arrays(
            deltas, rest_verts, rest_faces, inbetween_info, combination_info
        )

        return cls(
            deltas=deltas,
            rest_verts=rest_verts,
            rest_faces=rest_faces,
            inbetween_info=inbetween_info,
            combination_info=combination_info,
            model_name=model_name,
            alpha=alpha,
        )

    @staticmethod
    def _validate_arrays(
        deltas: npt.NDArray[np.floating],
        rest_verts: npt.NDArray[np.floating],
        rest_faces: npt.NDArray[np.integer],
        inbetween_info: Dict[str, Any],
        combination_info: Dict[str, Any],
    ):
        """Validate model data arrays before creating frozen instance.

        This static method validates data consistency before the immutable
        dataclass is constructed. This is cleaner than validating after
        construction since frozen dataclasses shouldn't be modified.

        Args:
            deltas: Blendshape delta vectors to validate.
            rest_verts: Rest pose vertex positions to validate.
            rest_faces: Mesh face indices to validate.
            inbetween_info: Inbetween shape definitions to validate.
            combination_info: Corrective shape definitions to validate.

        Raises:
            ValueError: If validation fails with descriptive error message.

        Note:
            This is called by from_npz() before creating the instance.
            Direct construction via __init__() does NOT validate - this is
            intentional to allow flexible testing/programmatic creation.
        """
        # Check deltas shape
        if len(deltas.shape) != _DELTAS_NDIM:
            raise ValueError(
                f"deltas must be 3D array (n_blendshapes, n_vertices, 3), "
                f"got shape {deltas.shape}"
            )

        if deltas.shape[2] != _SPATIAL_DIMS:
            raise ValueError(
                f"deltas last dimension must be 3 (x,y,z), got {deltas.shape[2]}"
            )

        # Check rest_verts shape
        if len(rest_verts.shape) != _VERTS_NDIM:
            raise ValueError(
                f"rest_verts must be 2D array (n_vertices, 3), "
                f"got shape {rest_verts.shape}"
            )

        if rest_verts.shape[1] != _SPATIAL_DIMS:
            raise ValueError(
                f"rest_verts second dimension must be 3 (x,y,z), "
                f"got {rest_verts.shape[1]}"
            )

        # Check consistency between deltas and rest_verts
        if deltas.shape[1] != rest_verts.shape[0]:
            raise ValueError(
                f"deltas vertex count ({deltas.shape[1]}) must match "
                f"rest_verts vertex count ({rest_verts.shape[0]})"
            )

        # Check rest_faces shape
        if len(rest_faces.shape) != _FACES_NDIM:
            raise ValueError(
                f"rest_faces must be 2D array (n_faces, 4), "
                f"got shape {rest_faces.shape}"
            )

        if rest_faces.shape[1] != _QUAD_VERTS:
            raise ValueError(
                f"rest_faces second dimension must be 4 (quads), "
                f"got {rest_faces.shape[1]}"
            )

        # Check that face indices are within valid range
        max_face_idx = rest_faces.max()
        if max_face_idx >= rest_verts.shape[0]:
            raise ValueError(
                f"rest_faces contains invalid vertex index {max_face_idx}, "
                f"max valid index is {rest_verts.shape[0] - 1}"
            )

        # Check data types
        if not np.issubdtype(deltas.dtype, np.floating):
            raise ValueError(
                f"deltas must have floating point dtype, got {deltas.dtype}"
            )

        if not np.issubdtype(rest_verts.dtype, np.floating):
            raise ValueError(
                f"rest_verts must have floating point dtype, got {rest_verts.dtype}"
            )

        if not np.issubdtype(rest_faces.dtype, np.integer):
            raise ValueError(
                f"rest_faces must have integer dtype, got {rest_faces.dtype}"
            )

        # Check dictionary types
        if not isinstance(inbetween_info, dict):
            raise ValueError(f"inbetween_info must be dict, got {type(inbetween_info)}")

        if not isinstance(combination_info, dict):
            raise ValueError(
                f"combination_info must be dict, got {type(combination_info)}"
            )

    @property
    def n_blendshapes(self) -> int:
        """Number of blendshapes in the source model.

        Returns:
            Number of blendshapes (S in paper notation).
        """
        return self.deltas.shape[0]

    @property
    def n_vertices(self) -> int:
        """Number of vertices in the mesh.

        Returns:
            Number of vertices (N in paper notation).
        """
        return self.rest_verts.shape[0]

    @property
    def n_faces(self) -> int:
        """Number of faces in the mesh.

        Returns:
            Number of quad faces.
        """
        return self.rest_faces.shape[0]

    def __repr__(self) -> str:
        """Return string representation of model data.

        Returns:
            Compact string showing model name and key dimensions.
        """
        return (
            f"BlendshapeModelData(model_name='{self.model_name}', "
            f"n_blendshapes={self.n_blendshapes}, "
            f"n_vertices={self.n_vertices})"
        )

    def print_details(self):
        """Print detailed information about all data arrays.

        Similar to the old print_content() method, shows shapes and dtypes
        for debugging purposes.
        """
        print(f"Model: {self.model_name}")
        print(f"  deltas: {self.deltas.shape} {self.deltas.dtype}")
        print(f"  rest_verts: {self.rest_verts.shape} {self.rest_verts.dtype}")
        print(f"  rest_faces: {self.rest_faces.shape} {self.rest_faces.dtype}")
        print(f"  inbetween_info: {type(self.inbetween_info)} (dict)")
        print(f"  combination_info: {type(self.combination_info)} (dict)")
