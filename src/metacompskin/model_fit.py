"""Implements the SkinCompressor class for blendshape compression and optimization.

Handles loading model data, running optimization, and saving results.
"""

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree.

import time
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import scipy as sp
import torch
from scipy.sparse.csgraph import dijkstra
from scipy.spatial.distance import cdist

from metacompskin.model_data import BlendshapeModelData
from metacompskin.utils import add_homogeneous_coordinate, npf

location = Path(__file__).parent
data_location = location / "data"

# Optimization constants
_SPARSITY_THRESHOLD = 1e-4  # Threshold for counting non-zero values in sparse matrices

_REST_JOINT_MATRICES_NDIM = (
    3  # Expected number of dimensions for rest_joint_matrices: (P, 4, 4)
)
_JOINT_MATRIX_SHAPE = (4, 4)  # Expected shape of each rest joint matrix
_DEFAULT_NUMBER_OF_BONES = 100  # P
_DEFAULT_MAX_INFLUENCES = 8  # K
_DEFAULT_SEED = 12345

_ROOT_WEIGHT_FLOOR = 1e-6  # smallest weight the root influence can hold
_ROOT_JOINT_NEW = "new"
_ROOT_JOINT_NONE = "none"
_WEIGHT_INITS = ("noise", "distance")
_STATIC_MOTION_FRACTION = (
    1e-3  # a vertex is static below this share of the largest motion
)


def _build_adjacency_matrix(faces: np.ndarray, n_verts: int) -> sp.sparse.csr_matrix:
    """Build a binary vertex adjacency matrix from polygon faces.

    Connects every pair of vertices that share a face (all combinations within
    each face), matching libigl's adjacency_matrix behaviour for quad meshes.

    Args:
        faces: Face index array, shape (n_faces, verts_per_face).
        n_verts: Total number of vertices.

    Returns:
        Symmetric binary CSR adjacency matrix, shape (n_verts, n_verts).
    """
    i_verts, j_verts = _face_vertex_pairs(faces)
    rows = np.concatenate([i_verts, j_verts])
    cols = np.concatenate([j_verts, i_verts])
    adj = sp.sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float64), (rows, cols)),
        shape=(n_verts, n_verts),
    )
    adj.data[:] = 1.0  # deduplicate: shared edges appear in two faces → clip to binary
    return adj


def _face_vertex_pairs(faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Lists every pair of vertices that share a face, with repeats.

    Args:
        faces: Face index array, shape (n_faces, verts_per_face).

    Returns:
        Two index arrays of equal length, one pair per entry.
    """
    verts_per_face = faces.shape[1]
    pair_cols = np.array(list(combinations(range(verts_per_face), 2)))
    return faces[:, pair_cols[:, 0]].ravel(), faces[:, pair_cols[:, 1]].ravel()


def _edge_length_graph(
    rest_verts: np.ndarray, rest_faces: np.ndarray
) -> sp.sparse.csr_matrix:
    """Builds the mesh graph weighted by Euclidean edge length.

    Every pair of vertices sharing a face is an edge (quad diagonals
    included), so the graph matches the Laplacian's adjacency.

    Args:
        rest_verts: Vertex positions, shape (N, 3).
        rest_faces: Face index array, shape (F, verts_per_face).

    Returns:
        Upper-triangular CSR matrix, shape (N, N), one length per edge.
    """
    i_verts, j_verts = _face_vertex_pairs(rest_faces)
    edges = np.unique(np.sort(np.stack([i_verts, j_verts], axis=1), axis=1), axis=0)
    lengths = np.linalg.norm(rest_verts[edges[:, 0]] - rest_verts[edges[:, 1]], axis=1)
    n_verts = len(rest_verts)
    return sp.sparse.csr_matrix(
        (lengths, (edges[:, 0], edges[:, 1])), shape=(n_verts, n_verts)
    )


def geodesic_joint_distances(
    rest_verts: np.ndarray, rest_faces: np.ndarray, joint_positions: np.ndarray
) -> np.ndarray:
    """Measures how far every vertex is from every joint along the mesh surface.

    Each joint is snapped to its nearest vertex; the distance to a vertex is
    that gap plus the shortest path along mesh edges (Dijkstra). Walking along
    the surface keeps the upper lip far from a lower-lip joint even though the
    two are close in space. Vertices no path can reach from a joint, such as a
    separate shell, get their straight-line distance plus the longest
    reachable distance, so joints on a vertex's own shell always rank first
    and the rest fall back to spatial order.

    Args:
        rest_verts: Rest vertex positions, shape (N, 3).
        rest_faces: Face index array, shape (F, verts_per_face).
        joint_positions: Joint rest positions in the same space, shape (P, 3).

    Returns:
        Distances, shape (P, N), float64 and finite.

    References:
        Dijkstra 1959, shortest paths on the graph of mesh edges.
    """
    verts = np.asarray(rest_verts, dtype=np.float64)
    joints = np.asarray(joint_positions, dtype=np.float64)
    euclidean = cdist(joints, verts)  # (P, N)
    sources = euclidean.argmin(axis=1)  # (P,)
    distances = dijkstra(
        _edge_length_graph(verts, rest_faces), directed=False, indices=sources
    )
    distances += euclidean[np.arange(len(joints)), sources][:, None]
    unreachable = ~np.isfinite(distances)
    if unreachable.any():
        distances[unreachable] = euclidean[unreachable] + distances[~unreachable].max()
    return distances


def candidate_joint_mask(
    distances: np.ndarray, candidates_per_vertex: int
) -> np.ndarray:
    """Marks, per vertex, the joints close enough to be allowed to drive it.

    Args:
        distances: Joint-to-vertex distances, shape (P, N).
        candidates_per_vertex: Number of nearest joints M kept per vertex.
            A value of P or more keeps every joint.

    Returns:
        Boolean mask, shape (P, N), with exactly ``min(M, P)`` True entries
        per column.

    Raises:
        ValueError: If ``candidates_per_vertex`` is below one.
    """
    if candidates_per_vertex < 1:
        raise ValueError(
            f"candidates_per_vertex={candidates_per_vertex} must be at least one"
        )
    n_joints = distances.shape[0]
    if candidates_per_vertex >= n_joints:
        return np.ones(distances.shape, dtype=bool)
    nearest = np.argpartition(distances, candidates_per_vertex - 1, axis=0)
    mask = np.zeros(distances.shape, dtype=bool)
    np.put_along_axis(mask, nearest[:candidates_per_vertex], True, axis=0)
    return mask


def distance_weights(distances: np.ndarray, max_influences: int) -> np.ndarray:
    r"""Builds skinning weights from each vertex's distance to the joints.

    Each vertex is weighted to its ``max_influences`` nearest joints with a
    falloff that reaches zero at the next nearest joint:
    :math:`w_j \propto (1 - d_j / d_{K+1})^2` for the K nearest, zero beyond,
    normalised to sum to one. The nearest joint always gets the largest
    weight, so a joint's support is the region around it. These are the
    starting weights of a solve with joint matrices (see
    :class:`SkinCompressor`), not a finished bind.

    Args:
        distances: Joint-to-vertex distances, shape (P, N), for example from
            :func:`geodesic_joint_distances`.
        max_influences: K, the number of joints each vertex starts on.

    Returns:
        Weights of shape (P, N), float64, every column summing to one.

    References:
        Baran & Popović 2007 (Pinocchio) and Dionne & de Lasa 2013 for
        distance-based binds; the falloff here is the simplest of that family.
    """
    n_joints, n_verts = distances.shape
    keep = min(max_influences, n_joints - 1)
    order = np.argsort(distances, axis=0)[: keep + 1]  # (K+1, N), nearest first
    nearest = np.take_along_axis(distances, order, axis=0)
    radius = np.maximum(nearest[-1], 1e-12)
    falloff = np.clip(1.0 - nearest[:-1] / radius, 0.0, None) ** 2  # (K, N)
    falloff[0, falloff.sum(axis=0) <= 0] = 1.0  # all K joints at one distance
    weights = np.zeros_like(distances, dtype=np.float64)
    np.put_along_axis(weights, order[:-1], falloff, axis=0)
    return weights / weights.sum(axis=0)


def static_vertex_mask(
    deltas: np.ndarray, fraction: float = _STATIC_MOTION_FRACTION
) -> np.ndarray:
    """Marks the vertices no blendshape moves.

    Args:
        deltas: Blendshape offsets, shape (S, N, 3).
        fraction: A vertex is static when its largest displacement over all
            shapes is below this share of the model's largest displacement.

    Returns:
        Boolean mask of shape (N,).
    """
    motion = np.linalg.norm(np.asarray(deltas, dtype=np.float64), axis=2).max(axis=0)
    return motion < fraction * motion.max()


def joint_chain_matrix(joint_parents: Sequence[int] | np.ndarray) -> np.ndarray:
    """Marks, for every joint, itself and each of its ancestors.

    Args:
        joint_parents: Index of each joint's parent, or -1 for a joint with
            no parent; shape (P,). Parents may be listed after their children.

    Returns:
        Matrix of shape (P, P), float64, where entry (j, a) is 1 when joint a
        is joint j or one of its ancestors, else 0.

    Raises:
        ValueError: If a parent index is outside [-1, P), a joint is its own
            parent, or the parents form a cycle.
    """
    parents = np.asarray(joint_parents, dtype=np.int64)
    n_joints = len(parents)
    if ((parents < -1) | (parents >= n_joints)).any():
        raise ValueError(
            f"joint parents must lie in [-1, {n_joints}), got {parents.tolist()}"
        )
    if (parents == np.arange(n_joints)).any():
        raise ValueError("a joint cannot be its own parent")
    chain = np.eye(n_joints)
    for joint in range(n_joints):
        ancestor, depth = parents[joint], 0
        while ancestor != -1:
            depth += 1
            if depth > n_joints:
                raise ValueError(f"joint parents form a cycle through joint {joint}")
            chain[joint, ancestor] = 1.0
            ancestor = parents[ancestor]
    return chain


def hierarchical_coefficients(
    local: torch.Tensor, pivots: torch.Tensor, chain: torch.Tensor
) -> torch.Tensor:
    r"""Turns per-joint local deltas into the joint transforms skinning uses.

    A local delta of joint j rotates about the joint's own rest position
    :math:`p_j` and translates, :math:`D v = r \times (v - p_j) + t`. In the
    shared frame that is rotation :math:`r` and translation
    :math:`t - r \times p_j`. A joint's transform is the sum of its own and
    its ancestors' deltas, :math:`N_j = \sum_{a \in chain(j)} D_a`, so a child
    moves with its parent and stores only what it adds.

    Args:
        local: Local coefficients (rotation x, y, z, translation x, y, z),
            shape (6, S, P, 1, 1).
        pivots: Joint rest positions in the solver's centred frame, shape (P, 3).
        chain: :func:`joint_chain_matrix` as a tensor, shape (P, P).

    Returns:
        Coefficients in the shared frame, shape (6, S, P, 1, 1), the layout
        the transform basis expects.
    """
    rotation, translation = local[:3], local[3:]
    pivot = pivots.T.reshape(3, 1, -1, 1, 1)
    pivot_motion = torch.stack(
        [
            rotation[1] * pivot[2] - rotation[2] * pivot[1],
            rotation[2] * pivot[0] - rotation[0] * pivot[2],
            rotation[0] * pivot[1] - rotation[1] * pivot[0],
        ]
    )  # r x p, shape (3, S, P, 1, 1)
    shared = torch.cat([rotation, translation - pivot_motion])
    return torch.einsum("isakl,ja->isjkl", shared, chain)


def accumulate_joint_transforms(
    local_shape_xform: np.ndarray, joint_parents: Sequence[int] | np.ndarray
) -> np.ndarray:
    """Adds every joint's ancestors' local deltas to its own.

    This is the runtime counterpart of :func:`hierarchical_coefficients` for
    the ``localShapeXform`` array of a compressed file.

    Args:
        local_shape_xform: Local deltas in the shared frame, shape (3S, 4P).
        joint_parents: Parent of each joint, -1 for none, shape (P,).

    Returns:
        Joint transforms with the layout of ``shapeXform``, shape (3S, 4P).
    """
    chain = joint_chain_matrix(joint_parents)
    rows = local_shape_xform.shape[0]
    blocks = local_shape_xform.reshape(rows, len(chain), 4)
    return np.einsum("rak,ja->rjk", blocks, chain).reshape(rows, -1)


@dataclass(frozen=True)
class TrainingPhase:
    """One stage of the optimisation with fixed sparsity budgets.

    Attributes:
        iterations: Adam steps in this phase.
        max_influences: K, weights kept per vertex by the projection.
        total_nnz_B_rt: L, delta coefficients kept globally by the projection.
        normalize_weights: Whether the forward pass divides each vertex's
            weights by their sum (partition of unity).

    Raises:
        ValueError: If any count is below one.

    References:
        Section 4 of the paper, projections (1)-(3).
    """

    iterations: int
    max_influences: int
    total_nnz_B_rt: int  # noqa: N815 (matches SkinCompressor)
    normalize_weights: bool

    def __post_init__(self) -> None:
        """Validates the counts after dataclass assignment.

        Raises:
            ValueError: If any count is below one.
        """
        for name in ("iterations", "max_influences", "total_nnz_B_rt"):
            if getattr(self, name) < 1:
                raise ValueError(
                    f"{name} must be at least 1, got {getattr(self, name)}"
                )


@dataclass(frozen=True)
class ReconstructionError:
    """Final fit error of a compression run, in model units.

    Attributes:
        max_abs: Largest per-axis absolute error over all vertices and shapes (MXE).
        mean_abs: Mean per-axis absolute error (MAE).

    References:
        Paper Table 1 ("MXE", "MAE"); docs/user_guide/evaluating_results.md.
    """

    max_abs: float
    mean_abs: float


def build_training_schedule(
    stage_iterations: Sequence[int],
    max_influences: int,
    total_nnz_B_rt: int,  # noqa: N803 (matches SkinCompressor)
    start_influences: int | None = None,
    warm_up_enabled: bool = True,
) -> tuple[TrainingPhase, ...]:
    """Builds the phases run() executes, one stage per entry of stage_iterations.

    Stage i runs ``stage_iterations[i]`` steps. The influence budget starts at
    ``start_influences`` and shrinks geometrically each stage down to
    ``max_influences`` in the last one; with the default ceiling of
    ``max_influences * 2 ** (N - 1)``, N being the number of stages, it simply
    halves each stage. The warm-up runs at ``start_influences`` when given,
    even for a single stage, so the top-K cut keeps every candidate while
    ownership is decided. The delta budget is constant. When enabled, the
    first stage is preceded by a warm-up of the same length with weight
    normalisation off; every stage itself runs normalised. One entry is the
    classic two-phase schedule.

    Args:
        stage_iterations: Steps per stage; non-empty.
        max_influences: K reached in the final stage.
        total_nnz_B_rt: L used in every phase.
        start_influences: Budget of the first stage; None doubles per stage.
            With a candidate-joint mask this is the mask width, the widest
            budget a vertex can use (see :func:`candidate_joint_mask`).
        warm_up_enabled: Whether to include an unnormalised warm-up phase
            before the first stage (default True). When False, only stages
            run, all normalised.

    Returns:
        ``len(stage_iterations) + 1`` phases if warm_up_enabled, else
        ``len(stage_iterations)`` phases.

    Raises:
        ValueError: If ``stage_iterations`` is empty or an entry is below one,
            or ``start_influences`` is below ``max_influences``.

    References:
        Zhu & Gupta 2017 (gradual magnitude pruning).
    """
    if not stage_iterations:
        raise ValueError("iterations must contain at least one stage")
    budgets = _annealed_influence_budgets(
        len(stage_iterations), max_influences, start_influences
    )
    stages = [
        TrainingPhase(steps, budget, total_nnz_B_rt, normalize_weights=True)
        for steps, budget in zip(stage_iterations, budgets, strict=True)
    ]

    if not warm_up_enabled:
        return tuple(stages)

    warm_up_budget = (
        start_influences if start_influences is not None else stages[0].max_influences
    )
    warm_up = TrainingPhase(
        stages[0].iterations,
        warm_up_budget,
        total_nnz_B_rt,
        normalize_weights=False,
    )
    return (warm_up, *stages)


def _annealed_influence_budgets(
    n_stages: int, max_influences: int, start_influences: int | None
) -> list[int]:
    """Interpolates the per-stage influence budget from a ceiling down to K.

    Args:
        n_stages: Number of stages N.
        max_influences: K, the budget of the last stage.
        start_influences: Budget of the first stage; None means
            ``K * 2 ** (N - 1)``, the halving rule.

    Returns:
        N budgets, geometrically spaced and rounded to integers; ``[K]`` for a
        single stage.

    Raises:
        ValueError: If ``start_influences`` is below ``max_influences``.
    """
    if start_influences is not None and start_influences < max_influences:
        raise ValueError(
            f"start_influences={start_influences} must not be below "
            f"max_influences={max_influences}"
        )
    if n_stages == 1:
        return [max_influences]
    ceiling = (
        max_influences * 2 ** (n_stages - 1)
        if start_influences is None
        else start_influences
    )
    ratio = ceiling / max_influences
    return [
        round(max_influences * ratio ** ((n_stages - 1 - i) / (n_stages - 1)))
        for i in range(n_stages)
    ]


def _resolve_root_joint(
    requested: int | str | None, has_joint_matrices: bool, number_of_bones: int
) -> tuple[int | None, int]:
    """Turns the root_joint argument into (root index or None, bone count).

    None means "new" with joint matrices and no root without; "new" appends
    a synthetic root as the last bone; an index designates a supplied bone;
    "none" turns the root off. Bools and out-of-range values raise.

    Args:
        requested: The constructor argument: None for the default, "new" to
            append a synthetic root, "none" to disable, or an int index.
        has_joint_matrices: Whether joint matrices are provided.
        number_of_bones: P, the number of bones supplied (not counting a synthetic root).

    Returns:
        (root_index or None, actual_bone_count). For a synthetic root,
        root_index is P and actual_bone_count is P + 1.

    Raises:
        ValueError: If requested is a bool, an out-of-range index, or an
            unrecognized string.
    """
    if isinstance(requested, bool):
        raise ValueError(
            f"root_joint must be an int, str or None, not {type(requested).__name__}"
        )

    if requested is None:
        if has_joint_matrices:
            return (number_of_bones, number_of_bones + 1)
        return (None, number_of_bones)

    if requested == _ROOT_JOINT_NEW:
        return (number_of_bones, number_of_bones + 1)

    if requested == _ROOT_JOINT_NONE:
        return (None, number_of_bones)

    if isinstance(requested, int):
        if not 0 <= requested < number_of_bones:
            raise ValueError(
                f"root_joint={requested} out of range [0, {number_of_bones})"
            )
        return (requested, number_of_bones)

    raise ValueError(
        f"root_joint={requested!r} must be None, 'new', 'none', or an int in [0, {number_of_bones})"
    )


def _resolve_weight_init(requested: str | None, has_joint_matrices: bool) -> str:
    """Turns the weight_init argument into "noise" or "distance".

    Args:
        requested: The constructor argument, or None for the default.
        has_joint_matrices: Whether joint positions are available.

    Returns:
        "distance" with joint matrices, "noise" without, unless requested.

    Raises:
        ValueError: If the value is unknown, or "distance" is requested
            without joint matrices.
    """
    if requested is None:
        return "distance" if has_joint_matrices else "noise"
    if requested not in _WEIGHT_INITS:
        raise ValueError(f"weight_init={requested!r} must be one of {_WEIGHT_INITS}")
    if requested == "distance" and not has_joint_matrices:
        raise ValueError(
            'weight_init="distance" needs rest_joint_matrices: the start is built '
            "from each vertex's distance to the joints"
        )
    return requested


def _resolve_candidate_joints(
    requested: int | None,
    has_joint_matrices: bool,
    max_influences: int,
    number_of_bones: int,
) -> int | None:
    """Turns the candidate_joints_per_vertex argument into a count or None.

    Args:
        requested: The constructor argument: None for the default, 0 to turn
            the filter off, otherwise the count M.
        has_joint_matrices: Whether joint positions are available.
        max_influences: K, the lower bound for M.
        number_of_bones: P, the exclusive upper bound for M.

    Returns:
        M, or None when there is no filter.

    Raises:
        ValueError: If a count is requested without joint matrices, or lies
            outside K <= M < P.
    """
    if not has_joint_matrices:
        if requested:
            raise ValueError(
                "candidate_joints_per_vertex needs rest_joint_matrices: the "
                "filter ranks joints by their distance to each vertex"
            )
        return None
    if requested is None:
        default = min(max_influences + 2, number_of_bones - 1)
        # Fewer joints than K + 1 cannot run at all; leave that to run().
        return default if default >= max_influences else None
    if requested == 0:
        return None
    if not max_influences <= requested < number_of_bones:
        raise ValueError(
            f"candidate_joints_per_vertex={requested} must satisfy "
            f"max_influences={max_influences} <= M < "
            f"number_of_bones={number_of_bones}; pass 0 to turn the filter off"
        )
    return requested


def _as_stage_iterations(iterations: int | Sequence[int]) -> tuple[int, ...]:
    """Normalises the ``iterations`` constructor argument to a tuple of stages.

    Args:
        iterations: A single step count (one stage), or one step count per
            stage.

    Returns:
        ``(iterations,)`` for an int, ``tuple(iterations)`` for a sequence.

    Raises:
        ValueError: If a sequence is given and it is empty.
    """
    if isinstance(iterations, int):
        return (iterations,)
    stages = tuple(iterations)
    if not stages:
        raise ValueError("iterations must contain at least one stage")
    return stages


class SkinCompressor:
    """Optimizer for sparse skinning decomposition of facial blendshapes.

    This class implements the compressed skinning decomposition method from
    "Compressed Skinning for Facial Blendshapes" (SIGGRAPH 2024). It converts
    blendshape deltas into a sparse linear blend skinning representation using
    proximal gradient methods with Adam optimization.

    The optimization solves the non-convex problem (Equation 9):
        min_{w,N} Σᵢ Σₖ (Eᵢₖ)^p
        where Eᵢₖ = ‖vₖ,ᵢ - v₀,ᵢ - Σⱼ wᵢ,ⱼ Nₖ,ⱼ v₀,ᵢ‖

    This decomposes the input matrix A ∈ ℝ^(3S×N) into:
        A ≈ B·C
    where:
        - B ∈ ℝ^(3S×4P): sparse transformation matrices (~90% zeros)
        - C ∈ ℝ^(4P×N): weighted rest pose with sparse weights (K non-zeros per vertex)

    Key features:
        - Proximal projection enforces sparsity constraints (Section 4)
        - Sparse transformation matrices achieve 5-7× memory savings
        - 2-3× speed improvement over dense methods (Dem Bones)
        - Laplacian regularization for smooth results
        - Two-phase training (unnormalized → normalized weights)

    Attributes:
        model_data: BlendshapeModelData with geometry and rig info.
        iterations: Value given to the constructor (int or a sequence),
            kept as given; use stage_iterations or schedule instead.
        stage_iterations: iterations normalised to a tuple, one entry per
            stage.
        schedule: The TrainingPhase objects run() executes, built from
            stage_iterations, max_influences and total_nnz_B_rt. May be
            reassigned before run() for a hand-built schedule.
        rest_joint_matrices_3x4: Optional 3×4 affine matrices for joint rest poses.
            Extracted from the 4×4 matrices provided during initialization.
            If None, identity matrices are used (default behavior).
        number_of_bones: Number of proxy bones P (default 100 when neither the
            argument nor rest_joint_matrices is given; otherwise the number of
            matrices provided). The paper's experiments use P=40 (Table 1).
        max_influences: Max non-zero weights per vertex K (default 8).
            Standard GPU skinning pipeline constraint (Section 2.2).
        candidate_joints_per_vertex: Nearest joints M each vertex may be
            weighted to, or None when no joint matrices were given or the
            filter was turned off. See candidate_joint_mask.
        weight_init: "distance" or "noise", where the weights start; see the
            constructor.
        candidate_mask: Boolean (P, N) tensor built by run() from the joint
            positions, or None without a filter.
        total_nnz_B_rt: Total non-zeros in B_rt matrix (default 6000).
            Corresponds to ~1000 transformations with 6-DOF representation.
            Achieves ~90% sparsity while maintaining accuracy (Table 1).
        init_weight: Initialization scale for random parameters (1e-3).
            Small value starts optimization near zero.
        power: Lp norm exponent for loss function (default 2).
            p=2 gives L2 norm, p=12 for HD fit (Section 4.1).
        seed: Torch random seed for the initial deltas and weights (default
            12345). Set explicitly to compare local minima the optimizer
            converges to (Section 3, non-convexity).
        alpha: Laplacian regularization strength from model_data.
            Controls smoothness: lower for high-density, higher for low-density.
        device: PyTorch device ('cuda' or 'cpu').
        loss_list: Training loss history.
        abserr_list: Training absolute error history.
        reconstruction_error: Final MXE/MAE as a ReconstructionError, set by
            run(). None until run() has completed.

    Example:
        >>> from metacompskin.model_data import BlendshapeModelData
        >>> model_data = BlendshapeModelData.from_npz("data/source_models/aura.npz")
        >>> compressor = SkinCompressor(model_data=model_data, iterations=10000)
        >>> compressor.run(output_location="output/aura_compressed.npz")
        Model: aura
          deltas: (230, 7306, 3) float32
          rest_verts: (7306, 3) float32
          rest_faces: (14408, 4) int32
        Using alpha value: 10.0 for model 'aura'
        00000(0.123) 1.23456e-02 4.56789e-01 850 47824
        ...
        maxDelta 5.82
        meanDelta 0.0384
        >>> # Results: 5-7× memory savings, 2-3× speed improvement vs Dem Bones

    References:
        Section 4 "Compressed Skinning Decomposition" of the paper.
        Tables 1-3 for performance metrics and comparisons.
    """

    def __init__(  # noqa: PLR0913, PLR0917
        self,
        model_data: BlendshapeModelData,
        iterations: int | Sequence[int] = 10000,
        rest_joint_matrices: np.ndarray | list | None = None,
        number_of_bones: int | None = None,
        max_influences: int = _DEFAULT_MAX_INFLUENCES,
        total_nnz_B_rt: int = 6000,
        init_weight: float = 1e-3,
        power: int = 2,
        seed: int = _DEFAULT_SEED,
        candidate_joints_per_vertex: int | None = None,
        root_joint: int | str | None = None,
        warm_up_enabled: bool = True,
        joint_parents: Sequence[int] | None = None,
        deltas_about_joints: bool = False,
        weight_init: str | None = None,
    ):
        """Initializes the SkinCompressor.

        Args:
            model_data: BlendshapeModelData instance containing model geometry and rig info.
            iterations: Steps for a single stage (int, the classic two-phase
                run), or steps per stage (a sequence). A sequence anneals the
                influence budget: it halves each stage from
                max_influences * 2**(N-1) down to max_influences, N being the
                number of stages; the unnormalised warm-up takes the first
                entry's length. See build_training_schedule.
            rest_joint_matrices: Optional array of 4×4 transformation matrices for joint rest poses.
                If provided, must be shape (P, 4, 4) where P is the number of bones.
                Each matrix should be a standard 4×4 homogeneous transformation matrix.
                When provided, the number of bones is the number of matrices supplied.
                If None (default), uses identity matrices at the origin, maintaining
                backward compatibility.
            number_of_bones: Number of proxy bones P. Defaults to 100 when omitted.
                Only needed without rest_joint_matrices; if both are given they
                must agree.
            max_influences: Maximum non-zero weights per vertex K (default 8).
                Must be smaller than the number of bones.
            total_nnz_B_rt: Sparsity budget L, the number of non-zero coefficients
                kept in B_rt across the whole model (default 6000). Cannot exceed
                6·S·P.
            init_weight: Scale of the random initial values of B_rt (default 1e-3).
            power: Exponent p of the error norm in the loss (default 2; 12 for a
                worst-case fit, Section 4.1).
            seed: Torch random seed for the initial deltas and weights
                (default 12345).
            candidate_joints_per_vertex: With rest_joint_matrices, the number
                of nearest joints M (by geodesic distance along the mesh) a
                vertex may take its weights from, so each joint drives the
                region around it. None picks ``min(K + 2, P - 1)``; 0 turns
                the filter off. Must satisfy K <= M < P. Also the ceiling of
                an annealed schedule. Not allowed without joint matrices.
            root_joint: Designates a root influence, one joint whose transforms
                are held at zero and whose weight can never be pruned, for
                static or unexplainable vertices. None (default) means "new"
                when joint matrices are given (a synthetic root appended as
                bone P), else no root. "new" appends a synthetic root with an
                identity rest transform (bone count becomes P + 1). "none"
                disables the root. An int in [0, P) designates a supplied
                joint as the root; the bone count stays P.
            warm_up_enabled: Whether to include an unnormalised warm-up phase
                before the first stage (default True). When False, only stages
                run, all normalised.
            joint_parents: With rest_joint_matrices, the index of each joint's
                parent (-1 for none), one entry per matrix. A joint's transform
                then adds its ancestors' deltas to its own, so a child follows
                its parent and the delta budget is spent on what each joint
                adds. None (default) solves every joint independently.
            deltas_about_joints: With joint_parents, solve each joint's own
                delta about the joint's rest position rather than about the
                mesh centre, so its rotation is directly the joint's local
                rotation. Default False. This changes which coefficients the
                delta budget keeps, which can cost accuracy.
            weight_init: Where the skinning weights start. "distance" builds
                them from each vertex's distance to the joints (see
                :func:`distance_weights`) and puts vertices no shape moves on
                the root; "noise" starts from tiny random values and lets the
                first steps pick the joints. None (default) means "distance"
                with rest_joint_matrices and "noise" without. "distance" is
                not allowed without joint matrices.

        Raises:
            ValueError: If rest_joint_matrices is not shaped (P, 4, 4), or if
                number_of_bones disagrees with the number of matrices given;
                if iterations is an empty sequence or any entry is below one;
                if candidate_joints_per_vertex is given without joint
                matrices or lies outside K <= M < P; if root_joint is a bool,
                an out-of-range int, or an unrecognized string; if
                joint_parents is given without joint matrices, with the wrong
                length, with invalid or cyclic parents, or with a designated
                root joint that has a parent; if deltas_about_joints is set
                without joint_parents; if weight_init is unknown or "distance"
                without joint matrices.

        Example:
            >>> from metacompskin.model_data import BlendshapeModelData
            >>> model_data = BlendshapeModelData.from_npz("data/source_models/aura.npz")
            >>> # Default behavior with identity matrices
            >>> compressor = SkinCompressor(model_data=model_data, iterations=10000)
            >>> compressor.run()
            >>>
            >>> # With custom joint matrices
            >>> import numpy as np
            >>> joint_matrices = np.array([np.eye(4) for _ in range(28)])
            >>> compressor = SkinCompressor(
            ...     model_data=model_data, iterations=10000, rest_joint_matrices=joint_matrices
            ... )
            >>> compressor.run()
            >>>
            >>> # Anneal the influence budget over three stages
            >>> compressor = SkinCompressor(
            ...     model_data=model_data, iterations=(5000, 5000, 10000)
            ... )
            >>>
            >>> # A hand-built schedule (e.g. a longer final stage) overrides it
            >>> compressor.schedule = (
            ...     TrainingPhase(2000, 32, 6000, normalize_weights=False),
            ...     TrainingPhase(8000, 8, 6000, normalize_weights=True),
            ... )
            >>> compressor.run()
        """
        self.model_data = model_data
        self.iterations = iterations

        # Handle rest joint matrices and set number of bones accordingly
        self.rest_joint_matrices_3x4: np.ndarray | None
        if rest_joint_matrices is not None:
            # Bind the array to its own name: the parameter also accepts a list,
            # so reusing it would keep the wider type.
            matrices = np.asarray(rest_joint_matrices)

            # Validate shape
            if (
                matrices.ndim != _REST_JOINT_MATRICES_NDIM
                or matrices.shape[1:] != _JOINT_MATRIX_SHAPE
            ):
                raise ValueError(
                    f"rest_joint_matrices must have shape (P, 4, 4), "
                    f"got shape {matrices.shape}"
                )

            if number_of_bones is not None and number_of_bones != len(matrices):
                raise ValueError(
                    f"number_of_bones={number_of_bones} conflicts with the "
                    f"{len(matrices)} rest_joint_matrices given; omit "
                    "number_of_bones or make them agree."
                )

            # Extract 3×4 affine portion (first 3 rows of each matrix)
            self.rest_joint_matrices_3x4 = matrices[:, :3, :].astype(np.float32)
            self.number_of_bones = len(matrices)
        else:
            self.rest_joint_matrices_3x4 = None
            self.number_of_bones = (
                _DEFAULT_NUMBER_OF_BONES if number_of_bones is None else number_of_bones
            )

        # Resolve root joint before candidate joints to pass the geodesic count
        n_geodesic_joints = self.number_of_bones
        self.root_joint, self.number_of_bones = _resolve_root_joint(
            root_joint,
            has_joint_matrices=self.rest_joint_matrices_3x4 is not None,
            number_of_bones=n_geodesic_joints,
        )

        self.joint_parents = self._resolve_joint_parents(
            joint_parents, n_geodesic_joints
        )
        if deltas_about_joints and self.joint_parents is None:
            raise ValueError("deltas_about_joints needs joint_parents")
        self.deltas_about_joints = deltas_about_joints
        self._joint_pivots: torch.Tensor | None = None
        self._joint_chain: torch.Tensor | None = None

        self.max_influences = max_influences  # number of weights per vertex
        self.total_nnz_B_rt = total_nnz_B_rt  # non-zero values kept in B_rt
        self.init_weight = init_weight
        self.power = power
        self.weight_init = _resolve_weight_init(
            weight_init, has_joint_matrices=self.rest_joint_matrices_3x4 is not None
        )
        self.candidate_joints_per_vertex = _resolve_candidate_joints(
            candidate_joints_per_vertex,
            has_joint_matrices=self.rest_joint_matrices_3x4 is not None,
            max_influences=max_influences,
            number_of_bones=n_geodesic_joints,
        )

        self.stage_iterations = _as_stage_iterations(iterations)
        self.schedule: Sequence[TrainingPhase] = build_training_schedule(
            self.stage_iterations,
            max_influences,
            total_nnz_B_rt,
            start_influences=self.candidate_joints_per_vertex,
            warm_up_enabled=warm_up_enabled,
        )

        self.seed = seed
        torch.manual_seed(self.seed)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Print model information
        self.model_data.print_details()

        # Get alpha from model data
        self.alpha = self.model_data.alpha
        print(
            f"Using alpha value: {self.alpha} for model '{self.model_data.model_name}'"
        )

        self.loss_list: list[float] = []
        self.abserr_list: list[float] = []
        self.reconstruction_error: ReconstructionError | None = None

        self.L: torch.Tensor | None = None
        self.rest_pose: torch.Tensor | None = None
        self.candidate_mask: torch.Tensor | None = None
        self.geodesic_distances: np.ndarray | None = None
        self.transform_mask: torch.Tensor | None = None
        self.optimizer: torch.optim.Adam | None = None

    def run(self, output_location: str | Path) -> None:
        """Runs complete skinning decomposition and saves compressed results.

        This is the main entry point that orchestrates the entire optimization
        process. It initializes parameters, runs two-phase training, computes
        error metrics, and saves the compressed skinning data to an NPZ file.

        Optimization Workflow:
            0. Check self.schedule fits this model (K < P, L <= 6*S*P)
            1. Construct target matrix A from blendshape deltas
            2. Build Laplacian regularization matrix
            3. Build transformation basis TR (6-DOF)
            4. Initialize B_rt (transformation parameters) with small random values
            5. Initialize W (skinning weights) with small random values
            6. Prepare rest pose with homogeneous coordinates
            7. Build the candidate-joint mask from the joint positions, if any
            8. Run every phase of self.schedule in order
            9. Compute final normalized weights
            10. Evaluate and report error metrics, and joint coverage when
                joint matrices were given
            11. Save compressed results to NPZ file

        Training Schedule:
            self.schedule is a sequence of TrainingPhase objects, built by
            build_training_schedule from stage_iterations, max_influences and
            total_nnz_B_rt (or assigned directly before calling run for a
            hand-built schedule). Its first phase is always unnormalised
            (weights grow freely, helping discover good sparse structure);
            every later phase enforces partition of unity
            (Σⱼ wᵢ,ⱼ = 1). With the default settings this is the classic
            two-phase run.

        Output NPZ File Contents:
            rest: Rest pose vertices, shape (N, 3)
                Centered at origin (mean subtracted)
            quads: Mesh faces, shape (F, 4)
                Copied from input model
            weights: Normalized skinning weights, shape (N, P)
                Satisfies: non-negative, partition of unity, K-sparse
            restXform: Rest pose transforms, shape (P, 3, 4)
                If rest_joint_matrices was provided during initialization, contains
                those matrices (as 3×4 affine). Otherwise, contains identity matrices.
                Not used at runtime but included for completeness.
            shapeXform: Learned transformation matrices, shape (3S, 4P)
                SPARSE matrix B, ~90% zeros, used at runtime (Equation 7)

        Args:
            output_location: Path where compressed NPZ file will be saved.
                Can be string or Path object. Parent directory must already exist.

        Raises:
            ValueError: If self.schedule is empty, or a phase keeps
                max_influences >= number_of_bones weights per vertex, or more
                than 6*S*P delta coefficients.

        Side Effects:
            - Prints model information and training progress
            - Updates self.loss_list and self.abserr_list
            - Sets self.reconstruction_error to the final MXE/MAE
            - Creates output file at output_location
            - Reports final error metrics (MAE, MXE)

        Error Metrics Reported:
            - maxDelta (MXE): Maximum absolute error across all vertices/shapes
            - meanDelta (MAE): Mean absolute error

            These match Table 1 in the paper for comparison with Dem Bones.

        Example:
            >>> compressor = SkinCompressor(model_data, iterations=10000)
            >>> compressor.run("output/aura_compressed.npz")
            Model: aura
              deltas: (230, 7306, 3) float32
            Using alpha value: 10.0 for model 'aura'
            00000(0.234) 1.234e-02 5.678e-01 850 47824
            ...
            maxDelta 5.82
            meanDelta 0.0384

        References:
            - Section 4: Complete optimization algorithm
            - Table 1: Expected error metrics for validation
        """
        self._check_schedule_fits_model()

        A = self.get_matrix_for_optimization()
        N = A.shape[1]  # number of vertices

        self.L = self.get_laplacian_regularization(self.model_data.rest_faces)

        TR = self.buildTR()

        # B_rt - 6 degree of freedom  per blendshape per bone  (6, numBlendshapes, numBones, 1, 1)
        B_rt = (
            (
                self.init_weight
                * torch.randn(
                    (6, self.model_data.n_blendshapes, self.number_of_bones, 1, 1)
                )
            )
            .clone()
            .float()
            .to(self.device)
            .requires_grad_()
        )

        rest_centered = self.model_data.rest_verts - self.model_data.rest_verts.mean(
            axis=0
        )
        # rest_pose  nx4 aray of vertices (with forth column 1)
        self.rest_pose = (
            torch.from_numpy(add_homogeneous_coordinate(rest_centered, 1))
            .float()
            .to(self.device)
        )
        self.candidate_mask = self._build_candidate_mask()
        self.transform_mask = self._build_transform_mask()
        self._prepare_joint_hierarchy()

        # Apply transform mask to B_rt initialization
        if self.transform_mask is not None:
            B_rt.data *= self.transform_mask

        W = self._initial_weights(N)  # (P', N), one weight per vertex per bone
        W.requires_grad_()

        for index, phase in enumerate(self.schedule, start=1):
            print(
                f"phase {index}/{len(self.schedule)}: iterations={phase.iterations} "
                f"K={phase.max_influences} L={phase.total_nnz_B_rt} "
                f"normalize_weights={phase.normalize_weights}"
            )
            self.train(B_rt=B_rt, TR=TR, A=A, W=W, phase=phase)

        Wn = self._final_weights(W)
        self._validate_weights(Wn)
        print(Wn.min().item(), Wn.max().item())
        BX, B, _ = self.compBX(
            Wn, B_rt, TR, self.model_data.n_blendshapes, self.number_of_bones
        )
        orig_deltas = npf(
            A.transpose(1, 0).reshape(-1, self.model_data.n_blendshapes, 3)
        )
        our_deltas = npf(
            BX.transpose(1, 0).reshape(-1, self.model_data.n_blendshapes, 3)
        )

        maxDelta = np.abs(orig_deltas - our_deltas).max()
        meanDelta = np.abs(orig_deltas - our_deltas).mean()
        print(f"maxDelta {maxDelta}")
        print(f"meanDelta {meanDelta}")
        self.reconstruction_error = ReconstructionError(
            max_abs=float(maxDelta), mean_abs=float(meanDelta)
        )
        if self.rest_joint_matrices_3x4 is not None:
            self._report_joint_coverage(Wn)
        if self.root_joint is not None:
            self._report_root_usage(Wn, A)

        shapeXforms = B.detach().cpu().numpy()
        rest_xform = self._rest_transforms_for_output()

        np.savez(
            output_location,
            rest=npf(self.rest_pose[:, :3]),
            quads=self.model_data.rest_faces,
            weights=npf(Wn).transpose(),
            restXform=rest_xform,
            shapeXform=shapeXforms,
            **self._hierarchy_outputs(B_rt, TR),
        )

    def _resolve_joint_parents(
        self, joint_parents: Sequence[int] | None, n_joints: int
    ) -> np.ndarray | None:
        """Validates joint_parents and extends it with the synthetic root.

        Args:
            joint_parents: The constructor argument.
            n_joints: Number of joint matrices supplied.

        Returns:
            Parent per bone, shape (P',), or None without a hierarchy.

        Raises:
            ValueError: See the constructor.
        """
        if joint_parents is None:
            return None
        if self.rest_joint_matrices_3x4 is None:
            raise ValueError(
                "joint_parents needs rest_joint_matrices: parents refer to joints"
            )
        parents = np.asarray(joint_parents, dtype=np.int64)
        if parents.shape != (n_joints,):
            raise ValueError(
                f"joint_parents needs one parent per joint ({n_joints}), got {parents.shape}"
            )
        joint_chain_matrix(parents)
        if (
            self.root_joint is not None
            and self.root_joint < n_joints
            and parents[self.root_joint] != -1
        ):
            raise ValueError(
                f"root_joint={self.root_joint} has a parent; the root must be a top-level "
                "joint so that its transform stays zero"
            )
        if self.number_of_bones > n_joints:  # synthetic root, no parent
            parents = np.append(parents, -1)
        return parents

    def _prepare_joint_hierarchy(self) -> None:
        """Puts the joint pivots and the chain matrix on the device for run().

        With deltas_about_joints the pivots are the joint rest positions in
        the solver's frame, where the rest vertices are centred on their mean;
        otherwise every delta acts about that centre (zero pivots).
        """
        if self.joint_parents is None or self.rest_joint_matrices_3x4 is None:
            return
        pivots = np.zeros((self.number_of_bones, 3))
        if self.deltas_about_joints:
            joints = self.rest_joint_matrices_3x4[:, :3, 3]
            pivots[: len(joints)] = joints - self.model_data.rest_verts.mean(axis=0)
        self._joint_pivots = torch.from_numpy(pivots).float().to(self.device)
        chain = joint_chain_matrix(self.joint_parents)
        self._joint_chain = torch.from_numpy(chain).float().to(self.device)

    def _joint_coefficients(self, B_rt: torch.Tensor) -> torch.Tensor:  # noqa: N803
        """The coefficients of each joint's transform, through the hierarchy if any.

        Args:
            B_rt: Optimised coefficients, shape (6, S, P, 1, 1).

        Returns:
            Coefficients in the shared frame, shape (6, S, P, 1, 1).
        """
        if self._joint_chain is None or self._joint_pivots is None:
            return B_rt
        return hierarchical_coefficients(B_rt, self._joint_pivots, self._joint_chain)

    def _hierarchy_outputs(
        self,
        B_rt: torch.Tensor,  # noqa: N803
        TR: torch.Tensor,  # noqa: N803
    ) -> dict[str, Any]:
        """The extra archive keys written when a joint hierarchy was solved.

        Args:
            B_rt: Optimised local coefficients, shape (6, S, P, 1, 1).
            TR: Transform basis, shape (6, 1, 1, 3, 4).

        Returns:
            ``jointParents`` (P',) and ``localShapeXform`` (3S, 4P'), or an
            empty dict without a hierarchy.
        """
        if self.joint_parents is None or self._joint_pivots is None:
            return {}
        identity = torch.eye(self.number_of_bones, device=self.device)
        local = hierarchical_coefficients(B_rt, self._joint_pivots, identity)
        local_matrix = self._transform_matrix(
            local, TR, self.model_data.n_blendshapes, self.number_of_bones
        )
        return {
            "jointParents": self.joint_parents.copy(),
            "localShapeXform": local_matrix.detach().cpu().numpy(),
        }

    def _check_schedule_fits_model(self) -> None:
        """Rejects a schedule the projections cannot apply to this model.

        Raises:
            ValueError: If self.schedule is empty, or a phase keeps
                max_influences >= number_of_bones weights per vertex, or more
                than 6*S*P delta coefficients.
        """
        if not self.schedule:
            raise ValueError("schedule must contain at least one TrainingPhase")
        n_coefficients = 6 * self.model_data.n_blendshapes * self.number_of_bones
        for index, phase in enumerate(self.schedule, start=1):
            if phase.max_influences >= self.number_of_bones:
                raise ValueError(
                    f"phase {index}: max_influences={phase.max_influences} must be "
                    f"smaller than number_of_bones={self.number_of_bones}"
                )
            if phase.total_nnz_B_rt > n_coefficients:
                raise ValueError(
                    f"phase {index}: total_nnz_B_rt={phase.total_nnz_B_rt} exceeds "
                    f"6*S*P={n_coefficients}"
                )

    def _build_transform_mask(self) -> torch.Tensor | None:
        """Builds the (1, 1, P, 1, 1) mask to zero the root's transform block.

        Returns:
            Boolean tensor with shape (1, 1, P', 1, 1), all true except the root
            row which is false. Returns None when there is no root.
        """
        if self.root_joint is None:
            return None
        mask = torch.ones(
            1, 1, self.number_of_bones, 1, 1, dtype=torch.bool, device=self.device
        )
        mask[0, 0, self.root_joint, 0, 0] = False
        return mask

    def _rest_transforms_for_output(self) -> np.ndarray:
        """Returns the (P', 3, 4) rest transforms for the output.

        Appends an identity row for a synthetic root, or uses the caller's
        matrices, or creates identities without matrices.

        Returns:
            Array of shape (P', 3, 4) where P' may be one larger than P.
        """
        if self.rest_joint_matrices_3x4 is not None:
            matrices = self.rest_joint_matrices_3x4.copy()
        else:
            matrices = np.eye(3, 4)[np.newaxis, :, :].repeat(
                self.number_of_bones - (1 if self.root_joint is not None else 0), axis=0
            )

        if self.root_joint is not None and self.root_joint == (
            self.number_of_bones - 1
        ):
            identity_row = np.eye(3, 4)[np.newaxis, :, :]
            matrices = np.vstack([matrices, identity_row])
        return matrices.astype(np.float32)

    def _build_candidate_mask(self) -> torch.Tensor | None:
        """Builds the (P', N) mask of joints allowed to weight each vertex.

        Returns:
            Boolean tensor on self.device with shape (P', N) where P' includes
            the synthetic root if present. Returns None when there is no filter.

        Raises:
            RuntimeError: If a filter is set but no joint matrices are held.
        """
        if self.candidate_joints_per_vertex is None:
            return None
        mask = candidate_joint_mask(
            self._joint_distances(), self.candidate_joints_per_vertex
        )
        mask_tensor = torch.from_numpy(mask).to(self.device)

        # Add root row (all true) only for synthetic root
        if self.root_joint is not None and self.root_joint == self.number_of_bones - 1:
            n_verts = mask.shape[1]
            root_row = torch.ones(1, n_verts, dtype=torch.bool, device=self.device)
            mask_tensor = torch.cat([mask_tensor, root_row], dim=0)
        return mask_tensor

    def _joint_distances(self) -> np.ndarray:
        """Geodesic distance from every joint to every vertex, computed once.

        Returns:
            Distances of shape (P, N) for the supplied joints (the synthetic
            root, if any, is not included).

        Raises:
            RuntimeError: If no joint matrices are held.
        """
        if self.rest_joint_matrices_3x4 is None:
            raise RuntimeError("joint distances need rest_joint_matrices")
        if self.geodesic_distances is None:
            self.geodesic_distances = geodesic_joint_distances(
                self.model_data.rest_verts,
                self.model_data.rest_faces,
                self.rest_joint_matrices_3x4[:, :3, 3],
            )
        return self.geodesic_distances

    def _initial_weights(self, n_verts: int) -> torch.Tensor:
        """Builds the starting weights according to weight_init.

        Args:
            n_verts: Number of vertices N.

        Returns:
            Float tensor on self.device, shape (P', N): tiny noise inside the
            candidate mask for "noise", or the distance bind of
            :func:`distance_weights` for "distance", with static vertices
            fully on the root. The root row, if any, is at least its floor.
        """
        if self.weight_init == "distance":
            start = distance_weights(self._joint_distances(), self.max_influences)
            if start.shape[0] < self.number_of_bones:  # synthetic root row
                start = np.vstack([start, np.zeros((1, n_verts))])
            if self.root_joint is not None:
                static = static_vertex_mask(self.model_data.deltas)
                start[:, static] = 0.0
                start[self.root_joint, static] = 1.0
                print(
                    f"initial weights: distance bind over the {self.max_influences} "
                    f"nearest joints; {int(static.sum())} static vertices start on the root"
                )
            W = torch.from_numpy(start).float().to(self.device)  # noqa: N806
        else:
            W = (
                (1e-8 * torch.randn(self.number_of_bones, n_verts))
                .float()
                .to(self.device)
            )  # noqa: N806
        if self.candidate_mask is not None:
            W *= self.candidate_mask
        if self.root_joint is not None:
            W[self.root_joint].clamp_(min=_ROOT_WEIGHT_FLOOR)
        return W

    def _report_joint_coverage(self, Wn: torch.Tensor) -> None:  # noqa: N803
        """Prints how many vertices each joint drives and lists idle joints.

        Args:
            Wn: Normalised weights, shape (P', N).
        """
        vertices_per_joint = (Wn > 0).sum(dim=1)  # (P',)
        print(
            "vertices per joint: "
            f"min {vertices_per_joint.min().item()} "
            f"median {vertices_per_joint.median().item()} "
            f"max {vertices_per_joint.max().item()}"
        )
        idle = torch.nonzero(vertices_per_joint == 0).flatten().tolist()

        # Skip root from idle list
        if self.root_joint is not None and self.root_joint in idle:
            idle.remove(self.root_joint)

        if idle:
            print(f"joints driving no vertex: {idle}")

    def _report_root_usage(self, Wn: torch.Tensor, A: torch.Tensor) -> None:
        """Reports how many vertices depend entirely or partially on the root.

        Args:
            Wn: Normalised weights, shape (P', N).
            A: Target blendshape matrix, shape (3S, N).
        """
        if self.root_joint is None:
            return

        root_weights = Wn[self.root_joint]
        full_root = (root_weights == 1.0).sum().item()
        partial_root = ((root_weights > 0) & (root_weights < 1.0)).sum().item()

        # Find largest delta for vertices on root
        if full_root > 0:
            root_verts = torch.nonzero(root_weights == 1.0).flatten()
            A_root_verts = A[:, root_verts]
            largest_delta = A_root_verts.abs().max().item()
        else:
            largest_delta = 0.0

        print(
            f"vertices on the root: {full_root} full, {partial_root} partial, "
            f"largest delta on root {largest_delta:.5e}"
        )

    def get_matrix_for_optimization(self) -> torch.Tensor:
        """Constructs target matrix A ∈ ℝ^(3S×N) from blendshape deltas (Equation 3).

        This method reshapes the input blendshape deltas into the matrix form
        required for optimization. The resulting matrix A contains all delta
        vectors stacked vertically, which will be approximated by B·C.

        Mathematical Context:
            From Equation 1, blendshapes are:
                v̂₀,ᵢ + Σₖ cₖ(v̂ₖ,ᵢ - v̂₀,ᵢ)

            The matrix A contains the deltas (v̂ₖ,ᵢ - v̂₀,ᵢ), organized as:

            A = [ δ₁,₁  δ₁,₂  ...  δ₁,ₙ ]   ← x components of shape 1
                [ δ₁,₁  δ₁,₂  ...  δ₁,ₙ ]   ← y components of shape 1
                [ δ₁,₁  δ₁,₂  ...  δ₁,ₙ ]   ← z components of shape 1
                [ δ₂,₁  δ₂,₂  ...  δ₂,ₙ ]   ← x components of shape 2
                [   ⋮      ⋮    ⋱     ⋮  ]
                [ δₛ,₁  δₛ,₂  ...  δₛ,ₙ ]   ← z components of shape S

            Shape: (3S, N) where S=blendshapes, N=vertices

        Reshape Operations:
            Input deltas: (S, N, 3)
            1. .transpose(1, 0, 2): → (N, S, 3)
            2. .reshape(-1, S*3): → (N, 3S)
            3. .transpose(): → (3S, N) = A

        Returns:
            Blendshape matrix A, shape (3S, N).
                Target for the optimization A ≈ B·C.
                Moved to self.device (cuda or cpu).

        Note:
            The matrix is converted to PyTorch tensor and moved to the
            training device. This is done once at initialization to avoid
            repeated CPU-GPU transfers during training.

        References:
            - Equation 3: Matrix A definition
            - Section 3: Framework and notation
        """
        deltas = (
            self.model_data.deltas.transpose(1, 0, 2)
            .reshape(-1, self.model_data.n_blendshapes * 3)
            .transpose()
        )
        return torch.from_numpy(deltas).float().to(self.device)

    def get_laplacian_regularization(self, rest_faces: np.ndarray) -> torch.Tensor:
        """Computes Laplacian regularization matrix for mesh smoothing.

        This method constructs the rigidity Laplacian used to enforce smooth
        deformations during optimization. The regularization term penalizes
        high-frequency details and helps avoid noisy or unrealistic skinning
        weights.

        Mathematical Formulation:
            The Laplacian matrix L enforces smoothness via:
                Regularization_term = α · ‖L·BX‖²

            where α (self.alpha) controls regularization strength.

            The rigidity Laplacian is defined as::

                Lᵢ,ⱼ = { -1          if j = i
                       { 1/deg(i)    if j ∈ N(i)  (1-ring neighbors)
                       { 0           otherwise

            where N(i) denotes all 1-ring neighbors of vertex i and deg(i)
            is their count.

        Construction Process:
            1. Build adjacency matrix from mesh faces
            2. Compute vertex degrees (number of neighbors)
            3. Construct normalized Laplacian: D⁻¹(A - D)
               where D is diagonal degree matrix, A is adjacency

        Args:
            rest_faces: Mesh face indices, shape (n_faces, verts_per_face).
                Mesh topology defining vertex connectivity. Supports both
                triangles (3) and quads (4).

        Returns:
            Sparse Laplacian matrix, shape (N, N) where N is number of vertices.
                Stored as sparse tensor for efficient computation.
                Used in loss function: α · (L @ BX.T).pow(2).mean()

        Note:
            The Laplacian is computed on the CPU even when training on GPU,
            then converted to sparse tensor on the target device. This is
            efficient because the Laplacian is sparse (~6-8 non-zeros per row
            for typical meshes) and only computed once at initialization.

        References:
            - Section 4: Mentions Laplacian regularization
            - Le & Deng 2014: Dem Bones (same regularization approach)
            - Botsch & Sorkine 2008: Differential coordinates (Laplacian theory)
        """
        # Adjacency matrix: represents the connectivity of the mesh
        adj = _build_adjacency_matrix(rest_faces, rest_faces.max() + 1)
        # Diagonal adjacency matrix: calculates the degree of each vertex
        adj_diag = np.asarray(adj.sum(axis=1)).squeeze().astype(float)
        # Rigidness Laplacian regularization.
        # ⎧-1        if k = i
        # ⎨1/|N(i)|, if k ∈ N(i)
        # ⎩0         otherwise.
        # Where N(i) denotes all the 1-ring neighbours of i
        Lg = sp.sparse.diags(1 / adj_diag) @ (adj - sp.sparse.diags(adj_diag))
        # Return the Laplacian regularization matrix as a sparse tensor
        return torch.from_numpy((Lg).todense()).float().to(self.device).to_sparse()

    def train(
        self,
        B_rt: torch.Tensor,
        TR: torch.Tensor,
        A: torch.Tensor,
        W: torch.Tensor,
        phase: TrainingPhase,
    ) -> None:
        """Trains skinning parameters using proximal Adam optimization (Section 4).

        This method implements the core optimization loop that learns sparse
        skinning weights W and transformation parameters B_rt to approximate
        the input blendshape matrix A. The approach combines Adam optimizer
        with proximal projection steps to enforce constraints.

        Optimization Algorithm:
            1. Forward pass: Compute B·C using compBX()
            2. Compute loss: Lp norm + Laplacian regularization (Equation 9)
            3. Backward pass: Automatic differentiation computes gradients
            4. Adam step: Update parameters with adaptive learning rate
            5. Proximal projection: Enforce sparsity and non-negativity constraints
            6. Repeat for specified number of iterations

        Loss Function (Equation 9):
            Loss = (Σᵢ,ₖ ‖Eᵢ,ₖ‖^p)^(2/p) + α·Laplacian_term

            where:
                Eᵢ,ₖ = vₖ,ᵢ - v₀,ᵢ - Σⱼ wᵢ,ⱼ Nₖ,ⱼ v₀,ᵢ
                p = self.power (default 2 for L2, 12 for HD fit)
                α = self.alpha (Laplacian regularization strength)

        Proximal Projection (Key Innovation):
            After each Adam step, we project parameters onto constraint sets:

            For W (skinning weights):
                1. Zero the joints outside the vertex's candidate set, when
                   self.candidate_mask is set (locality)
                2. Keep only K largest weights per vertex (spatial sparsity)
                3. Clamp to non-negative values (non-negativity constraint)
                4. If phase.normalize_weights, normalize to sum to 1 (partition of unity)

            For B_rt (transformations):
                1. Keep only L largest values globally by absolute value
                2. Zero out remaining values (transformation sparsity)

            This is the "prox" operator that makes this a proximal algorithm,
            inspired by proximal methods in convex optimization.

        Args:
            B_rt: Rotation-translation parameters to optimize, shape (6, S, P, 1, 1).
                Initialized with small random values, requires_grad=True.
            TR: Fixed transformation basis matrices, shape (6, 1, 1, 3, 4).
                Not optimized, constructed by buildTR().
            A: Target blendshape matrix, shape (3S, N).
                Ground truth deltas from input model.
            W: Bone weights to optimize, shape (P, N).
                Initialized with small random values, requires_grad=True.
            phase: The TrainingPhase to run: its iterations, max_influences
                and total_nnz_B_rt set the loop length and the two
                projections; its normalize_weights sets whether the forward
                pass enforces partition of unity.

        Note:
            run() executes self.schedule, a sequence of TrainingPhase built
            by build_training_schedule (see that function for how a staged
            iterations argument turns into phases). With the default
            settings this is the classic two phases:

            Phase 1 (normalize_weights=False):
                - Weights can grow freely during initial optimization
                - Helps escape local minima in early iterations
                - Focuses on finding good sparse structure

            Phase 2 (normalize_weights=True):
                - Enforces partition of unity constraint
                - Produces final skinning weights satisfying LBS requirements
                - Refines the solution from phase 1

        Side Effects:
            - Updates self.loss_list and self.abserr_list every 200 iterations
            - Prints progress information every 200 iterations
            - Modifies B_rt and W tensors in-place via optimizer

        References:
            - Section 4: Compressed skinning decomposition algorithm
            - Equation 9: Loss function formulation
            - Parikh et al. 2014: Proximal algorithms (theoretical foundation)
            - Kingma & Ba 2014: Adam optimizer
        """
        param_list = [B_rt, W]
        self.optimizer = torch.optim.Adam(param_list, lr=1e-3, betas=(0.9, 0.9))

        st = time.time()
        for i in range(phase.iterations):
            W_n = W / W.sum(dim=0) if phase.normalize_weights else W

            B_X, _, _ = self.compBX(
                W_n, B_rt, TR, self.model_data.n_blendshapes, self.number_of_bones
            )
            weighed_error = B_X - A

            loss = weighed_error.pow(self.power).mean().pow(2 / self.power)
            if self.alpha is not None:
                if self.L is None:
                    raise RuntimeError(
                        "Laplacian is not initialized; train must be called via run()."
                    )
                # add Laplacian regularization term
                loss += self.alpha * (self.L @ (B_X).transpose(0, 1)).pow(2).mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            with torch.no_grad():
                self._project_weights(W, phase)
                self._project_transforms(B_rt, phase)

            if i % 200 == 0:
                B_X, _, _ = self.compBX(
                    W_n, B_rt, TR, self.model_data.n_blendshapes, self.number_of_bones
                )
                trunc_err = (B_X - A).abs().max().item()

                if self.device == "cuda":
                    torch.cuda.synchronize()

                empty_cols = (W.sum(dim=0) == 0).sum().item()
                print(
                    f"{i:05d}({time.time() - st:.3f}) {loss.item():.5e} {trunc_err:.5e} {(B_rt.abs() > _SPARSITY_THRESHOLD).count_nonzero().item()} {(W.abs() > _SPARSITY_THRESHOLD).count_nonzero().item()} {empty_cols}"
                )
                self.loss_list.append(loss.item())
                self.abserr_list.append(trunc_err)

                st = time.time()

    def _project_weights(self, W: torch.Tensor, phase: TrainingPhase) -> None:
        """Applies the mask, the top-K cut, the sign clamp and the root floor in place.

        Args:
            W: Bone weights, shape (P', N).
            phase: The TrainingPhase defining the influence budget.
        """
        if self.candidate_mask is not None:
            W *= self.candidate_mask
        cutoff = torch.topk(W, phase.max_influences + 1, dim=0).values[-1, :]
        W.copy_((cutoff < W) * W)
        W.clamp_(min=0)
        if self.root_joint is not None:
            W[self.root_joint].clamp_(min=_ROOT_WEIGHT_FLOOR)

    def _project_transforms(self, B_rt: torch.Tensor, phase: TrainingPhase) -> None:
        """Zeroes the root's block and keeps the L largest coefficients in place.

        Args:
            B_rt: Transformation matrices, shape (6, S, P', 1, 1).
            phase: The TrainingPhase defining the delta budget.
        """
        if self.transform_mask is not None:
            B_rt *= self.transform_mask
        decider = B_rt.abs()
        cutoff = torch.topk(decider.flatten(), phase.total_nnz_B_rt).values[-1]
        B_rt.copy_((decider >= cutoff) * B_rt)

    def _final_weights(self, W: torch.Tensor) -> torch.Tensor:
        """Normalises the weights, dropping floor-level root entries first.

        Args:
            W: Raw weights, shape (P', N).

        Returns:
            Normalised weights where each column sums to one. Floor-level root
            entries beside other non-zero weights are zeroed before normalisation.
        """
        W = W.detach().clone()
        if self.root_joint is not None:
            others_alive = (W.sum(dim=0) - W[self.root_joint]) > 0
            at_floor = W[self.root_joint] <= _ROOT_WEIGHT_FLOOR
            W[self.root_joint, at_floor & others_alive] = 0
        return W / W.sum(dim=0)

    def _validate_weights(self, Wn: torch.Tensor) -> None:
        """Raises RuntimeError unless Wn is finite, non-negative, normalised and K-sparse.

        Args:
            Wn: Normalised weights, shape (P', N).

        Raises:
            RuntimeError: If any check fails.
        """
        if not torch.isfinite(Wn).all():
            raise RuntimeError("weights contain NaN or Inf")
        if (Wn < 0).any():
            raise RuntimeError("weights are negative")
        col_sums = Wn.sum(dim=0)
        if not torch.allclose(col_sums, torch.ones_like(col_sums), atol=1e-5):
            raise RuntimeError("weights do not sum to one per vertex")
        n_nonzero = (Wn > 0).sum(dim=0)
        if (n_nonzero > self.max_influences).any():
            raise RuntimeError(f"weights exceed max_influences={self.max_influences}")

    def compBX(
        self,
        Wn: torch.Tensor,
        B_rt: torch.Tensor,
        TR: torch.Tensor,
        n_bs: int,
        P: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes the skinning decomposition B·C ≈ A (Equations 3-5).

        This function implements the core matrix factorization that approximates
        blendshape deltas as a product of transformation matrices B and weighted
        rest pose C. The transformation matrices are constructed from learnable
        6-DOF parameters B_rt using the basis matrices TR (Equation 8).

        Mathematical Formulation:
            A ≈ B·C

            where:
                A ∈ ℝ^(3S×N): target blendshape deltas
                B ∈ ℝ^(3S×4P): transformation matrices (from B_rt and TR)
                C ∈ ℝ^(4P×N): weighted rest pose (from Wn and rest_pose)

            From Equation 5:
                Σⱼ wᵢ,ⱼ Nₖ,ⱼ v₀,ᵢ = v̂ₖ,ᵢ - v̂₀,ᵢ

            The transformation matrices are:
                Nₖ,ⱼ = B_rt combined with TR basis (Equation 8)

        Tensor Operations:
            1. X = C: Weighted rest pose
               Shape: (4P, N)
               Each column is rest_pose weighted by bone influence

            2. B: Transformation matrix from B_rt parameters
               Shape: (3S, 4P)
               Constructed by linear combination of TR basis matrices

            3. BX = B @ C: Final skinned result
               Shape: (3S, N)
               Approximates input blendshape matrix A

        Args:
            Wn: Normalized bone weights, shape (P, N).
                Each column sums to 1, satisfies partition of unity constraint.
            B_rt: Rotation-translation parameters, shape (6, S, P, 1, 1).
                6-DOF learnable parameters (3 rotation + 3 translation).
                These are the primary optimization variables.
            TR: Transformation basis matrices, shape (6, 1, 1, 3, 4).
                Fixed basis for constructing transformations (from buildTR).
                Represents linearized rotation and translation (Equation 8).
            n_bs: Number of blendshapes (S in paper notation).
            P: Number of proxy bones (P in paper notation).

        Returns:
            tuple containing:
                - BX: Skinned result, shape (3S, N).
                    Approximation of input blendshape matrix A.
                - B: Transformation matrix, shape (3S, 4P).
                    Can be saved for runtime evaluation (Equation 7).
                - X: Weighted rest pose, shape (4P, N).
                    Rest pose multiplied by skinning weights.

        Note:
            The ASCII art diagram in the code shows the tensor layout clearly.
            X is arranged as (4P, N) with bones vertically and vertices horizontally.
            B stacks transformation matrices for each blendshape and bone combination.

        References:
            - Equations 3-5: Matrix decomposition formulation
            - Equation 8: Linearized rotation representation
            - Section 4: Optimization details
        """
        # calculates Linear Blend Skinning
        # Wn ∈ PxN   (numBones x numVertices)
        # B_rt - 6 degree of freedom  per blendshape per bone  (6, n_bs, numBones, 1, 1)
        # TR  - 6 base matrices (n_bx, numBones, 3, 4)[6]  one per degree of freedom these are used to convert B_rt to B
        # rest_pose  ∈ nx4
        # X :  rest_pose.p * weight
        #         vertex...vertex
        #            0      N
        #         ┌           ┐
        # bone0  x│┌───┐      │
        #        y││  →│...   │
        #        z││w*p│      │
        #        w│└───┘      │
        # bone1  x│           │
        #        y│           │
        #        z│           │
        #        w│           │
        #         ┆           ┆
        # boneP  w│           │
        #         └           ┘
        if self.rest_pose is None:
            raise RuntimeError(
                "rest_pose is not initialized; compBX must be called via run()."
            )
        X = (Wn.unsqueeze(2) * self.rest_pose).permute(0, 2, 1).reshape(4 * P, -1)
        B = self._transform_matrix(self._joint_coefficients(B_rt), TR, n_bs, P)
        # B current bone transforms
        #               bone 0... bone N
        #                0123     0123
        #              ┌               ┐
        # blendshape0 0│┌────┐   ┌────┐│
        #             1││TM  │...│TM  ││
        #             2│└────┘   └────┘│
        # blendshape1 0│┌────┐   ┌────┐│
        #             1││TM  │...│TM  ││
        #             2│└────┘   └────┘│
        #              │               │
        #              ┆               ┆
        #              └               ┘
        return B @ X, B, X

    @staticmethod
    def _transform_matrix(
        coefficients: torch.Tensor,
        TR: torch.Tensor,
        n_bs: int,
        P: int,  # noqa: N803
    ) -> torch.Tensor:
        """Assembles B, shape (3S, 4P), from coefficients in the shared frame.

        Args:
            coefficients: Shape (6, S, P, 1, 1).
            TR: Transform basis, shape (6, 1, 1, 3, 4).
            n_bs: Number of blendshapes S.
            P: Number of bones.

        Returns:
            The transform matrix B, shape (3S, 4P).
        """
        B = coefficients[0, ...] * TR[0]
        for i in range(1, 6):
            B = B + coefficients[i, ...] * TR[i]
        return B.permute(0, 2, 1, 3).reshape(n_bs * 3, P * 4)

    def buildTR(self) -> torch.Tensor:
        """Builds the 6-DOF transformation basis matrices (Equation 8).

        This method constructs the fixed basis matrices used to represent affine
        transformations with linearized rotations. The 6 degrees of freedom consist
        of 3 for linearized rotation and 3 for translation.

        Mathematical Background:
            Each transformation Nₖ,ⱼ ∈ ℝ^(3×4) has the form (Equation 8):

            Nₖ,ⱼ = [  0      -r₃    r₂   t₁ ]
                   [  r₃      0    -r₁   t₂ ]
                   [ -r₂      r₁    0    t₃ ]

            where (r₁, r₂, r₃) are linearized rotation parameters and
            (t₁, t₂, t₃) are translation parameters.

            This representation is closed under linear blending (Equation 7):
                Mⱼ = I + Σₖ cₖ Nₖ,ⱼ

            Meaning we can blend transformations with simple linear combination,
            which is faster than quaternion interpolation and matches the
            blendshape linear blending paradigm.

        Basis Decomposition:
            The transformation is decomposed as:
                Nₖ,ⱼ = Σᵢ₌₁⁶ (B_rt)ᵢ,ₖ,ⱼ · (TR)ᵢ

            where (TR)ᵢ are the 6 basis matrices returned by this function:
                (TR)₁: rotation around z-axis (skew-symmetric in xy)
                (TR)₂: rotation around y-axis (skew-symmetric in xz)
                (TR)₃: rotation around x-axis (skew-symmetric in yz)
                (TR)₄: translation in x direction
                (TR)₅: translation in y direction
                (TR)₆: translation in z direction

        Returns:
            Transformation basis tensor, shape (6, 1, 1, 3, 4).
                - First dimension: 6 basis matrices (3 rotation + 3 translation)
                - Singleton dimensions for broadcasting with B_rt
                - Last two dimensions: 3×4 affine transformation matrices

        Note:
            These basis matrices are FIXED and never optimized. Only the
            coefficients B_rt are learned during training. The linearized
            rotation representation (skew-symmetric matrices) approximates
            small rotations and is exact for infinitesimal rotations.

        References:
            - Equation 8: Linearized rotation representation
            - Section 3.1: Transformation representation details
            - Goldstein et al. 2002: Classical Mechanics (linearized rotations)
        """
        # fmt: off
        ebase = torch.tensor([[[0, 0,  0, 0],
                               [0, 0, -1, 0],
                               [0, 1,  0, 0]],

                              [[ 0, 0, 1, 0],
                               [ 0, 0, 0, 0],
                               [-1, 0, 0, 0]],

                              [[0, -1, 0, 0],
                               [1,  0, 0, 0],
                               [0,  0, 0, 0]],

                              [[0, 0, 0, 1],
                               [0, 0, 0, 0],
                               [0, 0, 0, 0]],

                              [[0, 0, 0, 0],
                               [0, 0, 0, 1],
                               [0, 0, 0, 0]],

                              [[0, 0, 0, 0],
                               [0, 0, 0, 0],
                               [0, 0, 0, 1]]], dtype=torch.float32).to(self.device)
        return ebase.reshape(6, 1, 1, 3, 4)
        # fmt: on
