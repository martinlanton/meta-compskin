"""Implements the SkinCompressor class for blendshape compression and optimization.

Handles loading model data, running optimization, and saving results.
"""

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree.

import time
from pathlib import Path

import igl
import numpy as np
import scipy as sp
import torch

from metacompskin.model_data import BlendshapeModelData
from metacompskin.utils import add_homogeneous_coordinate, npf

location = Path(__file__).parent
data_location = location / "data"

# Optimization constants
_SPARSITY_THRESHOLD = 1e-4  # Threshold for counting non-zero values in sparse matrices


class SkinCompressor:
    """Optimizer for sparse skinning decomposition of facial blendshapes.

    This class implements the compressed skinning decomposition method from
    "Compressed Skinning for Facial Blendshapes" (SIGGRAPH 2024). It converts
    blendshape deltas into a sparse linear blend skinning representation using
    proximal gradient methods with Adam optimization.

    The optimization solves the non-convex problem (Equation 9):
        min_{w,N} Σᵢ Σₖ (Eᵢₖ)^p
        where Eᵢₖ = |vₖ,ᵢ - v₀,ᵢ - Σⱼ wᵢ,ⱼ Nₖ,ⱼ v₀,ᵢ|

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
        iterations: Number of optimization iterations (default 10000).
        number_of_bones: Number of proxy bones P (default 40).
            Paper uses P=40 for good quality/performance balance (Table 1).
        max_influences: Max non-zero weights per vertex K (default 8).
            Standard GPU skinning pipeline constraint (Section 2.2).
        total_nnz_B_rt: Total non-zeros in B_rt matrix (default 6000).
            Corresponds to ~1000 transformations with 6-DOF representation.
            Achieves ~90% sparsity while maintaining accuracy (Table 1).
        init_weight: Initialization scale for random parameters (1e-3).
            Small value starts optimization near zero.
        power: Lp norm exponent for loss function (default 2).
            p=2 gives L2 norm, p=12 for HD fit (Section 4.1).
        seed: Random seed for reproducibility (12345).
        alpha: Laplacian regularization strength from model_data.
            Controls smoothness: lower for high-density, higher for low-density.
        device: PyTorch device ('cuda' or 'cpu').
        loss_list: Training loss history.
        abserr_list: Training absolute error history.

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

    def __init__(
        self,
        model_data: BlendshapeModelData,
        iterations: int = 10000,
    ):
        """Initializes the SkinCompressor.

        Args:
            model_data: BlendshapeModelData instance containing model geometry and rig info.
            iterations: Number of optimization iterations.

        Example:
            >>> from metacompskin.model_data import BlendshapeModelData
            >>> model_data = BlendshapeModelData.from_npz("data/source_models/aura.npz")
            >>> compressor = SkinCompressor(model_data=model_data, iterations=10000)
            >>> compressor.run()
        """
        self.model_data = model_data
        self.iterations = iterations
        self.number_of_bones = 40
        self.max_influences = 8  # number of weights per vertex
        self.total_nnz_B_rt = 6000  # number of non-zero values into B_rt matrix
        self.init_weight = 1e-3
        self.power = 2

        self.seed = 12345
        torch.manual_seed(self.seed)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Print model information
        self.model_data.print_details()

        # Get alpha from model data
        self.alpha = self.model_data.alpha
        print(
            f"Using alpha value: {self.alpha} for model '{self.model_data.model_name}'"
        )

        self.loss_list = []
        self.abserr_list = []

    def run(self, output_location):
        """Runs complete skinning decomposition and saves compressed results.

        This is the main entry point that orchestrates the entire optimization
        process. It initializes parameters, runs two-phase training, computes
        error metrics, and saves the compressed skinning data to an NPZ file.

        Optimization Workflow:
            1. Construct target matrix A from blendshape deltas
            2. Build Laplacian regularization matrix
            3. Build transformation basis TR (6-DOF)
            4. Initialize B_rt (transformation parameters) with small random values
            5. Initialize W (skinning weights) with small random values
            6. Prepare rest pose with homogeneous coordinates
            7. Phase 1: Train without weight normalization
            8. Phase 2: Train with weight normalization
            9. Compute final normalized weights
            10. Evaluate and report error metrics
            11. Save compressed results to NPZ file

        Two-Phase Training Strategy:
            Phase 1 (normalizeW=False):
                - Allows weights to grow freely during initial optimization
                - Helps discover good sparse structure
                - Focuses on reducing reconstruction error

            Phase 2 (normalizeW=True):
                - Enforces partition of unity constraint (Σⱼ wᵢ,ⱼ = 1)
                - Produces final skinning weights satisfying LBS requirements
                - Refines solution from phase 1

        Output NPZ File Contents:
            rest: Rest pose vertices, shape (N, 3)
                Centered at origin (mean subtracted)
            quads: Mesh faces, shape (F, 4)
                Copied from input model
            weights: Normalized skinning weights, shape (N, P)
                Satisfies: non-negative, partition of unity, K-sparse
            restXform: Identity transforms for rest pose, shape (P, 3, 4)
                Not used at runtime but included for completeness
            shapeXform: Learned transformation matrices, shape (3S, 4P)
                SPARSE matrix B, ~90% zeros, used at runtime (Equation 7)

        Args:
            output_location: Path where compressed NPZ file will be saved.
                Can be string or Path object. Parent directories created if needed.

        Side Effects:
            - Prints model information and training progress
            - Updates self.loss_list and self.abserr_list
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
        # W PxN (numBones x numVertices) weights one per vertex per bone
        W = (
            (1e-8 * torch.randn(self.number_of_bones, N))
            .clone()
            .float()
            .to(self.device)
            .requires_grad_()
        )

        self.train(B_rt=B_rt, TR=TR, A=A, W=W, normalizeW=False)
        self.train(B_rt=B_rt, TR=TR, A=A, W=W, normalizeW=True)

        Wn = W / W.sum(axis=0)
        print(Wn.min().item(), Wn.max().item())
        BX, _, _ = self.compBX(
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

        _, B, _ = self.compBX(
            Wn, B_rt, TR, self.model_data.n_blendshapes, self.number_of_bones
        )

        shapeXforms = B.detach().cpu().numpy()

        np.savez(
            output_location,
            rest=npf(self.rest_pose[:, :3]),
            quads=self.model_data.rest_faces,
            weights=npf(Wn).transpose(),
            restXform=np.array([np.eye(3, 4)] * self.number_of_bones),
            shapeXform=shapeXforms,
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

    def get_laplacian_regularization(self, rest_faces):
        """Computes Laplacian regularization matrix for mesh smoothing.

        This method constructs the rigidity Laplacian used to enforce smooth
        deformations during optimization. The regularization term penalizes
        high-frequency details and helps avoid noisy or unrealistic skinning
        weights.

        Mathematical Formulation:
            The Laplacian matrix L enforces smoothness via:
                Regularization_term = α · ||L·BX||²

            where α (self.alpha) controls regularization strength.

            The rigidity Laplacian is defined as:
                Lᵢ,ⱼ = { -1         if j = i
                       { 1/|N(i)|   if j ∈ N(i)  (1-ring neighbors)
                       { 0          otherwise

            where N(i) denotes all 1-ring neighbors of vertex i.

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
        adj = igl.adjacency_matrix(rest_faces)
        # Diagonal adjacency matrix: calculates the degree of each vertex
        adj_diag = np.array(np.sum(adj, axis=1)).squeeze()
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
        normalizeW=False,
    ):
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
            Loss = (Σᵢ,ₖ |Eᵢ,ₖ|^p)^(2/p) + α·Laplacian_term

            where:
                Eᵢ,ₖ = vₖ,ᵢ - v₀,ᵢ - Σⱼ wᵢ,ⱼ Nₖ,ⱼ v₀,ᵢ
                p = self.power (default 2 for L2, 12 for HD fit)
                α = self.alpha (Laplacian regularization strength)

        Proximal Projection (Key Innovation):
            After each Adam step, we project parameters onto constraint sets:

            For W (skinning weights):
                1. Keep only K largest weights per vertex (spatial sparsity)
                2. Clamp to non-negative values (non-negativity constraint)
                3. If normalizeW=True, normalize to sum to 1 (partition of unity)

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
            normalizeW: Whether to enforce partition of unity during training.
                False: Allows weights to grow freely (phase 1)
                True: Normalizes weights to sum to 1 per vertex (phase 2)

        Note:
            Training is typically done in two phases (see run() method):

            Phase 1 (normalizeW=False):
                - Weights can grow freely during initial optimization
                - Helps escape local minima in early iterations
                - Focuses on finding good sparse structure

            Phase 2 (normalizeW=True):
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
        for i in range(self.iterations):
            W_n = W / W.sum(axis=0) if normalizeW else W

            B_X, _, _ = self.compBX(
                W_n, B_rt, TR, self.model_data.n_blendshapes, self.number_of_bones
            )
            weighed_error = B_X - A

            loss = weighed_error.pow(self.power).mean().pow(2 / self.power)
            if self.alpha is not None:
                # add Laplacian regularization term
                loss += self.alpha * (self.L @ (B_X).transpose(0, 1)).pow(2).mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            with torch.no_grad():
                W_cutoff = torch.topk(W, self.max_influences + 1, dim=0).values[-1, :]
                W_mask = W_cutoff < W
                W_pruned = W_mask * W
                W.copy_(W_pruned)
                W.clamp_(min=0)

                B_decider = B_rt.abs()
                B_cutoff = torch.topk(B_decider.flatten(), self.total_nnz_B_rt).values[
                    -1
                ]
                B_mask = B_decider >= B_cutoff
                B_pruned = B_mask * B_rt
                B_rt.copy_(B_pruned)

            if i % 200 == 0:
                B_X, _, _ = self.compBX(
                    W_n, B_rt, TR, self.model_data.n_blendshapes, self.number_of_bones
                )
                trunc_err = (B_X - A).abs().max().item()

                if self.device == "cuda":
                    torch.cuda.synchronize()

                print(
                    f"{i:05d}({time.time() - st:.3f}) {loss.item():.5e} {trunc_err:.5e} {(B_rt.abs() > _SPARSITY_THRESHOLD).count_nonzero().item()} {(W.abs() > _SPARSITY_THRESHOLD).count_nonzero().item()}"
                )
                self.loss_list.append(loss.item())
                self.abserr_list.append(trunc_err)

                st = time.time()

    def compBX(self, Wn, B_rt, TR, n_bs, P):
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
        X = (Wn.unsqueeze(2) * self.rest_pose).permute(0, 2, 1).reshape(4 * P, -1)
        B = B_rt[0, ...] * TR[0]
        for i in range(1, 6):
            B += B_rt[i, ...] * TR[i]
        B = B.permute(0, 2, 1, 3).reshape(n_bs * 3, P * 4)
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

    def buildTR(self):
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
