"""Generates animation frames from compressed skinning data.

This module provides functionality to generate animation frames using the output
from the SkinCompressor class along with blendshape model data.
"""

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree.

from pathlib import Path

import igl
import numpy as np
import torch

import metacompskin.rig.riglogic as rl
from metacompskin.model_data import BlendshapeModelData
from metacompskin.utils import add_homogeneous_coordinate


class AnimationFrameGenerator:
    """Generates animation frames from compressed skinning data and animation weights.

    This class takes the output NPZ file from SkinCompressor and generates
    animation frames by applying rig logic and skinning transformations.
    """

    def __init__(
        self,
        compressed_data_path: str | Path,
        model_data: BlendshapeModelData,
    ):
        """Initializes the AnimationFrameGenerator.

        Args:
            compressed_data_path: Path to the NPZ file output from SkinCompressor.
                Expected to contain: rest, quads, weights, restXform, shapeXform.
            model_data: BlendshapeModelData instance containing rig information
                (inbetween_info and combination_info).

        Example:
            >>> from metacompskin.model_data import BlendshapeModelData
            >>> from metacompskin.animation_generator import AnimationFrameGenerator
            >>> model_data = BlendshapeModelData.from_npz("data/source_models/aura.npz")
            >>> generator = AnimationFrameGenerator(
            ...     compressed_data_path="output/aura_compressed.npz", model_data=model_data
            ... )
            >>> generator.generate_frames(
            ...     animation_weights_path="data/source_models/test_anim.npz",
            ...     output_dir="output/frames",
            ... )
        """
        self.compressed_data_path = Path(compressed_data_path)
        self.model_data = model_data

        # Load compressed skinning data
        compressed_data = np.load(self.compressed_data_path)
        self.rest_verts = compressed_data["rest"]
        self.quads = compressed_data["quads"]
        self.weights = compressed_data["weights"]
        self.rest_xform = compressed_data["restXform"]
        self.shape_xform = compressed_data["shapeXform"]

        # Compute derived values
        self.num_bones = self.weights.shape[1]

        # Prepare rest pose with homogeneous coordinates
        self.rest_pose_homog = add_homogeneous_coordinate(self.rest_verts, 1)

    def _generate_skinning_transforms(
        self, blendshape_weights: np.ndarray
    ) -> np.ndarray:
        """Generates runtime skinning transforms from blend weights (Equation 7).

        This method implements the runtime transformation computation that converts
        rig-driven blend weights into skinning transformations suitable for GPU
        evaluation. This is the CPU-side computation mentioned in Figure 2 of the
        paper that happens once per frame before GPU skinning.

        Mathematical Formulation (Equation 7):
            Mⱼ = I + Σₖ cₖ Nₖ,ⱼ

            where:
                - Mⱼ: Final skinning transformation for bone j
                - I: 3×4 identity matrix [I₃ₓ₃ | 0₃ₓ₁]
                - cₖ: Blend weight for shape k (from rig)
                - Nₖ,ⱼ: Delta transformation for shape k, bone j (from shapeXform)
                - Summation is sparse due to ~90% zeros in Nₖ,ⱼ

        Implementation Details:
            This uses an efficient sparse matrix-vector product leveraging
            the sparsity in self.shape_xform. The weighted diagonal matrix Z
            is constructed to apply blend weights to each transformation:

            1. Create Z: (3, 3, S) diagonal matrices scaled by weights
            2. Reshape Z to (3, 3S) for matrix multiplication
            3. Compute weighted sum: Z @ shapeXform (sparse)
            4. Add identity offset to get final transforms

        The ASCII art in the code visualizes the tensor operations clearly.

        Args:
            blendshape_weights: Blend coefficients cₖ from rig, shape (S,).
                Typically in range [0, 1] but can exceed for overshoot.
                Obtained from rig logic (compute_rig_logic in generate_frames).

        Returns:
            Skinning transformations, shape (3, 4P) where P=number of bones.
                These are affine transformation matrices ready for GPU skinning.
                Layout: [M₁ | M₂ | ... | Mₚ] where each Mⱼ is 3×4.

        Performance Note:
            The sparse structure of self.shape_xform is critical for performance.
            With ~90% zeros, this sparse matrix-vector product is 2-3× faster
            than the dense equivalent (Table 3) while using 5-7× less memory
            (Table 2).

        References:
            - Equation 7: Runtime transformation computation
            - Section 3: Framework and conversion formulas
            - Figure 2: Shows this as "Compute Mⱼ" on CPU before GPU skinning
            - Table 3: Performance comparison sparse vs dense
        """
        # blendshape_weights: (num_shapes,), output of riglogic
        # self.shape_xform: (3*num_shapes, 4*num_bones) matrix
        # returns: (3, 4*num_bones) skinning transforms

        n_shapes = blendshape_weights.shape[0]
        n_bones = self.shape_xform.shape[1] // 4

        # Create weighted diagonal matrices for each blendshape
        # Z shape: (3, 3, n_shapes)
        Z = blendshape_weights.reshape(1, 1, n_shapes) * np.dstack(
            [np.eye(3)] * n_shapes
        )

        # Z:
        # ┌      ┐┌      ┐┌      ┐
        # │w₁   0││w₂   0││w₃   0│
        # │  w₁  ││  w₂  ││  w₃  │  ───▶ axis 2
        # │0   w₁││0   w₂││0   w₃│
        # └      ┘└      ┘└      ┘
        #
        # Z.transpose(0, 2, 1).reshape(3, -1)
        # ┌                  ┐
        # │w₁0 0 w₂0 0 w₃0 0 │
        # │0 w₁0 0 w₂0 0 w₃0 │
        # │0 0 w₁0 0 w₂0 0 w₃│
        # └                  ┘

        # Compute weighted sum of blendshape transforms
        weighted_transforms = Z.transpose(0, 2, 1).reshape(3, -1) @ self.shape_xform

        # Add identity to diagonals for each transform
        identity_offset = (
            np.array([np.eye(3, 4)] * n_bones).transpose(1, 0, 2).reshape(3, -1)
        )

        return weighted_transforms + identity_offset

    def generate_frames(
        self,
        animation_weights_path: str | Path,
        output_dir: str | Path,
        max_control_weights: int = 72,
    ):
        """Generates animation frames and saves them as OBJ files.

        Args:
            animation_weights_path: Path to NPZ file containing animation weights.
                Expected to contain a "weights" array of shape (num_frames, num_controls).
            output_dir: Directory where output OBJ frames will be saved.
            max_control_weights: Number of control weights to use from animation file.
                Default is 72 (standard rig control count).

        Raises:
            FileNotFoundError: If animation_weights_path does not exist.
            ValueError: If animation weights file doesn't contain required data.

        Runtime Process (per frame):
            1. Load control weights from animation file (e.g., 72 FACS controls)
            2. Apply rig logic: control weights → blendshape weights
               Uses inbetween_info and combination_info from model_data
            3. Compute skinning transforms Mⱼ (Equation 7, CPU)
               Sparse matrix-vector product leveraging compressed data
            4. Apply skinning: vertices = Σⱼ wᵢ,ⱼ Mⱼ v₀,ᵢ (GPU-ready)
            5. Save as OBJ file

        Performance:
            On Snapdragon 652 (mobile platform):
                - Sparse transform computation: 160-251 μs
                - Dense transform computation: 520-653 μs
                - Speed-up: 2-3× (Table 3)

            Memory requirements:
                - Sparse storage: 81-87 KB
                - Dense storage: 486-612 KB
                - Savings: 5-7× (Table 2)

        References:
            - Figure 2: Shows runtime process flow
            - Tables 2-3: Performance metrics and comparisons
            - Section 5: Results and validation
        """
        animation_weights_path = Path(animation_weights_path)
        output_dir = Path(output_dir)

        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load animation data
        if not animation_weights_path.exists():
            raise FileNotFoundError(
                f"Animation file not found: {animation_weights_path}"
            )

        anim_data = np.load(animation_weights_path)
        if "weights" not in anim_data:
            raise ValueError("Animation file must contain 'weights' array")

        control_weights = anim_data["weights"][:, :max_control_weights]

        # Apply rig logic to get blendshape weights
        blendshape_weights = rl.compute_rig_logic(
            torch.from_numpy(control_weights).float(),
            self.model_data.inbetween_info,
            self.model_data.combination_info,
        ).numpy()

        num_frames = blendshape_weights.shape[0]
        print(f"Generating {num_frames} frames...")
        print(f"Number of blendshapes: {blendshape_weights.shape[1]}")

        # Generate each frame
        for frame_idx in range(num_frames):
            # Get skinning transforms for this frame
            transforms = self._generate_skinning_transforms(
                blendshape_weights[frame_idx, :]
            )

            # Apply skinning to rest pose
            # weights: (num_vertices, num_bones)
            # rest_pose_homog: (num_vertices, 4)
            # X: (4 * num_bones, num_vertices)
            X = (
                self.weights.T.reshape(self.num_bones, 1, -1) * self.rest_pose_homog.T
            ).reshape(4 * self.num_bones, -1)

            # Apply transforms: (3, 4 * num_bones) @ (4 * num_bones, num_vertices)
            # Result: (3, num_vertices)
            animated_verts = transforms @ X

            # Save as OBJ file
            output_path = output_dir / f"anim_frame{frame_idx:05d}.obj"
            igl.write_obj(str(output_path), animated_verts.T, self.quads)

            if (frame_idx + 1) % 10 == 0 or frame_idx == 0:
                print(f"Generated frame {frame_idx + 1}/{num_frames}")

        print(f"All frames saved to {output_dir}")
