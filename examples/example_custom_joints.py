"""Example: Using SkinCompressor with custom joint matrices.

This example demonstrates how to use the SkinCompressor with custom
rest joint matrices, allowing compression to work with specific facial
rigs instead of generating joints at the center of the world.
"""

import json
import numpy as np
from pathlib import Path

from metacompskin.model_data import BlendshapeModelData
from metacompskin.model_fit import SkinCompressor


def example_with_custom_joints():
    """Example of using custom joint transformation matrices."""
    # Load your blendshape model data
    # Replace this path with your actual model file
    model_data = BlendshapeModelData.from_npz("path/to/your/model.npz")

    # Option 1: Load joint matrices from a JSON file
    # The matrices should be 4×4 homogeneous transformation matrices
    # stored as a flat list of 16 values per matrix
    with open("path/to/joint_matrices.json", encoding="utf-8") as f:
        matrices_flat = json.load(f)

    # Reshape from flat list to (N, 4, 4) array
    joint_matrices = np.array(matrices_flat).reshape(-1, 4, 4)

    # Create compressor with custom joint matrices
    # The number of bones will be automatically set to match
    # the number of matrices provided
    compressor = SkinCompressor(
        model_data=model_data,
        iterations=10000,
        rest_joint_matrices=joint_matrices
    )

    print(f"Number of bones: {compressor.number_of_bones}")

    # Run the compression
    compressor.run(output_location="output/custom_joints_compressed.npz")


def example_with_generated_joints():
    """Example of programmatically generating joint matrices."""
    model_data = BlendshapeModelData.from_npz("path/to/your/model.npz")

    # Generate 30 joint matrices positioned in a grid
    num_joints = 30
    joint_matrices = []

    for i in range(num_joints):
        # Create a 4×4 identity matrix
        matrix = np.eye(4)

        # Set the translation (last column, first 3 rows)
        # Position joints in a 3D grid
        x = (i % 5) * 2.0  # 5 joints along X
        y = ((i // 5) % 3) * 2.0  # 3 joints along Y
        z = (i // 15) * 2.0  # 2 joints along Z

        matrix[0, 3] = x
        matrix[1, 3] = y
        matrix[2, 3] = z

        joint_matrices.append(matrix)

    joint_matrices = np.array(joint_matrices)

    # Create compressor with generated joints
    compressor = SkinCompressor(
        model_data=model_data,
        iterations=10000,
        rest_joint_matrices=joint_matrices
    )

    compressor.run(output_location="output/generated_joints_compressed.npz")


def example_default_behavior():
    """Example using default behavior (identity matrices, 40 bones)."""
    model_data = BlendshapeModelData.from_npz("path/to/your/model.npz")

    # When rest_joint_matrices is not provided (or is None),
    # the compressor uses the default behavior:
    # - 40 bones
    # - Identity matrices at the origin
    compressor = SkinCompressor(
        model_data=model_data,
        iterations=10000
        # rest_joint_matrices=None  # This is the default
    )

    print(f"Number of bones: {compressor.number_of_bones}")  # Will print 40

    compressor.run(output_location="output/default_compressed.npz")


if __name__ == "__main__":
    print("Example: SkinCompressor with custom joint matrices")
    print("=" * 60)
    print("\nThis example demonstrates three ways to use SkinCompressor:")
    print("1. With joint matrices loaded from a JSON file")
    print("2. With programmatically generated joint matrices")
    print("3. With default behavior (no custom matrices)")
    print("\nUpdate the file paths in the code to run the examples.")

