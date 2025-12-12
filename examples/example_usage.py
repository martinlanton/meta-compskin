"""Example script demonstrating how to use the AnimationFrameGenerator.

This script shows how to:
1. Load model data
2. Run compression with SkinCompressor
3. Generate animation frames with AnimationFrameGenerator
"""

from pathlib import Path

from metacompskin.animation_generator import AnimationFrameGenerator
from metacompskin.model_data import BlendshapeModelData
from metacompskin.model_fit import SkinCompressor


def main():
    """Example workflow for compression and animation generation."""
    # Paths
    source_model_path = Path("data/source_models/aura.npz")
    animation_path = Path("data/source_models/test_anim.npz")
    compressed_output_path = Path("output/aura_compressed.npz")
    animation_output_dir = Path("output/frames")

    # Step 1: Load model data
    print("Loading model data...")
    model_data = BlendshapeModelData.from_npz(source_model_path)

    # Step 2: Run compression
    print("\nRunning compression...")
    compressor = SkinCompressor(model_data=model_data, iterations=10000)
    compressor.run(output_location=compressed_output_path)
    print(f"Compressed data saved to: {compressed_output_path}")

    # Step 3: Generate animation frames (optional)
    print("\nGenerating animation frames...")
    generator = AnimationFrameGenerator(
        compressed_data_path=compressed_output_path, model_data=model_data
    )
    generator.generate_frames(
        animation_weights_path=animation_path, output_dir=animation_output_dir
    )
    print(f"Animation frames saved to: {animation_output_dir}")


if __name__ == "__main__":
    main()
