# Compressed Skinning for Facial Blendshapes

**[Meta Reality Labs Research]**

[Ladislav Kavan], [John Doublestein], [Martin Prazak], [Matthew Cioffi], [Doug Roble]

[[`Paper`] https://arxiv.org/abs/2406.11597 ]

## Installation

Please install Python, PyTorch (https://pytorch.org/get-started/locally/) and libigl (https://github.com/libigl/libigl-python-bindings).

```bash
pip install torch numpy scipy igl
```

## <a name="GettingStarted"></a>Getting Started

### Basic Usage

The package provides a clean API for compressing blendshape data:

```python
from metacompskin.model_data import BlendshapeModelData
from metacompskin.model_fit import SkinCompressor

# Load your blendshape model
model_data = BlendshapeModelData.from_npz("data/source_models/aura.npz")

# Run compression
compressor = SkinCompressor(model_data=model_data, iterations=10000)
compressor.run(output_location="output/aura_compressed.npz")
```

### Generating Animation Frames

To generate animation frames from compressed data:

```python
from metacompskin.animation_generator import AnimationFrameGenerator

# Create generator with compressed data
generator = AnimationFrameGenerator(
    compressed_data_path="output/aura_compressed.npz",
    model_data=model_data
)

# Generate frames from animation weights
generator.generate_frames(
    animation_weights_path="data/source_models/test_anim.npz",
    output_dir="output/frames"
)
```

### Complete Workflow

See `examples/example_usage.py` for a complete workflow example.

## Project Structure

- `src/metacompskin/`
  - `model_data.py` - BlendshapeModelData class for loading and validating model data
  - `model_fit.py` - SkinCompressor class for optimization and compression
  - `animation_generator.py` - AnimationFrameGenerator for generating animation frames
  - `constants.py` - Model-specific constants and configurations
  - `rig/riglogic.py` - Rig logic for computing blendshape weights

## License

The model is licensed under the [Apache 2.0 license](LICENSE).


