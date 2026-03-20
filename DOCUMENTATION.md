# Compressed Skinning for Facial Blendshapes — Full Documentation

**[Meta Reality Labs Research]**

[Ladislav Kavan], [John Doublestein], [Martin Prazak], [Matthew Cioffi], [Doug Roble]

[[`Paper`] https://arxiv.org/abs/2406.11597 ]

---

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Getting Started](#getting-started)
  - [Basic Usage](#basic-usage)
  - [Generating Animation Frames](#generating-animation-frames)
  - [Complete Workflow](#complete-workflow)
- [Custom Joint Matrices](#custom-joint-matrices)
  - [Overview](#custom-joints-overview)
  - [Usage Examples](#usage-examples)
  - [Matrix Format](#matrix-format)
  - [Loading from JSON](#loading-from-json)
  - [Validation Rules](#validation-rules)
- [Maya Integration](#maya-integration)
  - [Loading OBJ Files](#loading-obj-files)
  - [Generating Compressed Data for Maya Rigs](#generating-compressed-data-for-maya-rigs)
  - [Building a Rig in Maya](#building-a-rig-in-maya)
- [Output Format](#output-format)
- [Project Structure](#project-structure)
- [Implementation Details](#implementation-details)
  - [Core Changes](#core-changes)
  - [Test Results](#test-results)
  - [Backward Compatibility](#backward-compatibility)
- [Testing](#testing)
  - [Running Tests (meta-compskin)](#running-tests-meta-compskin)
  - [Running Tests (private tests)](#running-tests-private-tests)
- [API Quick Reference](#api-quick-reference)
- [Limitations and Future Work](#limitations-and-future-work)
- [License](#license)

---

## Overview

This repository implements a novel method for converting facial animation blendshapes
into a fast linear blend skinning representation using sparse transformations. The
project is based on the SIGGRAPH 2024 paper "Compressed Skinning for Facial
Blendshapes."

**Core Technology:**
- PyTorch-based optimization using proximal algorithms with Adam optimizer
- Sparse skinning decomposition achieving ~90% sparsity in transformations
- 5–7× memory savings and 2–3× speed improvements over dense methods (Dem Bones)
- Targets low-spec mobile platforms (e.g., Snapdragon 652)

---

## Installation

### Prerequisites

- Python 3.8+
- PyTorch (https://pytorch.org/get-started/locally/)
- libigl (https://github.com/libigl/libigl-python-bindings)

```bash
pip install torch numpy scipy igl
```

### Development Installation

```bash
# Clone the repository
git clone https://github.com/martinlanton/meta-compskin.git
cd meta-compskin

# Install in development mode with dev dependencies
pip install -e ".[dev]"
```

### GPU Support (Recommended)

GPU acceleration provides ~60× faster compression:

```bash
# For CUDA 11.8
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia -y

# For CUDA 12.1
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

Verify GPU setup:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")
```

---

## Getting Started

### Basic Usage

The package provides a clean API for compressing blendshape data:

```python
from metacompskin.model_data import BlendshapeModelData
from metacompskin.model_fit import SkinCompressor

# Load your blendshape model
model_data = BlendshapeModelData.from_npz("data/source_models/aura.npz")

# Run compression (default: 40 bones, identity matrices)
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

---

## Custom Joint Matrices

<a name="custom-joints-overview"></a>
### Overview

The `SkinCompressor` class supports an optional `rest_joint_matrices` parameter
that allows you to provide custom 4×4 transformation matrices for joint rest poses.
This enables compression to work with specific facial rigs that have predefined
joint positions and orientations, instead of generating joints at the center of
the world.

**Key Features:**
- **Flexible Joint Positioning**: Specify exact positions and orientations for
  joints based on your facial rig
- **Automatic Bone Count**: The number of bones is automatically set to match the
  number of matrices provided
- **Backward Compatible**: When `rest_joint_matrices` is `None` (default), the
  behavior is identical to the previous implementation
- **Format Validation**: Input validation ensures matrices have the correct shape
  `(P, 4, 4)`

### Usage Examples

#### Option 1: Default Behavior (Identity Matrices, 40 Bones)

```python
from metacompskin.model_data import BlendshapeModelData
from metacompskin.model_fit import SkinCompressor

model_data = BlendshapeModelData.from_npz("model.npz")

# Default: 40 bones with identity matrices at origin
compressor = SkinCompressor(model_data=model_data, iterations=10000)
compressor.run(output_location="output.npz")
```

#### Option 2: Custom Joint Matrices (Auto-Sets Number of Bones)

```python
import numpy as np

joint_matrices = np.array([...])  # Shape: (N, 4, 4)
compressor = SkinCompressor(
    model_data=model_data,
    iterations=10000,
    rest_joint_matrices=joint_matrices  # N bones
)
compressor.run(output_location="output.npz")
```

#### Option 3: Loading Matrices from JSON

```python
import json
import numpy as np
from metacompskin.model_data import BlendshapeModelData
from metacompskin.model_fit import SkinCompressor

# Load joint matrices from JSON file
with open("matrices.json", encoding="utf-8") as f:
    matrices_flat = json.load(f)  # List of lists, 16 values each

joint_matrices = np.array(matrices_flat).reshape(-1, 4, 4)

# Load model and create compressor
model_data = BlendshapeModelData.from_npz("model.npz")
compressor = SkinCompressor(
    model_data=model_data,
    iterations=10000,
    rest_joint_matrices=joint_matrices
)
compressor.run(output_location="output.npz")
```

See `examples/example_custom_joints.py` for additional working examples.

### Matrix Format

Input: `(P, 4, 4)` homogeneous transformation matrices

```
[R₁₁  R₁₂  R₁₃  Tₓ]
[R₂₁  R₂₂  R₂₃  Tᵧ]
[R₃₁  R₃₂  R₃₃  Tᵤ]
[ 0    0    0   1 ]
```

Where:
- R₁₁ through R₃₃: 3×3 rotation matrix
- Tₓ, Tᵧ, Tᵤ: translation vector
- Bottom row: `[0, 0, 0, 1]` for homogeneous coordinates

### Loading from JSON

```python
import json
import numpy as np

with open("matrices.json", encoding="utf-8") as f:
    matrices_flat = json.load(f)  # List of lists, 16 values each

joint_matrices = np.array(matrices_flat).reshape(-1, 4, 4)
```

### Validation Rules

| Input | Status |
|-------|--------|
| `np.array` with shape `(P, 4, 4)` | ✅ Valid |
| `List` convertible to `(P, 4, 4)` | ✅ Valid |
| Shape `(P, 3, 4)` | ❌ `ValueError` |
| 2D array | ❌ `ValueError` |

---

## Maya Integration

### Loading OBJ Files

Use `MayaBlendshapeModelData` to load geometry from OBJ files exported from Maya:

```python
from pathlib import Path
from metacompskin.model_data import MayaBlendshapeModelData

rest = Path("maya/HEAD.obj")
shapes = sorted(Path("maya/").glob("*.obj"))
shapes = [s for s in shapes if s.name != "HEAD.obj"]

# Subprocess mode (standard Python)
mayapy = Path("C:/Program Files/Autodesk/Maya2025/bin/mayapy.exe")
model_data = MayaBlendshapeModelData.from_obj_files(
    rest_obj_path=rest,
    blendshape_paths=shapes,
    model_name="my_head",
    maya_interpreter_path=mayapy,
)
```

### Generating Compressed Data for Maya Rigs

Use the provided script to generate compressed data using custom joint matrices
and Maya OBJ files:

```bash
cd meta-compskin_private_tests
python scripts/generate_maya_compressed_data.py
```

See `meta-compskin_private_tests/scripts/generate_maya_compressed_data.py`
for full details. The script:

1. Loads joint matrices from `joint_matrices/matrices.json`
2. Loads HEAD.obj and all blendshape OBJs via mayapy
3. Runs the SkinCompressor with the custom joint matrices
4. Saves the compressed `.npz` output

### Building a Rig in Maya

Use the provided Maya script to build a visual rig from compressed data:

```bash
# Run inside Maya's Script Editor (Python tab) or via mayapy:
cd meta-compskin_private_tests
mayapy scripts/build_maya_rig.py
```

See `meta-compskin_private_tests/scripts/build_maya_rig.py` for full details. The script:

1. Loads the compressed `.npz` file
2. Creates Maya joints at the rest-pose positions from `restXform`
3. Imports the HEAD mesh from the OBJ file
4. Binds the mesh to the joints with the optimized skinning weights
5. Creates blendshape-driven animation controls

---

## Output Format

The compressed NPZ file produced by `SkinCompressor.run()` contains:

| Key | Shape | Description |
|-----|-------|-------------|
| `rest` | `(N, 3)` | Rest pose vertices (centered at origin) |
| `quads` | `(F, 4)` | Mesh faces |
| `weights` | `(N, P)` | Normalized skinning weights (non-negative, partition of unity, K-sparse) |
| `restXform` | `(P, 3, 4)` | Rest joint matrices (custom if provided, otherwise identity) |
| `shapeXform` | `(3S, 4P)` | Learned sparse transformation matrices (~90% zeros) |

---

## Project Structure

```
meta-compskin/
├── src/metacompskin/
│   ├── model_data.py            # BlendshapeModelData, MayaBlendshapeModelData
│   ├── model_fit.py             # SkinCompressor (core optimization)
│   ├── animation_generator.py   # AnimationFrameGenerator
│   ├── constants.py             # Model-specific constants
│   ├── maya_loader.py           # Maya OBJ loading utilities
│   ├── utils.py                 # Helper utilities
│   └── rig/riglogic.py          # Rig logic for blendshape weights
├── examples/
│   ├── example_usage.py         # Basic workflow example
│   └── example_custom_joints.py # Custom joint matrices example
├── scripts/
│   └── build_docs.py            # Sphinx documentation builder
├── tests/
│   └── test_default_output.py   # Backward compatibility tests
├── docs/                        # Sphinx documentation
├── DOCUMENTATION.md             # This file — combined documentation
└── pyproject.toml               # Project configuration
```

### Private Tests Repository

The companion repository `meta-compskin_private_tests` contains production-scale
test fixtures, validation tests, and Maya workflow scripts:

```
meta-compskin_private_tests/
├── scripts/
│   ├── generate_maya_compressed_data.py  # Generate compressed NPZ from Maya OBJs + joint matrices
│   └── build_maya_rig.py                 # Build visual rig inside Maya from compressed data
├── tests/
│   ├── conftest.py                    # Shared pytest fixtures
│   ├── test_fixtures_sample.py        # Fixture usage examples
│   ├── test_joint_matrices.py         # Custom joint matrix tests
│   ├── test_maya_compression.py       # Maya compression pipeline tests
│   ├── test_maya_integration.py       # API-level Maya integration tests
│   └── generate_expected_maya_output.py  # Regenerate expected test data
├── tests/test_data/
│   ├── joint_matrices/matrices.json   # 28 joint transformation matrices (4×4)
│   ├── maya/HEAD.obj                  # Neutral/rest pose mesh (5761 vertices)
│   ├── maya/AU1.obj ... AU20.obj      # Action Unit blendshapes
│   ├── maya/Shape*.obj                # 60+ named blendshape targets
│   └── expected_maya_compression.npz  # Regression test baseline
├── pyproject.toml
└── README.md                          # Comprehensive testing guide
```

For complete documentation on the private tests package (installation, running tests,
GPU acceleration, troubleshooting, etc.), see
**[meta-compskin_private_tests/README.md](../meta-compskin_private_tests/README.md)**.

---

## Implementation Details

### Core Changes

The custom joint matrices feature was implemented in `src/metacompskin/model_fit.py`:

#### `SkinCompressor.__init__()`
- **Added parameter**: `rest_joint_matrices: np.ndarray | list | None = None`
- **Validation**: Ensures input has shape `(P, 4, 4)` where P is number of bones
- **Automatic bone count**: When matrices are provided, `number_of_bones` is set
  to `len(rest_joint_matrices)`
- **Matrix extraction**: Extracts the 3×4 affine portion from the 4×4 matrices
- **Backward compatibility**: When `None`, uses default behavior (40 bones, identity)

#### `SkinCompressor.run()`
- Modified save logic to use `rest_joint_matrices_3x4` if available
- Falls back to identity matrices when custom matrices are not provided

#### Storage
- Internally, matrices are stored as `self.rest_joint_matrices_3x4` with shape `(P, 3, 4)`
- The 3×4 affine portion (first 3 rows) is extracted from each 4×4 input matrix
- Saved to the output NPZ file under the `restXform` key

### Test Results

#### Backward Compatibility Tests (meta-compskin)
```
tests/test_default_output.py::TestCompskin::test_default_output PASSED
tests/test_default_output.py::TestCompskin::test_default_output_short_iter PASSED

2 passed ✅
```

#### Custom Joint Matrices Tests (meta-compskin_private_tests)
```
tests/test_joint_matrices.py::TestJointMatrices::test_default_behavior_unchanged PASSED
tests/test_joint_matrices.py::TestJointMatrices::test_custom_joint_matrices_from_fixture PASSED
tests/test_joint_matrices.py::TestJointMatrices::test_invalid_shape_raises_error PASSED
tests/test_joint_matrices.py::TestJointMatrices::test_list_input_accepted PASSED
tests/test_joint_matrices.py::TestJointMatrices::test_output_differs_from_default PASSED

5 passed ✅
```

### Backward Compatibility

The feature is fully backward compatible:
- Existing code continues to work without modification
- Default behavior (40 bones, identity matrices) is preserved
- All existing tests pass without changes
- The `rest_joint_matrices` parameter defaults to `None`

---

## Testing

### Running Tests (meta-compskin)

```bash
cd meta-compskin

# Run backward compatibility tests (fast, ~1 min)
python -m pytest tests/test_default_output.py::TestCompskin::test_default_output_short_iter -v

# Run full default output test (slow, ~20 min CPU / ~50s GPU)
python -m pytest tests/test_default_output.py::TestCompskin::test_default_output -v
```

### Running Tests (private tests)

```bash
cd meta-compskin_private_tests

# Run fixture/sample tests (fast, no Maya needed)
python -m pytest tests/test_fixtures_sample.py -v

# Run custom joint matrices tests (requires aura.npz test data)
python -m pytest tests/test_joint_matrices.py -v

# Run Maya integration tests (requires mayapy)
python -m pytest tests/test_maya_integration.py -v

# Run Maya compression regression test (requires mayapy, ~8–10 min)
python -m pytest tests/test_maya_compression.py::test_maya_compression_regression -v

# Run comprehensive quality test (slow, requires mayapy)
python -m pytest tests/test_maya_compression.py -m slow -v -s
```

For full details on the private test suite — including GPU acceleration, performance
benchmarks, troubleshooting, and CI/CD integration — see
**[meta-compskin_private_tests/README.md](../meta-compskin_private_tests/README.md)**.

---

## API Quick Reference

### SkinCompressor

```python
from metacompskin.model_fit import SkinCompressor

# Default (40 bones, identity matrices)
compressor = SkinCompressor(model_data=model_data, iterations=10000)

# Custom joint matrices (auto-sets number of bones)
compressor = SkinCompressor(
    model_data=model_data,
    iterations=10000,
    rest_joint_matrices=joint_matrices  # shape (P, 4, 4)
)

compressor.run(output_location="output.npz")
```

### BlendshapeModelData

```python
from metacompskin.model_data import BlendshapeModelData

# From NPZ file
model_data = BlendshapeModelData.from_npz("model.npz")
```

### MayaBlendshapeModelData

```python
from metacompskin.model_data import MayaBlendshapeModelData

model_data = MayaBlendshapeModelData.from_obj_files(
    rest_obj_path=Path("HEAD.obj"),
    blendshape_paths=[Path("AU1.obj"), Path("AU2.obj")],
    maya_interpreter_path=Path("C:/.../mayapy.exe"),
)
```

### AnimationFrameGenerator

```python
from metacompskin.animation_generator import AnimationFrameGenerator

generator = AnimationFrameGenerator(
    compressed_data_path="compressed.npz",
    model_data=model_data,
)
generator.generate_frames(
    animation_weights_path="anim.npz",
    output_dir="frames/",
)
```

---

## Limitations and Future Work

### Current Limitations

1. Input joint matrices must be 4×4 (not 3×4)
2. All matrices must be provided together (no partial specification)
3. Number of bones is fixed once matrices are provided
4. Compression is lossy — typical MAE < 0.05 mm

### Potential Future Enhancements

- Validation of homogeneous coordinate convention (`[0, 0, 0, 1]` in last row)
- Warning for non-standard transformation matrices
- Support for rotation + translation representation (quaternion + vector)
- Per-bone naming/labeling for better debugging
- GPU-accelerated proximal projection
- Adaptive learning rate scheduling

---

## License

The model is licensed under the [Apache 2.0 license](LICENSE.txt).

