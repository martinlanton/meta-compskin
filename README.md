# Compressed Skinning for Facial Blendshapes

**[Meta Reality Labs Research]**

[Ladislav Kavan], [John Doublestein], [Martin Prazak], [Matthew Cioffi], [Doug Roble]

[[`Paper`] https://arxiv.org/abs/2406.11597 ]

## Installation

Requires **Python 3.10+**. The core dependencies (PyTorch, NumPy, SciPy) are
installed automatically.

### Standard installation

```bash
git clone https://github.com/martinlanton/meta-compskin.git
cd meta-compskin
pip install .
```

By default this installs the CPU build of PyTorch. The optimization is ~60×
faster on GPU — for CUDA support, install a CUDA build of PyTorch **first**
(pick your platform/CUDA version at https://pytorch.org/get-started/locally/),
then install this package:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128  # example
pip install .
```

Verify the installation:

```bash
python -c "import metacompskin; print(metacompskin.__version__)"
```

### Development installation

For working on the code itself (editable install with test/lint/docs tooling):

```bash
pip install -e ".[dev]"
pre-commit install   # optional: run ruff/mypy automatically on commit
pytest               # run the test suite
```

Optional extras:

- `.[dev]` — pytest, pytest-cov, ruff, mypy, pre-commit, Sphinx (docs)
- `.[viz]` — matplotlib, for visualization helpers

### Inside Autodesk Maya (for the exporter)

Only needed if you want to use `MayaBlendshapeExporter` to export model data
from Maya scenes. Maya's Python only needs **numpy** (bundled with recent Maya
versions) — install the package **without dependencies** so PyTorch is not
pulled into Maya:

```bash
mayapy -m pip install --no-deps <path-to-meta-compskin>
```

(`mayapy` lives in Maya's `bin` directory, e.g.
`C:\Program Files\Autodesk\Maya2025\bin\mayapy.exe` on Windows or
`/Applications/Autodesk/maya2025/Maya.app/Contents/bin/mayapy` on macOS.)

Alternatively, skip installation entirely and add the source directory to the
path at the top of your Maya script:

```python
import sys
sys.path.insert(0, r"D:/path/to/meta-compskin/src")
```

Verify from a shell:

```bash
mayapy -c "from metacompskin.maya_exporter import MayaBlendshapeExporter; print('OK')"
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

### Exporting Model Data from Autodesk Maya

`MayaBlendshapeExporter` runs inside Maya (or mayapy) and writes an NPZ file
that `BlendshapeModelData.from_npz` can load anywhere — so you export in Maya,
then compress outside Maya where PyTorch/CUDA is available. It only needs
numpy inside Maya; importing `metacompskin` in Maya does **not** require
PyTorch.

Inside Maya:

```python
from metacompskin.maya_exporter import MayaBlendshapeExporter

# Simplest: auto-discovers the blendShape node on the mesh and captures
# every target (weights and envelope are restored afterwards).
MayaBlendshapeExporter("head_GEO").export("exports/head.npz")

# Or diff separate target meshes against the rest mesh (points are read in
# object space, so targets translated aside for layout are handled):
MayaBlendshapeExporter(
    "head_GEO",
    target_meshes=["smile_GEO", "frown_GEO"],
).export("exports/head.npz")

# Optionally export skeleton joint rest matrices for
# SkinCompressor(rest_joint_matrices=...):
MayaBlendshapeExporter(
    "head_GEO",
    joints=["jaw_JNT", "cheek_L_JNT", "cheek_R_JNT"],  # >= 9 for compression
).export("exports/head.npz")
```

Then outside Maya:

```python
from metacompskin.model_data import BlendshapeModelData
from metacompskin.model_fit import SkinCompressor

model_data = BlendshapeModelData.from_npz("exports/head.npz")
SkinCompressor(model_data=model_data).run("exports/head_compressed.npz")
```

See `examples/example_maya_export.py` for the full workflow, including
compressing with the exported joint matrices.

### Complete Workflow

See `examples/example_usage.py` for a complete workflow example.

### Using the Output in a Rig

Riggers who need to bring the compressed weights and joint motion into Maya (or any
other skinning pipeline) should read
[Using Compressed Skinning Output in Maya](docs/guides/using_compressed_skinning_in_maya.md).
It explains the data, the mixing rule that turns blendshape values into joint
transforms, and what to validate, without assuming a particular tool or rig setup.

## Project Structure

- `src/metacompskin/`
  - `model_data.py` - BlendshapeModelData class for loading and validating model data
  - `model_fit.py` - SkinCompressor class for optimization and compression
  - `animation_generator.py` - AnimationFrameGenerator for generating animation frames
  - `maya_exporter.py` - MayaBlendshapeExporter for exporting model data from inside Maya
  - `maya_loader.py` - Loading blendshape geometry from OBJ files via mayapy
  - `constants.py` - Model-specific constants and configurations
  - `rig/riglogic.py` - Rig logic for computing blendshape weights

## License

The model is licensed under the [Apache 2.0 license](LICENSE).


