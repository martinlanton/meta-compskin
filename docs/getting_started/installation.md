# Installation

`metacompskin` is a Python package. The compression step needs PyTorch and
ideally a CUDA GPU; the Maya-side exporter needs only numpy. Pick the section
that matches where you are going to run it.

## Requirements

| Component | Requirement |
|-----------|-------------|
| Python | 3.10 or later |
| PyTorch | 1.9 or later. Installed automatically (CPU build). A CUDA build is strongly recommended for compression, see below. |
| NumPy, SciPy | Installed automatically. |
| GPU | Optional. Compression is roughly 60× faster on an NVIDIA GPU than on CPU: about a minute for 10 000 iterations instead of most of an hour. |
| Autodesk Maya | Optional. Only needed to export blendshapes from Maya scenes or to load OBJ files through Maya's importer. Maya 2024 or later ships numpy; older versions need it installed into `mayapy`. |

## Standard installation

```bash
git clone https://github.com/martinlanton/meta-compskin.git
cd meta-compskin
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install .
```

### GPU support

`pip install .` pulls in the CPU build of PyTorch if none is present. To use a
GPU, install a CUDA build of PyTorch **before** installing the package, using
the command for your platform and CUDA version from
[pytorch.org/get-started](https://pytorch.org/get-started/locally/). For example:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install .
```

Check that the GPU is visible:

```python
import torch
print(torch.cuda.is_available())      # True
print(torch.cuda.get_device_name(0))  # e.g. NVIDIA RTX A6000
```

The compressor selects CUDA automatically when it is available and falls back to
CPU otherwise. No code changes are needed either way.

### Verify

```bash
python -c "import metacompskin; print(metacompskin.__version__)"
```

## Development installation

For working on the package itself, install it editable with the `dev` extra:

```bash
pip install -e ".[dev]"
pre-commit install      # optional: run ruff and mypy on every commit
pytest                  # run the test suite
```

The `dev` extra contains pytest and pytest-cov, ruff, mypy, pre-commit, and
Sphinx with the RTD theme and MyST parser for building this documentation.
The `viz` extra adds matplotlib.

## Inside Autodesk Maya

Only needed to run `MayaBlendshapeExporter` (exporting a blendshape model from a
scene) or `MayaBlendshapeModelData.from_obj_files` without a `mayapy` path.

The package imports its heavy dependencies lazily, so **PyTorch is not required
inside Maya**. Install without dependencies to keep it that way:

```bash
mayapy -m pip install --no-deps <path-to-meta-compskin>
```

`mayapy` lives in Maya's `bin` directory:

| Platform | Path |
|----------|------|
| Windows | `C:\Program Files\Autodesk\Maya2025\bin\mayapy.exe` |
| macOS | `/Applications/Autodesk/maya2025/Maya.app/Contents/bin/mayapy` |
| Linux | `/usr/autodesk/maya2025/bin/mayapy` |

If you would rather not install anything into Maya, add the source folder to the
path at the top of your script instead:

```python
import sys
sys.path.insert(0, r"D:/path/to/meta-compskin/src")
```

Verify from a shell:

```bash
mayapy -c "from metacompskin.maya_exporter import MayaBlendshapeExporter; print('OK')"
```

Maya 2023 and older do not bundle numpy. Install it with
`mayapy -m pip install numpy` first.

## Typical setups

- **Rigger with a Maya workstation and no GPU.** Install into `mayapy` (above)
  to export; hand the exported file to whoever runs compression, or install the
  standard package and accept CPU compression times.
- **TD with a GPU box.** Standard installation with the CUDA build of PyTorch.
  Exports arrive as `.npz` files from Maya machines.
- **Contributor.** Development installation, plus Maya if you touch the Maya
  modules.

Next: [Quick start](quickstart.md).
