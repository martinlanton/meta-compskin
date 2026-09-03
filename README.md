# Compressed Skinning for Facial Blendshapes

**[Meta Reality Labs Research]**

[Ladislav Kavan], [John Doublestein], [Martin Prazak], [Matthew Cioffi], [Doug Roble]

[[`Paper`] https://arxiv.org/abs/2406.11597 ]

`metacompskin` converts a facial blendshape model into a linear blend skinning
model driven by about 40 virtual joints. The skinned result reproduces the
blendshapes to a mean error below 0.05 mm while using 5 to 7× less memory and
evaluating 2 to 3× faster than dense skinning decompositions such as Dem Bones.
It is the reference implementation of the SIGGRAPH 2024 paper.

## Documentation

The full documentation lives in [`docs/`](docs/index.md) and is published from
`main`. Start with:

- **Riggers:** [Overview](docs/concepts/overview.md) →
  [Preparing data](docs/user_guide/preparing_data.md) →
  [Maya rig workflow](docs/user_guide/maya_rig_workflow.md)
- **TDs and engineers:** [Installation](docs/getting_started/installation.md) →
  [Quick start](docs/getting_started/quickstart.md) →
  [Compressing](docs/user_guide/compressing.md) →
  [Pipeline integration](docs/user_guide/pipeline_integration.md)
- **The method:** [From blendshapes to skinning](docs/concepts/blendshapes_to_skinning.md),
  [How the solver works](docs/concepts/how_the_solver_works.md),
  [Data formats](docs/concepts/data_formats.md)
- **Contributors:** [Architecture](docs/developer/architecture.md),
  [Development](docs/developer/development.md)

## Installation

Requires Python 3.10+. For GPU compression (about 60× faster), install a CUDA
build of PyTorch first from [pytorch.org](https://pytorch.org/get-started/locally/).

```bash
git clone https://github.com/martinlanton/meta-compskin.git
cd meta-compskin
pip install .                      # or: pip install -e ".[dev]" for development
```

Inside Maya only numpy is needed: `mayapy -m pip install --no-deps <path-to-meta-compskin>`.
Details and platform paths: [Installation](docs/getting_started/installation.md).

## Usage in three lines

```python
from metacompskin import BlendshapeModelData, SkinCompressor

model_data = BlendshapeModelData.from_npz("exports/head.npz")
SkinCompressor(model_data=model_data, iterations=10000).run("exports/head_compressed.npz")
```

Export `head.npz` from a Maya scene with `MayaBlendshapeExporter("head_GEO").export(...)`,
or build it from any DCC ([Preparing data](docs/user_guide/preparing_data.md)).
The output holds skin weights and, per blendshape, the motion of every joint;
what to do with it is in [Maya rig workflow](docs/user_guide/maya_rig_workflow.md)
and [Pipeline integration](docs/user_guide/pipeline_integration.md).
Complete scripts are in [`examples/`](examples/).

## Project structure

```
src/metacompskin/
├── model_data.py            BlendshapeModelData, MayaBlendshapeModelData   (input)
├── model_fit.py             SkinCompressor                                  (solver)
├── animation_generator.py   AnimationFrameGenerator                         (reference runtime)
├── maya_exporter.py         MayaBlendshapeExporter        (runs inside Maya, numpy only)
├── maya_loader.py           OBJ loading through Maya
├── constants.py             per-model defaults
└── rig/riglogic.py          rig logic of the sample heads
docs/                        documentation sources (Sphinx + MyST)
examples/                    end-to-end example scripts
tests/                       test suite and sample data
paper/                       the paper in Markdown
```

## License

The model is licensed under the [Apache 2.0 license](LICENSE.txt).
