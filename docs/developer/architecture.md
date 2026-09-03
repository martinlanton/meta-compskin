# Architecture

A map of the package for contributors: what each module owns, how data flows
between them, and the design decisions that are not obvious from the code.

## Modules

```
src/metacompskin/
├── __init__.py            version + lazy re-exports of the public classes
├── model_data.py          BlendshapeModelData, MayaBlendshapeModelData   (input)
├── model_fit.py           SkinCompressor                                  (solver)
├── animation_generator.py AnimationFrameGenerator                         (reference runtime)
├── maya_exporter.py       MayaBlendshapeExporter        (runs inside Maya, numpy only)
├── maya_loader.py         load_obj_with_maya            (OBJ via Maya, direct or subprocess)
├── constants.py           per-model alpha defaults
├── utils.py               homogeneous coordinates, tensor → numpy
└── rig/riglogic.py        rig logic of the sample heads (controls → shape coefficients)
```

| Module | Owns | Depends on |
|--------|------|------------|
| `model_data` | Validated, immutable input container. Loading from NPZ; loading from OBJ through `maya_loader`. | numpy, `constants`, `maya_loader` |
| `model_fit` | The optimisation: matrix $\mathbf{A}$, Laplacian, basis $\mathbf{TR}$, two-phase training with projections, output file. | torch, scipy, `utils` |
| `animation_generator` | Reading the output file, Equation 7, linear blend skinning, OBJ writing, sample-rig animation playback through `riglogic`. | numpy, torch (for `riglogic` only), `utils` |
| `maya_exporter` | Capturing deltas from a live Maya scene; joint matrices. Writes the same file `model_data` reads. | numpy, `maya.cmds`, `maya.api.OpenMaya` at call time |
| `maya_loader` | Getting vertices and faces out of an OBJ using Maya's importer, in-process or via a spawned `mayapy`. | numpy, subprocess |

## Data flow

```
                Maya scene                  OBJ files                 any DCC / script
                    │                           │                           │
        MayaBlendshapeExporter     MayaBlendshapeModelData.from_obj_files   np.savez
                    │                           │                           │
                    └──────────────► input .npz ◄───────────────────────────┘
                                          │
                              BlendshapeModelData.from_npz     validate shapes, pick alpha
                                          │
                                   SkinCompressor.run          A = deltas as (3S, N)
                                          │                    L = mesh Laplacian
                                          │                    phase 1: train(normalizeW=False)
                                          │                    phase 2: train(normalizeW=True)
                                          │                    report MXE / MAE
                                          ▼
                                   compressed .npz            rest, quads, weights, restXform, shapeXform
                                          │
                     ┌────────────────────┼─────────────────────────┐
                     ▼                    ▼                         ▼
          AnimationFrameGenerator     rig builders             engine importers
          (reference runtime,         (Maya rig workflow)      (pipeline integration)
           tests, evaluation)
```

Shapes at each stage, using $N$ vertices, $S$ shapes, $P$ joints, $F$ faces:

| Stage | Arrays |
|-------|--------|
| input | `deltas (S, N, 3)`, `rest_verts (N, 3)`, `rest_faces (F, 3|4)` |
| optimisation | `A (3S, N)`, `B_rt (6, S, P, 1, 1)`, `TR (6, 1, 1, 3, 4)`, `W (P, N)`, `rest_pose (N, 4)`, `L (N, N)` sparse |
| output | `rest (N, 3)`, `weights (N, P)`, `restXform (P, 3, 4)`, `shapeXform (3S, 4P)` |
| runtime | coefficients `(S,)` → transforms `(3, 4P)` → vertices `(N, 3)` |

## Design decisions

**Torch-free import path.** `metacompskin/__init__.py` re-exports the public
classes through a module `__getattr__` (PEP 562) instead of importing them
eagerly. `maya_exporter` and `model_data` import only numpy at module level.
This is what allows `pip install --no-deps` into `mayapy`: a rigger can export
from Maya without PyTorch ever being installed there. Keep it that way: any
new module that Maya needs must not import torch at the top level, and
`animation_generator` (which imports torch for `riglogic`) must stay out of the
Maya path.

**Frozen input container.** `BlendshapeModelData` is a frozen dataclass
validated in `from_npz` (and by the exporter, which reuses the same validator
before writing). Direct construction skips validation on purpose so tests can
build small synthetic models.

**Solver state lives on the instance.** `SkinCompressor` keeps settings as
plain attributes set in `__init__` so callers can override them between
construction and `run`. `run` builds all tensors, trains twice, evaluates and
saves; `train` is the inner loop and is re-entrant on the same tensors.

**Projections, not penalties.** All constraints are enforced by projection
after each Adam step under `torch.no_grad` (see
[How the solver works](../concepts/how_the_solver_works.md)). The only penalty
in the loss is the Laplacian term.

**Custom joints are bookkeeping.** `rest_joint_matrices` sets $P$ and is echoed
to `restXform`; the solve itself always uses identity rest transforms at the
centred origin. This keeps Equation 7 trivially valid and moves the
"where are the joints" question to the rig builder, where the bind pose
answers it.

**Maya loading has two modes.** `maya_loader` either imports `maya.cmds`
directly (when already inside Maya) or writes a small script to a temp file
and runs it with `mayapy`, parsing JSON from stdout. The subprocess mode is
slow (Maya start-up per file) but lets a plain Python session use Maya's OBJ
importer, which matches how the shapes were exported.

**Sample rig logic is a black box.** `rig/riglogic.py` is the original
authors' rig evaluation for the sample heads, kept verbatim (ruff and coverage
exclude it). It is used only by `generate_frames` and the vertex-position
regression tests.

## Testing strategy

| Test | What it pins down | Cost |
|------|-------------------|------|
| `test_model_data.py` | Loading, validation errors, alpha lookup. | seconds |
| `test_pipeline_smoke.py` | Full pipeline on a tiny synthetic grid: shapes, weight constraints, finiteness. Runs on every platform in CI. | seconds |
| `test_default_output*.py` | Bit-exact regression of the compressed output for the Aura sample at 600 and 10 000 iterations, per platform (Windows and macOS data sets). | 1 min / 20 min CPU |
| `test_vertex_positions.py` | Regression of skinned vertex positions on 30 diverse animation frames (macOS data). Catches a broken runtime with correct weights. | 1 min |

Bit-exact tests are platform-specific because torch results differ across
BLAS backends and hardware; each expected data set records the platform it was
generated on (`tests/test_data/<platform>/SETUP.md`). The Maya modules are
exercised by the private companion repository, which has the production-scale
fixtures and a `mayapy` on the machine.

## Documentation build

Sphinx with MyST. Narrative pages are Markdown under `docs/`, the API
reference is reStructuredText under `docs/api/` (one `automodule` page per
module, private members excluded). `docs/README.md` describes the layout;
[Development](development.md) has the commands. The site deploys from `main`
via `.github/workflows/docs.yml`.
