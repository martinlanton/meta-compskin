# Compressed Skinning for Facial Blendshapes

`metacompskin` converts a facial blendshape model into a **linear blend
skinning** model driven by a small set of virtual joints. The result reproduces
the original blendshapes to within a fraction of a millimetre, while using
5 to 7 times less memory and evaluating 2 to 3 times faster than existing
skinning decompositions. It is the reference implementation of the SIGGRAPH
2024 paper *Compressed Skinning for Facial Blendshapes* from Meta Reality Labs
Research ([arXiv 2406.11597](https://arxiv.org/abs/2406.11597)).

```
blendshape targets  ──►  SkinCompressor  ──►  skin weights + per-shape joint motion
   (S × N offsets)        (offline, GPU)          (small, sparse, engine-ready)
```

## Where to start

**I am a rigger.** I want to understand what this produces and how to use it in
my rig, without writing code.

1. [Overview](concepts/overview.md): the idea in plain language.
2. [Preparing data](user_guide/preparing_data.md): getting blendshapes out of Maya.
3. [Maya rig workflow](user_guide/maya_rig_workflow.md): using the output in a rig.
4. [Troubleshooting](user_guide/troubleshooting.md): symptoms and causes.

**I am a TD or pipeline engineer.** I want to run the tool, tune it, and
integrate the output in a pipeline or an engine.

1. [Installation](getting_started/installation.md) and [Quick start](getting_started/quickstart.md).
2. [From blendshapes to skinning](concepts/blendshapes_to_skinning.md) and [How the solver works](concepts/how_the_solver_works.md).
3. [Data formats](concepts/data_formats.md): every file, key and convention.
4. [Compressing](user_guide/compressing.md), [Evaluating results](user_guide/evaluating_results.md) and [Pipeline integration](user_guide/pipeline_integration.md).
5. [API reference](api/index.rst).

## What you get

| | Blendshapes | Compressed skinning |
|---|---|---|
| Stored per shape | one offset per vertex | one small transform per joint, mostly zeros |
| Per-frame cost | proportional to vertices × active shapes | a sparse sum, then standard GPU skinning |
| Memory (human head, the paper's 40-joint setup) | dense | 81 to 87 KB, 5 to 7× less than dense skinning |
| Accuracy | exact | mean error < 0.05 mm, worst case a few mm (paper, Table 1) |
| Runs on | anything that evaluates blendshapes | anything with a skinning shader that accepts full affine matrices |

```{toctree}
:hidden:
:maxdepth: 2
:caption: Getting started

getting_started/installation
getting_started/quickstart
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Concepts

concepts/overview
concepts/blendshapes_to_skinning
concepts/how_the_solver_works
concepts/data_formats
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: User guide

user_guide/preparing_data
user_guide/compressing
user_guide/evaluating_results
user_guide/pipeline_integration
user_guide/maya_rig_workflow
user_guide/troubleshooting
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Reference

api/index
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Developer

developer/architecture
developer/development
```
