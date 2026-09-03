# Documentation

The documentation has moved to the [`docs/`](docs/index.md) folder, where it is
organised by audience and built with Sphinx.

| I want to... | Read |
|--------------|------|
| understand what the tool produces, without code | [Overview](docs/concepts/overview.md) |
| install it | [Installation](docs/getting_started/installation.md) |
| run it once end to end | [Quick start](docs/getting_started/quickstart.md) |
| get blendshapes out of Maya or another DCC | [Preparing data](docs/user_guide/preparing_data.md) |
| tune the compression, use custom joints | [Compressing](docs/user_guide/compressing.md) |
| know whether the result is good | [Evaluating results](docs/user_guide/evaluating_results.md) |
| use the output in a Maya rig | [Maya rig workflow](docs/user_guide/maya_rig_workflow.md) |
| use the output in an engine or runtime | [Pipeline integration](docs/user_guide/pipeline_integration.md) |
| know every file key and convention | [Data formats](docs/concepts/data_formats.md) |
| understand the maths | [From blendshapes to skinning](docs/concepts/blendshapes_to_skinning.md), [How the solver works](docs/concepts/how_the_solver_works.md) |
| fix an error | [Troubleshooting](docs/user_guide/troubleshooting.md) |
| contribute | [Architecture](docs/developer/architecture.md), [Development](docs/developer/development.md) |
| look up a class or function | [API reference](docs/api/index.rst) (rendered in the built site) |

The companion `meta-compskin_private_tests` repository holds production-scale
Maya fixtures, the Maya integration tests, and two scripts:
`generate_maya_compressed_data.py` (OBJ files plus joint matrices to compressed
output) and `build_maya_rig.py` (builds a verification rig in Maya from the
compressed output).
