API Reference
=============

Reference documentation generated from the docstrings in ``src/metacompskin``.
For task-oriented instructions see the :doc:`user guide <../user_guide/compressing>`;
for the file layouts consumed and produced by these classes see
:doc:`../concepts/data_formats`.

The package re-exports its public classes lazily, so the short import form works
everywhere, including inside Maya where PyTorch is not installed:

.. code-block:: python

   from metacompskin import (
       AnimationFrameGenerator,
       BlendshapeModelData,
       MayaBlendshapeExporter,
       MayaBlendshapeModelData,
       SkinCompressor,
       build_skinned_rig,
       compress_and_build_rig,
   )

Pipeline classes
----------------

.. toctree::
   :maxdepth: 2

   model_data
   model_fit
   animation_generator

Maya integration
----------------

.. toctree::
   :maxdepth: 2

   maya_pipeline
   maya_exporter
   maya_rig_builder
   maya_loader

Supporting modules
------------------

.. toctree::
   :maxdepth: 1

   cli
   constants
   utils
   riglogic
