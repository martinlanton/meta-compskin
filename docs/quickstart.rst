Quick Start
===========

This guide will get you started with using Compressed Skinning for Facial Blendshapes.

Basic Usage
-----------

The simplest way to use the library is with the ``SkinCompressor`` class:

.. code-block:: python

   from metacompskin.model_fit import SkinCompressor

   # Create a compressor with default settings
   compressor = SkinCompressor(iterations=10000)

   # Run the optimization
   compressor.run(output_filename="result")

This will:

1. Load the default model (Aura)
2. Run 10,000 iterations of optimization
3. Save the compressed skinning weights and transformations to ``result.npz``

Using Custom Models
-------------------

To use your own model data:

.. code-block:: python

   from metacompskin.model_fit import SkinCompressor

   # Create compressor with custom model file
   compressor = SkinCompressor(
       model_file="path/to/your/model.npz",
       iterations=10000
   )

   # Run optimization
   compressor.run(output_filename="my_result")

Your model file should be a NumPy ``.npz`` archive containing:

* ``rest_verts``: Rest pose vertices, shape ``(N, 3)``
* ``deltas``: Blendshape deltas, shape ``(S, N, 3)``
* ``rest_faces``: Triangle faces, shape ``(F, 3)``

Where:

* ``N`` = number of vertices
* ``S`` = number of blendshapes
* ``F`` = number of faces

Configuring Parameters
----------------------

The ``SkinCompressor`` accepts several parameters:

.. code-block:: python

   compressor = SkinCompressor(
       model_file="path/to/model.npz",
       iterations=10000,           # Number of optimization iterations
   )

   # Configure additional parameters
   compressor.number_of_bones = 40       # Number of proxy bones
   compressor.max_influences = 8         # Max weights per vertex
   compressor.total_nnz_Brt = 6000      # Non-zero values in transform matrix
   compressor.alpha = 10                 # Laplacian regularization weight

Understanding the Output
------------------------

The output ``.npz`` file contains:

* ``w``: Skinning weights, sparse matrix of shape ``(N, P)``
* ``Brt``: Transformation matrices, sparse matrix of shape ``(S, P, 3, 4)``

Where ``P`` is the number of bones.

These can be loaded and used for runtime skinning:

.. code-block:: python

   import numpy as np

   # Load the result
   data = np.load("result.npz")
   weights = data['w']
   transforms = data['Brt']

   # Use for runtime evaluation
   # (See User Guide for detailed runtime usage)

Next Steps
----------

* Read the :doc:`guides/index` for detailed usage
* Explore the :doc:`api/modules` for API reference
* See the paper for mathematical details

