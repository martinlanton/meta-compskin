Compressed Skinning for Facial Blendshapes
===========================================

Welcome to the documentation for **Compressed Skinning for Facial Blendshapes**,
a novel method for converting facial animation blendshapes into fast linear blend
skinning representation using sparse transformations.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   api/index
   guides/index

Overview
--------

This repository implements the method described in the SIGGRAPH 2024 paper
"Compressed Skinning for Facial Blendshapes" by researchers from Meta Reality Labs.

**Key Features:**

* PyTorch-based optimization using proximal algorithms with Adam optimizer
* Sparse skinning decomposition achieving ~90% sparsity in transformations
* 5-7× memory savings compared to dense methods (Dem Bones)
* 2-3× speed improvements over dense methods
* Optimized for low-spec mobile platforms (e.g., Snapdragon 652)

**Performance Targets:**

* Mean Absolute Error (MAE): < 0.05 mm for typical models
* Maximum Error (MXE): < 10 mm for typical models
* Memory: 5-7× reduction vs. dense methods
* Speed: 2-3× improvement vs. dense methods
* Sparsity: ~90% zeros in transformation matrices

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

