Installation
============

Requirements
------------

* Python 3.10 or later
* PyTorch 1.9.0 or later (installed automatically; CUDA build recommended)
* NumPy 1.20.0 or later (installed automatically)
* SciPy 1.7.0 or later (installed automatically)

Standard Installation
---------------------

1. Clone the repository:

.. code-block:: bash

   git clone https://github.com/martinlanton/meta-compskin.git
   cd meta-compskin

2. Create a virtual environment (recommended):

.. code-block:: bash

   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

3. (Optional but recommended) Install a CUDA build of PyTorch first — the
   optimization is roughly 60× faster on GPU. Pick the command for your
   platform and CUDA version at https://pytorch.org/get-started/locally/,
   for example:

.. code-block:: bash

   pip install torch --index-url https://download.pytorch.org/whl/cu128

4. Install the package (pulls in any missing core dependencies):

.. code-block:: bash

   pip install .

Development Installation
------------------------

For development work, install in editable mode with the ``dev`` extra:

.. code-block:: bash

   pip install -e ".[dev]"

This installs additional tools for testing, linting, and documentation:

* pytest and pytest-cov for testing
* ruff for formatting and linting
* mypy for type checking
* pre-commit for git hooks
* Sphinx (with the RTD theme and MyST parser) for documentation

Set up pre-commit hooks:

.. code-block:: bash

   pre-commit install

The ``viz`` extra adds matplotlib for visualization helpers:

.. code-block:: bash

   pip install -e ".[viz]"

Installation Inside Autodesk Maya
---------------------------------

This is only needed to use :class:`~metacompskin.maya_exporter.MayaBlendshapeExporter`
for exporting model data from Maya scenes. Inside Maya the package only needs
numpy (bundled with recent Maya versions) — PyTorch is **not** required, so
install without dependencies:

.. code-block:: bash

   mayapy -m pip install --no-deps <path-to-meta-compskin>

``mayapy`` lives in Maya's ``bin`` directory, e.g.
``C:\Program Files\Autodesk\Maya2025\bin\mayapy.exe`` on Windows or
``/Applications/Autodesk/maya2025/Maya.app/Contents/bin/mayapy`` on macOS.

Alternatively, skip installation and prepend the source directory to the path
at the top of your Maya script:

.. code-block:: python

   import sys

   sys.path.insert(0, r"D:/path/to/meta-compskin/src")

Verify from a shell:

.. code-block:: bash

   mayapy -c "from metacompskin.maya_exporter import MayaBlendshapeExporter; print('OK')"

Verify Installation
-------------------

Check that the package imports and reports its version:

.. code-block:: bash

   python -c "import metacompskin; print(metacompskin.__version__)"

With a development installation, run the test suite:

.. code-block:: bash

   pytest
