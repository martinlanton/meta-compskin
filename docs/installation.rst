Installation
============

Requirements
------------

* Python 3.11 or later
* PyTorch 2.0.0 or later
* NumPy 1.24.0 or later
* SciPy 1.10.0 or later
* libigl Python bindings

Basic Installation
------------------

1. Clone the repository:

.. code-block:: bash

   git clone https://github.com/meta/compskin.git
   cd compskin

2. Create a virtual environment (recommended):

.. code-block:: bash

   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install PyTorch (visit https://pytorch.org/get-started/locally/ for GPU support):

.. code-block:: bash

   pip install torch torchvision torchaudio

4. Install libigl:

.. code-block:: bash

   pip install libigl

5. Install the package:

.. code-block:: bash

   pip install -e .

Development Installation
------------------------

For development work, install with development dependencies:

.. code-block:: bash

   pip install -e .[dev,docs]

This installs additional tools for testing, linting, and documentation:

* pytest and pytest-cov for testing
* ruff for formatting and linting
* mypy for type checking
* pre-commit for git hooks
* Sphinx for documentation

Set up pre-commit hooks:

.. code-block:: bash

   pre-commit install

Verify Installation
-------------------

Run the tests to verify everything is working:

.. code-block:: bash

   pytest

Run a simple example:

.. code-block:: bash

   python -m metacompskin.model_fit

