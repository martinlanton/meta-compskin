Alpha Estimation (experimental)
===============================

.. warning::

   Experimental, and for heads in centimetres only: in other units it does not
   work yet. The rule is calibrated on few heads and its constant may change; it
   is not used by default. Using it raises an ``ExperimentalWarning``.

Estimates the Laplacian smoothness weight ``alpha`` from the head's geometry
instead of looking it up by model name in :doc:`constants`. See
:doc:`../user_guide/compressing` for how to compress with the estimate.

.. automodule:: metacompskin.alpha_estimation
   :members:
   :undoc-members:
   :show-inheritance:
   :private-members: False
