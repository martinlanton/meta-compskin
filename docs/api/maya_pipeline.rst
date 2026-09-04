Maya Pipeline
=============

Runs inside Maya or ``mayapy``. One call exports the selected mesh, compresses
it in a subprocess (with a PyTorch interpreter, on CUDA when available) and
builds the rig back in the same session. See
:doc:`../user_guide/maya_rig_workflow`, section 4.2.

.. automodule:: metacompskin.maya_pipeline
   :members:
   :undoc-members:
   :show-inheritance:
   :private-members: False
