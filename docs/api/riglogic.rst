Rig Logic
=========

Evaluation of the sample rigs' control logic: maps animator controls to
blendshape coefficients, resolving in-between and corrective shapes. This
module is only used by :class:`~metacompskin.animation_generator.AnimationFrameGenerator`
when the model file carries ``inbetween_info`` and ``combination_info``
metadata (the sample models do; data exported from your own rig normally does
not, see :doc:`../user_guide/evaluating_results`).

.. automodule:: metacompskin.rig.riglogic
   :members: compute_rig_logic
   :show-inheritance:
   :private-members: False
