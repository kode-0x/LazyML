Model Modules (lazyml.models)
===============================

The ``lazyml.models`` package provides model training, selection, and ensemble learning capabilities.

ModelTrainer
------------

.. autoclass:: lazyml.models.trainer.ModelTrainer
   :members:
   :undoc-members:
   :show-inheritance:

ModelSelector
-------------

.. autoclass:: lazyml.models.selector.ModelSelector
   :members:
   :undoc-members:
   :show-inheritance:

Ensemble Learning
-----------------

.. autofunction:: lazyml.models.ensemble.build_voting_ensemble

.. autofunction:: lazyml.models.ensemble.build_stacking_ensemble

.. autofunction:: lazyml.models.ensemble.create_ensemble_from_results
