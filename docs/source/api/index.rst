API Reference
=============

This section provides detailed API documentation for LazyML modules.

Core Modules
------------

.. toctree::
   :maxdepth: 2

   core
   models
   cli
   optim

Module Overview
---------------

**lazyml.core**
   Data loading, profiling, preprocessing, and pipeline orchestration.

**lazyml.models**
   Model training, selection, and ensemble learning.

**lazyml.cli**
   Command-line interface built with Typer.

**lazyml.optim**
   Hyperparameter optimization with Optuna.

**lazyml.reports**
   HTML and JSON report generation.

Package Structure
-----------------

.. code-block:: text

   lazyml/
   ├── core/
   │   ├── loader.py          # Data loading
   │   ├── profiler.py        # Dataset profiling
   │   ├── preprocessor.py    # Data preprocessing
   │   ├── feature_selection.py  # Feature selection
   │   ├── imbalance_handler.py    # Class imbalance handling
   │   └── pipeline.py        # Main LazyML pipeline
   ├── models/
   │   ├── trainer.py         # Model training
   │   ├── selector.py        # Model selection
   │   └── ensemble.py        # Ensemble learning
   ├── cli/
   │   └── main.py            # CLI commands
   ├── optim/
   │   ├── tuner.py           # Optuna tuner
   │   └── search_spaces.py   # Hyperparameter spaces
   └── reports/
       └── generator.py       # Report generation
