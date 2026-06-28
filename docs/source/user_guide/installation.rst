Installation
============

Requirements
------------

LazyML requires Python 3.9 or higher. The core dependencies are:

- pandas >= 2.0.0
- scikit-learn >= 1.3.0
- numpy >= 1.24.0
- typer >= 0.9.0
- rich >= 13.0.0

Installing with uv (Recommended)
--------------------------------

`uv <https://github.com/astral-sh/uv>`_ is a fast Python package installer and resolver.
It's the recommended way to install LazyML.

.. code-block:: bash

   git clone https://github.com/username/lazyml.git
   cd lazyml
   uv sync

Installing with pip
-------------------

You can also install LazyML using pip:

.. code-block:: bash

   git clone https://github.com/username/lazyml.git
   cd lazyml
   pip install -e .

Optional Dependencies
---------------------

LazyML supports several optional dependencies that enhance its capabilities:

Gradient Boosting Libraries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For XGBoost, LightGBM, and CatBoost support:

.. code-block:: bash

   uv sync --extra boost

SMOTE for Class Imbalance
~~~~~~~~~~~~~~~~~~~~~~~~~

For handling imbalanced datasets with SMOTE oversampling:

.. code-block:: bash

   uv sync --extra imbalanced

Hyperparameter Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For Optuna-based Bayesian optimization:

.. code-block:: bash

   uv sync --extra optuna

Terminal User Interface
~~~~~~~~~~~~~~~~~~~~~~~

For the interactive TUI dashboard:

.. code-block:: bash

   uv sync --extra tui

All Extras
~~~~~~~~~~~

To install all optional dependencies at once:

.. code-block:: bash

   uv sync --all-extras

Development Dependencies
------------------------

For development and testing:

.. code-block:: bash

   uv sync --extra dev

Verifying Installation
------------------------

Verify that LazyML is installed correctly:

.. code-block:: bash

   uv run lazyml --version

You should see output like:

.. code-block:: text

   LazyML 0.1.0
