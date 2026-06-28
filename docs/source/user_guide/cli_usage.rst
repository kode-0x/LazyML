CLI Usage
=========

LazyML provides a command-line interface (CLI) built with Typer. All commands support ``--help`` for detailed information.

Main Commands
-------------

run
~~~

Train models on a dataset.

.. code-block:: bash

   uv run lazyml run [FILE] [OPTIONS]

**Arguments:**

- ``FILE`` - Path to CSV dataset (optional, runs interactive mode if not provided)

**Options:**

.. list-table::
   :header-rows: 1
   :widths: 30 50 20

   * - Option
     - Description
     - Default
   * - ``--target, -t``
     - Target column name
     - auto-detect
   * - ``--task``
     - Task type (classification/regression)
     - auto-detect
   * - ``--time-budget``
     - Maximum training time in seconds
     - 300
   * - ``--cv-folds``
     - Number of cross-validation folds
     - 5
   * - ``--output-dir, -o``
     - Directory for output files
     - ./output
   * - ``--ensemble``
     - Enable ensemble methods
     - False
   * - ``--feature-selection``
     - Automatic feature selection
     - True
   * - ``--handle-imbalance``
     - Handle class imbalance (SMOTE)
     - True
   * - ``--parallel``
     - Train models in parallel
     - True
   * - ``--max-workers``
     - Max parallel workers
     - auto
   * - ``--optuna``
     - Enable Optuna hyperparameter tuning
     - False
   * - ``--optuna-trials``
     - Number of trials per model
     - 30

**Examples:**

.. code-block:: bash

   # Basic training
   uv run lazyml run data.csv --target price

   # With hyperparameter tuning
   uv run lazyml run data.csv --target price --optuna --optuna-trials 50

   # Enable ensemble learning
   uv run lazyml run data.csv --target price --ensemble

   # Full configuration
   uv run lazyml run data.csv \
       --target price \
       --task regression \
       --time-budget 600 \
       --cv-folds 5 \
       --ensemble \
       --optuna

predict
~~~~~~~~

Generate predictions using a trained model.

.. code-block:: bash

   uv run lazyml predict MODEL DATA [OPTIONS]

**Arguments:**

- ``MODEL`` - Path to trained model pickle file
- ``DATA`` - Path to CSV file for prediction

**Options:**

- ``--output, -o`` - Output file path (default: predictions.csv)

**Example:**

.. code-block:: bash

   uv run lazyml predict model.pkl new_data.csv --output results.csv

profile
~~~~~~~

Analyze dataset without training.

.. code-block:: bash

   uv run lazyml profile FILE

**Arguments:**

- ``FILE`` - Path to CSV file

**Example:**

.. code-block:: bash

   uv run lazyml profile data.csv

Global Options
--------------

All commands support:

- ``--version, -v`` - Show version and exit
- ``--help`` - Show help message

Examples
--------

Classification Task
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Train on classification dataset
   uv run lazyml run iris.csv --target species

   # With class imbalance handling
   uv run lazyml run imbalanced.csv --target class --handle-imbalance

Regression Task
~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Train on regression dataset
   uv run lazyml run housing.csv --target price --task regression

   # With ensemble
   uv run lazyml run housing.csv --target price --ensemble --optuna

Interactive Mode
~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Launch interactive mode
   uv run lazyml run

   # Follow the prompts to:
   # 1. Enter CSV file path
   # 2. Select target column
   # 3. Choose task type
