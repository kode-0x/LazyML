Quick Start
===========

This guide will get you up and running with LazyML in minutes.

Basic Usage
-----------

Train a model on your CSV dataset:

.. code-block:: bash

   uv run lazyml run data.csv --target target_column

LazyML will:

1. Profile your dataset
2. Preprocess the data (handle missing values, encoding, scaling)
3. Train multiple ML models
4. Select the best model
5. Generate an HTML report

Interactive Mode
-----------------

If you don't specify a file, LazyML enters interactive mode:

.. code-block:: bash

   uv run lazyml run

You'll be prompted for:
- Path to CSV file
- Target column name
- Task type (classification/regression)

Making Predictions
------------------

Use a trained model to make predictions on new data:

.. code-block:: bash

   uv run lazyml predict model.pkl new_data.csv --output predictions.csv

The output CSV will contain the predicted values.

Profiling a Dataset
-------------------

Analyze a dataset without training:

.. code-block:: bash

   uv run lazyml profile data.csv

This shows:
- Dataset shape and memory usage
- Column types and missing values
- Class distribution (for classification)
- Suggested target column

Output Structure
----------------

After training, the following files are created in the output directory:

.. code-block:: text

   output/
   ├── model.pkl              # Best trained model
   ├── preprocessor.pkl       # Fitted preprocessing pipeline
   ├── report.html            # Evaluation report with leaderboard
   ├── leaderboard.json       # Model rankings and scores
   └── predictions.csv        # Validation predictions

Next Steps
----------

- Learn about :doc:`CLI commands <cli_usage>`
- Explore :doc:`advanced features <features>`
- Check the :doc:`API reference <../api/index>`
