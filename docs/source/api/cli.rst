CLI Module (lazyml.cli)
========================

The ``lazyml.cli`` module provides the command-line interface built with Typer.

Main CLI Application
---------------------

The CLI is built using Typer and provides three main commands:

- ``run`` - Train models on a dataset
- ``predict`` - Generate predictions using a trained model
- ``profile`` - Analyze dataset without training

Commands
--------

.. code-block:: python

   from lazyml.cli.main import app

CLI Functions
~~~~~~~~~~~~~

These functions are used internally by the CLI:

.. autofunction:: lazyml.cli.main.run

.. autofunction:: lazyml.cli.main.predict

.. autofunction:: lazyml.cli.main.profile

Usage Examples
--------------

See the :doc:`../user_guide/cli_usage` section for detailed usage examples.

Command Structure
-----------------

.. code-block:: text

   lazyml [GLOBAL OPTIONS] COMMAND [ARGS] [COMMAND OPTIONS]

Global Options:

- ``--version, -v`` - Show version and exit
- ``--help`` - Show help message
