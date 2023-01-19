=====
Usage
=====

**Tristan** is a python package that provides a set of prototype tools for processing data collected on Tristan, 
the experimental timepix3-based event-mode detector in use at Diamond Light Source.

Installation
============

.. code-block:: console

    pip install tristan


Bin events to images
====================

Aggregate the events from a LATRD Tristan data collection into one or more images.


Single image tool
^^^^^^^^^^^^^^^^^

.. code-block:: console

    images single /path/to/file


Multiple image tool
^^^^^^^^^^^^^^^^^^^

.. code-block:: console

    images multi /path/to/file -n 1750


Alternatively, 

.. code-block:: console

    images multi /path/to/file -e .1ms


Static sequence pump-probe tool
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: console

    images pp -n 20 -t TTL-rising /path/to/file


Multiple sequence pump-probe tool
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: console

    images sequences -x 20 -n 180 -t TTL-rising /path/to/file



Apply the flatfield correction
==============================



Getting help
============

Every command in the tristan package has a help message that explains its usage and shows a list of accepted 
positional and optional arguments. 
The help message is printed by passing the option --help, alias -h, to any of the commands.

.. code-block:: console

    images --help


.. code-block:: console

    images multi -h