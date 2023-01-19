============
Installation
============

.. code-block:: console

    pip install tristan


=============
Image binning
=============

**Tristan** is a python package that provides a set of prototype tools for processing data collected on Tristan,
the experimental timepix3-based event-mode detector in use at Diamond Light Source.

Intead of images, this detector collects an event stream recording the pixel where the photon hit the detector, its timestamp (time of arrival) and
energy (time over threshold). The processing consists in binning these events into one or more images.


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

A tool to apply the flat-field correction to the binnes images. It is possible to choose whether to multiply or divide the images by the
flat-field.


.. code-block:: console

    apply-flat-field /path/to/binned_img_file /path/to/flatfield_file {multiply, divide}
