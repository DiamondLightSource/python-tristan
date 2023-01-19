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

To bin all the events into a single image, for powder processing or similar, use the images single command, alias images 1. 
As input, this accepts either the <file-name-stem>.nxs file or, if there is only a single data set in a given directory, you can just pass the directory.

.. code-block:: console

    images single /path/to/file


Multiple image tool
^^^^^^^^^^^^^^^^^^^

To bin the events into a chronological image sequence, like a rotation scan, use images multi. 
As before, this accepts the _meta.h5 file or (if unique) its parent directory, but now we also need to specify either the number of images, with -n, or the exposure time, with -e. -e uses accepts most human-readable specifications of units, like -e 100us, -e 100µs, -e .1ms, etc..

.. code-block:: console

    images multi /path/to/file -n 1750


Alternatively,

.. code-block:: console

    images multi /path/to/file -e .1ms


Static sequence pump-probe tool
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The tool images pump-probe, alias images pp, aggregates all the events from a pump-probe measurement, divides the pump rep period into bins of equal 'width' in time and creates an image for each bin. The resulting sequence of images describes the evolution of the system following a pump pulse, averaged over all pump pulses.

This tool could, for example, be used to create a 'waterfall plot' of the intensity of a single reflection from a static sample, as it evolves in response to pump pulses.

This tool requires that you specify the trigger type with -t, and the bin 'width' with -e or the number of bins with -n, in a similar manner as for images multi. 

.. code-block:: console

    images pp -n 20 -t TTL-rising /path/to/file


Multiple sequence pump-probe tool
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The tool images sequences, alias images sweeps, divides the pump rep period into bins of equal duration, as for images pp above. It then creates a sweep of images, in the same manner as images multi, for each bin, using only the events that fall into that bin. The result is a sequence, or sweep, of images for each pump-probe delay bin. This could be used to deconstruct a rotation data collection into several rotation data image sets, each corresponding to a different pump-probe delay window.

In the same manner as images multi, you can set the exposure time of the images with -e, or the number of images per sweep with -n. As for images pp, the trigger signal is specified with -t. The pump-probe delay intervals are specified either by duration, with -i, or by number, with -x.

.. code-block:: console

    images sequences -x 20 -n 180 -t TTL-rising /path/to/file



Apply the flatfield correction
==============================

A tool to apply the flat-field correction to the binnes images. It is possible to choose whether to multiply or divide the images by the
flat-field.


.. code-block:: console

    apply-flat-field /path/to/binned_img_file /path/to/flatfield_file {multiply, divide}
