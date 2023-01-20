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

To bin all the events into a single image, for powder processing or similar, use the `images single`` command, alias `images 1`.

This accepts as input either the <file-name-stem>.nxs file, the <file-name-stem>_meta.h5 file or just the collection directory,
if only a single data set has been saved there.

.. code-block:: console

    images single /path/to/file


Multiple image tool
^^^^^^^^^^^^^^^^^^^

To bin the events into a chronological image sequence, for example a rotation scan, use `images multi`.

As input, this command also accepts the <file-name-stem>.nxs file, the <file-name-stem>_meta.h5 file or the collection parent directory if unique.
Additionally, it is also necessary to specify either the number of images, with `-n`, or the exposure time, with `-e`, to know how many images the events should be binned into.

.. code-block:: console

    images multi /path/to/file -n 1750


Alternatively,

.. code-block:: console

    images multi /path/to/file -e .1ms


.. note::

    `-e` accepts most human-readable specifications of units, eg. `-e 100us`, `-e 100µs`, `-e .1ms`, etc...


Another availabe option for this tool is the `-a` flag, alias `--align-trigger`, which aligns the image start time with the first specified trigger signal.
This is useful for examining changes in the sample after a trigger signal.


.. code-block:: console

    images multi -n 400 -a TTL-rising /path/to/file


Static sequence pump-probe tool
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The tool `images pump-probe`, alias `images pp`, aggregates all the events from a pump-probe measurement, divides the pump rep period into bins of equal 'width' in time and creates an image for each bin.
The resulting sequence of images describes the evolution of the system following a pump pulse, averaged over all pump pulses.

Similarily to `images multi`, this tool requires the trigger type to be specified with `-t`, and the bin 'width' with `-e`` or the number of bins with `-n`.

.. code-block:: console

    images pp -n 20 -t TTL-rising /path/to/file


For example, this tool could be used to create a 'waterfall plot' of the intensity of a single reflection from a static sample, as it evolves in response to pump pulses.


Multiple sequence pump-probe tool
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To bin events into images representing different pump-probe delays, use `images sequences`, alias `images sweeps`. This tool first divides the pump rep period into bins of equal
duration and then creates a sweep of images for each bin, using only the events that fall into that bin. The result is a sequence, or sweep, of images for each pump-probe delay bin.

In the same manner as `images multi`, it is required to set either the exposure time of the images with `-e`, or the number of images per sweep with `-n`.
As for the triggers, the trigger signal is specified with `-t`, as in `images pp`. It is also necessary to provide the pump-probe delay intervals either
by duration, with `-i`, or by number, with `-x`.

.. code-block:: console

    images sequences -x 20 -n 180 -t TTL-rising /path/to/file


For example, this could be used to deconstruct a rotation data collection into several rotation datasets, each corresponding to a different pump-probe delay window.


Apply the flatfield correction
==============================

A tool to apply the flat-field correction to the binned images if needed. It is possible to choose whether to multiply or divide the images by the
flat-field.


.. code-block:: console

    apply-flat-field /path/to/binned_img_file /path/to/flatfield_file {multiply, divide}
