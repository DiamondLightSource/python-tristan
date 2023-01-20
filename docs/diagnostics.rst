================
Diagnostic tools
================

Trigger inspection tool
=======================

This tool runs a quick check on the trigger signals - recorded as cue messages - in a Tristan dataset:

   - Looks for shutter opening and closing cues and their timestamps
   - Calculates the number of TTL rising edges and LVDS rising and falling edges and looks for their timestamps
   - Looks for SYNC triggers and timestamps if running a serial crystallography experiment (to run:  add the -e/--expt ssx option to the command line)
   - Calculates the time interval between triggers

This check is run on every module of the detector to be sure that all are correctly saving the cue messages. 
If one module doesn't show some or all of the triggers/timestamps, it's a sign that something might be wrong with the setup.

A copy of the results is saved as a `.log` file in the working directory, unless otherwise specified using the `-o` option.

.. code-block:: console

    find-trigger-intervals /path/to/collection/directory filename_root -o .


Valid events check
==================

This tool checks that all modules contain at least some valid events ie. events whose timestamp falls
in the interval between the shutter open and close signals. It is useful to diagnose synchronization problems during a collection.

.. code-block:: console

    valid-events /path/to/collection/directory filename_root -o . -s 101.43 801.43

As this process has to look through the full dataset, it might take some time to run. Thus, it should only be
used when synchronization issues are suspected, for example in case the image binning returns datasets full of 0.

Modules inspection tool
=======================

This tool checks that all files from all detector modules contain valid data, and assigns each file to the correct module.
If everything is as it should be for a 10M detector, there will be 10 consecutive files listed for each module.

A copy of the results is saved as a `.log` file in the working directory, unless otherwise specified using the `-o` option.

.. code-block:: console

    check-tristan-files /path/to/collection/directory filename_root -o .


