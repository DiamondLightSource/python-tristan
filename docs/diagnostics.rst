================
Diagnostic tools
================

Trigger inspection tool
=======================

This tool runs a quick check on the trigger signals - recorded as cue messages - in a Tristan data set:

    Looks for shutter opening and closing cues and their timestamps
    Calculates the number of TTL rising edges and LVDS rising and falling edges and looks for their timestamps
    Looks for SYNC triggers and timestamps if running a serial crystallography experiment (to run:  add the -e/--expt ssx option to the command line)
    Calculates the time interval between triggers

This check is run on every module of the detector to be sure that all are correctly saving the cue messages. If one module doesn't show some or all of the triggers/timestamps, it's a sign that something might be wrong with the set-up.

Valid events check
==================

Modules inspection tool
=======================


