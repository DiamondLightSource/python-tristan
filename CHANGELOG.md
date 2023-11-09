# CHANGES

## 0.#.#
- Tidier diagnostics, possibility to only print out selected trigger information and additional warning in case of missing module files.

## 0.2.2
- Fixed the axis dimensions for `images pp`.
- Added timestamp check and warning on triggers if they happen before/after shutters in `find-tristan-triggers`.
- Added `images serial` for gated access binning of events.
- Added python3.11 support.

## 0.2.1
- Added dagnostic tool `valid-events` for checking that there are events recorded after the shutter open signal in case the binned image is blank(asynchronicity issue). Also, a couple of small improvements on the other diagnostic tools.
- Set up documentation and published a first version with basic information.

## 0.2.0
- All the `images` tools have had an overhaul and should now be much more robust, even when binning large numbers of events to large numbers of images.

## 0.1.17
- You can now launch any of the `images` commands with the `<file-name>.nxs` file as valid input, as an alternative to the `<file-name>_meta.h5` file.

## 0.1.16
- Added diagnostic tools `check-tristan-files` for checking that all files from all detector modules contain valid data, and `find-trigger-intervals` for discovering the intervals between related cue messages.
