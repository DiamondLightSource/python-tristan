#!/usr/bin/env python
from __future__ import division, print_function

import os
import sys
import xml.etree.ElementTree as ET

import h5py

# import time
import numpy
from h5py import AttributeManager


def xml_reader(xml_file):
    # Parse xml file
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # What I need lives in extendedCollectRequest
    main = root.findall("extendedCollectRequest")[0]
    osc = main[0].findall("oscillation_sequence")[0]

    # Find scan range
    osc_range = float(osc.findall("range")[0].text)
    if osc_range == 0.0:
        scan_range = float(osc.findall("start")[0].text)
    else:
        start = float(osc.findall("start")[0].text)
        num = float(osc.findall("number_of_images")[0].text)
        stop = start + (osc_range * num)
        scan_range = (start, stop)
        # scan_range = numpy.arange(start, stop, osc_range)

    # Determine whether it's an omega or a phi scan and the can range
    if main.find("axisChoice").text == "Phi":
        scan_axis = "Phi"
        omega = float(main.findall("otherAxis")[0].text)
        phi = scan_range
    else:
        scan_axis = "Omega"
        phi = float(main.findall("otherAxis")[0].text)
        omega = scan_range

    # Save all relevant information in a dictionary to fill nexus file
    experiment_info = {
        "detector_distance": [
            float(main.findall("sampleDetectorDistanceInMM")[0].text),
            "mm",
        ],
        "transmission": float(main.findall("transmissionInPerCent")[0].text),
        "visitPath": main.findall("visitPath")[0].text,
        "scan_axis": scan_axis,
        "kappa": float(main.findall("kappa")[0].text),
        "omega": omega,
        "two_theta": float(main.findall("twoTheta")[0].text),
        "phi": phi,
    }
    return experiment_info


def get_detector_params():
    beam = {
        "incident_wavelength": [0.4859, "angstrom"],
        "total_flux": [0.0, "Hz"],
    }
    source = {
        "name": ["Diamond Light Source", "DLS"],
        "type": "Synchrotron X-ray Source",
    }
    module = {
        # offset, transformation, units, vector
        "fast_pixel_direction": [[0, 0, 0], "translation", "m", [-1, 0, 0]],
        "slow_pixel_direction": [[0, 0, 0], "translation", "m", [0, 1, 0]],
    }
    detector_params = {
        "beam": beam,
        "source": source,
        "beam_center_xy": [1890.389, 275.653],
        "xy_pixel_size": [5.5e-05, 5.5e-05],
        "sensor_material": "Si",
        "sensor_thickness": [0.00045, "m"],
        "data_size": [2069, 1147],
        "module": module,
    }
    return detector_params


def get_axes_geometry():
    det_z = {
        "name": "det_z",
        "depends_on": "two_theta",
        "type": "translation",
        "units": "mm",
        "vector": [0, 0, -1],
    }
    kappa = {
        "name": "kappa",
        "depends_on": "omega",
        "type": "rotation",
        "units": "deg",
        "vector": [-0.76641398, -0.64234696, 0],
    }
    omega = {
        "name": "omega",
        "depends_on": ".",
        "type": "rotation",
        "units": "deg",
        "vector": [-1, 0, 0],
    }
    phi = {
        "name": "phi",
        "depends_on": "kappa",
        "type": "rotation",
        "units": "deg",
        "vector": [-1, 0, 0],
    }
    twotheta = {
        "name": "two_theta",
        "depends_on": ".",
        "type": "rotation",
        "units": "deg",
        "vector": [-1, 0, 0],
    }

    axes_geometry = {
        "geometry": "mcstas",
        "sample_depends_on": "phi",
        "detector_z": det_z,
        "kappa": kappa,
        "omega": omega,
        "phi": phi,
        "two_theta": twotheta,
    }
    return axes_geometry


class NexusWriter(object):
    """
    Class to write NeXus file with experiment metadata
    """

    def __init__(self, vds_file, xml_file, exposure_time, msg=None):
        self._vds = vds_file
        self._nxs = h5py.File(vds_file.split("_vds")[0] + ".nxs", "x")
        self._xml = xml_file
        self._time = exposure_time
        self._msg = msg

    def _get_attributes(self, obj, names, values):
        for n, v in zip(names, values):
            if type(v) is str:
                v = numpy.string_(v)
            AttributeManager.create(obj, name=n, data=v)

    def find_depends_on(self, d_info, path=None):
        _d = d_info
        if _d == ".":
            return numpy.string_(_d)
        else:
            _s = path + _d
            return numpy.string_(_s)

    def write_NXdata(self, nxentry, experiment_info, axes_geometry):
        # Get scan axis
        _scan = experiment_info["scan_axis"].lower()
        nxdata = nxentry.create_group("data")
        self._get_attributes(
            nxdata, ("NX_class", "axes", "signal"), ("NXdata", _scan, "data")
        )
        nxdata["data"] = h5py.ExternalLink(self._vds, "/")
        # Add links to all raw data files in directory
        wdir = os.path.dirname(self._vds)
        for filename in os.listdir(wdir):
            name, ext = os.path.splitext(filename)
            if ext == ".h5":
                if "vds" in filename or "meta" in filename:
                    continue
                try:
                    _link = os.path.join(wdir, filename)
                    nxdata[name] = h5py.ExternalLink(_link, "/")
                except OSError:
                    continue
        ax = nxdata.create_dataset(_scan, data=experiment_info[_scan])
        self._get_attributes(
            ax,
            ("depends_on", "transformation_type", "units", "vector"),
            (
                self.find_depends_on(
                    axes_geometry[_scan]["depends_on"],
                    path="/entry/sample/transformations/",
                ),
                axes_geometry[_scan]["type"],
                axes_geometry[_scan]["units"],
                axes_geometry[_scan]["vector"],
            ),
        )

    def write_NXdetector(self, nxinstr, experiment_info, detector_params):
        nxdet = nxinstr.create_group("detector")
        self._get_attributes(nxdet, ("NX_class",), ("NXdetector",))
        nxdet.create_dataset(
            "depends_on", data="/entry/instrument/transformations/det_z"
        )

        self.write_NXdetector_module(nxdet, detector_params)

        beam_center_x = nxdet.create_dataset(
            "beam_center_x", data=detector_params["beam_center_xy"][0]
        )
        self._get_attributes(beam_center_x, ("units",), ("pixels",))
        beam_center_y = nxdet.create_dataset(
            "beam_center_y", data=detector_params["beam_center_xy"][1]
        )
        self._get_attributes(beam_center_y, ("units",), ("pixels",))

        nxdet.create_dataset("count_time", data=self._time)
        nxdet.create_dataset("description", data=numpy.string_("Timepix"))

        # dist = nxdet.create_dataset(
        #    "detector_distance", data=experiment_info["detector_distance"][0] / 1000
        # )
        # self._get_attributes(dist, ("units",), ("m",))

        # nxdet.create_dataset(
        #    "saturation_value", data=detector_params["saturation_value"]
        # )
        nxdet.create_dataset(
            "sensor_material", data=numpy.string_(detector_params["sensor_material"])
        )
        thick = nxdet.create_dataset(
            "sensor_thickness", data=detector_params["sensor_thickness"][0]
        )
        self._get_attributes(
            thick, ("units",), (detector_params["sensor_thickness"][1],)
        )

        nxdet.create_dataset("type", data=numpy.string_("Pixel"))

        x_pix_size = nxdet.create_dataset(
            "x_pixel_size", data=detector_params["xy_pixel_size"][0]
        )
        self._get_attributes(x_pix_size, ("units",), ("m",))
        y_pix_size = nxdet.create_dataset(
            "y_pixel_size", data=detector_params["xy_pixel_size"][1]
        )
        self._get_attributes(y_pix_size, ("units",), ("m",))

    def write_NXdetector_module(self, nxdet, detector_params):
        nxmod = nxdet.create_group("module")
        self._get_attributes(nxmod, ("NX_class",), ("NXdetector_module",))
        nxmod.create_dataset("data_origin", data=numpy.array([0, 0]))
        nxmod.create_dataset("data_size", data=detector_params["data_size"])
        nxmod.create_dataset("data_stride", data=numpy.array([1, 1]))

        fast_pixel = nxmod.create_dataset(
            "fast_pixel_direction", data=detector_params["xy_pixel_size"][0]
        )
        # offset, transformation, units, vector
        self._get_attributes(
            fast_pixel,
            ("depends_on", "offset", "transformation_type", "units", "vector"),
            (
                "/entry/instrument/detector/module/module_offset",
                detector_params["module"]["fast_pixel_direction"][0],
                detector_params["module"]["fast_pixel_direction"][1],
                detector_params["module"]["fast_pixel_direction"][2],
                detector_params["module"]["fast_pixel_direction"][3],
            ),
        )

        slow_pixel = nxmod.create_dataset(
            "slow_pixel_direction", data=detector_params["xy_pixel_size"][0]
        )
        self._get_attributes(
            slow_pixel,
            ("depends_on", "offset", "transformation_type", "units", "vector"),
            (
                "/entry/instrument/detector/module/module_offset",
                detector_params["module"]["slow_pixel_direction"][0],
                detector_params["module"]["slow_pixel_direction"][1],
                detector_params["module"]["slow_pixel_direction"][2],
                detector_params["module"]["slow_pixel_direction"][3],
            ),
        )

        scaled_center_x = (
            detector_params["beam_center_xy"][0] * detector_params["xy_pixel_size"][0]
        )
        scaled_center_y = (
            detector_params["beam_center_xy"][1] * detector_params["xy_pixel_size"][1]
        )
        det_origin = scaled_center_x * numpy.array(
            detector_params["module"]["fast_pixel_direction"][3]
        ) + scaled_center_y * numpy.array(
            detector_params["module"]["slow_pixel_direction"][3]
        )
        det_origin = -det_origin

        module_offset = nxmod.create_dataset("module_offset", data=([0.0]))
        self._get_attributes(
            module_offset,
            ("depends_on", "offset", "transformation_type", "units", "vector"),
            (
                "/entry/instrument/transformations/det_z",
                det_origin,
                "translation",
                "m",
                [1, 0, 0],
            ),
        )

    def write_detZ(self, nxinstr, nxtransf, experiment_info, axes_geometry):
        nxdet_z = nxinstr.create_group("detector_z")
        self._get_attributes(nxdet_z, ("NX_class",), ("NXpositioner",))
        det_z = nxdet_z.create_dataset(
            "det_z", data=experiment_info["detector_distance"][0]
        )
        self._get_attributes(
            det_z,
            ("depends_on", "transformation_type", "units", "vector"),
            (
                self.find_depends_on(
                    axes_geometry["detector_z"]["depends_on"],
                    path="/entry/sample/transformations/",
                ),
                axes_geometry["detector_z"]["type"],
                axes_geometry["detector_z"]["units"],
                axes_geometry["detector_z"]["vector"],
            ),
        )

        # Hardlink in /entry/instrument/transformations
        _link = "/entry/instrument/detector_z/det_z"
        nxtransf["det_z"] = self._nxs[_link]

    def write_NXpositioner(
        self, nxinstr, experiment_info, axes_geometry, detector_params
    ):
        nxtransf = nxinstr.create_group("transformations")
        self._get_attributes(nxtransf, ("NX_class",), ("NXtransformations",))

        # Detector_z
        self.write_detZ(nxinstr, nxtransf, experiment_info, axes_geometry)

        # Two_theta
        twotheta = nxtransf.create_dataset(
            "two_theta", data=experiment_info["two_theta"]
        )
        self._get_attributes(
            twotheta,
            ("depends_on", "transformation_type", "units", "vector"),
            (
                self.find_depends_on(
                    axes_geometry["two_theta"]["depends_on"],
                    path="/entry/sample/transformations/",
                ),
                axes_geometry["two_theta"]["type"],
                axes_geometry["two_theta"]["units"],
                axes_geometry["two_theta"]["vector"],
            ),
        )

        # Hardlink
        nx2theta = nxinstr.create_group("twotheta")
        self._get_attributes(nx2theta, ("NX_class",), ("NXpositioner",))
        nx2theta["twotheta"] = self._nxs["/entry/instrument/transformations/two_theta"]

    def write_NXinstrument(
        self, nxentry, experiment_info, axes_geometry, detector_params
    ):
        nxinstr = nxentry.create_group("instrument")
        self._get_attributes(
            nxinstr, ("NX_class", "short_name"), ("NXinstrument", "I19-2")
        )

        # NXattenuator
        nxatt = nxinstr.create_group("attenuator")
        self._get_attributes(nxatt, ("NX_class",), ("NXattenuator",))
        nxatt.create_dataset(
            "attenuator_transmission", data=experiment_info["transmission"]
        )

        # NXbeam
        nxbeam = nxinstr.create_group("beam")
        self._get_attributes(nxbeam, ("NX_class",), ("NXbeam",))
        wl = nxbeam.create_dataset(
            "incident_wavelength",
            data=detector_params["beam"]["incident_wavelength"][0],
        )
        self._get_attributes(
            wl, ("units",), (detector_params["beam"]["incident_wavelength"][1],)
        )
        flux = nxbeam.create_dataset("total_flux", data=0.0)
        self._get_attributes(flux, ("units",), ("Hz",))

        # NXdetector
        self.write_NXdetector(nxinstr, experiment_info, detector_params)

        # NXsource
        nxsource = nxinstr.create_group("source")
        self._get_attributes(nxsource, ("NX_class",), ("NXsource",))
        nxsource.create_dataset("name", data="Diamond Light Source")
        self._get_attributes(nxsource["name"], ("short_name",), ("DLS",))
        nxsource.create_dataset("type", data="Synchrotron X-ray Source")

        # NXpositioner
        self.write_NXpositioner(
            nxinstr, experiment_info, axes_geometry, detector_params
        )

    def write_NXsample(self, nxentry, experiment_info, axes_geometry):
        nxsample = nxentry.create_group("sample")
        self._get_attributes(nxsample, ("NX_class",), ("NXsample",))
        nxsample.create_dataset(
            "depends_on",
            data=self.find_depends_on(
                axes_geometry["sample_depends_on"],
                path="/entry/sample/transformations/",
            ),
        )
        nxsample["beam"] = self._nxs["/entry/instrument/beam"]

        nxkappa = nxsample.create_group("sample_kappa")
        self._get_attributes(nxkappa, ("NX_class",), ("NXpositioner",))
        kappa = nxkappa.create_dataset("kappa", data=experiment_info["kappa"])
        self._get_attributes(
            kappa,
            ("depends_on", "transformation_type", "units", "vector"),
            (
                self.find_depends_on(
                    axes_geometry["kappa"]["depends_on"],
                    path="/entry/sample/transformations/",
                ),
                axes_geometry["kappa"]["type"],
                axes_geometry["kappa"]["units"],
                axes_geometry["kappa"]["vector"],
            ),
        )

        # Scan_axis and other_axis; scan_axis is just a link to NXdata
        _scan = experiment_info["scan_axis"].lower()
        if _scan == "omega":
            _other = "phi"
        else:
            _other = "omega"

        nxscan = nxsample.create_group("sample_" + _scan)
        self._get_attributes(nxscan, ("NX_class",), ("NXpositioner",))
        nxscan[_scan] = self._nxs["/entry/data/" + _scan]

        nxother = nxsample.create_group("sample_" + _other)
        self._get_attributes(nxother, ("NX_class",), ("NXpositioner",))
        other_ax = nxother.create_dataset(_other, data=experiment_info[_other])
        self._get_attributes(
            other_ax,
            ("depends_on", "transformation_type", "units", "vector"),
            (
                self.find_depends_on(
                    axes_geometry[_other]["depends_on"],
                    path="/entry/sample/transformations/",
                ),
                axes_geometry[_other]["type"],
                axes_geometry[_other]["units"],
                axes_geometry[_other]["vector"],
            ),
        )

        nxtr = nxsample.create_group("transformations")
        self._get_attributes(nxtr, ("NX_class",), ("NXtransformations",))
        nxtr["kappa"] = self._nxs["/entry/sample/sample_kappa/kappa"]
        nxtr["omega"] = self._nxs["/entry/sample/sample_omega/omega"]
        nxtr["phi"] = self._nxs["/entry/sample/sample_phi/phi"]
        nxtr["two_theta"] = self._nxs["entry/instrument/transformations/two_theta"]
        nxtr["det_z"] = self._nxs["entry/instrument/transformations/det_z"]

    def write_NXnote(self, nxentry):
        nxnote = nxentry.create_group("Comments")
        self._get_attributes(nxnote, ("NX_class",), ("NXnote",))
        nxnote.create_dataset("Note", data=numpy.string_(self._msg))

    def write(self):
        # Get information to fill the tree
        experiment_info = xml_reader(self._xml)
        axes_geometry = get_axes_geometry()
        detector_params = get_detector_params()

        # Start writing the NeXus tree
        nxentry = self._nxs.create_group("entry")
        self._get_attributes(nxentry, ("NX_class",), ("NXentry",))

        # Definition: /entry/definition
        nxentry.create_dataset("definition", data=numpy.string_("NXmx"))

        # Data: /entry/data
        self.write_NXdata(nxentry, experiment_info, axes_geometry)

        # Instrument: /entry/instrument
        self.write_NXinstrument(
            nxentry, experiment_info, axes_geometry, detector_params
        )

        # Sample: /entry/sample
        self.write_NXsample(nxentry, experiment_info, axes_geometry)

        # Comments section
        if self._msg is not None:
            self.write_NXnote(nxentry)
        # /entry/start_time-end_time
        # nxentry.create_dataset("start_time", data=start)
        # nxentry.create_dataset("end_time", data=end)

        self._nxs.close()


if __name__ == "__main__":
    # Input arguments order: vdsfile, xmlfile, exposure_time, comments
    # Last option is comments
    if len(sys.argv) > 4:
        NexusWriter(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]).write()
    else:
        NexusWriter(sys.argv[1], sys.argv[2], sys.argv[3], None).write()
