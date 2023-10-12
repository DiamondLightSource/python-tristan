import pytest

from tristan.diagnostics.utils import define_modules, module_cooordinates


def test_define_modules_for_1M():
    m = define_modules("1M")

    assert len(m) == 1
    assert m["0"] == ([0, 515], [0, 2069])


def test_module_coord():
    mc = module_cooordinates("2M")

    assert len(mc) == 2
    assert "1" in list(mc.keys())
    assert mc["0"] == (0, 0)
    assert mc["1"] == (1, 0)


def test_define_modules_raises_error_if_unknown_config():
    with pytest.raises(ValueError):
        define_modules("3M")


def test_module_coordinates_raises_error_if_unknown_config():
    with pytest.raises(ValueError):
        module_cooordinates("4M")
