import pytest
import numpy as np
import xarray as xr

import sike
import sike.post_processing as spp


def test_solve():
    """Test a simple hydrogen example using the solve() method"""
    nx = 3
    Te = np.linspace(1, 10, nx)
    ne = 1e20 * np.ones(nx)

    c = sike.SIKERun(Te=Te, ne=ne, element="H")
    ds = c.solve()

    assert isinstance(ds, xr.Dataset)

    Zavg = spp.get_Zavg(ds)
    assert isinstance(ds, xr.Dataset)

    expected_vals = [0.0661866, 0.99995676, 0.999989]
    assert all(np.isclose(Zavg.values, expected_vals))


def test_evolve():
    """Test a simple hydrogen example using the solve() method"""
    nx = 2
    Te = np.linspace(1, 10, nx)
    ne = 1e20 * np.ones(nx)

    for el in ["He", "C", "Ar"]:
        c = sike.SIKERun(Te=Te, ne=ne, element=el, saha_boltzmann_init=False)
        ds = c.evolve(0.0)
        Zavg_0 = spp.get_Zavg(ds)
        assert all(np.isclose(Zavg_0, 0))

        ds_evolve = c.evolve(1e3)
        ds_solve = c.solve()
        assert all(np.isclose(spp.get_Zavg(ds_evolve), spp.get_Zavg(ds_solve)))


def test_all_elements_n_resolved():
    """Ensure SIKERun object can be created for all species using n-resolved data"""
    nx = 2
    Te = np.linspace(1, 10, nx)
    ne = 1e20 * np.ones(nx)
    for el in sike.SYMBOL2ELEMENT.keys():
        c = sike.SIKERun(Te=Te, ne=ne, element=el)
