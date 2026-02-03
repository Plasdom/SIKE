from contextlib import suppress
from importlib.metadata import PackageNotFoundError, version

import sike.plotting  # noqa: F401
from sike.constants import (
    ATOMIC_DATA_BASE_URL,
    ATOMIC_DATA_LOCATION,
    BOHR_RADIUS,
    BOLTZMANN_K,
    CONFIG_FILENAME,
    EL_CHARGE,
    EL_MASS,
    ELEMENT2SYMBOL,
    EPSILON_0,
    ION_MASS,
    LIGHT_SPEED,
    MARCHAND_SCREENING_COEFFS,
    NUCLEAR_CHARGE_DICT,
    PLANCK_H,
    SOLKIT_VGRID,
    SYMBOL2ELEMENT,
    TEST_DATA_LOCATION,
)
from sike.core import SIKERun, get_atomic_data_savedir, get_test_data_dir
from sike.plasma_utils import generate_vgrid, get_bimaxwellians, get_maxwellians
from sike.setup import setup

with suppress(PackageNotFoundError):
    __version__ = version(__name__)

__all__ = [
    "ATOMIC_DATA_BASE_URL",
    "ATOMIC_DATA_LOCATION",
    "BOHR_RADIUS",
    "BOLTZMANN_K",
    "CONFIG_FILENAME",
    "ELEMENT2SYMBOL",
    "EL_CHARGE",
    "EL_MASS",
    "EPSILON_0",
    "ION_MASS",
    "LIGHT_SPEED",
    "MARCHAND_SCREENING_COEFFS",
    "NUCLEAR_CHARGE_DICT",
    "PLANCK_H",
    "SOLKIT_VGRID",
    "SYMBOL2ELEMENT",
    "TEST_DATA_LOCATION",
    "SIKERun",
    "generate_vgrid",
    "get_atomic_data_savedir",
    "get_bimaxwellians",
    "get_maxwellians",
    "get_test_data_dir",
    "setup",
]
