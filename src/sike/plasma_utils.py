import numpy as np
from numba import jit

from sike.constants import *


@jit(nopython=True)
def lambda_ei(n: float, T: float, T_0: float, n_0: float, Z_0: float) -> float:
    """e-i Coulomb logarithm

    :param n: density
    :param T: temperature
    :param T_0: temperature normalisation
    :param n_0: density normalisation
    :param Z_0: ion charge
    :return: lambda_ei
    """

    if T * T_0 < 10.00 * Z_0**2:
        return 23.00 - np.log(
            np.sqrt(n * n_0 * 1.00e-6) * Z_0 * (T * T_0) ** (-3.00 / 2.00)
        )
    else:
        return 24.00 - np.log(np.sqrt(n * n_0 * 1.00e-6) / (T * T_0))


@jit(nopython=True)
def maxwellian(T: float, n: float, vgrid: np.ndarray) -> np.ndarray:
    """Return a normalised (to n_0 / v_th,0 ** 3) Maxwellian electron distribution (isotropic, as function of velocity magnitude).

    :param T: Normalised electron temperature
    :param n: Normalised electron density
    :param vgrid: Normalised velocity grid on which to define Maxwellian distribution. If None, create using vgrid = np.arange(0.00001, 10, 1. / 1000.)
    :return: numpy array of Maxwellian
    """

    f = [0.0 for i in range(len(vgrid))]
    for i, v in enumerate(vgrid):
        f[i] = n * (np.pi * T) ** (-3 / 2) * np.exp(-(v**2) / T)
    f = np.array(f)

    return f


@jit(nopython=True)
def bimaxwellian(
    T1: float, n1: float, T2: float, n2: float, vgrid: np.ndarray
) -> np.ndarray:
    """Return a normalised (to n_0 / v_th,0 ** 3) Maxwellian electron distribution (isotropic, as function of velocity magnitude).

    :param T1: First population electron temperature
    :param n1: First population electron density
    :param T2: Second population electron temperature
    :param n2: Second population electron density
    :param vgrid: Velocity grid on which to define Maxwellian distribution
    :return: numpy array of Maxwellian
    """

    f = [0.0 for i in range(len(vgrid))]
    for i, v in enumerate(vgrid):
        f[i] = (n1 * (np.pi * T1) ** (-3 / 2) * np.exp(-(v**2) / T1)) + (
            n2 * (np.pi * T2) ** (-3 / 2) * np.exp(-(v**2) / T2)
        )
    f = np.array(f)

    return f


def get_maxwellians(
    ne: np.ndarray, Te: np.ndarray, vgrid: np.ndarray, normalised: bool = True
) -> np.ndarray:
    """Return an array of Maxwellian electron distributions with the given densities and temperatures.

    :param ne: Electron densities
    :param Te: Electron temperatures
    :param vgrid: Velocity grid on which to calculate Maxwellians
    :param normalised: specify whether inputs (and therefore outputs) are normalised or not, defaults to True
    :return: 2d numpy array of Maxwellians at each location in x
    """

    if normalised is False:
        T_norm = 10
        n_norm = 1e19
        v_th = np.sqrt(2 * EL_CHARGE * T_norm / EL_MASS)
        ne /= n_norm
        Te /= T_norm
        vgrid = vgrid.copy()
        vgrid /= v_th

    f0_max = [[0.0 for i in range(len(ne))] for j in range(len(vgrid))]
    for i in range(len(ne)):
        f0_max_loc = maxwellian(Te[i], ne[i], vgrid)
        for j in range(len(vgrid)):
            f0_max[j][i] = f0_max_loc[j]
    f0_max = np.array(f0_max)

    if normalised is False:
        f0_max *= n_norm / v_th**3

    return f0_max


def get_bimaxwellians(
    n1: np.ndarray,
    n2: np.ndarray,
    T1: np.ndarray,
    T2: np.ndarray,
    vgrid: np.ndarray,
    normalised: bool = True,
) -> np.ndarray:
    """Return an array of bi-Maxwellian electron distributions with the given densities and temperatures.

    :param n1: First population electron densities
    :param n2: Second population electron densities
    :param T1: First population electron temperatures
    :param T2: Second population electron temperatures
    :param vgrid: Velocity grid on which to calculate Maxwellians
    :param normalised: specify whether inputs (and therefore outputs) are normalised or not, defaults to True
    :return: 2d numpy array of bi-Maxwellians at each location in x
    """

    if normalised is False:
        T_norm = 10
        n_norm = 1e19
        v_th = np.sqrt(2 * EL_CHARGE * T_norm / EL_MASS)
        n1 = n1.copy()
        n2 = n2.copy()
        T1 = T1.copy()
        T2 = T2.copy()
        n1 /= n_norm
        n2 /= n_norm
        T1 /= T_norm
        T2 /= T_norm
        vgrid = vgrid.copy()
        vgrid /= v_th

    f0_bimax = np.zeros([len(vgrid), len(n1)])
    for i in range(len(n1)):
        f0_bimax_loc = bimaxwellian(T1[i], n1[i], T2[i], n2[i], vgrid)
        for j in range(len(vgrid)):
            f0_bimax[j, i] = f0_bimax_loc[j]
    f0_bimax = np.array(f0_bimax)

    if normalised is False:
        f0_bimax *= n_norm / v_th**3

    return f0_bimax


@jit(nopython=True)
def density_moment(f0: np.ndarray, vgrid: np.ndarray, dvc: np.ndarray) -> float:
    """Calculate density moment of input electron distribution

    :param f0: Electron distribution
    :param vgrid: Velocity grid
    :param dvc: Velocity grid widths
    :return: Density. Units are normalised or m**-3 depending on whether inputs are normalised.
    TODO: Should be a normalised argument here?
    """
    n = 4 * np.pi * np.sum(f0 * vgrid**2 * dvc)
    return n


@jit(nopython=True)
def temperature_moment(
    f0: np.ndarray, vgrid: np.ndarray, dvc: np.ndarray, normalised: bool = True
) -> float:
    """Calculate the temperature moment of input electron distribution

    :param f0: Electron distribution
    :param vgrid: Velocity grid
    :param dvc: Velocity grid widths
    :param normalised: specify whether inputs (and therefore outputs) are normalised or not, defaults to True
    :return: temperature. Units are dimensionless or eV depending on normalised argument
    """
    n = density_moment(f0, vgrid, dvc)
    if normalised:
        T = (2 / 3) * 4 * np.pi * np.sum(f0 * vgrid**4 * dvc) / n
    else:
        T = (2 / 3) * 4 * np.pi * 0.5 * EL_MASS * np.sum(f0 * vgrid**4 * dvc) / n
        T /= EL_CHARGE

    return T
