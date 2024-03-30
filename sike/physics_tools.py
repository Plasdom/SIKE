import numpy as np
from numba import jit
from scipy import interpolate
from constants import *


@jit(nopython=True)
def lambda_ei(n, T, T_0, n_0, Z_0):
    """e-i Coulomb logarithm

    Args:
        n (float): density
        T (float): temperature
        T_0 (float): temperature normalisation
        n_0 (float): density normalisation
        Z_0 (float): Ion charge

    Returns:
        float: lambda_ei
    """
    if T * T_0 < 10.00 * Z_0**2:
        return 23.00 - np.log(
            np.sqrt(n * n_0 * 1.00e-6) * Z_0 * (T * T_0) ** (-3.00 / 2.00)
        )
    else:
        return 24.00 - np.log(np.sqrt(n * n_0 * 1.00e-6) / (T * T_0))


@jit(nopython=True)
def maxwellian(T, n, vgrid):
    """Return a normalised (to n_0 / v_th,0 ** 3) Maxwellian electron distribution (isotropic, as function of velocity magnitude).

    Args:
        T (float): Normalised electron temperature
        n (float): Normalised electron density
        vgrid (np.array, optional): Normalised velocity grid on which to define Maxwellian distribution. If None, create using vgrid = np.arange(0.00001, 10, 1. / 1000.)

    Returns:
        np.array(num_v): numpy array of Maxwellian
    """

    f = [0.0 for i in range(len(vgrid))]
    for i, v in enumerate(vgrid):
        f[i] = n * (np.pi * T) ** (-3 / 2) * np.exp(-(v**2) / T)
    f = np.array(f)

    return f


@jit(nopython=True)
def bimaxwellian(T1, n1, T2, n2, vgrid):
    """Return a normalised (to n_0 / v_th,0 ** 3) Maxwellian electron distribution (isotropic, as function of velocity magnitude).

    Args:
        T1 (float): First population electron temperature
        n1 (float): First population electron density
        T2 (float): Second population electron temperature
        n2 (float): Second population electron density
        vgrid (np.array, optional): Velocity grid on which to define Maxwellian distribution

    Returns:
        np.array(num_v): numpy array of Maxwellian
    """

    f = [0.0 for i in range(len(vgrid))]
    for i, v in enumerate(vgrid):
        f[i] = (n1 * (np.pi * T1) ** (-3 / 2) * np.exp(-(v**2) / T1)) + (
            n2 * (np.pi * T2) ** (-3 / 2) * np.exp(-(v**2) / T2)
        )
    f = np.array(f)

    return f


def boltzmann_dist(Te, energies, stat_weights, gnormalise=False):
    """Generate a boltzmann distribution for the given set of energies and statistical weights

    Args:
        Te (np.ndarray): Electron temperature array [eV]
        energies (np.ndarray): Atomic state energies [eV]
        stat_weights (np.ndarray): Atomic state staistical weights
        gnormalise (bool, optional): Option to normalise output densities by their statistical weights. Defaults to False.

    Returns:
        np.ndarray: Boltzmann-distributed densities, relative to ground state
    """
    rel_dens = np.zeros(len(energies))
    for i in range(len(energies)):
        rel_dens[i] = (stat_weights[i] / stat_weights[0]) * np.exp(
            -(energies[i] - energies[0]) / Te
        )
        if gnormalise:
            rel_dens[i] /= stat_weights[i]
    return rel_dens


def saha_dist(Te, ne, imp_dens_tot, impurity):
    """Generate a Saha distribution of ionization stage densities for the given electron temperature

    Args:
        Te (_type_): _description_
        ne (_type_): _description_
        imp_dens_tot (_type_): _description_
        r (_type_): _description_
        el (_type_): _description_
    """
    el_mass = 9.10938e-31
    el_charge = 1.602189e-19
    planck_h = 6.62607004e-34

    ground_states = [s for s in impurity.states if s.ground is True]
    ground_states = list(reversed(sorted(ground_states, key=lambda x: x.num_el)))

    de_broglie_l = np.sqrt((planck_h**2) / (2 * np.pi * el_mass * el_charge * Te))

    # Compute ratios
    dens_ratios = np.zeros(impurity.num_Z - 1)
    for z in range(1, impurity.num_Z):
        eps = -(ground_states[z - 1].energy - ground_states[z].energy)
        stat_weight_zm1 = ground_states[z - 1].stat_weight
        stat_weight = ground_states[z].stat_weight

        dens_ratios[z - 1] = (
            2 * (stat_weight / stat_weight_zm1) * np.exp(-eps / Te)
        ) / (ne * (de_broglie_l**3))

    # Fill densities
    denom_sum = 1.0 + np.sum(
        [np.prod(dens_ratios[: z + 1]) for z in range(impurity.num_Z - 1)]
    )
    dens_saha = np.zeros(impurity.num_Z)
    dens_saha[0] = imp_dens_tot / denom_sum
    for z in range(1, impurity.num_Z):
        dens_saha[z] = dens_saha[z - 1] * dens_ratios[z - 1]

    return dens_saha


def get_maxwellians(ne, Te, vgrid, normalised=True):
    """Return an array of Maxwellian electron distributions with the given densities and temperatures.

    Args:
        ne (np.array): Normalised electron densities
        Te (np.array): Normalised electron temperatures
        vgrid (np.array): Normalised velocity grid on which to calculate Maxwellians

    Returns:
        np.array(num_v, num_x): 2d numpy array of Maxwellians at each location in x
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


def get_bimaxwellians(n1, n2, T1, T2, vgrid, normalised=True):
    """Return an array of bi-Maxwellian electron distributions with the given densities and temperatures.

    Args:
        T1 (np.ndarray): First population electron temperatures
        n1 (np.ndarray): First population electron densities
        T2 (np.ndarray): Second population electron temperatures
        n2 (np.ndarray): Second population electron densities
        vgrid (np.array): Velocity grid on which to calculate bi-Maxwellians
        normalised (bool):

    Returns:
        np.array(num_v, num_x): 2d numpy array of Maxwellians at each location in x
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
def density_moment(f0, vgrid, dvc):
    """Calculate density moment of input electron distribution

    Args:
        f0 (np.array): Electron distribution
        vgrid (_type_): Velocity grid
        dvc (_type_): Velocity grid widths
        normalised (bool, optional): Specify if inputs and output are normalised. Defaults to False.

    Returns:
        float: density. Units are normalised or m**-3 depending on whether inputs are normalised.
    """
    n = 4 * np.pi * np.sum(f0 * vgrid**2 * dvc)
    return n


@jit(nopython=True)
def temperature_moment(f0, vgrid, dvc, normalised=True):
    """_summary_

    Args:
        f0 (_type_): _description_
        vgrid (_type_): _description_
        dvc (_type_): _description_
        normalised (bool, optional): _description_. Defaults to True.

    Returns:
        float: temperature. Units are dimensionless or eV depending on normalised argument
    """

    n = density_moment(f0, vgrid, dvc)
    if normalised:
        T = (2 / 3) * 4 * np.pi * np.sum(f0 * vgrid**4 * dvc) / n
    else:
        T = (2 / 3) * 4 * np.pi * 0.5 * EL_MASS * np.sum(f0 * vgrid**4 * dvc) / n
        T /= EL_CHARGE

    return T
