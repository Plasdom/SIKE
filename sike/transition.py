from scipy import interpolate
import numpy as np
from numpy.typing import ArrayLike
from numba import jit

import atomic_state
import physics_tools


class Transition:
    """Base transition class"""

    def __init__(
        self, type: str, element: str, from_id: int, to_id: int, delta_E: float
    ):
        # TODO: delta_E was previously passed un-normalised, then divided by T_norm. Now T_norm is not provided so will need to make sure this step happens when the transitions are initialised
        """Initialise

        :param type: Type of transition (e.g. "excitation"))
        :type type: str
        :param element: Atomic element
        :type element: str
        :param from_id: ID of the initial state
        :type from_id: int
        :param to_id: ID of the final state
        :type to_id: int
        :param delta_E: Transition energy
        :type delta_E: float
        """
        self.type = type
        self.element = element
        self.from_id = from_id
        self.to_id = to_id
        self.delta_E = delta_E


class ExTrans(Transition):
    """Excitation transition"""

    def __init__(
        self,
        sigma: ArrayLike,
        collrate_const: float,
        sigma_norm: float,
        from_stat_weight: float | None = None,
        born_bethe_coeffs: ArrayLike | None = None,
        **transition_kwargs,
    ):
        """Initialise

        :param sigma: Cross-sections
        :type sigma: ArrayLike
        :param collrate_const: Normalisation constant for collision rate calculation
        :type collrate_const: float
        :param sigma_norm: Normalisation constant for cross-section
        :type sigma_norm: float
        :param from_stat_weight: Statistical weight of the initial state, defaults to None
        :type from_stat_weight: float | None, optional
        :param born_bethe_coeffs: Born-Bethe coefficients, defaults to None
        :type born_bethe_coeffs: ArrayLike | None, optional
        :param transition_kwargs: Arguments for base Transition class
        :type: Keyword arguments
        """
        self.super().__init__(self, transition_kwargs)

        self.sigma = 1e-4 * np.array(sigma) / sigma_norm
        self.sigma[np.where(self.sigma < 0.0)] = 0.0
        self.collrate_const = collrate_const
        self.from_stat_weight = from_stat_weight
        self.born_bethe_coeffs = born_bethe_coeffs

    def set_sigma_deex(self, g_ratio: float, vgrid: ArrayLike) -> ArrayLike:
        """Calculate the de-excitation cross-section

        :param g_ratio: the ratio of statistical weights of from/to states
        :type g_ratio: float
        :param vgrid: Velocity grid
        :type vgrid: ArrayLike
        :return: De-excitation cross-sections
        :rtype: ArrayLike
        """
        vgrid_inv = np.sqrt(vgrid**2 + self.delta_E)
        sigma_interp_func = interpolate.interp1d(
            vgrid, self.sigma, fill_value=0.0, bounds_error=False, kind="linear"
        )
        sigma_interp = sigma_interp_func(vgrid_inv)
        self.sigma_deex = self.get_sigma_deex(vgrid, vgrid_inv, sigma_interp, g_ratio)

    def get_mat_value(self, fe: ArrayLike, vgrid: ArrayLike, dvc: ArrayLike) -> float:
        """Get the matrix value for this transition. For excitation transitions, this is ne * rate coefficient

        :param fe: local electron distribution
        :type fe: ArrayLike
        :param vgrid: velocity grid
        :type vgrid: ArrayLike
        :param dvc: velocity grid widths
        :type dvc: ArrayLike
        :return: Matrix value
        :rtype: float
        """
        K_ex = physics_tools.calc_rate(vgrid, dvc, fe, self.sigma, self.collrate_const)
        return K_ex

    def get_mat_value_inv(
        self, fe: ArrayLike, vgrid: ArrayLike, dvc: ArrayLike
    ) -> float:
        """Get the matrix value for the inverse of transition. For excitation transitions, this is three-body recombination

        :param fe: local electron distribution
        :type fe: ArrayLike
        :param vgrid: velocity grid
        :type vgrid: ArrayLike
        :param dvc: velocity grid widths
        :type dvc: ArrayLike
        :return: electron density multiplied by three-body recombination rate coefficient
        :rtype: float
        """
        K_deex = physics_tools.calc_rate(
            vgrid, dvc, fe, self.sigma_deex, self.collrate_const
        )
        return K_deex

    def get_sigma_deex(
        self,
        vgrid: ArrayLike,
        vgrid_inv: ArrayLike,
        sigma_interp: ArrayLike,
        g_ratio: float,
    ) -> ArrayLike:
        """Get the de-excitation cross-section, assuming detailed balance

        :param vgrid: velocity grid
        :type vgrid: ArrayLike
        :param vgrid_inv: velocity grid of post-collision electrons
        :type vgrid_inv: ArrayLike
        :param sigma_interp: excitation cross-section interpolated on to vgrid_inv
        :type sigma_interp: ArrayLike
        :param g_ratio: the ratio of statistical weights (free / bound)
        :type g_ratio: float
        :return: local de-excitation cross-section
        :rtype: ArrayLike
        """
        sigma_deex = physics_tools.get_sigma_deex(
            vgrid, vgrid_inv, sigma_interp, g_ratio
        )

        return sigma_deex


class IzTrans(Transition):
    """Ionization transition class. Derived from Transition class."""

    def __init__(
        self,
        sigma: ArrayLike,
        collrate_const: float,
        tbrec_norm: float,
        sigma_norm: float,
        from_stat_weight: float | None = None,
        fit_params: ArrayLike | None = None,
        **transition_kwargs,
    ):
        """Initialise

        :param sigma: Cross-sections
        :type sigma: ArrayLike
        :param collrate_const: Normalisation constant for collision rate calculation
        :type collrate_const: float
        :param tbrec_norm: Normalisation constant for three-body recombination rate
        :type tbrec_norm: float
        :param sigma_norm: Normalisation constant for cross-section
        :type sigma_norm: float
        :param from_stat_weight: Statistical weight of the initial state, defaults to None, defaults to None
        :type from_stat_weight: float | None, optional
        :param fit_params: Parameters for the high-energy cross-section fit, defaults to None
        :type fit_params: ArrayLike | None, optional
        :param transition_kwargs: Arguments for base Transition class
        :type: Keyword arguments
        """
        self.super().__init__(self, transition_kwargs)
        self.sigma = 1e-4 * np.array(sigma) / sigma_norm
        self.sigma[np.where(self.sigma < 0.0)] = 0.0
        self.collrate_const = collrate_const
        self.tbrec_norm = tbrec_norm
        self.from_stat_weight = from_stat_weight
        self.fit_params = fit_params

    def set_inv_data(self, g_ratio: float, vgrid: ArrayLike):
        """Store some useful data for calculating the inverse (3b-recombination) cross-sections

        :param g_ratio: statistical weight ratio of from/to states
        :type g_ratio: float
        :param vgrid: the velocity grid
        :type vgrid: ArrayLike
        """
        self.g_ratio = g_ratio
        self.vgrid_inv = np.sqrt(vgrid**2 + self.delta_E)
        sigma_interp_func = interpolate.interp1d(
            vgrid, self.sigma, fill_value=0.0, bounds_error=False, kind="linear"
        )
        self.sigma_interp = sigma_interp_func(self.vgrid_inv)

    def get_mat_value(self, fe: ArrayLike, vgrid: ArrayLike, dvc: ArrayLike) -> float:
        """Get the matrix value for this transition. For ionization transitions, this is ne * rate coefficient

        :param fe: local electron distribution
        :type fe: ArrayLike
        :param vgrid: velocity grid
        :type vgrid: ArrayLike
        :param dvc: velocity grid widths
        :type dvc: ArrayLike
        :return: Matrix value
        :rtype: float
        """
        K_ion = physics_tools.calc_rate(vgrid, dvc, fe, self.sigma, self.collrate_const)
        return K_ion

    def get_mat_value_inv(
        self, fe: ArrayLike, vgrid: ArrayLike, dvc: ArrayLike, ne: float, Te: float
    ):
        """Get the matrix value for the inverse of transition. For ionization transitions, this is three-body recombination

        :param fe: local electron distribution
        :type fe: ArrayLike
        :param vgrid: velocity grid
        :type vgrid: ArrayLike
        :param dvc: velocity grid widths
        :type dvc: ArrayLike
        :param ne: Electron density
        :type ne: float
        :param Te: Electron temperature
        :type Te: float
        :return: electron density multiplied by three-body recombination rate coefficient
        :rtype: float
        """
        sigma_tbrec = self.get_sigma_tbrec(vgrid, Te)
        K_tbrec = physics_tools.calc_rate(
            vgrid, dvc, fe, sigma_tbrec, ne * self.tbrec_norm * self.collrate_const
        )
        return K_tbrec

    def get_sigma_tbrec(self, vgrid: ArrayLike, Te: float) -> ArrayLike:
        """Get the three-body recombination cross-section, assuming detailed balance

        :param vgrid: Velocity grid
        :type vgrid: ArrayLike
        :param Te: Electron temperature
        :type Te: float
        :return: Three-body recombination cross-section
        :rtype: ArrayLike
        """
        sigma_tbrec = physics_tools.get_sigma_tbr(
            vgrid, self.vgrid_inv, self.sigma_interp, self.g_ratio, Te
        )

        return sigma_tbrec


class RRTrans(Transition):
    """Radiative recombination transition class. Derived from Transition class."""

    def __init__(
        self,
        sigma: ArrayLike,
        collrate_const: float,
        sigma_norm: float,
        from_stat_weight: float | None,
        to_stat_weight: float | None,
        l: int | None,
        fit_params: ArrayLike | None,
        **transition_kwargs,
    ):
        """Initialise

        :param sigma: Cross-sections
        :type sigma: ArrayLike
        :param collrate_const: Normalisation constant for collision rate calculation
        :type collrate_const: float
        :param sigma_norm: Normalisation constant for cross-section
        :type sigma_norm: float
        :param from_stat_weight: Statistical weight of the initial state, defaults to None
        :type from_stat_weight: float | None
        :param to_stat_weight: Statistical weight of the final state, defaults to None
        :type to_stat_weight: float | None
        :param l: Orbital angular momentum quantum number of final state (TODO: Check this)
        :type l: int | None
        :param fit_params: Parameters for high-energy cross-section fit
        :type fit_params: ArrayLike | None
        """
        self.super().__init__(self, transition_kwargs)

        self.sigma = 1e-4 * np.array(sigma) / sigma_norm
        self.collrate_const = collrate_const
        self.from_stat_weight = from_stat_weight
        self.to_stat_weight = to_stat_weight
        self.l = l
        self.fit_params = fit_params

    def get_mat_value(self, fe: ArrayLike, vgrid: ArrayLike, dvc: ArrayLike) -> float:
        """Get the matrix value for this transition.

        :param fe: local electron distribution
        :type fe: ArrayLike
        :param vgrid: velocity grid
        :type vgrid: ArrayLike
        :param dvc: velocity grid widths
        :type dvc: ArrayLike
        :return: Matrix value
        :rtype: float
        """
        K_radrec = physics_tools.calc_rate(
            vgrid, dvc, fe, self.sigma, self.collrate_const
        )
        return K_radrec


class EmTrans(Transition):
    """Spontaneous emission transition class. Derived from Transition class."""

    def __init__(
        self,
        rate: float,
        time_norm: float,
        gf: float | None = None,
        **transition_kwargs,
    ):
        """Initialise

        :param rate: Spontaneous emission rate (TODO: units?)
        :type rate: float
        :param time_norm: Time units normalisation constant
        :type time_norm: float
        :param gf: _description_ (TODO: check), defaults to None
        :type gf: float | None, optional
        :param transition_kwargs: Arguments for base Transition class
        :type: Keyword arguments
        """
        self.super().__init__(self, transition_kwargs)

        self.gf = gf
        self.rate = rate * time_norm

    def get_mat_value(self) -> float:
        """Get the matrix value for this transition. For spontaneous emission, this is the emission rate

        :return: Matrix value
        :rtype: float
        """
        A_em = self.rate
        return A_em


class AiTrans(Transition):
    """Autoionization transition class. Derived from Transition class."""

    def __init__(self, rate: float, time_norm: float, **transition_kwargs):
        """Initialise

        :param rate: Autoionization rate (TODO: units?)
        :type rate: float
        :param time_norm: Time units normalisation constant
        :type time_norm: float
        :param transition_kwargs: Arguments for base Transition class
        :type: Keyword arguments
        """
        self.super().__init__(self, transition_kwargs)
        self.rate = rate * time_norm

    def get_mat_value(self) -> float:
        """Get the matrix value for this transition. For spontaneous emission, this is the emission rate

        :return: Matrix value
        :rtype: float
        """
        A_ai = self.rate
        return A_ai


@jit(nopython=True)
def calc_rate(
    vgrid: np.ndarray,
    dvc: np.ndarray,
    fe: np.ndarray,
    sigma: np.ndarray,
    const: float = 1.0,
) -> float:
    """Efficiently compute the collisional rate for a given process

    :param vgrid: velocity grid
    :type vgrid: np.ndarray
    :param dvc: velocity grid widths
    :type dvc: np.ndarray
    :param fe: local electron velocity distribution
    :type fe: np.ndarray
    :param sigma: cross-section
    :type sigma: np.ndarray
    :param const: normalisation cross-section (defaults to 1), defaults to 1.0
    :type const: float, optional
    :return: _description_
    :rtype: float
    """
    rate = 0.0
    for i in range(len(vgrid)):
        rate += vgrid[i] ** 3 * dvc[i] * fe[i] * sigma[i]
    rate *= const * 4.0 * np.pi
    return rate


@jit(nopython=True)
def get_sigma_tbr(vgrid, vgrid_inv, sigma_interp, g_ratio, Te):
    """Calculate the three-body recombination cross-section

    Args:
        vgrid (nd.ndarray): velocity grid
        vgrid_inv (nd.ndarray): post-collision velocity grid
        sigma_interp (np.ndarray): Ionization cross-section interpolated to vgrid_inv
        g_ratio (float): ratio of statistical weights
        Te (float): local electron temperature

    Returns:
        _type_: _description_
    """
    sigma_tbrec = (
        0.5
        * g_ratio
        * (1 / (np.sqrt(Te) ** 3))
        * sigma_interp
        * ((vgrid_inv / vgrid) ** 2)
    )
    return sigma_tbrec


@jit(nopython=True)
def get_sigma_deex(vgrid, vgrid_inv, sigma_interp, g_ratio):
    """Calculate the deexcitation cross-section

    Args:
        vgrid (nd.ndarray): velocity grid
        vgrid_inv (nd.ndarray): post-collision velocity grid
        sigma_interp (np.ndarray): Excitation cross-section interpolated to vgrid_inv
        g_ratio (float): ratio of statistical weights

    Returns:
        nd.ndarray: deexcitation cross-section
    """
    sigma_deex = g_ratio * sigma_interp * ((vgrid_inv / vgrid) ** 2)
    return sigma_deex


@jit(nopython=True)
def get_associated_transitions(state_id, from_ids, to_ids):
    """Efficiently find the positions of all transitions associated with a given state ID

    Args:
        state_id (int): ID for a given state
        from_ids (np.ndarray): list of all from IDs for each transition
        to_ids (np.ndarray): list of all to IDs for each transition

    Returns:
        list: a list of the indices of all associated transitions
    """
    associated_transition_indices = []
    for i in range(len(from_ids)):
        if from_ids[i] == state_id or to_ids[i] == state_id:
            associated_transition_indices.append(i)
    return associated_transition_indices


def interpolate_adf11_data(adas_file, Te, ne, num_z):
    num_x = len(Te)
    interp_data = np.zeros([num_x, num_z - 1])
    for z in range(num_z - 1):
        adas_file_interp = interpolate.interp2d(
            adas_file.logNe, adas_file.logT, adas_file.data[z], kind="linear"
        )
        for i in range(num_x):
            log_ne = np.log10(1e-6 * ne[i])
            log_Te = np.log10(Te[i])
            interp_result = adas_file_interp(log_ne, log_Te)
            interp_data[i, z] = 1e-6 * (10 ** interp_result[0])

    return interp_data
