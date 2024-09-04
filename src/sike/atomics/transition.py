from scipy import interpolate
import numpy as np
from numba import jit

from sike.utils.constants import *


class Transition:
    """Base transition class"""

    def __init__(
        self, type: str, element: str, from_id: int, to_id: int, delta_E: float, **_
    ):
        # TODO: delta_E was previously passed un-normalised, then divided by T_norm. Now T_norm is not provided so will need to make sure this step happens when the transitions are initialised
        """Initialise

        :param type: Type of transition (e.g. "excitation"))
        :param element: Atomic element
        :param from_id: ID of the initial state
        :param to_id: ID of the final state
        :param delta_E: Transition energy
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
        sigma: np.ndarray,
        collrate_const: float,
        sigma_norm: float,
        from_stat_weight: float | None = None,
        born_bethe_coeffs: np.ndarray | None = None,
        **transition_kwargs,
    ):
        """Initialise

        :param sigma: Cross-sections
        :param collrate_const: Normalisation constant for collision rate calculation
        :param sigma_norm: Normalisation constant for cross-section
        :param from_stat_weight: Statistical weight of the initial state, defaults to None
        :param born_bethe_coeffs: Born-Bethe coefficients, defaults to None
        :param transition_kwargs: Arguments for base Transition class
        """
        super().__init__(**transition_kwargs)

        self.sigma = 1e-4 * np.array(sigma) / sigma_norm
        self.sigma[np.where(self.sigma < 0.0)] = 0.0
        self.collrate_const = collrate_const
        self.from_stat_weight = from_stat_weight
        self.born_bethe_coeffs = born_bethe_coeffs

    def set_sigma_deex(self, g_ratio: float, vgrid: np.ndarray) -> np.ndarray:
        """Calculate the de-excitation cross-section

        :param g_ratio: the ratio of statistical weights of from/to states
        :param vgrid: Velocity grid
        :return: De-excitation cross-sections
        """
        vgrid_inv = np.sqrt(vgrid**2 + self.delta_E)
        sigma_interp_func = interpolate.interp1d(
            vgrid, self.sigma, fill_value=0.0, bounds_error=False, kind="linear"
        )
        sigma_interp = sigma_interp_func(vgrid_inv)
        self.sigma_deex = self.get_sigma_deex(vgrid, vgrid_inv, sigma_interp, g_ratio)

    def get_mat_value(
        self, fe: np.ndarray, vgrid: np.ndarray, dvc: np.ndarray
    ) -> float:
        """Get the matrix value for this transition. For excitation transitions, this is ne * rate coefficient

        :param fe: local electron distribution
        :param vgrid: velocity grid
        :param dvc: velocity grid widths
        :return: Matrix value
        """
        self.rate = calc_rate(vgrid, dvc, fe, self.sigma, self.collrate_const)
        return self.rate

    def get_mat_value_inv(
        self, fe: np.ndarray, vgrid: np.ndarray, dvc: np.ndarray
    ) -> float:
        """Get the matrix value for the inverse of transition. For excitation transitions, this is three-body recombination

        :param fe: local electron distribution
        :param vgrid: velocity grid
        :param dvc: velocity grid widths
        :return: electron density multiplied by three-body recombination rate coefficient
        """
        self.rate_inv = calc_rate(vgrid, dvc, fe, self.sigma_deex, self.collrate_const)
        return self.rate_inv

    def get_sigma_deex(
        self,
        vgrid: np.ndarray,
        vgrid_inv: np.ndarray,
        sigma_interp: np.ndarray,
        g_ratio: float,
    ) -> np.ndarray:
        """Get the de-excitation cross-section, assuming detailed balance

        :param vgrid: velocity grid
        :param vgrid_inv: velocity grid of post-collision electrons
        :param sigma_interp: excitation cross-section interpolated on to vgrid_inv
        :param g_ratio: the ratio of statistical weights (free / bound)
        :return: local de-excitation cross-section
        """
        sigma_deex = get_sigma_deex(vgrid, vgrid_inv, sigma_interp, g_ratio)

        return sigma_deex


class IzTrans(Transition):
    """Ionization transition class. Derived from Transition class."""

    def __init__(
        self,
        sigma: np.ndarray,
        collrate_const: float,
        tbrec_norm: float,
        sigma_norm: float,
        from_stat_weight: float | None = None,
        fit_params: np.ndarray | None = None,
        **transition_kwargs,
    ):
        """Initialise

        :param sigma: Cross-sections
        :param collrate_const: Normalisation constant for collision rate calculation
        :param tbrec_norm: Normalisation constant for three-body recombination rate
        :param sigma_norm: Normalisation constant for cross-section
        :param from_stat_weight: Statistical weight of the initial state, defaults to None, defaults to None
        :param fit_params: Parameters for the high-energy cross-section fit, defaults to None
        :param transition_kwargs: Arguments for base Transition class
        """
        super().__init__(**transition_kwargs)
        self.sigma = 1e-4 * np.array(sigma) / sigma_norm
        self.sigma[np.where(self.sigma < 0.0)] = 0.0
        self.collrate_const = collrate_const
        self.tbrec_norm = tbrec_norm
        self.from_stat_weight = from_stat_weight
        self.fit_params = fit_params

    def set_inv_data(self, g_ratio: float, vgrid: np.ndarray):
        """Store some useful data for calculating the inverse (3b-recombination) cross-sections

        :param g_ratio: statistical weight ratio of from/to states
        :param vgrid: the velocity grid
        """
        self.g_ratio = g_ratio
        self.vgrid_inv = np.sqrt(vgrid**2 + self.delta_E)
        sigma_interp_func = interpolate.interp1d(
            vgrid, self.sigma, fill_value=0.0, bounds_error=False, kind="linear"
        )
        self.sigma_interp = sigma_interp_func(self.vgrid_inv)

    def get_mat_value(
        self, fe: np.ndarray, vgrid: np.ndarray, dvc: np.ndarray
    ) -> float:
        """Get the matrix value for this transition. For ionization transitions, this is ne * rate coefficient

        :param fe: local electron distribution
        :param vgrid: velocity grid
        :param dvc: velocity grid widths
        :return: Matrix value
        """
        self.rate = calc_rate(vgrid, dvc, fe, self.sigma, self.collrate_const)
        return self.rate

    def get_mat_value_inv(
        self, fe: np.ndarray, vgrid: np.ndarray, dvc: np.ndarray, ne: float, Te: float
    ):
        """Get the matrix value for the inverse of transition. For ionization transitions, this is three-body recombination

        :param fe: local electron distribution
        :param vgrid: velocity grid
        :param dvc: velocity grid widths
        :param ne: Electron density
        :param Te: Electron temperature
        :return: electron density multiplied by three-body recombination rate coefficient
        """
        sigma_tbrec = self.get_sigma_tbrec(vgrid, Te)
        self.rate_inv = calc_rate(
            vgrid, dvc, fe, sigma_tbrec, ne * self.tbrec_norm * self.collrate_const
        )
        return self.rate_inv

    def get_sigma_tbrec(self, vgrid: np.ndarray, Te: float) -> np.ndarray:
        """Get the three-body recombination cross-section, assuming detailed balance

        :param vgrid: Velocity grid
        :param Te: Electron temperature
        :return: Three-body recombination cross-section
        """
        sigma_tbrec = get_sigma_tbr(
            vgrid, self.vgrid_inv, self.sigma_interp, self.g_ratio, Te
        )

        return sigma_tbrec


class RRTrans(Transition):
    """Radiative recombination transition class. Derived from Transition class."""

    def __init__(
        self,
        sigma: np.ndarray,
        collrate_const: float,
        sigma_norm: float,
        from_stat_weight: float | None,
        to_stat_weight: float | None,
        l: int | None,
        fit_params: np.ndarray | None,
        **transition_kwargs,
    ):
        """Initialise

        :param sigma: Cross-sections
        :param collrate_const: Normalisation constant for collision rate calculation
        :param sigma_norm: Normalisation constant for cross-section
        :param from_stat_weight: Statistical weight of the initial state, defaults to None
        :param to_stat_weight: Statistical weight of the final state, defaults to None
        :param l: Orbital angular momentum quantum number of final state (TODO: Check this)
        :param fit_params: Parameters for high-energy cross-section fit
        """
        super().__init__(**transition_kwargs)

        self.sigma = 1e-4 * np.array(sigma) / sigma_norm
        self.collrate_const = collrate_const
        self.from_stat_weight = from_stat_weight
        self.to_stat_weight = to_stat_weight
        self.l = l
        self.fit_params = fit_params

    def get_mat_value(
        self, fe: np.ndarray, vgrid: np.ndarray, dvc: np.ndarray
    ) -> float:
        """Get the matrix value for this transition.

        :param fe: local electron distribution
        :param vgrid: velocity grid
        :param dvc: velocity grid widths
        :return: Matrix value
        """
        self.rate = calc_rate(vgrid, dvc, fe, self.sigma, self.collrate_const)
        return self.rate


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
        :param time_norm: Time units normalisation constant
        :param gf: _description_ (TODO: check), defaults to None
        :param transition_kwargs: Arguments for base Transition class
        """
        super().__init__(**transition_kwargs)

        self.gf = gf
        self.rate = rate * time_norm

    def get_mat_value(self, *_) -> float:
        """Get the matrix value for this transition. For spontaneous emission, this is the emission rate

        :return: Matrix value
        """
        self.rate = self.rate
        return self.rate


class AiTrans(Transition):
    """Autoionization transition class. Derived from Transition class."""

    def __init__(self, rate: float, time_norm: float, **transition_kwargs):
        """Initialise

        :param rate: Autoionization rate (TODO: units?)
        :param time_norm: Time units normalisation constant
        :param transition_kwargs: Arguments for base Transition class
        """
        super().__init__(**transition_kwargs)
        self.rate = rate * time_norm

    def get_mat_value(self, *_) -> float:
        """Get the matrix value for this transition. For spontaneous emission, this is the emission rate

        :return: Matrix value
        """
        self.rate = self.rate
        return self.rate


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
    :param dvc: velocity grid widths
    :param fe: local electron velocity distribution
    :param sigma: cross-section
    :param const: normalisation cross-section (defaults to 1), defaults to 1.0
    :return: _description_
    """
    rate = 0.0
    for i in range(len(vgrid)):
        rate += vgrid[i] ** 3 * dvc[i] * fe[i] * sigma[i]
    rate *= const * 4.0 * np.pi
    return rate


@jit(nopython=True)
def get_sigma_tbr(
    vgrid: np.ndarray,
    vgrid_inv: np.ndarray,
    sigma_interp: np.ndarray,
    g_ratio: float,
    Te: float,
) -> np.ndarray:
    """Get three-body recombination cross-section via detailed balance

    :param vgrid: Velocity grid
    :param vgrid_inv: Post-collision velocity grid
    :param sigma_interp: Ionisation cross-section interpolated to vgrid_inv
    :param g_ratio: Ratio of statistical weights
    :param Te: Electron temperature
    :return: Three-body recombination cross-section
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
def get_sigma_deex(
    vgrid: np.ndarray, vgrid_inv: np.ndarray, sigma_interp: np.ndarray, g_ratio: float
) -> np.ndarray:
    """_summary_

    :param vgrid: Velocity grid
    :param vgrid_inv: Post-collision velocity grid
    :param sigma_interp: Excitation cross-section interpolated to vgrid_inv
    :param g_ratio: Ratio of statistical weights
    :return: De-excitation cross-section
    """
    sigma_deex = g_ratio * sigma_interp * ((vgrid_inv / vgrid) ** 2)
    return sigma_deex


@jit(nopython=True)
def get_associated_transitions(
    state_id: int, from_ids: list[int], to_ids: list[int]
) -> list[int]:
    """Efficiently find the positions of all transitions associated with a given state ID

    :param state_id: ID for a given state
    :param from_ids: list of all from IDs for each transition
    :param to_ids: list of all to IDs for each transition
    :return: A list of the indices of all associated transitions
    """
    associated_transition_indices = []
    for i in range(len(from_ids)):
        if from_ids[i] == state_id or to_ids[i] == state_id:
            associated_transition_indices.append(i)
    return associated_transition_indices
