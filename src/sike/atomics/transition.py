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
        E_grid: np.ndarray | None = None,
        **transition_kwargs,
    ):
        """Initialise

        :param sigma: Cross-sections
        :param collrate_const: Normalisation constant for collision rate calculation
        :param sigma_norm: Normalisation constant for cross-section
        :param from_stat_weight: Statistical weight of the initial state, defaults to None
        :param born_bethe_coeffs: Born-Bethe coefficients, defaults to None
        :param Egrid: Energy grid (pre_collision), in eV, on which cross-sections were calculation
        :param transition_kwargs: Arguments for base Transition class
        """
        super().__init__(**transition_kwargs)

        self.sigma = 1e-4 * np.array(sigma) / sigma_norm
        self.sigma[np.where(self.sigma < 0.0)] = 0.0
        self.collrate_const = collrate_const
        self.from_stat_weight = from_stat_weight
        if born_bethe_coeffs is not None and E_grid is not None:
            self.born_bethe_coeffs = born_bethe_coeffs
            if E_grid is not None:
                self.Egrid = E_grid  # TODO: Normalisation?

    def interpolate_cross_section(self, new_Egrid: np.ndarray):
        interp_func = interpolate.interp1d(
            self.Egrid,
            np.log(self.sigma),
            kind="linear",
            bounds_error=False,
            fill_value=np.nan,
        )
        sigma_new = np.exp(interp_func(new_Egrid))

        # Apply Born-Bethe approximation at energies higher than 200 * transition energy
        bb_thresh = min(self.delta_E * 200, self.Egrid[-1])
        b0 = self.born_bethe_coeffs[0]
        b1 = self.born_bethe_coeffs[1]
        E_grid_bb = new_Egrid[np.where(new_Egrid > bb_thresh)]
        sigma_new[np.where(new_Egrid > bb_thresh)] = (
            1.1969e-15
            * (1 / (self.from_stat_weight * E_grid_bb))
            * (b0 * np.log(E_grid_bb / self.delta_E) + b1)
        )

        # Set below-threshold sigma to zero
        sigma_new[np.where(new_Egrid <= self.delta_E)] = 0.0

        # Interpolate values which are nan but above threshold
        # isnans = np.isnan(sigma_new)
        # if isnans.any():
        #     nan_locs = np.argwhere(isnans)
        #     first_nonnan_E = new_Egrid[nan_locs[0][-1] + 1]
        #     first_nonnan_sigma = sigma_new[nan_locs[0][-1] + 1]
        #     for nan_loc in nan_locs[0]:
        #         nan_E = new_Egrid[nan_loc]
        #         d1 = nan_E - self.delta_E
        #         d2 = first_nonnan_E - nan_E
        #         interp_val = (d1 * first_nonnan_sigma + d2 * 0.0) / (d1 + d2)
        #         sigma_new[nan_loc] = interp_val
        isnans = np.isnan(sigma_new)
        if isnans.any():
            nan_locs = np.argwhere(isnans)
            first_nonnan_E = new_Egrid[nan_locs[-1][0] + 1]
            first_nonnan_sigma = sigma_new[nan_locs[-1][0] + 1]
            for nan_loc in nan_locs:
                nan_E = new_Egrid[nan_loc[0]]
                d1 = nan_E - self.delta_E
                d2 = first_nonnan_E - nan_E
                interp_val = (d1 * first_nonnan_sigma + d2 * 0.0) / (d1 + d2)
                sigma_new[nan_loc[0]] = interp_val

        # If nans still exist, use the Born-Bethe approx for all elements
        isnans = np.isnan(sigma_new)
        if isnans.any():
            bb_thresh = self.delta_E
            b0 = self.born_bethe_coeffs[0]
            b1 = self.born_bethe_coeffs[1]
            E_grid_bb = new_Egrid[np.where(new_Egrid > bb_thresh)]
            sigma_new[np.where(new_Egrid > bb_thresh)] = (
                1.1969e-15
                * (1 / (self.from_stat_weight * E_grid_bb))
                * (b0 * np.log(E_grid_bb / self.delta_E) + b1)
            )

        # Update cross-section
        self.sigma = sigma_new

        # Final nan check
        if np.isnan(self.sigma).any():
            print(
                "Found some nans in interpolated excitation cross-sections. This is likely to causing issues when solving the rate equations."
            )

        # Delete the energy grid associated with the transition
        del self.Egrid

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
        E_grid: np.ndarray | None = None,
        **transition_kwargs,
    ):
        """Initialise

        :param sigma: Cross-sections
        :param collrate_const: Normalisation constant for collision rate calculation
        :param tbrec_norm: Normalisation constant for three-body recombination rate
        :param sigma_norm: Normalisation constant for cross-section
        :param from_stat_weight: Statistical weight of the initial state, defaults to None, defaults to None
        :param fit_params: Parameters for the high-energy cross-section fit, defaults to None
        :param Egrid: Energy grid (pre_collision), in eV, on which cross-sections were calculation
        :param transition_kwargs: Arguments for base Transition class
        """
        super().__init__(**transition_kwargs)
        self.sigma = 1e-4 * np.array(sigma) / sigma_norm
        self.sigma[np.where(self.sigma < 0.0)] = 0.0
        self.collrate_const = collrate_const
        self.tbrec_norm = tbrec_norm
        self.from_stat_weight = from_stat_weight
        self.fit_params = fit_params
        if E_grid is not None:
            self.Egrid = E_grid  # TODO: Normalisation?

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

    def interpolate_cross_section(self, new_Egrid: np.ndarray):
        # try:
        interp_func = interpolate.interp1d(
            self.Egrid,
            np.log(self.sigma),
            kind="linear",
            bounds_error=False,
            fill_value=np.nan,
        )
        # except:
        #     print("hey")
        sigma_new = np.exp(interp_func(new_Egrid))

        # Apply fit to energies higher than FAC calculated
        p = self.fit_params
        E_grid_fit = new_Egrid[np.where(new_Egrid > self.Egrid[-1])]
        x = E_grid_fit / self.delta_E
        y = 1 - (1 / x)
        sigma_new[np.where(new_Egrid > self.Egrid[-1])] = (
            1.1969e-15
            * (1 / (np.pi * self.from_stat_weight * E_grid_fit))
            * (p[0] * np.log(x) + p[1] * y**2 + p[2] * (y / x) + p[3] * (y / x**2))
        )

        # Set below-threshold sigma to zero
        sigma_new[np.where(new_Egrid <= self.delta_E)] = 0.0

        # Interpolate values which are nan but above threshold
        isnans = np.isnan(sigma_new)
        if isnans.any():
            nan_locs = np.argwhere(isnans)
            first_nonnan_E = new_Egrid[nan_locs[-1][0] + 1]
            first_nonnan_sigma = sigma_new[nan_locs[-1][0] + 1]
            for nan_loc in nan_locs:
                nan_E = new_Egrid[nan_loc[0]]
                d1 = nan_E - self.delta_E
                d2 = first_nonnan_E - nan_E
                interp_val = (d1 * first_nonnan_sigma + d2 * 0.0) / (d1 + d2)
                sigma_new[nan_loc[0]] = interp_val

        # Update cross-section
        self.sigma = sigma_new

        # Final nan check
        if np.isnan(self.sigma).any():
            print(
                "Found some nans in interpolated ionization cross-sections. This is likely to causing issues when solving the rate equations."
            )

        # Delete the energy grid associated with the transition
        del self.Egrid


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
        E_grid: np.ndarray | None = None,  # TODO: Unify use of underscores
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
        :param Egrid: Energy grid (pre_collision), in eV, on which cross-sections were calculation
        """
        super().__init__(**transition_kwargs)

        self.sigma = 1e-4 * np.array(sigma) / sigma_norm
        self.collrate_const = collrate_const
        self.from_stat_weight = from_stat_weight
        self.to_stat_weight = to_stat_weight
        self.l = l
        self.fit_params = fit_params
        if E_grid is not None:
            self.Egrid = E_grid  # TODO: Normalisation

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

    def interpolate_cross_section(self, new_Egrid: np.ndarray):
        interp_func = interpolate.interp1d(
            self.Egrid,
            np.log(self.sigma),
            kind="linear",
            bounds_error=False,
            fill_value=np.nan,
        )
        sigma_new = np.exp(interp_func(new_Egrid))

        # Apply fit to energies higher than FAC calculated
        p = self.fit_params
        E_grid_fit = new_Egrid[np.where(new_Egrid > self.Egrid[-1])]
        x = E_grid_fit / self.delta_E
        y = 1 - (1 / x)
        sigma_new[np.where(new_Egrid > self.Egrid[-1])] = (
            1.1969e-15
            * (1 / (np.pi * self.from_stat_weight * E_grid_fit))
            * (p[0] * np.log(x) + p[1] * y**2 + p[2] * (y / x) + p[3] * (y / x**2))
        )

        E_h = 27.211
        p = self.fit_params
        E_grid_fit = new_Egrid[np.where(new_Egrid > self.Egrid[-1])]
        eps = E_grid_fit / E_h
        w = E_grid_fit + self.delta_E
        x = (E_grid_fit + p[3]) / p[3]
        y = (1.0 + p[2]) / (np.sqrt(x) + p[2])
        dgf_dE = (
            (w / (E_grid_fit + p[3]))
            * p[0]
            * x ** (-3.5 - self.l + (p[1] / 2))
            * y ** p[1]
        )
        alpha = 1 / 137
        g_i = self.to_stat_weight
        g_f = self.from_stat_weight
        sigma_pi = (
            (2 * np.pi * alpha / g_i)
            * ((1 + alpha**2 * eps) / (1 + 0.5 * alpha**2 * eps))
            * dgf_dE
        )

        arb_const = 4e-20  # TODO: This constant works, but not sure exactly what it
        sigma_new[np.where(new_Egrid > self.Egrid[-1])] = (
            (alpha**2 / 2)
            * (g_i / g_f)
            * (w**2 / (eps * (1.0 + 0.5 * alpha**2 * eps)))
            * sigma_pi
            * arb_const
        )

        # Update cross-section
        self.sigma = sigma_new

        # Final nan check
        if np.isnan(self.sigma).any():
            print(
                "Found some nans in interpolated radiative recombination cross-sections. This is likely to causing issues when solving the rate equations."
            )

        # Delete the energy grid associated with the transition
        del self.Egrid


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
