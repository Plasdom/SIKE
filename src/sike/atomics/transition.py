from scipy import interpolate
import numpy as np
from numba import jit

from sike.constants import *
from sike.atomics.atomic_state import State


class Transition:
    """Base transition class"""

    def __init__(
        self,
        type: str,
        element: str,
        from_id: int,
        to_id: int,
        delta_E: float,
        T_norm: float,
        **_,
    ):
        # TODO: delta_E was previously passed un-normalised, then divided by T_norm. Now T_norm is not provided so will need to make sure this step happens when the transitions are initialised
        """Initialise

        :param type: Type of transition (e.g. "excitation"))
        :param element: Atomic element
        :param from_id: ID of the initial state
        :param to_id: ID of the final state
        :param delta_E: Transition energy [eV]
        :param T_norm: Normalisation constant [eV] for energy/temperature
        """
        self.type = type
        self.element = element
        self.from_id = from_id
        self.to_id = to_id
        self.delta_E = delta_E / T_norm
        self.T_norm = T_norm


class ExTrans(Transition):
    """Excitation transition"""

    def __init__(
        self,
        collrate_const: float,
        sigma_norm: float,
        simulation_E_grid: np.ndarray,
        from_stat_weight: float | None = None,
        born_bethe_coeffs: np.ndarray | None = None,
        E_grid: np.ndarray | None = None,
        sigma: np.ndarray | None = None,
        osc_str: float | None = None,
        **transition_kwargs,
    ):
        """Initialise

        :param sigma: Cross-sections
        :param collrate_const: Normalisation constant [s^-1] for collision rate calculation
        :param sigma_norm: Normalisation constant [m^-2] for cross-section
        :param simulation_E_grid: Energy grid (pre_collision), in eV, on which all cross-sections will be evaluated when calculating the rate matrix
        :param from_stat_weight: Statistical weight of the initial state, defaults to None
        :param born_bethe_coeffs: Born-Bethe coefficients, defaults to None
        :param E_grid: Energy grid (pre_collision), in eV, on which cross-sections were calculation, defaults to None
        :param transition_kwargs: Arguments for base Transition class
        :param sigma: Cross-sections, defaults to None
        :param osc_str: Oscillator strength, used to geneerate cross-sections if not provided, defaults to None
        """
        super().__init__(**transition_kwargs)

        self.collrate_const = collrate_const
        self.from_stat_weight = from_stat_weight

        if sigma is not None and E_grid is not None and born_bethe_coeffs is not None:
            # Interpolate the given cross-section to the simulation energy grid (used for FAC-derived nlj-resolved data)
            sigma = 1e-4 * np.array(sigma) / sigma_norm
            sigma[np.where(sigma < 0.0)] = 0.0
            self.sigma = self.interpolate_cross_section(
                sigma, born_bethe_coeffs, E_grid, simulation_E_grid
            )
        elif osc_str is not None:
            # Compute cross-sections using van  Regemorter's formula
            sigma = self.compute_cross_section(simulation_E_grid, osc_str)
            sigma = 1e-4 * np.array(sigma) / sigma_norm
            sigma[np.where(sigma < 0.0)] = 0.0
            self.sigma = sigma
        else:
            raise Exception(
                "To calculate excitation cross-sections on the simulation energy grid, either provide:"
                + "\n    1. existing cross-sections, energy grid and Born-Bethe coefficients for high-energy fit, or"
                + "\n    2. oscillator strength"
            )

    def compute_cross_section(self, Egrid: np.ndarray, osc_str: float) -> np.ndarray:
        """Compute the excitation cross-sections using van Regemorter's formula

        :param Egrid: Energy grid, in eV, on which to calculate cross-sections
        :param osc_str: Oscillator strength
        :return: Cross-sections on the input energy grid
        """

        def g(U):
            A = 0.15
            B = 0.0
            C = 0.0
            D = 0.28
            return A + B / U + C / U**2 + D * np.log(U)

        a_0 = 5.29177e-11
        I_H = 13.6058

        eps = self.delta_E * self.T_norm
        f_ij = osc_str

        # Calculate cross-section
        cs = np.zeros(len(Egrid))
        for i in range(len(Egrid)):
            # E = 0.5 * EL_MASS * vgrid[i] ** 2 / EL_CHARGE
            E = Egrid[i]
            U = E / eps
            if E >= eps:
                cs[i] = (
                    (8.0 * np.pi**2 * a_0**2 / np.sqrt(3))
                    * (I_H / eps) ** 2
                    * f_ij
                    * g(U)
                    / U
                )

        if any(np.array(cs) < 0.0):
            print("cs below zero")
        if any(np.isnan(cs)):
            print("nans found in excitation cross-section - setting to zero ")

        return 1e4 * cs

    def interpolate_cross_section(
        self,
        sigma: np.ndarray,
        born_bethe_coeffs: list[float],
        old_Egrid: np.ndarray,
        new_Egrid: np.ndarray,
    ):
        """Inteprolate cross-sections to a new energy grid

        :param sigma: Cross-sections calculated on old_Egrid
        :param born_bethe_coeffs: Born-Bethe coefficients for the high-energy fit
        :param old_Egrid: Energy grid on which cross-sections were calculated
        :param new_Egrid: New energy grid to interpolate to
        :return: Interpolated cross-sections
        """
        interp_func = interpolate.interp1d(
            old_Egrid,
            np.log(sigma),
            kind="linear",
            bounds_error=False,
            fill_value=np.nan,
        )
        sigma_new = np.exp(interp_func(new_Egrid))

        # Apply Born-Bethe approximation at energies higher than 200 * transition energy
        bb_thresh = min(self.delta_E * self.T_norm * 200, old_Egrid[-1])
        b0 = born_bethe_coeffs[0]
        b1 = born_bethe_coeffs[1]
        E_grid_bb = new_Egrid[np.where(new_Egrid > bb_thresh)]
        sigma_new[np.where(new_Egrid > bb_thresh)] = (
            1.1969e-15
            * (1 / (self.from_stat_weight * E_grid_bb))
            * (b0 * np.log(E_grid_bb / self.delta_E * self.T_norm) + b1)
        )

        # Set below-threshold sigma to zero
        sigma_new[np.where(new_Egrid <= self.delta_E * self.T_norm)] = 0.0

        # Interpolate values which are nan but above threshold
        isnans = np.isnan(sigma_new)
        if isnans.any():
            nan_locs = np.argwhere(isnans)
            first_nonnan_E = new_Egrid[nan_locs[-1][0] + 1]
            first_nonnan_sigma = sigma_new[nan_locs[-1][0] + 1]
            for nan_loc in nan_locs:
                nan_E = new_Egrid[nan_loc[0]]
                d1 = nan_E - self.delta_E * self.T_norm
                d2 = first_nonnan_E - nan_E
                interp_val = (d1 * first_nonnan_sigma + d2 * 0.0) / (d1 + d2)
                sigma_new[nan_loc[0]] = interp_val

        # If nans still exist, use the Born-Bethe approx for all elements
        isnans = np.isnan(sigma_new)
        if isnans.any():
            bb_thresh = self.delta_E * self.T_norm
            b0 = self.born_bethe_coeffs[0]
            b1 = self.born_bethe_coeffs[1]
            E_grid_bb = new_Egrid[np.where(new_Egrid > bb_thresh)]
            sigma_new[np.where(new_Egrid > bb_thresh)] = (
                1.1969e-15
                * (1 / (self.from_stat_weight * E_grid_bb))
                * (b0 * np.log(E_grid_bb / (self.delta_E * self.T_norm)) + b1)
            )

        # Final nan check
        if np.isnan(sigma_new).any():
            print(
                "Found some nans in interpolated excitation cross-sections. This is likely to causing issues when solving the rate equations."
            )

        return sigma_new

    def set_sigma_deex(self, g_ratio: float, Egrid: np.ndarray) -> np.ndarray:
        """Calculate the de-excitation cross-section

        :param g_ratio: the ratio of statistical weights of from/to states
        :param vgrid: Velocity grid
        :param v_th: Normalisation constant [ms^-1] for electron velocities
        :return: De-excitation cross-sections
        """
        Egrid_inv = Egrid + self.delta_E
        sigma_interp_func = interpolate.interp1d(
            Egrid, self.sigma, fill_value=0.0, bounds_error=False, kind="linear"
        )
        sigma_interp = sigma_interp_func(Egrid_inv)
        self.sigma_deex = self.get_sigma_deex(Egrid, Egrid_inv, sigma_interp, g_ratio)

    def get_mat_value(self, fe: np.ndarray, Egrid: np.ndarray, dE: np.ndarray) -> float:
        """Get the matrix value for this transition. For excitation transitions, this is ne * rate coefficient

        :param fe: local electron distribution
        :param vgrid: velocity grid
        :param dvc: velocity grid widths
        :return: Matrix value
        """
        self.rate = calc_rate_en(Egrid, dE, fe, self.sigma, self.collrate_const)
        return self.rate

    def get_mat_value_inv(
        self, fe: np.ndarray, Egrid: np.ndarray, dE: np.ndarray
    ) -> float:
        """Get the matrix value for the inverse of transition. For excitation transitions, this is three-body recombination

        :param fe: local electron distribution
        :param vgrid: velocity grid
        :param dvc: velocity grid widths
        :return: electron density multiplied by three-body recombination rate coefficient
        """
        self.rate_inv = calc_rate_en(
            Egrid, dE, fe, self.sigma_deex, self.collrate_const
        )
        return self.rate_inv

    def get_sigma_deex(
        self,
        Egrid: np.ndarray,
        Egrid_inv: np.ndarray,
        sigma_interp: np.ndarray,
        g_ratio: float,
    ) -> np.ndarray:
        """Get the de-excitation cross-section, assuming detailed balance

        :param vgrid: velocity grid
        :param vgrid_inv: velocity grid of post-collision electrons
        :param sigma_interp: excitation cross-section interpolated on to vgrid_inv
        :param g_ratio: the ratio of statistical weights (free / bound)
        :param v_th: Normalisation constant [ms^-1] for electron velocities
        :return: local de-excitation cross-section
        """
        sigma_deex = get_sigma_deex(Egrid, Egrid_inv, sigma_interp, g_ratio)

        return sigma_deex


class IzTrans(Transition):
    """Ionization transition class. Derived from Transition class."""

    def __init__(
        self,
        collrate_const: float,
        tbrec_norm: float,
        sigma_norm: float,
        simulation_E_grid: np.ndarray,
        from_state: State,
        to_state: State,
        from_stat_weight: float | None = None,
        fit_params: np.ndarray | None = None,
        E_grid: np.ndarray | None = None,
        sigma: np.ndarray | None = None,
        **transition_kwargs,
    ):
        """Initialise

        :param collrate_const: Normalisation constant [s^-1] for collision rate calculation
        :param tbrec_norm: Normalisation constant for three-body recombination rate
        :param sigma_norm: Normalisation constant [m^-2] for cross-section
        :param simulation_E_grid: Energy grid (pre_collision), in eV, on which all cross-sections will be evaluated when calculating the rate matrix
        :param from_state: Initial state
        :param to_state: Final state
        :param from_stat_weight: Statistical weight of the initial state, defaults to None, defaults to None
        :param fit_params: Parameters for the high-energy cross-section fit, defaults to None
        :param E_grid: Energy grid (pre_collision), in eV, on which cross-sections were calculation
        :param transition_kwargs: Arguments for base Transition class
        :param sigma: Cross-sections, defaults to None
        """
        super().__init__(**transition_kwargs)
        self.collrate_const = collrate_const
        self.tbrec_norm = tbrec_norm
        self.from_stat_weight = from_stat_weight

        if sigma is not None and E_grid is not None and fit_params is not None:
            # Interpolate existing cross-sections to the simulation energy grid
            sigma = 1e-4 * np.array(sigma) / sigma_norm
            sigma[np.where(sigma < 0.0)] = 0.0
            self.sigma = self.interpolate_cross_section(
                sigma, fit_params, E_grid, simulation_E_grid
            )
        elif isinstance(from_state.config, list) or isinstance(
            from_state.config, np.ndarray
        ):
            # Compute cross-sections from scratch
            sigma = self.compute_cross_section(simulation_E_grid, from_state, to_state)
            sigma = 1e-4 * np.array(sigma) / sigma_norm
            sigma[np.where(sigma < 0.0)] = 0.0
            self.sigma = sigma
        else:
            raise Exception(
                "To calculate ionization cross-sections on the simulation energy grid, either provide:"
                + "\n    1. existing cross-sections, energy grid and fit parameters for high-energy fit, or"
                + "\n    2. n-shell occupancy as a list of integers in the atomic state object for the initial (from) state"
            )

    def set_inv_data(self, g_ratio: float, Egrid: np.ndarray):
        """Store some useful data for calculating the inverse (3b-recombination) cross-sections

        :param g_ratio: statistical weight ratio of from/to states
        :param vgrid: the velocity grid
        :param v_th: Normalisation constant [ms^-1] for electron velocities
        """
        self.g_ratio = g_ratio
        self.Egrid_inv = Egrid + self.delta_E
        sigma_interp_func = interpolate.interp1d(
            Egrid, self.sigma, fill_value=0.0, bounds_error=False, kind="linear"
        )
        self.sigma_interp = sigma_interp_func(self.Egrid_inv)

    def get_mat_value(self, fe: np.ndarray, Egrid: np.ndarray, dE: np.ndarray) -> float:
        """Get the matrix value for this transition. For ionization transitions, this is ne * rate coefficient

        :param fe: local electron distribution
        :param vgrid: velocity grid
        :param dvc: velocity grid widths
        :return: Matrix value
        """
        self.rate = calc_rate_en(Egrid, dE, fe, self.sigma, self.collrate_const)
        return self.rate

    def get_mat_value_inv(
        self,
        fe: np.ndarray,
        Egrid: np.ndarray,
        dE: np.ndarray,
        ne: float,
        Te: float,
    ):
        """Get the matrix value for the inverse of transition. For ionization transitions, this is three-body recombination

        :param fe: local electron distribution
        :param vgrid: velocity grid
        :param dvc: velocity grid widths
        :param ne: Electron density
        :param Te: Electron temperature
        :param v_th: Normalisation constant [ms^-1] for electron velocities
        :return: electron density multiplied by three-body recombination rate coefficient
        """
        sigma_tbrec = self.get_sigma_tbrec(Egrid, Te)
        self.rate_inv = calc_rate_en(
            Egrid,
            dE,
            fe,
            sigma_tbrec,
            ne * self.tbrec_norm * self.collrate_const,
        )
        return self.rate_inv

    def get_sigma_tbrec(self, Egrid: np.ndarray, Te: float) -> np.ndarray:
        """Get the three-body recombination cross-section, assuming detailed balance

        :param vgrid: Velocity grid
        :param Te: Electron temperature
        :param v_th: Normalisation constant [ms^-1] for electron velocities
        :return: Three-body recombination cross-section
        """
        sigma_tbrec = get_sigma_tbr(
            Egrid, self.Egrid_inv, self.sigma_interp, self.g_ratio, Te
        )

        return sigma_tbrec

    def compute_cross_section(
        self, Egrid: np.ndarray, from_state: State, to_state: State
    ) -> np.ndarray:
        """Calculate the ionization cross-sections using the Burgess-Chidichimo formula

        :param Egrid: Energy grid, in eV, on which to calculate cross-sections
        :param from_state: Atomic state object of the intial state
        :param to_state: Atomic state object of the final state
        :return: Cross-sections computed on Egrid
        """
        # Note
        z = from_state.nuc_chg - from_state.num_el
        # z = delta_z

        I_H = 13.6058
        a_0 = 5.29177e-11
        cs = np.zeros(len(Egrid))
        # nu = 0.25 * (np.sqrt((100 * z + 91) / (4 * z + 3)) - 1)
        C = 2.0

        zeta = [c for c in from_state.config if c != 0][-1]
        I_n = to_state.energy - from_state.energy

        for i in range(len(Egrid)):
            E = Egrid[i]
            if E >= I_n:
                beta = 0.25 * (((100 * z + 91) / (4 * z + 3)) ** 0.5 - 5)
                W = np.log(E / I_n) ** (beta * I_n / E)
                cs[i] = (
                    np.pi
                    * a_0**2
                    * C
                    * zeta
                    * (I_H / I_n) ** 2
                    * (I_n / E)
                    * np.log(E / I_n)
                    * W
                )

        return 1e4 * cs

    def interpolate_cross_section(
        self,
        sigma: np.ndarray,
        fit_params: list[float],
        old_Egrid: np.ndarray,
        new_Egrid: np.ndarray,
    ):
        """Inteprolate cross-sections to a new energy grid

        :param sigma: Cross-sections calculated on old_Egrid
        :param fit_params: Parameters for the high-energy fit
        :param old_Egrid: Energy grid on which cross-sections were calculated
        :param new_Egrid: New energy grid to interpolate to
        :return: Interpolated cross-sections
        """
        interp_func = interpolate.interp1d(
            old_Egrid,
            np.log(sigma),
            kind="linear",
            bounds_error=False,
            fill_value=np.nan,
        )
        sigma_new = np.exp(interp_func(new_Egrid))

        # Apply fit to energies higher than FAC calculated
        p = fit_params
        E_grid_fit = new_Egrid[np.where(new_Egrid > old_Egrid[-1])]
        x = E_grid_fit / (self.delta_E * self.T_norm)
        y = 1 - (1 / x)
        sigma_new[np.where(new_Egrid > old_Egrid[-1])] = (
            1.1969e-15
            * (1 / (np.pi * self.from_stat_weight * E_grid_fit))
            * (p[0] * np.log(x) + p[1] * y**2 + p[2] * (y / x) + p[3] * (y / x**2))
        )

        # Set below-threshold sigma to zero
        sigma_new[np.where(new_Egrid <= self.delta_E * self.T_norm)] = 0.0

        # Interpolate values which are nan but above threshold
        isnans = np.isnan(sigma_new)
        if isnans.any():
            nan_locs = np.argwhere(isnans)
            first_nonnan_E = new_Egrid[nan_locs[-1][0] + 1]
            first_nonnan_sigma = sigma_new[nan_locs[-1][0] + 1]
            for nan_loc in nan_locs:
                nan_E = new_Egrid[nan_loc[0]]
                d1 = nan_E - self.delta_E * self.T_norm
                d2 = first_nonnan_E - nan_E
                interp_val = (d1 * first_nonnan_sigma + d2 * 0.0) / (d1 + d2)
                sigma_new[nan_loc[0]] = interp_val

        # Final nan check
        if np.isnan(sigma_new).any():
            print(
                "Found some nans in interpolated ionization cross-sections. This is likely to causing issues when solving the rate equations."
            )

        return sigma_new


class RRTrans(Transition):
    """Radiative recombination transition class. Derived from Transition class."""

    def __init__(
        self,
        collrate_const: float,
        sigma_norm: float,
        simulation_E_grid: np.ndarray,
        from_state: State,
        to_state: State,
        from_stat_weight: float | None = None,
        to_stat_weight: float | None = None,
        l: int | None = None,
        fit_params: np.ndarray | None = None,
        E_grid: np.ndarray | None = None,  # TODO: Unify use of underscores
        sigma: np.ndarray | None = None,
        **transition_kwargs,
    ):
        """Initialise

        :param collrate_const: Normalisation constant [s^-1] for collision rate calculation
        :param sigma_norm: Normalisation constant [m^-2] for cross-section
        :param from_state: Initial state
        :param to_state: Final state
        :param from_stat_weight: Statistical weight of the initial state, defaults to None
        :param to_stat_weight: Statistical weight of the final state, defaults to None
        :param l: Orbital angular momentum quantum number of final state (TODO: Check this)
        :param fit_params: Parameters for high-energy cross-section fit
        :param Egrid: Energy grid (pre_collision), in eV, on which cross-sections were calculation
        :param sigma: Cross-sections, defaults to None
        """
        super().__init__(**transition_kwargs)
        self.collrate_const = collrate_const
        self.from_stat_weight = from_stat_weight
        self.to_stat_weight = to_stat_weight
        self.l = l

        if sigma is not None and E_grid is not None and fit_params is not None:
            # Interpolate existing cross-sections to the simulation energy grid
            sigma = 1e-4 * np.array(sigma) / sigma_norm
            sigma[np.where(sigma < 0.0)] = 0.0
            self.sigma = self.interpolate_cross_section(
                sigma, fit_params, E_grid, simulation_E_grid
            )
        elif isinstance(from_state.config, list) or isinstance(
            from_state.config, np.ndarray
        ):
            # Compute cross-sections from scratch
            sigma = self.compute_cross_section(simulation_E_grid, from_state, to_state)
            sigma = 1e-4 * np.array(sigma) / sigma_norm
            sigma[np.where(sigma < 0.0)] = 0.0
            self.sigma = sigma
        else:
            raise Exception(
                "To calculate radiative recombination cross-sections on the simulation energy grid, either provide:"
                + "\n    1. existing cross-sections, energy grid and fit parameters for high-energy fit, or"
                + "\n    2. n-shell occupancy as a list of integers in the atomic state object for the initial (from) state"
            )

    def get_mat_value(self, fe: np.ndarray, Egrid: np.ndarray, dE: np.ndarray) -> float:
        """Get the matrix value for this transition.

        :param fe: local electron distribution
        :param vgrid: velocity grid
        :param dvc: velocity grid widths
        :return: Matrix value
        """
        self.rate = calc_rate_en(Egrid, dE, fe, self.sigma, self.collrate_const)
        return self.rate

    def compute_cross_section(
        self, Egrid: np.ndarray, from_state: State, to_state: State
    ) -> np.ndarray:
        """Compute the radiative recombination cross-sections by taking the inverse of the photoionization cross-sections, calculated using Kramers' formula

        :param Egrid: Energy grid, in eV, on which to compute the cross-sections
        :param from_state: Atomic state object of the initial state
        :param to_state: Atomic state object of the final state
        :return: Cross-sections evaluated on Egrid
        """
        alpha = 1 / 137
        bohr_radius = 5.292e-11
        Ry = 13.60569
        c = 299792458

        # Calculation of radiative recombination rate cross-sections
        def photo_iz_sigma(I_n, Q_n, E):
            return (
                (64 * np.pi * alpha * bohr_radius**2 / (3**1.5))
                * I_n**2.5
                * Ry**0.5
                / (Q_n * (E**3))
            )

        # Calculate iz potential
        n = to_state.n
        P = to_state.config
        Z = to_state.nuc_chg
        sum_val = 0
        for m in range(1, min(n, 10)):
            sum_val += MARCHAND_SCREENING_COEFFS[min(n, 10) - 1, m - 1] * P[m - 1]
        Q_n = (
            Z
            - sum_val
            - 0.5
            * MARCHAND_SCREENING_COEFFS[min(n, 10) - 1, min(n, 10) - 1]
            * max(P[n - 1] - 1, 0)
        )
        I_n = (
            Ry
            * (Q_n**2)
            / n**2
            * (1 + ((alpha * Q_n / n) ** 2) * (((2 * n) / (n + 1)) - (3 / 4)))
        )

        # Calculate rad rec sigma
        sigma_rr = np.zeros(len(Egrid))
        for i, E in enumerate(Egrid):
            hnu = I_n + E
            sigma_rr[i] = (
                (to_state.stat_weight / from_state.stat_weight)
                * (hnu**2 / (2.0 * EL_MASS * c**2 / EL_CHARGE))
                * photo_iz_sigma(I_n, Q_n, hnu)
                / E
            )

        return 1e4 * sigma_rr

    def interpolate_cross_section(
        self,
        sigma: np.ndarray,
        fit_params: list[float],
        old_Egrid: np.ndarray,
        new_Egrid: np.ndarray,
    ) -> np.ndarray:
        """Inteprolate given cross-section to the provided energy grid

        :param sigma: Input cross-section
        :param fit_params: Fit parameters to use at high energies
        :param old_Egrid: Old energy grid [eV]
        :param new_Egrid: New energy grid [eV]
        :return: Interpolated cross-section
        """
        interp_func = interpolate.interp1d(
            old_Egrid,
            np.log(sigma),
            kind="linear",
            bounds_error=False,
            fill_value=np.nan,
        )
        sigma_new = np.exp(interp_func(new_Egrid))

        # Apply fit to energies higher than FAC calculated
        p = self.fit_params
        E_grid_fit = new_Egrid[np.where(new_Egrid > old_Egrid[-1])]
        x = E_grid_fit / (self.delta_E * self.T_norm)
        y = 1 - (1 / x)
        sigma_new[np.where(new_Egrid > old_Egrid[-1])] = (
            1.1969e-15
            * (1 / (np.pi * self.from_stat_weight * E_grid_fit))
            * (p[0] * np.log(x) + p[1] * y**2 + p[2] * (y / x) + p[3] * (y / x**2))
        )

        E_h = 27.211
        p = fit_params
        E_grid_fit = new_Egrid[np.where(new_Egrid > old_Egrid[-1])]
        eps = E_grid_fit / E_h
        w = E_grid_fit + self.delta_E * self.T_norm
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

        const = 4e-20
        sigma_new[np.where(new_Egrid > old_Egrid[-1])] = (
            (alpha**2 / 2)
            * (g_i / g_f)
            * (w**2 / (eps * (1.0 + 0.5 * alpha**2 * eps)))
            * sigma_pi
            * const
        )

        # Final nan check
        if np.isnan(sigma_new).any():
            print(
                "Found some nans in interpolated radiative recombination cross-sections. This is likely to causing issues when solving the rate equations."
            )

        return sigma_new


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
def calc_rate_en(
    Egrid: np.ndarray,
    dE: np.ndarray,
    fe: np.ndarray,
    sigma: np.ndarray,
    const: float = 1.0,
) -> float:
    """Efficiently compute the collisional rate for a given process (by integrating over energy instead of velocity)

    :param Egrid: energy grid (normalised)
    :param dE: energy grid widths (normalised)
    :param fe: local electron distribution (normalised)
    :param sigma: cross-section (normalised)
    :param const: normalisation cross-section (defaults to 1), defaults to 1.0
    :return: Collisional rate
    """
    rate = 0.0
    for i in range(len(Egrid)):
        rate += Egrid[i] * dE[i] * fe[i] * sigma[i]
    rate *= const * 0.5 * 4.0 * np.pi
    return rate


@jit(nopython=True)
def get_sigma_tbr(
    Egrid: np.ndarray,
    Egrid_inv: np.ndarray,
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
    :param v_th: Normalisation constant [ms^-1] for electron velocities
    :return: Three-body recombination cross-section
    """
    sigma_tbrec = (
        0.5 * g_ratio * ((1 / (np.sqrt(Te) ** 3)) * sigma_interp * (Egrid_inv / Egrid))
    )
    return sigma_tbrec


@jit(nopython=True)
def get_sigma_deex(
    Egrid: np.ndarray,
    Egrid_inv: np.ndarray,
    sigma_interp: np.ndarray,
    g_ratio: float,
) -> np.ndarray:
    """Get de-excitation cross section via principle of detailed balance

    :param vgrid: Velocity grid
    :param vgrid_inv: Post-collision velocity grid
    :param sigma_interp: Excitation cross-section interpolated to vgrid_inv
    :param g_ratio: Ratio of statistical weights
    :param v_th: Normalisation constant [ms^-1] for electron velocities
    :return: De-excitation cross-section
    """
    sigma_deex = g_ratio * sigma_interp * (Egrid_inv / Egrid)
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
