from scipy import interpolate
import numpy as np
from numpy.typing import ArrayLike

import atomic_state
import SIKE_tools


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
        :param T_norm: Normalisation constant for temperature
        :type T_norm: float
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
        K_ex = SIKE_tools.calc_rate(vgrid, dvc, fe, self.sigma, self.collrate_const)
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
        K_deex = SIKE_tools.calc_rate(
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
        sigma_deex = SIKE_tools.get_sigma_deex(vgrid, vgrid_inv, sigma_interp, g_ratio)

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
        K_ion = SIKE_tools.calc_rate(vgrid, dvc, fe, self.sigma, self.collrate_const)
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
        K_tbrec = SIKE_tools.calc_rate(
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
        sigma_tbrec = SIKE_tools.get_sigma_tbr(
            vgrid, self.vgrid_inv, self.sigma_interp, self.g_ratio, Te
        )

        return sigma_tbrec


class RRTrans(Transition):
    """Radiative recombination transition class. Derived from Transition class.

    Attributes:
        sigma (np.array): radiative recombination cross-section in cm^2
    """

    def __init__(self, trans_dict, collrate_const, sigma_0, T_norm):
        Transition.__init__(
            self,
            trans_dict["type"],
            trans_dict["element"],
            trans_dict["from_id"],
            trans_dict["to_id"],
            trans_dict["delta_E"] / T_norm,
        )
        self.sigma = 1e-4 * np.array(trans_dict["sigma"]) / sigma_0
        self.collrate_const = collrate_const
        if "from_stat_weight" in trans_dict.keys():
            self.from_stat_weight = trans_dict["from_stat_weight"]
        if "to_stat_weight" in trans_dict.keys():
            self.to_stat_weight = trans_dict["to_stat_weight"]
        if "l" in trans_dict.keys():
            self.l = trans_dict["l"]
        if "fit_params" in trans_dict.keys():
            self.fit_params = trans_dict["fit_params"]

    def get_mat_value(self, fe, vgrid, dvc):
        """Get the matrix value for this transition. For radiative recombination transitions, this is ne * rate coefficient

        Args:
            fe (np.array): local electron distribution
            vgrid (np.array): velocity grid
            dvc (np.array): velocity grid widths

        Returns:
            float: electron density multiplied by ionization rate coefficient
        """
        K_radrec = SIKE_tools.calc_rate(vgrid, dvc, fe, self.sigma, self.collrate_const)
        return K_radrec


class EmTrans(Transition):
    """Spontaneous emission transition class. Derived from Transition class.

    Attributes:
        rate (float): spontaneous emission rate in s^-1
    """

    def __init__(self, trans_dict, time_norm, T_norm):
        Transition.__init__(
            self,
            trans_dict["type"],
            trans_dict["element"],
            trans_dict["from_id"],
            trans_dict["to_id"],
            trans_dict["delta_E"] / T_norm,
        )
        if "gf" in trans_dict.keys():
            self.gf = trans_dict["gf"]
        self.rate = trans_dict["rate"] * time_norm

    def get_mat_value(self, _=None, __=None, ___=None):
        """Get the matrix value for this transition. For spontaneous emission, this is the emission rate

        Returns:
            float: electron density multiplied by ionization rate coefficient
        """
        A_em = self.rate
        return A_em


class AiTrans(Transition):
    """Autoionization transition class. Derived from Transition class.

    Attributes:
        rate (float): autoionization rate in s^-1
    """

    def __init__(self, trans_dict, time_norm, T_norm):
        Transition.__init__(
            self,
            trans_dict["type"],
            trans_dict["element"],
            trans_dict["from_id"],
            trans_dict["to_id"],
            trans_dict["delta_E"] / T_norm,
        )
        self.rate = trans_dict["rate"] * time_norm

    def get_mat_value(self, _=None, __=None, ___=None):
        """Get the matrix value for this transition. For spontaneous emission, this is the emission rate

        Returns:
            float: electron density multiplied by ionization rate coefficient
        """
        A_ai = self.rate
        return A_ai
