from multiprocessing.spawn import get_preparation_data
from scipy import interpolate
import numpy as np
import re
import matplotlib.pyplot as plt
import atomic_state
import SIKE_tools


class Transition:
    """Base transition class.

    Attributes:
        type (str): transition type. Expected values: 'ionization', 'excitation', 'emission', 'autoionization', 'radiative recombination')
        element (str): element name (chemical symbol)
        from_id (int): level id of the initial state
        to_id (int): level id of the final state
        delta_E (float): transition energy in eV
    """

    def __init__(self, type, element, from_id, to_id, delta_E):

        self.type = type
        self.element = element
        self.from_id = from_id
        self.to_id = to_id
        self.delta_E = delta_E


class ExTrans(Transition):
    """Excitation transition class. Derived from Transition class.

    Attributes:
        sigma (np.array): excitation cross-section in cm^2
    """

    def __init__(self, trans_dict, collrate_const, sigma_norm, T_norm):
        Transition.__init__(self, trans_dict['type'], trans_dict['element'],
                             trans_dict['from_id'], trans_dict['to_id'], trans_dict['delta_E']/T_norm)

        self.sigma = 1e-4 * np.array(trans_dict['sigma']) / sigma_norm
        self.sigma[np.where(self.sigma < 0.0)] = 0.0
        self.collrate_const = collrate_const
        if 'from_stat_weight' in trans_dict.keys():
            self.from_stat_weight = trans_dict['from_stat_weight']
        if 'born_bethe_coeffs' in trans_dict.keys():
            self.born_bethe_coeffs = trans_dict['born_bethe_coeffs']

    def set_sigma_deex(self, g_ratio, vgrid):
        """Calculate the de-excitation cross-section

        Args:
            g_ratio (float): the ratio of statistical weights of from/to states
            vgrid (np.ndarray): the velocity grid
        """
        vgrid_inv = np.sqrt(vgrid ** 2 + self.delta_E)
        sigma_interp_func = interpolate.interp1d(
            vgrid, self.sigma, fill_value=0.0, bounds_error=False, kind='linear')
        sigma_interp = sigma_interp_func(vgrid_inv)
        self.sigma_deex = self.get_sigma_deex(
            vgrid, vgrid_inv, sigma_interp, g_ratio)

    def get_mat_value(self, fe, vgrid, dvc):
        """Get the matrix value for this transition. For excitation transitions, this is ne * rate coefficient

        Args:
            fe (np.array): local electron distribution
            vgrid (np.array): velocity grid
            dvc (np.array): velocity grid widths

        Returns:
            float: electron density multiplied by ionization rate coefficient
        """
        K_ex = SIKE_tools.calc_rate(vgrid, dvc, fe, self.sigma, self.collrate_const)
        return K_ex

    def get_mat_value_inv(self, fe, vgrid, dvc):
        """Get the matrix value for the inverse of transition. For ionization transitions, this is three-body recombination

        Args:
            fe (np.array): local electron distribution
            vgrid (np.array): velocity grid
            dvc (np.array): velocity grid widths
            Te (float): local electron temperature
            g_ratio (float): the ratio of statistical weights (free / bound)
        Returns:
            float: electron density multiplied by three-body recombination rate coefficient
        """

        K_deex = SIKE_tools.calc_rate(
            vgrid, dvc, fe, self.sigma_deex, self.collrate_const)
        return K_deex

    def get_sigma_deex(self, vgrid, vgrid_inv, sigma_interp, g_ratio):
        """Get the de-excitation cross-section, assuming detailed balance

        Args:
            vgrid (np.array): velocity grid
            vgrid_inv (np.array): velocity grid of post-collision electrons
            sigma_interp (np.array): excitation cross-section interpolated on to vgrid_inv
            g_ratio (float): the ratio of statistical weights (free / bound)

        Returns:
            np.array: local de-excitation cross-section
        """
        sigma_deex = SIKE_tools.get_sigma_deex(
            vgrid, vgrid_inv, sigma_interp, g_ratio)

        return sigma_deex


class IzTrans(Transition):
    """Ionization transition class. Derived from Transition class.

    Attributes:
        sigma (np.array): ionization cross-section in cm^2
    """

    def __init__(self, trans_dict, collrate_const, tbrec_norm, sigma_norm, T_norm):

        Transition.__init__(self, trans_dict['type'], trans_dict['element'],
                             trans_dict['from_id'], trans_dict['to_id'], trans_dict['delta_E']/T_norm)
        self.sigma = 1e-4 * np.array(trans_dict['sigma']) / sigma_norm
        self.sigma[np.where(self.sigma < 0.0)] = 0.0
        self.collrate_const = collrate_const
        self.tbrec_norm = tbrec_norm
        if 'from_stat_weight' in trans_dict.keys():
            self.from_stat_weight = trans_dict['from_stat_weight']
        if 'fit_params' in trans_dict.keys():
            self.fit_params = trans_dict['fit_params']

    def set_inv_data(self, g_ratio, vgrid):
        """Store some useful data for calculating the inverse (3b-recombination) cross-sections

        Args:
            g_ratio (float): statistical weight ratio of from/to states
            vgrid (np.ndarray): the velocity grid
        """
        self.g_ratio = g_ratio
        self.vgrid_inv = np.sqrt(vgrid ** 2 + self.delta_E)
        sigma_interp_func = interpolate.interp1d(
            vgrid, self.sigma, fill_value=0.0, bounds_error=False, kind='linear')
        self.sigma_interp = sigma_interp_func(self.vgrid_inv)

    def get_mat_value(self, fe, vgrid, dvc):
        """Get the matrix value for this transition. For ionization transitions, this is ne * rate coefficient

        Args:
            fe (np.array): local electron distribution
            vgrid (np.array): velocity grid
            dvc (np.array): velocity grid widths

        Returns:
            float: electron density multiplied by ionization rate coefficient
        """
        K_ion = SIKE_tools.calc_rate(
            vgrid, dvc, fe, self.sigma, self.collrate_const)
        return K_ion

    def get_mat_value_inv(self, fe, vgrid, dvc, ne, Te):
        """Get the matrix value for the inverse of transition. For ionization transitions, this is three-body recombination

        Args:
            fe (np.array): local electron distribution
            vgrid (np.array): velocity grid
            dvc (np.array): velocity grid widths
            g_ratio (float): the ratio of statistical weights (free / bound)
            Te (float): local electron temperature
        Returns:
            float: electron density multiplied by three-body recombination rate coefficient
        """
        sigma_tbrec = self.get_sigma_tbrec(vgrid, Te)
        K_tbrec = SIKE_tools.calc_rate(
            vgrid, dvc, fe, sigma_tbrec, ne * self.tbrec_norm * self.collrate_const)
        return K_tbrec

    def get_sigma_tbrec(self, vgrid, Te):
        """Get the three-body recombination cross-section, assuming detailed balance

        Args:
            vgrid (np.array): velocity grid
            g_ratio (_type_): the ratio of statistical weights (free / bound)
            Te (_type_): local electron temperature

        Returns:
            np.array: local three-body recombination cross-section
        """

        sigma_tbrec = SIKE_tools.get_sigma_tbr(
            vgrid, self.vgrid_inv, self.sigma_interp, self.g_ratio, Te)

        return sigma_tbrec


class RRTrans(Transition):
    """Radiative recombination transition class. Derived from Transition class.

    Attributes:
        sigma (np.array): radiative recombination cross-section in cm^2
    """

    def __init__(self, trans_dict, collrate_const, sigma_0, T_norm):
        Transition.__init__(self, trans_dict['type'], trans_dict['element'],
                             trans_dict['from_id'], trans_dict['to_id'], trans_dict['delta_E']/T_norm)
        self.sigma = 1e-4 * np.array(trans_dict['sigma']) / sigma_0
        self.collrate_const = collrate_const
        if 'from_stat_weight' in trans_dict.keys():
            self.from_stat_weight = trans_dict['from_stat_weight']
        if 'to_stat_weight' in trans_dict.keys():
            self.to_stat_weight = trans_dict['to_stat_weight']
        if 'l' in trans_dict.keys():
            self.l = trans_dict['l']
        if 'fit_params' in trans_dict.keys():
            self.fit_params = trans_dict['fit_params']

    def get_mat_value(self, fe, vgrid, dvc):
        """Get the matrix value for this transition. For radiative recombination transitions, this is ne * rate coefficient

        Args:
            fe (np.array): local electron distribution
            vgrid (np.array): velocity grid
            dvc (np.array): velocity grid widths

        Returns:
            float: electron density multiplied by ionization rate coefficient
        """
        K_radrec = SIKE_tools.calc_rate(
            vgrid, dvc, fe, self.sigma, self.collrate_const)
        return K_radrec


class EmTrans(Transition):
    """Spontaneous emission transition class. Derived from Transition class.

    Attributes:
        rate (float): spontaneous emission rate in s^-1
    """

    def __init__(self, trans_dict, time_norm, T_norm):
        Transition.__init__(self, trans_dict['type'], trans_dict['element'],
                             trans_dict['from_id'], trans_dict['to_id'], trans_dict['delta_E']/T_norm)
        if 'gf' in trans_dict.keys():
            self.gf = trans_dict['gf']
        self.rate = trans_dict['rate'] * time_norm

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
        Transition.__init__(self, trans_dict['type'], trans_dict['element'],
                             trans_dict['from_id'], trans_dict['to_id'], trans_dict['delta_E']/T_norm)
        self.rate = trans_dict['rate'] * time_norm

    def get_mat_value(self, _=None, __=None, ___=None):
        """Get the matrix value for this transition. For spontaneous emission, this is the emission rate

        Returns:
            float: electron density multiplied by ionization rate coefficient
        """
        A_ai = self.rate
        return A_ai
