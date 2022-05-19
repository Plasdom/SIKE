import input
from scipy import interpolate
import numpy as np
import re
import matplotlib.pyplot as plt
import atomic_state


class Transition:
    def __init__(self, trans_type, imp_name, from_state, to_state, vgrid=None, T_norm=None, sigma_0=None, n_norm=None, t_norm=None, dtype=None, spontem_rate=None, opts=None):
        self.trans_type = trans_type
        self.imp_name = imp_name
        self.vgrid = vgrid
        self.T_norm = T_norm
        self.sigma_0 = sigma_0
        self.n_norm = n_norm
        self.t_norm = t_norm
        self.from_state = from_state
        self.from_loc = from_state.loc
        self.to_state = to_state
        self.to_loc = to_state.loc
        self.dtype = dtype
        if self.trans_type == 'ionization':
            self.load_iz_cross_section(opts)
        elif self.trans_type == 'radrec':
            self.load_rate_data()
        elif self.trans_type == 'excitation':
            self.load_ex_cross_section()
        elif self.trans_type == 'spontem':
            self.spontem_rate = spontem_rate*t_norm

    def load_ex_cross_section(self):

        if self.dtype == 'SunoKato':
            self.sigma, self.thresh, self.coll_strength = input.load_sunokato_ex_sigma(
                self.vgrid, self.from_state, self.to_state, self.T_norm, self.sigma_0, self.from_state.statw)
        elif self.dtype == 'NIFS':
            self.sigma, self.thresh, self.coll_strength = input.load_nifs_ex_sigma(
                self.vgrid, self.from_state, self.to_state, self.sigma_0, self.T_norm)

    def load_iz_cross_section(self, opts):

        if self.dtype == 'SunoKato':
            self.sigma, self.thresh = input.load_sunokato_iz_sigma(
                self.vgrid, self.from_state, self.to_state, self.T_norm, self.sigma_0)
        elif self.dtype == 'Lotz':
            self.sigma, self.thresh = input.get_lotz_iz_cs(
                self.vgrid, self.T_norm, self.from_state, self.sigma_0)
        elif self.dtype == 'BurgessChidichimo':
            self.sigma, self.thresh = input.get_BC_iz_cs(
                self.vgrid, self.T_norm, self.from_state, self.to_state, self.sigma_0)

    def load_rate_data(self):

        if self.dtype == 'ADAS':
            self.radrec_rate, self.radrec_Te = input.load_adas_radrec_rates(
                self.imp_name, self.from_state, self.to_state, self.T_norm, self.n_norm, self.t_norm)
        self.radrec_interp = interpolate.interp1d(
            self.radrec_Te, self.radrec_rate, fill_value='extrapolate')

    def plot_cs(self, ax=None, form='cross_section', units='cm2', logx=False, logy=False):
        if ax is None:
            fig, ax = plt.subplots(1)
            ax.set_xlabel('E [eV]')
        if form == 'collision_strength':
            ax.set_xscale('log')
            ax.set_ylabel('Collision strength')
            label = str(self.from_state.iz) + '+  ' + self.from_state.statename + \
                ' -> ' + \
                '+ ' + self.to_state.statename + ''
            ax.plot((self.vgrid ** 2 * self.T_norm) / self.thresh,
                    self.coll_strength, label=label)
        elif form == 'cross_section':
            ax.set_ylabel('Cross-section [' + units + ']')
            if logx:
                ax.set_xscale('log')
            if logy:
                ax.set_yscale('log')
            ax.axvline(self.thresh, linestyle='--', color='grey')
            label = str(self.from_state.iz) + '+  (' + self.from_state.statename + \
                ') -> ' + str(self.to_state.iz) + \
                '+ (' + self.to_state.statename + ')'
            if units == 'm2':
                y = self.sigma * self.sigma_0
            elif units == 'cm2':
                y = self.sigma * self.sigma_0 * 1e4
            elif units == 'bohr':
                a_0 = 5.29177e-11
                y = self.sigma * self.sigma_0 / (a_0 * a_0)
            ax.plot(self.vgrid ** 2 * self.T_norm,
                    y, label=label)
        ax.legend()
