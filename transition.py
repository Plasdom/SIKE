import input
from scipy import interpolate


class State:
    def __init__(self, iz_stage, statename, loc, statw=1, energy=0):
        self.iz = iz_stage
        self.statename = statename
        self.loc = loc
        self.statw = statw
        self.energy = energy


class Transition:
    def __init__(self, trans_type, imp_name, from_state, to_state, vgrid=None, T_norm=None, sigma_0=None, n_norm=None, t_norm=None, dtype=None, spontem_rate=None):
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
            self.load_iz_cross_section()
        elif self.trans_type == 'radrec':
            self.load_rate_data()
        elif self.trans_type == 'excitation':
            self.load_ex_cross_section()
        elif self.trans_type == 'spontem':
            self.spontem_rate = spontem_rate*t_norm

    def load_ex_cross_section(self):

        if self.dtype == 'SunoKato':
            self.sigma, self.thresh = input.load_sunokato_ex_sigma(
                self.vgrid, self.from_state, self.to_state, self.T_norm, self.sigma_0, self.from_state.statw)

    def load_iz_cross_section(self):

        if self.dtype == 'SunoKato':
            self.sigma, self.thresh = input.load_sunokato_iz_sigma(
                self.vgrid, self.from_state, self.to_state, self.T_norm, self.sigma_0)

    def load_rate_data(self):

        if self.dtype == 'ADAS':
            self.radrec_rate, self.radrec_Te = input.load_adas_radrec_rates(
                self.imp_name, self.from_state, self.to_state, self.T_norm, self.n_norm, self.t_norm)
        self.radrec_interp = interpolate.interp1d(
            self.radrec_Te, self.radrec_rate, fill_value='extrapolate')
