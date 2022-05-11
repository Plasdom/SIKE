import numpy as np
import input
import rates
import os
import sk_plotting_functions as spf
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import interpolate
import transition
import pandas as pd
import aurora


class Impurity:
    def __init__(self, name, skrun, opts):
        self.name = name
        self.skrun = skrun
        self.opts = opts
        if self.name == 'C':
            self.num_z = 7
            self.longname = 'Carbon'
        elif self.name == 'W':
            self.num_z = 11
            self.longname = 'Tungsten'
        self.load_states()
        self.load_transitions()
        self.init_dens()

    def load_states(self):
        self.states = []
        if self.opts['GS_ONLY']:
            statedata_file = os.path.join(
                'imp_data', self.longname, 'states_gsonly.txt')
        else:
            statedata_file = os.path.join(
                'imp_data', self.longname, 'states.txt')
        with open(statedata_file) as f:
            lines = f.readlines()
            i = 0
            for l in lines[1:]:
                iz = int(l.split('\t')[0])
                statename = l.split('\t')[1]
                statw = int(l.split('\t')[-2].strip('\n'))
                energy = float(l.split('\t')[-1].strip('\n'))
                if iz < self.num_z:
                    self.states.append(
                        transition.State(iz, statename, i, statw, energy))
                    i += 1
        self.tot_states = len(self.states)

    def load_transitions(self):

        self.iz_transitions = []
        self.ex_transitions = []
        self.radrec_transitions = []
        self.spontem_transitions = []

        if self.opts['GS_ONLY_RADREC']:
            trans_file = os.path.join(
                'imp_data', self.longname, 'transitions_gsradrec.txt')
        else:
            trans_file = os.path.join(
                'imp_data', self.longname, 'transitions.txt')
        with open(trans_file) as f:
            lines = f.readlines()
            for l in lines[1:]:
                line_data = l.split('\t')
                trans_type = line_data[0]

                from_iz = int(line_data[1])
                from_statename = line_data[2]
                from_state = self.get_state(from_iz, from_statename)

                to_iz = int(line_data[3])
                to_statename = line_data[4]
                to_state = self.get_state(to_iz, to_statename)

                if from_state is not None and to_state is not None:

                    dtype = line_data[5].strip('\n')

                    if trans_type == 'ionization' and self.opts['COLL_ION_REC']:
                        self.iz_transitions.append(transition.Transition('ionization',
                                                                         self.longname,
                                                                         from_state,
                                                                         to_state,
                                                                         vgrid=self.skrun.vgrid/self.skrun.v_th,
                                                                         T_norm=self.skrun.T_norm,
                                                                         sigma_0=self.skrun.sigma_0,
                                                                         dtype=dtype))

                    if trans_type == 'radrec' and self.opts['RAD_REC']:
                        self.radrec_transitions.append(transition.Transition('radrec',
                                                                             self.longname,
                                                                             from_state,
                                                                             to_state,
                                                                             T_norm=self.skrun.T_norm,
                                                                             n_norm=self.skrun.n_norm,
                                                                             t_norm=self.skrun.t_norm,
                                                                             dtype=dtype))

                    if trans_type == 'excitation' and self.opts['COLL_EX_DEEX']:
                        self.ex_transitions.append(transition.Transition('excitation',
                                                                         self.longname,
                                                                         from_state,
                                                                         to_state,
                                                                         vgrid=self.skrun.vgrid/self.skrun.v_th,
                                                                         T_norm=self.skrun.T_norm,
                                                                         sigma_0=self.skrun.sigma_0,
                                                                         dtype=dtype))

        if self.opts['SPONT_EM']:
            self.load_spontem_transitions()

    def load_spontem_transitions(self):
        # Load spontem file data
        spontem_file = os.path.join(
            'imp_data', self.longname, 'nist_spontem.txt')
        spontem_data = pd.read_csv(spontem_file, sep='\t')

        # For each z, loop through all permutations and find matching transitions
        for z in range(self.num_z):
            for from_state in self.states:
                for to_state in self.states:
                    if (from_state.iz == to_state.iz) and from_state.statename != to_state.statename:
                        matched_transitions = spontem_data[(spontem_data['iz'] == z) & (spontem_data['LowerstatenameImpZeff'] == to_state.statename) & (
                            spontem_data['UpperstatenameImpZeff'] == from_state.statename)]
                        if len(matched_transitions['iz'].values) > 0:
                            # Add a transition for the matched states
                            ls_avg_rate = pd.to_numeric(
                                matched_transitions['rate(s-1)']).mean()
                            self.spontem_transitions.append(transition.Transition('spontem',
                                                                                  self.longname,
                                                                                  from_state,
                                                                                  to_state,
                                                                                  t_norm=self.skrun.t_norm,
                                                                                  spontem_rate=ls_avg_rate))

    def get_state(self, iz, statename):
        for state in self.states:
            if state.iz == iz and state.statename == statename:
                return state

    def get_ground_state(self, z):
        for state in self.states:
            if state.iz == z:
                return state

    def init_dens(self):
        self.dens = np.zeros((self.skrun.num_x, self.tot_states))
        self.dens_max = np.zeros((self.skrun.num_x, self.tot_states))
        self.dens_saha = np.zeros((self.skrun.num_x, self.tot_states))
        self.dens[:, 0] = self.opts['FRAC_IMP_DENS'] * \
            self.skrun.data['DENSITY']
        self.dens_max[:, 0] = self.opts['FRAC_IMP_DENS'] * \
            self.skrun.data['DENSITY']
        self.tmp_dens = np.zeros((self.skrun.num_x, self.tot_states))

    def get_saha_eq(self):

        gs_locs = []
        for z in range(self.num_z):
            gs = self.get_ground_state(z)
            gs_locs.append(gs.loc)

        for i in range(self.skrun.num_x):
            de_broglie_l = np.sqrt(
                (spf.planck_h ** 2) / (2 * np.pi * spf.el_mass * spf.el_charge * self.skrun.T_norm * self.skrun.data['TEMPERATURE'][i]))

            # Compute ratios
            dens_ratios = np.zeros(self.num_z - 1)
            for z in range(1, self.num_z):
                eps = self.iz_transitions[z-1].thresh
                dens_ratios[z-1] = (2 * (self.states[gs_locs[z-1]].statw / self.states[gs_locs[z]].statw) * np.exp(-eps / (self.skrun.data['TEMPERATURE'][i]*self.skrun.T_norm))) / (
                    (self.skrun.data['DENSITY'][i] * self.skrun.n_norm) * (de_broglie_l ** 3))
            # Fill densities
            imp_dens_tot = np.sum(self.dens[i, :])
            denom_sum = 1.0 + np.sum([np.prod(dens_ratios[:z+1])
                                      for z in range(self.num_z-1)])
            self.dens_saha[i, 0] = imp_dens_tot / denom_sum
            for z in range(1, self.num_z):
                self.dens_saha[i, gs_locs[z]] = self.dens_saha[i,
                                                               gs_locs[z-1]] * dens_ratios[z-1]

    def build_rate_matrices(self):

        self.skrun.load_dist()
        f0 = np.transpose(self.skrun.data['DIST_F'][0])
        f0 = f0[:, :]
        # ne = np.ones(skrun.num_x)
        ne = self.skrun.data['DENSITY']
        Te = self.skrun.data['TEMPERATURE']
        # Te = np.linspace(2.0, 0.01, skrun.num_x)
        f0_max = get_maxwellians(
            self.skrun.num_x, ne, Te, self.skrun.vgrid, self.skrun.v_th, self.skrun.num_v)

        collrate_const = self.skrun.n_norm * self.skrun.v_th * \
            self.skrun.sigma_0 * self.skrun.t_norm
        op_mat = [np.zeros((self.tot_states, self.tot_states))
                  for _ in range(self.skrun.num_x)]
        rate_mat = [np.zeros((self.tot_states, self.tot_states))
                    for _ in range(self.skrun.num_x)]
        op_mat_max = [np.zeros((self.tot_states, self.tot_states))
                      for _ in range(self.skrun.num_x)]
        rate_mat_max = [np.zeros((self.tot_states, self.tot_states))
                        for _ in range(self.skrun.num_x)]
        tbrec_norm = self.skrun.n_norm * np.sqrt((spf.planck_h ** 2) / (2 * np.pi *
                                                                        spf.el_mass * self.skrun.T_norm * spf.el_charge)) ** 3

        # Build kinetic rate matrices
        for i in range(self.skrun.num_x):

            if self.opts['COLL_ION_REC']:
                for iz_trans in self.iz_transitions:

                    # Ionization
                    K_ion = collrate_const * rates.ion_rate(
                        f0[i, :],
                        self.skrun.vgrid/self.skrun.v_th,
                        self.skrun.dvc/self.skrun.v_th,
                        iz_trans.sigma
                    )
                    K_ion_max = collrate_const * rates.ion_rate(
                        f0_max[i, :],
                        self.skrun.vgrid/self.skrun.v_th,
                        self.skrun.dvc/self.skrun.v_th,
                        iz_trans.sigma
                    )
                    # ...loss
                    row = iz_trans.from_loc
                    col = iz_trans.from_loc
                    rate_mat[i][row, col] += -K_ion
                    rate_mat_max[i][row, col] += -K_ion_max
                    # ...gain
                    row = iz_trans.to_loc
                    col = iz_trans.from_loc
                    rate_mat[i][row, col] += K_ion
                    rate_mat_max[i][row, col] += K_ion_max

                    # Three-body recombination
                    eps = iz_trans.thresh / self.skrun.T_norm
                    statw_ratio = iz_trans.to_state.statw / iz_trans.from_state.statw
                    K_rec = ne[i] * collrate_const * tbrec_norm * rates.tbrec_rate(
                        f0[i, :],
                        Te[i],
                        self.skrun.vgrid/self.skrun.v_th,
                        self.skrun.dvc/self.skrun.v_th,
                        eps,
                        statw_ratio,
                        iz_trans.sigma)
                    K_rec_max = ne[i] * collrate_const * tbrec_norm * rates.tbrec_rate(
                        f0_max[i, :],
                        Te[i],
                        self.skrun.vgrid/self.skrun.v_th,
                        self.skrun.dvc/self.skrun.v_th,
                        eps,
                        statw_ratio,
                        iz_trans.sigma)
                    # ...loss
                    row = iz_trans.to_loc
                    col = iz_trans.to_loc
                    rate_mat[i][row, col] += -K_rec
                    rate_mat_max[i][row, col] += -K_rec_max
                    # ...gain
                    row = iz_trans.from_loc
                    col = iz_trans.to_loc
                    rate_mat[i][row, col] += K_rec
                    rate_mat_max[i][row, col] += K_rec_max

            if self.opts['RAD_REC']:
                for radrec_trans in self.radrec_transitions:

                    # Radiative recombination
                    alpha_radrec = max(radrec_trans.radrec_interp(
                        Te[i]), 0.0)
                    # ...loss
                    row = radrec_trans.from_loc
                    col = radrec_trans.from_loc
                    rate_mat[i][row, col] += -(ne[i] * alpha_radrec)
                    rate_mat_max[i][row, col] += -(ne[i] * alpha_radrec)
                    # ...gain
                    row = radrec_trans.to_loc
                    col = radrec_trans.from_loc
                    rate_mat[i][row, col] += (ne[i] * alpha_radrec)
                    rate_mat_max[i][row, col] += (ne[i] * alpha_radrec)

            if self.opts['COLL_EX_DEEX']:
                for ex_trans in self.ex_transitions:

                    # Excitation
                    K_ex = collrate_const * rates.ex_rate(
                        f0[i, :],
                        self.skrun.vgrid/self.skrun.v_th,
                        self.skrun.dvc/self.skrun.v_th,
                        ex_trans.sigma
                    )
                    K_ex_max = collrate_const * rates.ex_rate(
                        f0_max[i, :],
                        self.skrun.vgrid/self.skrun.v_th,
                        self.skrun.dvc/self.skrun.v_th,
                        ex_trans.sigma
                    )
                    # ...loss
                    row = ex_trans.from_loc
                    col = ex_trans.from_loc
                    rate_mat[i][row, col] += -K_ex
                    rate_mat_max[i][row, col] += -K_ex_max
                    # ...gain
                    row = ex_trans.to_loc
                    col = ex_trans.from_loc
                    rate_mat[i][row, col] += K_ex
                    rate_mat_max[i][row, col] += K_ex_max

                    # Deexcitation
                    eps = ex_trans.thresh / self.skrun.T_norm
                    statw_ratio = ex_trans.from_state.statw / ex_trans.to_state.statw
                    K_deex = collrate_const * rates.deex_rate(
                        f0[i, :],
                        self.skrun.vgrid/self.skrun.v_th,
                        self.skrun.dvc/self.skrun.v_th,
                        eps,
                        statw_ratio,
                        ex_trans.sigma
                    )
                    K_deex_max = collrate_const * rates.deex_rate(
                        f0_max[i, :],
                        self.skrun.vgrid/self.skrun.v_th,
                        self.skrun.dvc/self.skrun.v_th,
                        eps,
                        statw_ratio,
                        ex_trans.sigma
                    )
                    # ...loss
                    row = ex_trans.to_loc
                    col = ex_trans.to_loc
                    rate_mat[i][row, col] += -K_deex
                    rate_mat_max[i][row, col] += -K_deex_max
                    # ...gain
                    row = ex_trans.from_loc
                    col = ex_trans.to_loc
                    rate_mat[i][row, col] += K_deex
                    rate_mat_max[i][row, col] += K_deex_max

            if self.opts['SPONT_EM']:
                for em_trans in self.spontem_transitions:

                    # Spontaneous emission
                    beta_spontem = em_trans.spontem_rate
                    # ...loss
                    row = em_trans.from_loc
                    col = em_trans.from_loc
                    rate_mat[i][row, col] += -beta_spontem
                    rate_mat_max[i][row, col] += -beta_spontem
                    # ...gain
                    row = em_trans.to_loc
                    col = em_trans.from_loc
                    rate_mat[i][row, col] += beta_spontem
                    rate_mat_max[i][row, col] += beta_spontem

            op_mat[i] = np.linalg.inv(np.identity(
                self.tot_states) - self.opts['DELTA_T'] * rate_mat[i])
            op_mat_max[i] = np.linalg.inv(np.identity(
                self.tot_states) - self.opts['DELTA_T'] * rate_mat_max[i])

        self.op_mat = op_mat
        self.rate_mat = rate_mat
        self.op_mat_max = op_mat_max
        self.rate_mat_max = rate_mat_max

    def solve(self):
        for i in range(self.skrun.num_x):
            loc_mat = self.rate_mat[i]
            loc_mat[-1, :] = 1.0
            rhs = np.zeros(self.tot_states)
            rhs[-1] = np.sum(self.dens[i, :])
            loc_mat_inv = np.linalg.inv(loc_mat)
            dens = loc_mat_inv.dot(rhs)
            self.dens[i, :] = dens.copy()

        for i in range(self.skrun.num_x):
            loc_mat = self.rate_mat_max[i]
            loc_mat[-1, :] = 1.0
            rhs = np.zeros(self.tot_states)
            rhs[-1] = np.sum(self.dens_max[i, :])
            loc_mat_inv = np.linalg.inv(loc_mat)
            dens = loc_mat_inv.dot(rhs)
            self.dens_max[i, :] = dens.copy()

    def evolve(self):

        # Compute the particle source for each ionization state
        for i in range(self.skrun.num_x):

            # Calculate new densities
            self.tmp_dens[i, :] = self.op_mat[i].dot(self.dens[i, :])

        residual = np.max(np.abs(self.tmp_dens - self.dens))
        self.dens = self.tmp_dens.copy()

        # Compute the particle source for each ionization state (Maxwellian)
        for i in range(self.skrun.num_x):

            # Calculate new densities
            self.tmp_dens[i, :] = self.op_mat_max[i].dot(self.dens_max[i, :])

        residual_max = np.max(np.abs(self.tmp_dens - self.dens_max))
        self.dens_max = self.tmp_dens.copy()

        return max(residual, residual_max)

    def get_Zeff(self):

        self.Zeff = np.zeros(self.skrun.num_x)
        self.Zeff_max = np.zeros(self.skrun.num_x)
        self.Zeff_saha = np.zeros(self.skrun.num_x)
        dens_tot = np.sum(self.dens, 1)
        dens_tot_max = np.sum(self.dens_max, 1)
        dens_tot_saha = np.sum(self.dens_saha, 1)

        for i in range(self.skrun.num_x):

            for k, state in enumerate(self.states):
                z = state.iz
                k = state.loc
                self.Zeff[i] += self.dens[i, k] * (float(z) ** 2)
                self.Zeff_max[i] += self.dens_max[i, k] * (float(z) ** 2)
                self.Zeff_saha[i] += self.dens_saha[i, k] * (float(z) ** 2)
            if dens_tot[i] > 0:
                self.Zeff[i] = self.Zeff[i] / self.skrun.data['DENSITY'][i]
            if dens_tot_max[i] > 0:
                self.Zeff_max[i] = self.Zeff_max[i] / \
                    self.skrun.data['DENSITY'][i]
            if dens_tot_saha[i] > 0:
                self.Zeff_saha[i] = self.Zeff_saha[i] / \
                    self.skrun.data['DENSITY'][i]

    def get_Zavg(self):

        self.Zavg = np.zeros(self.skrun.num_x)
        self.Zavg_max = np.zeros(self.skrun.num_x)
        self.Zavg_saha = np.zeros(self.skrun.num_x)
        dens_tot = np.sum(self.dens, 1)
        dens_tot_max = np.sum(self.dens_max, 1)
        dens_tot_saha = np.sum(self.dens_saha, 1)

        for i in range(self.skrun.num_x):

            for k, state in enumerate(self.states):
                z = state.iz
                k = state.loc
                self.Zavg[i] += self.dens[i, k] * float(z)
                self.Zavg_max[i] += self.dens_max[i, k] * float(z)
                self.Zavg_saha[i] += self.dens_saha[i, k] * float(z)
            if dens_tot[i] > 0:
                self.Zavg[i] = self.Zavg[i] / dens_tot[i]
            if dens_tot_max[i] > 0:
                self.Zavg_max[i] = self.Zavg_max[i] / dens_tot_max[i]
            if dens_tot_saha[i] > 0:
                self.Zavg_saha[i] = self.Zavg_saha[i] / dens_tot_saha[i]

    def plot_Zeff(self, plot_saha=False):
        self.get_Zeff()
        fig, ax = plt.subplots(1)
        ax.plot(self.skrun.xgrid[:-1], self.Zeff_max[:-1],
                label='Maxwellian $f_0$', color='black')
        if plot_saha:
            ax.plot(self.skrun.xgrid[:-1], self.Zeff_saha[:-1], '--',
                    label='Saha equilibrium', color='blue')
        ax.plot(self.skrun.xgrid[:-1], self.Zeff[:-1], '--',
                label='SOL-KiT $f_0$', color='red')
        ax.grid()
        ax.legend()
        ax.set_xlabel('x [m]')
        ax.set_ylabel('$Z_{eff}=\Sigma_i Z^2_i n_{Z}^i / n_e$')
        ax.set_title('$Z_{eff}$ profile, ' + self.longname)
        # ax.set_xlim([10.4, 11.8])

    def plot_Zavg(self, plot_saha=False):
        self.get_Zavg()
        fig, ax = plt.subplots(1)
        ax.plot(self.skrun.xgrid[:-1], self.Zavg_max[:-1],
                label='Maxwellian $f_0$', color='black')
        if plot_saha:
            ax.plot(self.skrun.xgrid[:-1], self.Zavg_saha[:-1], '--',
                    label='Saha equilibrium', color='blue')
        ax.plot(self.skrun.xgrid[:-1], self.Zavg[:-1], '--',
                label='SOL-KiT $f_0$', color='red')
        ax.grid()
        ax.legend()
        ax.set_xlabel('x [m]')
        ax.set_ylabel('$Z_{avg}=\Sigma_i Z_i n_{Z}^i / n_Z^{tot}$')
        ax.set_title('$Z_{avg}$ profile, ' + self.longname)
        # ax.set_xlim([10.4, 11.8])

    def plot_dens(self, xaxis='normal', plot_kin=True, plot_max=True, plot_saha=False, normalise=False):
        fig, ax = plt.subplots(1)
        cmap = plt.cm.get_cmap('Paired')
        z_dens, z_dens_max, z_dens_saha = self.get_z_dens()
        if normalise:
            for i in range(self.skrun.num_x):
                z_dens[i, :] = z_dens[i, :] / np.sum(z_dens[i, :])
                z_dens_max[i, :] = z_dens_max[i, :] / np.sum(z_dens_max[i, :])
                z_dens_saha[i, :] = z_dens_saha[i, :] / \
                    np.sum(z_dens_saha[i, :])
        else:
            z_dens = z_dens * self.skrun.n_norm
            z_dens_max = z_dens_max * self.skrun.n_norm
            z_dens_saha = z_dens_saha * self.skrun.n_norm
        min_z = 0
        max_z = self.num_z
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(
            vmin=min_z, vmax=max_z))
        legend_lines = []
        legend_labels = []
        if xaxis == 'normal':
            x = self.skrun.xgrid[:-1]
            ax.set_xlabel('x [m]')
        else:
            x = self.skrun.xgrid[-1] - self.skrun.xgrid[:-1]
            ax.set_xlabel('Distance from sheath [m]')
            ax.set_xscale('log')
        for z in range(min_z, max_z):
            if plot_kin:
                ax.plot(x, z_dens[:-1, z], '--', color=sm.to_rgba(z))
            if plot_max:
                ax.plot(x, z_dens_max[:-1, z], color=sm.to_rgba(z))
            if plot_saha:
                ax.plot(x, z_dens_saha[:-1, z], '-.', color=sm.to_rgba(z))
            legend_labels.append('$Z=$' + str(z))
            legend_lines.append(Line2D([0], [0], color=sm.to_rgba(z)))
        if plot_max:
            legend_lines.append(Line2D([0], [0], linestyle='-', color='black'))
        if plot_kin:
            legend_lines.append(
                Line2D([0], [0], linestyle='--', color='black'))
        if plot_saha:
            legend_lines.append(
                Line2D([0], [0], linestyle='-.', color='black'))
        if plot_max:
            legend_labels.append('Maxwellian $f_0$')
        if plot_kin:
            legend_labels.append('SOL-KiT $f_0$')
        if plot_saha:
            legend_labels.append('Saha equilibrium')
        ax.legend(legend_lines, legend_labels)
        ax.grid()
        if normalise:
            ax.set_ylabel('$n_Z / n_Z^{tot}$')
        else:
            ax.set_ylabel('$n_Z$')
        ax.set_title('$n_Z$ profiles, ' + self.longname)

    def get_z_dens(self):
        z_dens = np.zeros([self.skrun.num_x, self.num_z])
        z_dens_max = np.zeros([self.skrun.num_x, self.num_z])
        z_dens_saha = np.zeros([self.skrun.num_x, self.num_z])
        for state in self.states:
            z = state.iz
            z_dens[:, z] += self.dens[:, state.loc]
            z_dens_max[:, z] += self.dens_max[:, state.loc]
            z_dens_saha[:, z] += self.dens_saha[:, state.loc]
        return z_dens, z_dens_max, z_dens_saha

    def get_state_dens(self, all_dens):
        state_dens = []

        start_loc = 0
        for z in range(self.num_z):
            # Get number of states in current z
            num_states = 0
            for i, state in enumerate(self.states):
                if state.iz == z:
                    num_states += 1
            end_loc = start_loc + num_states
            # Extract these densities
            state_dens.append(all_dens[:, start_loc:end_loc])
            start_loc = end_loc

        return state_dens

    def gnormalise(self, dens, z):
        norm_dens = dens.copy()
        i = 0
        for state in self.states:
            if state.iz == z:
                norm_dens[i] = norm_dens[i] / state.statw
                i += 1
        return norm_dens

    def get_boltzmann_dist(self, locstate_dens, z, cell, gnormalise):
        boltz_dens = np.zeros(len(locstate_dens))
        ex_sum = 0
        Te = self.skrun.data['TEMPERATURE'][cell] * self.skrun.T_norm
        n_tot = np.sum(locstate_dens)
        for state in self.states:
            if state.iz == z:
                ex_sum += state.statw * np.exp(-state.energy/Te)
        i = 0
        for state in self.states:
            if state.iz == z:
                if gnormalise:
                    boltz_dens[i] = n_tot * \
                        np.exp(-state.energy/Te) / ex_sum
                else:
                    boltz_dens[i] = n_tot * state.statw * \
                        np.exp(-state.energy/Te) / ex_sum
                i += 1

        return boltz_dens

    def plot_atomdist(self, z=0, cells=-1, gnormalise=False):
        if isinstance(cells, int):
            cells = [cells]
            cell_pref = ['']
        else:
            cell_pref = []
            for i in range(len(cells)):
                cell_pref.append('Cell ' + str(cells[i]) + ', ')

        # Get statenames
        statenames = []
        for state in self.states:
            if state.iz == z:
                statenames.append(state.statename)

        # Plot atomic distribution
        fig, ax = plt.subplots(1)
        allstates_dens = self.get_state_dens(self.dens)
        allstates_dens_max = self.get_state_dens(self.dens_max)
        for cell in cells:
            locstate_dens = allstates_dens[z][cell, :]
            locstate_dens_max = allstates_dens_max[z][cell, :]
            boltz_dens = self.get_boltzmann_dist(
                locstate_dens, z, cells, gnormalise)
            if gnormalise:
                locstate_dens = self.gnormalise(locstate_dens, z)
                locstate_dens_max = self.gnormalise(locstate_dens_max, z)

            for i, cell in enumerate(cells):
                # ax.plot(range(len(locstate_dens_max)), locstate_dens_max,
                #         label=cell_pref[i] + 'Maxwellian $f_0$', color='black',  alpha=1.0-0.2*i)
                # ax.plot(range(len(locstate_dens)), locstate_dens, '--',
                #         label=cell_pref[i] + 'SOL-KiT $f_0$', color='red',  alpha=1.0-0.2*i)
                # ax.plot(range(len(locstate_dens)), boltz_dens, '--',
                #         label=cell_pref[i] + 'Boltzmann distribution', color='blue',  alpha=1.0-0.2*i)
                ax.plot(statenames, locstate_dens_max,
                        label=cell_pref[i] + 'Maxwellian $f_0$', color='black',  alpha=1.0-0.2*i)
                ax.plot(statenames, locstate_dens, '--',
                        label=cell_pref[i] + 'SOL-KiT $f_0$', color='red',  alpha=1.0-0.2*i)
                ax.plot(statenames, boltz_dens, '--',
                        label=cell_pref[i] + 'Boltzmann distribution', color='blue',  alpha=1.0-0.2*i)
        ax.grid()
        ax.legend()
        ax.set_yscale('log')
        ax.set_xlabel('State')
        ax.set_ylabel(r'$n^{C^{' + str(z) + '+}}_i' + ' / n_0$')
        ax.set_title('Atomic state densities, ' + self.longname + '+' + str(z))
        plt.xticks(rotation=45)
        fig.tight_layout(pad=2)

    def get_radrec_E(self, radrec_trans):
        for iz_trans in self.iz_transitions:
            if iz_trans.from_state.iz == radrec_trans.to_state.iz and iz_trans.to_state.iz == radrec_trans.from_state.iz:
                eps = iz_trans.thresh - radrec_trans.to_state.energy

        return eps

    def get_radrec_E_rates(self, per_z=False):
        Te = self.skrun.data['TEMPERATURE']
        ne = self.skrun.data['DENSITY']
        if per_z:
            radrec_E_rates = np.zeros((self.skrun.num_x, self.num_z))
            radrec_E_rates_max = np.zeros((self.skrun.num_x, self.num_z))
        else:
            radrec_E_rates = np.zeros(self.skrun.num_x)
            radrec_E_rates_max = np.zeros(self.skrun.num_x)

        # Calculate rad-rec rates
        for i in range(self.skrun.num_x):
            if self.opts['RAD_REC']:
                for radrec_trans in self.radrec_transitions:
                    alpha_radrec = max(radrec_trans.radrec_interp(
                        Te[i]), 0.0)
                    eps = self.get_radrec_E(radrec_trans) * spf.el_charge
                    rate = eps * self.skrun.n_norm * alpha_radrec * ne[i] * \
                        self.dens[i, radrec_trans.from_loc] / self.skrun.t_norm
                    rate_max = eps * self.skrun.n_norm * alpha_radrec * ne[i] * \
                        self.dens_max[i, radrec_trans.from_loc] / \
                        self.skrun.t_norm
                    if per_z:
                        z = radrec_trans.from_state.iz
                        radrec_E_rates[i, z] += rate
                        radrec_E_rates_max[i, z] += rate_max
                    else:
                        radrec_E_rates[i] += rate
                        radrec_E_rates_max[i] += rate_max
        return radrec_E_rates, radrec_E_rates_max

    def get_ex_E_rates(self):

        collrate_const = self.skrun.n_norm * self.skrun.v_th * \
            self.skrun.sigma_0 * self.skrun.t_norm

        Te = self.skrun.data['TEMPERATURE']
        ne = self.skrun.data['DENSITY']
        f0 = np.transpose(self.skrun.data['DIST_F'][0])
        f0 = f0[:, :]
        f0_max = get_maxwellians(
            self.skrun.num_x, ne, Te, self.skrun.vgrid, self.skrun.v_th, self.skrun.num_v)

        ex_E_rates = np.zeros(self.skrun.num_x)
        ex_E_rates_max = np.zeros(self.skrun.num_x)

        # Calculate collisional excitation rates
        for i in range(self.skrun.num_x):
            if self.opts['COLL_EX_DEEX']:
                for ex_trans in self.ex_transitions:
                    K_ex = collrate_const * rates.ex_rate(
                        f0[i, :],
                        self.skrun.vgrid/self.skrun.v_th,
                        self.skrun.dvc/self.skrun.v_th,
                        ex_trans.sigma
                    )
                    K_ex_max = collrate_const * rates.ex_rate(
                        f0_max[i, :],
                        self.skrun.vgrid/self.skrun.v_th,
                        self.skrun.dvc/self.skrun.v_th,
                        ex_trans.sigma
                    )
                    eps = ex_trans.thresh * spf.el_charge
                    ex_E_rates[i] += eps * self.skrun.n_norm * K_ex * \
                        self.dens[i, ex_trans.from_loc] / self.skrun.t_norm
                    ex_E_rates_max[i] += eps * self.skrun.n_norm * K_ex_max * \
                        self.dens_max[i, ex_trans.from_loc] / self.skrun.t_norm

        return ex_E_rates, ex_E_rates_max

    def get_deex_E_rates(self):

        collrate_const = self.skrun.n_norm * self.skrun.v_th * \
            self.skrun.sigma_0 * self.skrun.t_norm

        Te = self.skrun.data['TEMPERATURE']
        ne = self.skrun.data['DENSITY']
        f0 = np.transpose(self.skrun.data['DIST_F'][0])
        f0 = f0[:, :]
        f0_max = get_maxwellians(
            self.skrun.num_x, ne, Te, self.skrun.vgrid, self.skrun.v_th, self.skrun.num_v)

        deex_E_rates = np.zeros(self.skrun.num_x)
        deex_E_rates_max = np.zeros(self.skrun.num_x)

        # Calculate collisional deexcitation rates
        for i in range(self.skrun.num_x):
            if self.opts['COLL_EX_DEEX']:
                for ex_trans in self.ex_transitions:
                    eps = ex_trans.thresh / self.skrun.T_norm
                    statw_ratio = ex_trans.from_state.statw / ex_trans.to_state.statw
                    K_deex = collrate_const * rates.deex_rate(
                        f0[i, :],
                        self.skrun.vgrid/self.skrun.v_th,
                        self.skrun.dvc/self.skrun.v_th,
                        eps,
                        statw_ratio,
                        ex_trans.sigma
                    )
                    K_deex_max = collrate_const * rates.deex_rate(
                        f0_max[i, :],
                        self.skrun.vgrid/self.skrun.v_th,
                        self.skrun.dvc/self.skrun.v_th,
                        eps,
                        statw_ratio,
                        ex_trans.sigma
                    )
                    eps = ex_trans.thresh * spf.el_charge
                    deex_E_rates[i] += eps * self.skrun.n_norm * K_deex * \
                        self.dens[i, ex_trans.to_loc] / self.skrun.t_norm
                    deex_E_rates_max[i] += eps * self.skrun.n_norm * K_deex_max * \
                        self.dens_max[i, ex_trans.to_loc] / \
                        self.skrun.t_norm

        return deex_E_rates, deex_E_rates_max

    def get_spontem_E_rates(self, per_z=False):

        if per_z:
            spontem_E_rates = np.zeros((self.skrun.num_x, self.num_z))
            spontem_E_rates_max = np.zeros((self.skrun.num_x, self.num_z))
        else:
            spontem_E_rates = np.zeros(self.skrun.num_x)
            spontem_E_rates_max = np.zeros(self.skrun.num_x)

        # Calculate spontaneous emission rates
        for i in range(self.skrun.num_x):
            if self.opts['SPONT_EM']:
                for em_trans in self.spontem_transitions:
                    beta_spontem = em_trans.spontem_rate
                    eps = (em_trans.from_state.energy -
                           em_trans.to_state.energy) * spf.el_charge
                    rate = eps * self.skrun.n_norm * beta_spontem * \
                        self.dens[i, em_trans.from_loc] / self.skrun.t_norm
                    rate_max = eps * self.skrun.n_norm * beta_spontem * \
                        self.dens_max[i, em_trans.from_loc] / \
                        self.skrun.t_norm
                    if per_z:
                        z = em_trans.from_state.iz
                        spontem_E_rates[i, z] += rate
                        spontem_E_rates_max[i, z] += rate_max
                    else:
                        spontem_E_rates[i] += rate
                        spontem_E_rates_max[i] += rate_max

        return spontem_E_rates, spontem_E_rates_max

    def get_adas_ex_E_rates(self, per_z=False):

        if per_z:
            adas_ex_E_rates = np.zeros((self.skrun.num_x, self.num_z))
            adas_ex_E_rates_max = np.zeros((self.skrun.num_x, self.num_z))
        else:
            adas_ex_E_rates = np.zeros(self.skrun.num_x)
            adas_ex_E_rates_max = np.zeros(self.skrun.num_x)

        adas_plt, adas_eff_plt_max, adas_eff_plt = self.get_adas_PLT()

        # Calculate ADAS excitation radiation rates
        ne = self.skrun.data['DENSITY']
        if per_z is False:
            adas_ex_E_rates[:] = adas_eff_plt * \
                ne * np.sum(self.dens, 1)
            adas_ex_E_rates_max[:] = adas_eff_plt_max * \
                ne * np.sum(self.dens_max, 1)
        else:
            z_dens, z_dens_max, _ = self.get_z_dens()
            for z in range(self.num_z-1):
                for i in range(self.skrun.num_x):
                    adas_ex_E_rates[i, z] += adas_plt[i, z] * \
                        z_dens[i, z] * ne[i]
                    adas_ex_E_rates[i, z] += adas_plt[i, z] * \
                        z_dens_max[i, z] * ne[i]

        adas_ex_E_rates *= (self.skrun.n_norm ** 2)
        adas_ex_E_rates_max *= (self.skrun.n_norm ** 2)

        return adas_ex_E_rates, adas_ex_E_rates_max

    def get_ion_rates(self, per_z=False):
        collrate_const = self.skrun.n_norm * self.skrun.v_th * \
            self.skrun.sigma_0 * self.skrun.t_norm

        Te = self.skrun.data['TEMPERATURE']
        ne = self.skrun.data['DENSITY']
        f0 = np.transpose(self.skrun.data['DIST_F'][0])
        f0 = f0[:, :]
        f0_max = get_maxwellians(
            self.skrun.num_x, ne, Te, self.skrun.vgrid, self.skrun.v_th, self.skrun.num_v)
        if per_z:
            ion_rates = np.zeros((self.skrun.num_x, self.num_z))
            ion_rates_max = np.zeros((self.skrun.num_x, self.num_z))
        else:
            ion_rates = np.zeros(self.skrun.num_x)
            ion_rates_max = np.zeros(self.skrun.num_x)

        # Calculate collisional excitation rates
        for i in range(self.skrun.num_x):
            if self.opts['COLL_ION_REC']:
                for iz_trans in self.iz_transitions:
                    K_ion = collrate_const * rates.ex_rate(
                        f0[i, :],
                        self.skrun.vgrid/self.skrun.v_th,
                        self.skrun.dvc/self.skrun.v_th,
                        iz_trans.sigma
                    )
                    K_ion_max = collrate_const * rates.ex_rate(
                        f0_max[i, :],
                        self.skrun.vgrid/self.skrun.v_th,
                        self.skrun.dvc/self.skrun.v_th,
                        iz_trans.sigma
                    )
                    # eps = ex_trans.thresh * spf.el_charge
                    rate = self.skrun.n_norm * K_ion * \
                        self.dens[i, iz_trans.from_loc] / self.skrun.t_norm
                    rate_max = self.skrun.n_norm * K_ion_max * \
                        self.dens_max[i, iz_trans.from_loc] / \
                        self.skrun.t_norm
                    if per_z:
                        z = iz_trans.from_state.iz
                        ion_rates[i, z] += rate
                        ion_rates_max[i, z] += rate_max
                    else:
                        ion_rates[i] += rate
                        ion_rates_max[i] += rate_max

        return ion_rates, ion_rates_max

    def get_PLT(self):
        spontem_E_rates, spontem_E_rates_max = self.get_spontem_E_rates(
            per_z=True)

        z_dens, z_dens_max, z_dens_saha = self.get_z_dens()
        PLT = np.zeros((self.skrun.num_x, self.num_z))
        PLT_max = np.zeros((self.skrun.num_x, self.num_z))
        eff_PLT = np.zeros((self.skrun.num_x))
        eff_PLT_max = np.zeros((self.skrun.num_x))

        for z in range(self.num_z-1):
            PLT[:, z] = (spontem_E_rates[:, z]) / (z_dens[:, z] *
                                                   self.skrun.data['DENSITY'] * self.skrun.n_norm * self.skrun.n_norm)
            PLT_max[:, z] = (spontem_E_rates_max[:, z]) / (z_dens_max[:, z] *
                                                           self.skrun.data['DENSITY'] * self.skrun.n_norm * self.skrun.n_norm)
            eff_PLT[:] += (PLT[:, z] * z_dens[:, z]) / np.sum(z_dens, 1)
            eff_PLT_max[:] += (PLT_max[:, z] *
                               z_dens_max[:, z]) / np.sum(z_dens, 1)
            # PLT[z_dens[:, z] < 1e-16] = np.nan
            # PLT_max[z_dens_max[:, z] < 1e-16] = np.nan

        return PLT, PLT_max, eff_PLT, eff_PLT_max

    def get_adas_PLT(self):
        if self.name == 'C':
            adas_plt_file = aurora.adas_file('imp_data/Carbon/plt96_c.dat')

        # Interpolate adas data to skrun profile
        plt_interp = np.zeros([self.skrun.num_x, self.num_z-1])
        for z in range(self.num_z-1):
            adas_file_interp = interpolate.interp2d(
                adas_plt_file.logNe, adas_plt_file.logT, adas_plt_file.data[z], kind='cubic')
            for i in range(self.skrun.num_x):
                log_ne = np.log10(
                    1e-6 * (self.skrun.data['DENSITY'][i] * self.skrun.n_norm))
                log_Te = np.log10(
                    (self.skrun.data['TEMPERATURE'][i] * self.skrun.T_norm))
                interp_result = adas_file_interp(log_ne, log_Te)
                plt_interp[i, z] = 1e-6 * \
                    (10 ** interp_result[0])

        # Caluclate effective PLT
        z_dens, z_dens_max, _ = self.get_z_dens()
        adas_eff_plt_max = np.zeros(self.skrun.num_x)
        adas_eff_plt = np.zeros(self.skrun.num_x)
        for z in range(self.num_z-1):
            adas_eff_plt_max[:] += (plt_interp[:, z] *
                                    z_dens_max[:, z]) / np.sum(z_dens, 1)
            adas_eff_plt[:] += (plt_interp[:, z] *
                                z_dens[:, z]) / np.sum(z_dens, 1)

        return plt_interp, adas_eff_plt_max, adas_eff_plt

    def plot_PLT(self, compare_adas=False, plot_eff=True, plot_sk=True, plot_max=True, plot_stages=False):
        fig, ax = plt.subplots(1)

        PLT, PLT_max, eff_PLT, eff_PLT_max = self.get_PLT()
        if compare_adas:
            adas_PLT, adas_eff_PLT_max, adas_eff_PLT = self.get_adas_PLT()

        colours = ['orange', 'green', 'blue', 'cyan', 'brown', 'pink']
        if plot_stages:
            for z in range(self.num_z-1):
                if plot_max:
                    ax.plot(self.skrun.data['TEMPERATURE'] *
                            self.skrun.T_norm, PLT_max[:, z], color=colours[z], label=self.name + '$^{' + str(z) + '+}$')
                if plot_sk:
                    ax.plot(self.skrun.data['TEMPERATURE'] * self.skrun.T_norm, PLT[:, z], '--',
                            color=colours[z])
        if plot_sk and plot_eff:
            ax.plot(self.skrun.data['TEMPERATURE'] * self.skrun.T_norm, eff_PLT[:], '--',
                    color='black', label='effective PLT (SK)')
        if plot_max and plot_eff:
            ax.plot(self.skrun.data['TEMPERATURE'] * self.skrun.T_norm, eff_PLT_max[:], '-',
                    color='black', label='effective PLT (Max)')

        if compare_adas:
            if plot_stages:
                for z in range(self.num_z-1):
                    if plot_max:
                        ax.plot(self.skrun.data['TEMPERATURE'] *
                                self.skrun.T_norm, adas_PLT[:, z], linestyle=(0, (1, 1)), color=colours[z])
            if plot_sk and plot_eff:
                ax.plot(self.skrun.data['TEMPERATURE'] * self.skrun.T_norm, adas_eff_PLT[:], '--',
                        color='grey', label='ADAS effective PLT (SK)')
            if plot_max and plot_eff:
                ax.plot(self.skrun.data['TEMPERATURE'] * self.skrun.T_norm, adas_eff_PLT_max[:], linestyle=(0, (1, 1)),
                        color='grey', label='ADAS effective PLT (Max)')

        ax.legend()
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('$T_e$ [eV]')
        ax.set_ylabel('Carbon excitation radiation [Wm$^{3}$]')
        # ax.set_ylim([1e-36, None])
        # ax.set_xlim([1, 100])

    def get_q_rad(self, ex_only=False, compare_adas=False):

        spontem_E_rates, spontem_E_rates_max = self.get_spontem_E_rates()
        q_spontem = np.sum(self.skrun.dxc[::2] * 1e-6*spontem_E_rates[::2])
        q_spontem_max = np.sum(
            self.skrun.dxc[::2] * 1e-6*spontem_E_rates_max[::2])

        if ex_only is False:
            radrec_E_rates, radrec_E_rates_max = self.get_radrec_E_rates()
            q_radrec = np.sum(self.skrun.dxc[::2] * 1e-6*radrec_E_rates[::2])
            q_radrec_max = np.sum(
                self.skrun.dxc[::2] * 1e-6*radrec_E_rates_max[::2])

            q_rad = q_radrec + q_spontem
            q_rad_max = q_radrec_max + q_spontem_max

        else:
            q_rad = q_spontem
            q_rad_max = q_spontem_max

        if compare_adas:
            adas_ex_E_rates, adas_ex_E_rates_max = self.get_adas_ex_E_rates()
            q_ex_adas = np.sum(self.skrun.dxc[::2] * 1e-6*adas_ex_E_rates[::2])
            q_ex_adas_max = np.sum(
                self.skrun.dxc[::2] * 1e-6*adas_ex_E_rates_max[::2])
            return q_rad, q_rad_max, q_ex_adas, q_ex_adas_max
        else:
            return q_rad, q_rad_max

    def plot_radiation(self, log=False):

        radrec_E_rates, radrec_E_rates_max = self.get_radrec_E_rates()
        spontem_E_rates, spontem_E_rates_max = self.get_spontem_E_rates()

        fig, ax = plt.subplots(1)

        ax.plot(self.skrun.xgrid, 1e-6*radrec_E_rates_max, '-',
                color='blue', label='rad-rec (Max)')
        ax.plot(self.skrun.xgrid, 1e-6*radrec_E_rates, '--',
                color='blue', label='rad-rec (SK)')
        ax.plot(self.skrun.xgrid, 1e-6*spontem_E_rates_max, '-',
                color='red', label='spont-em (Max)')
        ax.plot(self.skrun.xgrid, 1e-6*spontem_E_rates, '--',
                color='red', label='spont-em (SK)')

        q_radrec = np.sum(self.skrun.dxc[::2] * 1e-6*radrec_E_rates[::2])
        q_radrec_max = np.sum(
            self.skrun.dxc[::2] * 1e-6*radrec_E_rates_max[::2])
        q_spontem = np.sum(self.skrun.dxc[::2] * 1e-6*spontem_E_rates[::2])
        q_spontem_max = np.sum(
            self.skrun.dxc[::2] * 1e-6*spontem_E_rates_max[::2])

        print('q_imp (SOL-KiT $f_0$) = ',
              '{:.2f}MW/m^2'.format(q_radrec + q_spontem))
        print('q_imp (Maxwellian $f_0$) = ',
              '{:.2f}MW/m^2'.format(q_radrec_max + q_spontem_max))

        ax.legend()
        ax.set_xlabel('x [m]')
        if log:
            ax.set_yscale('log')
        ax.set_ylabel('Radiatiative power loss [MWm$^{-3}$]')

    def plot_Zdist(self, cells=-1):
        if isinstance(cells, int):
            cells = [cells]
            cell_pref = ['']
        else:
            cell_pref = []
            for i in range(len(cells)):
                cell_pref.append('Cell ' + str(cells[i]) + ', ')

        fig, ax = plt.subplots(1)
        min_z = 0
        max_z = self.num_z
        zs = [i for i in range(min_z, max_z)]
        for i, cell in enumerate(cells):
            ax.plot(zs, self.dens_max[cell, min_z:max_z],
                    label=cell_pref[i] + 'Maxwellian $f_0$', color='black', alpha=1.0-0.2*i)
            ax.plot(zs, self.dens[cell, min_z:max_z],
                    '--', label=cell_pref[i] + 'SOL-KiT $f_0$', color='red', alpha=1.0-0.2*i)
            ax.plot(zs, self.dens_saha[cell, min_z:max_z],
                    '--', label=cell_pref[i] + 'Saha equilibrium', color='blue', alpha=1.0-0.2*i)
        ax.grid()
        ax.legend()
        ax.set_yscale('log')
        ax.set_xlabel('$Z$')
        ax.set_ylabel('$n_Z / n_0$')
        ax.set_title('Impurity state densities, ' + self.longname)


def get_maxwellians(num_x, ne, Te, vgrid, v_th, num_v):
    f0_max = np.zeros([num_x, num_v])
    vgrid = vgrid / v_th
    for i in range(num_x):
        f0_max[i, :] = spf.maxwellian(Te[i], ne[i], vgrid)
    return f0_max


def interp_2d(x, y, z, xnew, ynew):
    pass
