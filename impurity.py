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
import atomic_state


class Impurity:
    def __init__(self, name, opts, sktrun=None, sk_timestep=-1, Te=None, ne=None, xgrid=None, dxc=None):
        self.name = name
        self.opts = opts
        if sktrun is not None:
            self.load_from_solkit(sktrun, sk_timestep)
        elif Te is not None and ne is not None:
            self.load_from_arrays(Te, ne, xgrid, dxc)
        if self.name == 'C':
            self.nuc_chg = 6
            self.num_z = 7
            self.longname = 'Carbon'
        elif self.name == 'W':
            self.nuc_chg = 74
            self.num_z = 11
            self.longname = 'Tungsten'
        self.load_states()
        self.load_transitions()
        self.init_dens()

    def load_from_solkit(self, sktrun, sk_timestep):
        # Load in plasma profile
        sktrun.load_tvars(['TEMPERATURE', 'DENSITY'], normalised=True)
        self.Te = sktrun.tdata['TEMPERATURE'][sk_timestep]
        self.ne = sktrun.tdata['DENSITY'][sk_timestep]

        # Read grids and norms
        self.vgrid = sktrun.vgrid
        self.dvc = sktrun.dvc
        self.num_v = sktrun.num_v
        self.xgrid = sktrun.xgrid
        self.dxc = sktrun.dxc
        self.num_x = sktrun.num_x
        self.v_th = sktrun.v_th
        self.T_norm = sktrun.T_norm
        self.n_norm = sktrun.n_norm
        self.t_norm = sktrun.t_norm
        self.x_norm = sktrun.x_norm
        self.sigma_0 = sktrun.sigma_0

        # Load in f0 and generate equivalent Maxwellians
        sktrun.load_tvars('DIST_F', normalised=True)
        self.f0 = np.transpose(sktrun.tdata['DIST_F'][sk_timestep, 0, :, :])
        self.f0_max = get_maxwellians(
            self.num_x, self.ne, self.Te, self.vgrid, self.v_th, self.num_v)

    def load_from_arrays(self, Te, ne, xgrid=None, dxc=None):

        # Define normalisation
        self.T_norm = 10.0
        self.n_norm = 1.0e+20
        z = 1
        self.sigma_0 = 8.797355066696007e-21

        # Load in plasma profile
        self.Te = Te / self.T_norm
        self.ne = ne / self.n_norm

        # Generate grids and norms
        self.num_v = 80
        dv = 0.05
        vgrid_mult = 1.025
        self.vgrid = np.zeros(self.num_v)
        self.dvc = np.zeros(self.num_v)
        self.vgrid[0] = dv / 2
        self.dvc[0] = dv
        for i in range(1, self.num_v):
            self.dvc[i] = vgrid_mult * self.dvc[i-1]
            self.vgrid[i] = self.vgrid[i-1] + 0.5*(self.dvc[i-1]+self.dvc[i])

        if xgrid is None:
            self.num_x = len(Te)
            length = 10
            self.xgrid = np.linspace(0, length, self.num_x)
            self.dxc = np.array(
                [length / self.num_x for _ in range(self.num_x)])
        else:
            self.num_x = len(xgrid)
            self.xgrid = xgrid
            self.dxc = dxc

        self.v_th = np.sqrt(2 * self.T_norm * spf.el_charge / spf.el_mass)
        self.vgrid *= self.v_th
        self.dvc *= self.v_th
        gamma_ee_0 = spf.el_charge ** 4 / \
            (4 * np.pi * (spf.el_mass * spf.epsilon_0) ** 2)
        gamma_ei_0 = z * gamma_ee_0
        self.t_norm = self.v_th ** 3 / \
            (gamma_ei_0 * self.n_norm *
             spf.lambda_ei(1.0, 1.0, self.T_norm, self.n_norm, z)/z)
        self.x_norm = self.v_th * self.t_norm

        # Load in f0 and generate equivalent Maxwellians
        self.f0_max = get_maxwellians(
            self.num_x, self.ne, self.Te, self.vgrid, self.v_th, self.num_v)
        self.f0 = self.f0_max

    def load_states(self):
        self.states = []
        statedata_file = os.path.join(os.path.dirname(__file__),
                                      'imp_data', self.longname, 'states.txt')
        with open(statedata_file) as f:
            lines = f.readlines()
            i = 0
            for l in lines[1:]:
                line_data = l.split('\t')
                line_data[-1] = line_data[-1].strip('\n')
                iz = int(line_data[0])
                lower_shells = line_data[1]
                statename = line_data[2]
                statw = int(line_data[5])
                energy = float(line_data[6])
                I_0 = float(line_data[7])
                metastable = int(line_data[8])
                mask = bool(int(line_data[9]))
                if iz < self.num_z and not mask:

                    if self.opts['MODELLED_STATES'] == 'ground':
                        if energy == 0.0:
                            self.states.append(
                                atomic_state.State(self.nuc_chg, iz, lower_shells, statename, i, statw, energy, I_0, metastable))
                            i += 1

                    elif self.opts['MODELLED_STATES'] == 'metastable':
                        if energy == 0.0:
                            self.states.append(
                                atomic_state.State(self.nuc_chg, iz, lower_shells, statename, i, statw, energy, I_0, metastable))
                            i += 1
                        elif metastable == 1:
                            self.states.append(
                                atomic_state.State(self.nuc_chg, iz, lower_shells, statename, i, statw, energy, I_0, metastable))
                            i += 1

                    elif self.opts['MODELLED_STATES'] == 'all':
                        self.states.append(
                            atomic_state.State(self.nuc_chg, iz, lower_shells, statename, i, statw, energy, I_0, metastable))
                        i += 1

        self.tot_states = len(self.states)
        for state in self.states:
            state.get_shell_iz_energies(self.states)

    def load_transitions(self):

        self.iz_transitions = []
        self.ex_transitions = []
        self.radrec_transitions = []
        self.spontem_transitions = []

        trans_file = os.path.join(os.path.dirname(__file__),
                                  'imp_data', self.longname, 'transitions.txt')
        with open(trans_file) as f:
            lines = f.readlines()
            for l in lines[1:]:
                line_data = l.split('\t')
                line_data[-1] = line_data[-1].strip('\n')
                trans_type = line_data[0]

                from_iz = int(line_data[1])
                from_statename = line_data[2]
                from_state = self.get_state(from_iz, from_statename)

                to_iz = int(line_data[3])
                to_statename = line_data[4]
                to_state = self.get_state(to_iz, to_statename)

                if from_state is not None and to_state is not None:

                    dtype = line_data[5].strip('\n')

                    if trans_type == 'ionization' and self.opts['COLL_ION_REC'] and dtype != 'IGNORE':
                        self.iz_transitions.append(transition.Transition('ionization',
                                                                         self.longname,
                                                                         from_state,
                                                                         to_state,
                                                                         vgrid=self.vgrid/self.v_th,
                                                                         T_norm=self.T_norm,
                                                                         sigma_0=self.sigma_0,
                                                                         dtype=dtype,
                                                                         opts=self.opts))

                    if trans_type == 'radrec' and self.opts['RAD_REC']:
                        self.radrec_transitions.append(transition.Transition('radrec',
                                                                             self.longname,
                                                                             from_state,
                                                                             to_state,
                                                                             T_norm=self.T_norm,
                                                                             n_norm=self.n_norm,
                                                                             t_norm=self.t_norm,
                                                                             dtype=dtype))

                    if trans_type == 'excitation' and self.opts['COLL_EX_DEEX']:
                        self.ex_transitions.append(transition.Transition('excitation',
                                                                         self.longname,
                                                                         from_state,
                                                                         to_state,
                                                                         vgrid=self.vgrid/self.v_th,
                                                                         T_norm=self.T_norm,
                                                                         sigma_0=self.sigma_0,
                                                                         dtype=dtype))

        if self.opts['SPONT_EM']:
            self.load_spontem_transitions()

    def load_spontem_transitions(self):
        # Load spontem file data
        spontem_file = os.path.join(os.path.dirname(__file__),
                                    'imp_data', self.longname, 'nist_spontem.txt')
        # spontem_data = pd.read_csv(spontem_file, sep='\t')
        with open(spontem_file) as f:
            spontem_data = f.readlines()
        from_statenames = [[] for z in range(self.num_z)]
        to_statenames = [[] for z in range(self.num_z)]
        rates = [[] for z in range(self.num_z)]
        for l in spontem_data[1:]:
            data = l.split('\t')
            z = data[0]
            from_statenames[int(z)].append(data[1])
            to_statenames[int(z)].append(data[2])
            rates[int(z)].append(float(data[3].strip('/n')))

        # For each z, loop through all transitions
        for z in range(self.num_z):
            for i in range(len(from_statenames[z])):

                from_statename = from_statenames[z][i]
                to_statename = to_statenames[z][i]
                rate = rates[z][i]

                # Match from and to states
                from_state = None
                to_state = None
                for state in self.states:
                    if state.iz == z and state.statename == from_statename:
                        from_state = state
                    if state.iz == z and state.statename == to_statename:
                        to_state = state

                # Create transition object
                if from_state is not None and to_state is not None:
                    self.spontem_transitions.append(transition.Transition('spontem',
                                                                          self.longname,
                                                                          from_state,
                                                                          to_state,
                                                                          t_norm=self.t_norm,
                                                                          spontem_rate=rate))

    def get_state(self, iz, statename):
        for state in self.states:
            if state.iz == iz and state.statename == statename:
                return state

    def get_ground_state(self, z):
        for state in self.states:
            if state.iz == z:
                return state

    def init_dens(self):
        self.dens = np.zeros((self.num_x, self.tot_states))
        self.dens_max = np.zeros((self.num_x, self.tot_states))
        self.dens_saha = np.zeros((self.num_x, self.tot_states))
        self.dens[:, 0] = self.opts['FRAC_IMP_DENS'] * \
            self.ne
        self.dens_max[:, 0] = self.opts['FRAC_IMP_DENS'] * \
            self.ne
        self.tmp_dens = np.zeros((self.num_x, self.tot_states))

        if self.opts['COMPARE_ADAS']:
            self.dens_adas = np.zeros((self.num_x, self.tot_states))
            self.dens_adas[:, 0] = self.opts['FRAC_IMP_DENS'] * \
                self.ne

    def get_saha_eq(self):

        gs_locs = []
        for z in range(self.num_z):
            gs = self.get_ground_state(z)
            gs_locs.append(gs.loc)

        for i in range(self.num_x):
            de_broglie_l = np.sqrt(
                (spf.planck_h ** 2) / (2 * np.pi * spf.el_mass * spf.el_charge * self.T_norm * self.Te[i]))

            # Compute ratios
            dens_ratios = np.zeros(self.num_z - 1)
            for z in range(1, self.num_z):
                eps = self.iz_transitions[z-1].thresh
                dens_ratios[z-1] = (2 * (self.states[gs_locs[z-1]].statw / self.states[gs_locs[z]].statw) * np.exp(-eps / (self.Te[i]*self.T_norm))) / (
                    (self.ne[i] * self.n_norm) * (de_broglie_l ** 3))
            # Fill densities
            imp_dens_tot = np.sum(self.dens[i, :])
            denom_sum = 1.0 + np.sum([np.prod(dens_ratios[:z+1])
                                      for z in range(self.num_z-1)])
            self.dens_saha[i, 0] = imp_dens_tot / denom_sum
            for z in range(1, self.num_z):
                self.dens_saha[i, gs_locs[z]] = self.dens_saha[i,
                                                               gs_locs[z-1]] * dens_ratios[z-1]

    def build_rate_matrices(self):

        collrate_const = self.n_norm * self.v_th * \
            self.sigma_0 * self.t_norm
        # collrate_const = self.n_norm * self.v_th * self.t_norm
        op_mat = [np.zeros((self.tot_states, self.tot_states))
                  for _ in range(self.num_x)]
        rate_mat = [np.zeros((self.tot_states, self.tot_states))
                    for _ in range(self.num_x)]
        op_mat_max = [np.zeros((self.tot_states, self.tot_states))
                      for _ in range(self.num_x)]
        rate_mat_max = [np.zeros((self.tot_states, self.tot_states))
                        for _ in range(self.num_x)]
        if self.opts['COMPARE_ADAS']:
            op_mat_adas = [np.zeros((self.tot_states, self.tot_states))
                           for _ in range(self.num_x)]
            rate_mat_adas = [np.zeros((self.tot_states, self.tot_states))
                             for _ in range(self.num_x)]

        tbrec_norm = self.n_norm * np.sqrt((spf.planck_h ** 2) / (2 * np.pi *
                                                                  spf.el_mass * self.T_norm * spf.el_charge)) ** 3

        # Build kinetic rate matrices
        for i in range(self.num_x):

            if self.opts['COLL_ION_REC']:
                for iz_trans in self.iz_transitions:

                    # Ionization
                    K_ion = collrate_const * rates.ion_rate(
                        self.f0[i, :],
                        self.vgrid/self.v_th,
                        self.dvc/self.v_th,
                        iz_trans.sigma
                    )
                    K_ion_max = collrate_const * rates.ion_rate(
                        self.f0_max[i, :],
                        self.vgrid/self.v_th,
                        self.dvc/self.v_th,
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
                    eps = iz_trans.thresh / self.T_norm
                    statw_ratio = iz_trans.to_state.statw / iz_trans.from_state.statw
                    K_rec = self.ne[i] * collrate_const * tbrec_norm * rates.tbrec_rate(
                        self.f0[i, :],
                        self.Te[i],
                        self.vgrid/self.v_th,
                        self.dvc/self.v_th,
                        eps,
                        statw_ratio,
                        iz_trans.sigma)
                    K_rec_max = self.ne[i] * collrate_const * tbrec_norm * rates.tbrec_rate(
                        self.f0_max[i, :],
                        self.Te[i],
                        self.vgrid/self.v_th,
                        self.dvc/self.v_th,
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
                        self.Te[i]), 0.0)
                    # ...loss
                    row = radrec_trans.from_loc
                    col = radrec_trans.from_loc
                    rate_mat[i][row, col] += -(self.ne[i] * alpha_radrec)
                    rate_mat_max[i][row, col] += -(self.ne[i] * alpha_radrec)
                    # ...gain
                    row = radrec_trans.to_loc
                    col = radrec_trans.from_loc
                    rate_mat[i][row, col] += (self.ne[i] * alpha_radrec)
                    rate_mat_max[i][row, col] += (self.ne[i] * alpha_radrec)

            if self.opts['COLL_EX_DEEX']:
                for ex_trans in self.ex_transitions:

                    # Excitation
                    K_ex = collrate_const * rates.ex_rate(
                        self.f0[i, :],
                        self.vgrid/self.v_th,
                        self.dvc/self.v_th,
                        ex_trans.sigma
                    )
                    K_ex_max = collrate_const * rates.ex_rate(
                        self.f0_max[i, :],
                        self.vgrid/self.v_th,
                        self.dvc/self.v_th,
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
                    eps = ex_trans.thresh / self.T_norm
                    statw_ratio = ex_trans.from_state.statw / ex_trans.to_state.statw
                    K_deex = collrate_const * rates.deex_rate(
                        self.f0[i, :],
                        self.vgrid/self.v_th,
                        self.dvc/self.v_th,
                        eps,
                        statw_ratio,
                        ex_trans.sigma
                    )
                    K_deex_max = collrate_const * rates.deex_rate(
                        self.f0_max[i, :],
                        self.vgrid/self.v_th,
                        self.dvc/self.v_th,
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

            if self.opts['COMPARE_ADAS']:

                # Ionization and recombination
                adas_iz_coeffs = self.get_adas_iz_coeffs()
                adas_rec_coeffs = self.get_adas_rec_coeffs()
                for iz_trans in self.iz_transitions:

                    # Check that transition is ground state to ground state
                    from_state = iz_trans.from_state
                    to_state = iz_trans.to_state
                    gs_from = self.get_ground_state(from_state.iz)
                    gs_to = self.get_ground_state(to_state.iz)
                    if from_state.equals(gs_from) and to_state.equals(gs_to):

                        # Ionization
                        K_ion = adas_iz_coeffs[i, from_state.iz] * \
                            self.ne[i] * self.n_norm * self.t_norm
                        # ...loss
                        row = iz_trans.from_loc
                        col = iz_trans.from_loc
                        rate_mat_adas[i][row, col] += -K_ion
                        # ...gain
                        row = iz_trans.to_loc
                        col = iz_trans.from_loc
                        rate_mat_adas[i][row, col] += K_ion

                        # Recombination
                        K_rec = adas_rec_coeffs[i, from_state.iz] * \
                            self.ne[i] * self.n_norm * self.t_norm
                        # ...loss
                        row = iz_trans.to_loc
                        col = iz_trans.to_loc
                        rate_mat_adas[i][row, col] += -K_rec
                        # ...gain
                        row = iz_trans.from_loc
                        col = iz_trans.to_loc
                        rate_mat_adas[i][row, col] += K_rec

                op_mat_adas[i] = np.linalg.inv(np.identity(
                    self.tot_states) - self.opts['DELTA_T'] * rate_mat_adas[i])

        self.op_mat = op_mat
        self.rate_mat = rate_mat
        self.op_mat_max = op_mat_max
        self.rate_mat_max = rate_mat_max
        if self.opts['COMPARE_ADAS']:
            self.op_mat_adas = op_mat_adas
            self.rate_mat_adas = rate_mat_adas

    def solve(self):
        for i in range(self.num_x):
            loc_mat = self.rate_mat[i]
            loc_mat[-1, :] = 1.0
            rhs = np.zeros(self.tot_states)
            rhs[-1] = np.sum(self.dens[i, :])
            loc_mat_inv = np.linalg.inv(loc_mat)
            dens = loc_mat_inv.dot(rhs)
            self.dens[i, :] = dens.copy()

        for i in range(self.num_x):
            loc_mat = self.rate_mat_max[i]
            loc_mat[-1, :] = 1.0
            rhs = np.zeros(self.tot_states)
            rhs[-1] = np.sum(self.dens_max[i, :])
            loc_mat_inv = np.linalg.inv(loc_mat)
            dens = loc_mat_inv.dot(rhs)
            self.dens_max[i, :] = dens.copy()

        if self.opts['COMPARE_ADAS']:
            for i in range(self.num_x):

                loc_mat = self.rate_mat_adas[i]
                loc_mat[-1, :] = 1.0
                rhs = np.zeros(self.tot_states)
                rhs[-1] = np.sum(self.dens_adas[i, :])
                loc_mat_inv = np.linalg.inv(loc_mat)
                dens = loc_mat_inv.dot(rhs)
                self.dens_adas[i, :] = dens.copy()

    def evolve(self):

        # Compute the particle source for each ionization state
        for i in range(self.num_x):

            # Calculate new densities
            self.tmp_dens[i, :] = self.op_mat[i].dot(self.dens[i, :])

        residual = np.max(np.abs(self.tmp_dens - self.dens))
        self.dens = self.tmp_dens.copy()

        # Compute the particle source for each ionization state (Maxwellian)
        for i in range(self.num_x):

            # Calculate new densities
            self.tmp_dens[i, :] = self.op_mat_max[i].dot(self.dens_max[i, :])

        residual_max = np.max(np.abs(self.tmp_dens - self.dens_max))
        self.dens_max = self.tmp_dens.copy()

        # Compute the particle source for each ionization state (ADAS)
        if self.opts['COMPARE_ADAS']:
            for i in range(self.num_x):

                # Calculate new densities
                self.tmp_dens[i, :] = self.op_mat_adas[i].dot(
                    self.dens_adas[i, :])

            residual_adas = np.max(np.abs(self.tmp_dens - self.dens_adas))
            self.dens_adas = self.tmp_dens.copy()

            return max(residual, residual_max, residual_adas)
        else:
            return max(residual, residual_max)

    def get_Zeff(self):

        self.Zeff = np.zeros(self.num_x)
        self.Zeff_max = np.zeros(self.num_x)
        self.Zeff_saha = np.zeros(self.num_x)
        dens_tot = np.sum(self.dens, 1)
        dens_tot_max = np.sum(self.dens_max, 1)
        dens_tot_saha = np.sum(self.dens_saha, 1)

        for i in range(self.num_x):

            for k, state in enumerate(self.states):
                z = state.iz
                k = state.loc
                self.Zeff[i] += self.dens[i, k] * (float(z) ** 2)
                self.Zeff_max[i] += self.dens_max[i, k] * (float(z) ** 2)
                self.Zeff_saha[i] += self.dens_saha[i, k] * (float(z) ** 2)
            if dens_tot[i] > 0:
                self.Zeff[i] = self.Zeff[i] / self.ne[i]
            if dens_tot_max[i] > 0:
                self.Zeff_max[i] = self.Zeff_max[i] / \
                    self.ne[i]
            if dens_tot_saha[i] > 0:
                self.Zeff_saha[i] = self.Zeff_saha[i] / \
                    self.ne[i]

    def get_Zavg(self, compare_saha=False):

        self.Zavg = np.zeros(self.num_x)
        self.Zavg_max = np.zeros(self.num_x)
        self.Zavg_saha = np.zeros(self.num_x)
        dens_tot = np.sum(self.dens, 1)
        dens_tot_max = np.sum(self.dens_max, 1)
        dens_tot_saha = np.sum(self.dens_saha, 1)
        if self.opts['COMPARE_ADAS']:
            self.Zavg_adas = np.zeros(self.num_x)
            dens_tot_adas = np.sum(self.dens_adas, 1)

        for i in range(self.num_x):

            for k, state in enumerate(self.states):
                z = state.iz
                k = state.loc
                self.Zavg[i] += self.dens[i, k] * float(z)
                self.Zavg_max[i] += self.dens_max[i, k] * float(z)
                self.Zavg_saha[i] += self.dens_saha[i, k] * float(z)
                if self.opts['COMPARE_ADAS']:
                    self.Zavg_adas[i] += self.dens_adas[i, k] * float(z)
            if dens_tot[i] > 0:
                self.Zavg[i] = self.Zavg[i] / dens_tot[i]
            if dens_tot_max[i] > 0:
                self.Zavg_max[i] = self.Zavg_max[i] / dens_tot_max[i]
            if dens_tot_saha[i] > 0:
                self.Zavg_saha[i] = self.Zavg_saha[i] / dens_tot_saha[i]
            if self.opts['COMPARE_ADAS'] and dens_tot_adas[i] > 0:
                self.Zavg_adas[i] = self.Zavg_adas[i] / dens_tot_adas[i]

    def plot_Zeff(self, plot_saha=False):
        self.get_Zeff()
        fig, ax = plt.subplots(1)
        ax.plot(self.xgrid[:-1], self.Zeff_max[:-1],
                label='Maxwellian $f_0$', color='black')
        if plot_saha:
            ax.plot(self.xgrid[:-1], self.Zeff_saha[:-1], '--',
                    label='Saha equilibrium', color='blue')
        ax.plot(self.xgrid[:-1], self.Zeff[:-1], '--',
                label='SOL-KiT $f_0$', color='red')
        ax.grid()
        ax.legend()
        ax.set_xlabel('x [m]')
        ax.set_ylabel('$Z_{eff}=\Sigma_i Z^2_i n_{Z}^i / n_e$')
        ax.set_title('$Z_{eff}$ profile, ' + self.longname)
        # ax.set_xlim([10.4, 11.8])

    def plot_Zavg(self, xaxis='Te', logx=False, plot_saha=False, plot_sk=True, compare_adas=False):
        self.get_Zavg()
        fig, ax = plt.subplots(1)
        if xaxis == 'x':
            x = self.xgrid[:-1]
            ax.set_xlabel('x [m]')
        elif xaxis == 'Te':
            x = self.Te[:-1] * self.T_norm
            ax.set_xlabel('Te [eV]')
        elif xaxis == 'distance_from_sheath':
            x = self.xgrid[-1] - self.xgrid[:-1]
            ax.set_xlabel('Distance from sheath [m]')
        if logx:
            ax.set_xscale('log')
        ax.plot(x, self.Zavg_max[:-1],
                label='Maxwellian $f_0$', color='black')
        if plot_saha:
            ax.plot(x, self.Zavg_saha[:-1], '--',
                    label='Saha equilibrium', color='green')
        if plot_sk:
            ax.plot(x, self.Zavg[:-1], '--',
                    label='SOL-KiT $f_0$', color='red')
        if compare_adas and self.opts['COMPARE_ADAS']:
            ax.plot(x, self.Zavg_adas[:-1], '--',
                    label='ADAS', color='blue')
        ax.grid()
        ax.legend()
        ax.set_ylabel('$Z_{avg}=\Sigma_i Z_i n_{Z}^i / n_Z^{tot}$')
        ax.set_title('$Z_{avg}$ profile, ' + self.longname)
        # ax.set_xlim([10.4, 11.8])

    def plot_dens(self, xaxis='x', plot_kin=True, plot_max=True, plot_saha=False, compare_adas=False, normalise=False, logx=False):
        fig, ax = plt.subplots(1)
        # colours = ['orange', 'green', 'blue',
        #            'cyan', 'brown', 'pink', 'red']
        cmap = plt.cm.get_cmap('Paired')
        z_dens, z_dens_max, z_dens_saha = self.get_z_dens()
        if compare_adas:
            z_dens_adas = self.get_z_dens_adas()
        if normalise:
            for i in range(self.num_x):
                z_dens[i, :] = z_dens[i, :] / np.sum(z_dens[i, :])
                z_dens_max[i, :] = z_dens_max[i, :] / np.sum(z_dens_max[i, :])
                z_dens_saha[i, :] = z_dens_saha[i, :] / \
                    np.sum(z_dens_saha[i, :])
                if compare_adas:
                    z_dens_adas[i, :] = z_dens_adas[i, :] / \
                        np.sum(z_dens_adas[i, :])

        else:
            z_dens = z_dens * self.n_norm
            z_dens_max = z_dens_max * self.n_norm
            z_dens_saha = z_dens_saha * self.n_norm
            if compare_adas:
                z_dens_adas = z_dens_adas * self.n_norm

        min_z = 0
        max_z = self.num_z
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(
            vmin=min_z, vmax=max_z))
        legend_lines = []
        legend_labels = []
        if xaxis == 'x':
            x = self.xgrid[:-1]
            ax.set_xlabel('x [m]')
        elif xaxis == 'Te':
            x = self.Te[:-1] * self.T_norm
            ax.set_xlabel('$T_e$ [eV]')
        else:
            x = self.xgrid[-1] - self.xgrid[:-1]
            ax.set_xlabel('Distance from sheath [m]')
        if logx:
            ax.set_xscale('log')
        for z in range(min_z, max_z):
            if plot_kin:
                ax.plot(x, z_dens[:-1, z], '--', color=sm.to_rgba(z))
                # ax.plot(x, z_dens[:-1, z], '--', color=colours[z])
            if plot_max:
                ax.plot(x, z_dens_max[:-1, z], color=sm.to_rgba(z))
                # ax.plot(x, z_dens_max[:-1, z], color=colours[z])
            if plot_saha:
                ax.plot(x, z_dens_saha[:-1, z], '-.', color=sm.to_rgba(z))
                # ax.plot(x, z_dens_saha[:-1, z], '-.', color=colours[z])
            if compare_adas:
                ax.plot(x, z_dens_adas[:-1, z], '-.', color=sm.to_rgba(z))
                # ax.plot(x, z_dens_adas[:-1, z], '-.', color=colours[z])
            legend_labels.append('$Z=$' + str(z))
            legend_lines.append(Line2D([0], [0], color=sm.to_rgba(z)))
            # legend_lines.append(Line2D([0], [0], color=colours[z]))
        if plot_max:
            legend_lines.append(Line2D([0], [0], linestyle='-', color='black'))
        if plot_kin:
            legend_lines.append(
                Line2D([0], [0], linestyle='--', color='black'))
        if plot_saha:
            legend_lines.append(
                Line2D([0], [0], linestyle='-.', color='black'))
        if compare_adas:
            legend_lines.append(
                Line2D([0], [0], linestyle='-.', color='black'))
        if plot_max:
            legend_labels.append('Maxwellian $f_0$')
        if plot_kin:
            legend_labels.append('SOL-KiT $f_0$')
        if plot_saha:
            legend_labels.append('Saha equilibrium')
        if compare_adas:
            legend_labels.append('ADAS equilibrium')
        ax.legend(legend_lines, legend_labels)
        ax.grid()
        if normalise:
            ax.set_ylabel('$n_Z / n_Z^{tot}$')
        else:
            ax.set_ylabel('$n_Z$')
        ax.set_title('$n_Z$ profiles, ' + self.longname)

    def get_z_dens(self):
        z_dens = np.zeros([self.num_x, self.num_z])
        z_dens_max = np.zeros([self.num_x, self.num_z])
        z_dens_saha = np.zeros([self.num_x, self.num_z])
        for state in self.states:
            z = state.iz
            z_dens[:, z] += self.dens[:, state.loc]
            z_dens_max[:, z] += self.dens_max[:, state.loc]
            z_dens_saha[:, z] += self.dens_saha[:, state.loc]
        return z_dens, z_dens_max, z_dens_saha

    def get_z_dens_adas(self):
        z_dens_adas = np.zeros([self.num_x, self.num_z])
        for state in self.states:
            z = state.iz
            z_dens_adas[:, z] += self.dens_adas[:, state.loc]
        return z_dens_adas

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
        Te = self.Te[cell] * self.T_norm
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

    def plot_atomdist(self, z=0, cells=-1, gnormalise=False, znormalise=True):
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
        if znormalise:
            zdens, zdens_max, zdens_saha = self.get_z_dens()
        for cell in cells:
            locstate_dens = allstates_dens[z][cell, :]
            locstate_dens_max = allstates_dens_max[z][cell, :]
            boltz_dens = self.get_boltzmann_dist(
                locstate_dens, z, cells, gnormalise)
            if gnormalise:
                locstate_dens = self.gnormalise(locstate_dens, z)
                locstate_dens_max = self.gnormalise(locstate_dens_max, z)

            if znormalise:
                locstate_dens /= zdens[cell, z]
                locstate_dens_max /= zdens_max[cell, z]
                boltz_dens /= zdens_max[cell, z]

            for i, cell in enumerate(cells):
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
        ax.set_title('Atomic state densities, ' + self.longname +
                     '+' + str(z) + ', x={:.2f}m'.format(self.xgrid[cells[0]]))
        plt.xticks(rotation=45)
        fig.tight_layout(pad=2)

    def get_radrec_E(self, radrec_trans):
        for iz_trans in self.iz_transitions:
            if iz_trans.from_state.iz == radrec_trans.to_state.iz and iz_trans.to_state.iz == radrec_trans.from_state.iz:
                eps = iz_trans.thresh - radrec_trans.to_state.energy

        return eps

    def get_radrec_rates(self, per_z=False):
        Te = self.Te
        ne = self.ne
        if per_z:
            radrec_E_rates = np.zeros((self.num_x, self.num_z))
            radrec_E_rates_max = np.zeros((self.num_x, self.num_z))
        else:
            radrec_E_rates = np.zeros(self.num_x)
            radrec_E_rates_max = np.zeros(self.num_x)

        # Calculate rad-rec rates
        for i in range(self.num_x):
            if self.opts['RAD_REC']:
                for radrec_trans in self.radrec_transitions:
                    alpha_radrec = max(radrec_trans.radrec_interp(
                        Te[i]), 0.0)
                    rate = self.n_norm * alpha_radrec * ne[i] * \
                        self.dens[i, radrec_trans.from_loc]
                    rate_max = self.n_norm * alpha_radrec * ne[i] * \
                        self.dens_max[i, radrec_trans.from_loc]
                    if per_z:
                        z = radrec_trans.from_state.iz
                        radrec_E_rates[i, z] += rate
                        radrec_E_rates_max[i, z] += rate_max
                    else:
                        radrec_E_rates[i] += rate
                        radrec_E_rates_max[i] += rate_max
        return radrec_E_rates, radrec_E_rates_max

    def get_radrec_E_rates(self, per_z=False):
        Te = self.Te
        ne = self.ne
        if per_z:
            radrec_E_rates = np.zeros((self.num_x, self.num_z))
            radrec_E_rates_max = np.zeros((self.num_x, self.num_z))
        else:
            radrec_E_rates = np.zeros(self.num_x)
            radrec_E_rates_max = np.zeros(self.num_x)

        # Calculate rad-rec rates
        for i in range(self.num_x):
            if self.opts['RAD_REC']:
                for radrec_trans in self.radrec_transitions:
                    alpha_radrec = max(radrec_trans.radrec_interp(
                        Te[i]), 0.0)
                    eps = self.get_radrec_E(radrec_trans) * spf.el_charge
                    rate = eps * self.n_norm * alpha_radrec * ne[i] * \
                        self.dens[i, radrec_trans.from_loc] / self.t_norm
                    rate_max = eps * self.n_norm * alpha_radrec * ne[i] * \
                        self.dens_max[i, radrec_trans.from_loc] / \
                        self.t_norm
                    if per_z:
                        z = radrec_trans.from_state.iz
                        radrec_E_rates[i, z] += rate
                        radrec_E_rates_max[i, z] += rate_max
                    else:
                        radrec_E_rates[i] += rate
                        radrec_E_rates_max[i] += rate_max
        return radrec_E_rates, radrec_E_rates_max

    def get_ex_E_rates(self, per_z=False):

        collrate_const = self.n_norm * self.v_th * \
            self.sigma_0 * self.t_norm

        if per_z:
            ex_E_rates = np.zeros((self.num_x, self.num_z))
            ex_E_rates_max = np.zeros((self.num_x, self.num_z))
        else:
            ex_E_rates = np.zeros(self.num_x)
            ex_E_rates_max = np.zeros(self.num_x)

        # Calculate collisional excitation rates
        for i in range(self.num_x):
            if self.opts['COLL_EX_DEEX']:
                for ex_trans in self.ex_transitions:
                    K_ex = collrate_const * rates.ex_rate(
                        self.f0[i, :],
                        self.vgrid/self.v_th,
                        self.dvc/self.v_th,
                        ex_trans.sigma
                    )
                    K_ex_max = collrate_const * rates.ex_rate(
                        self.f0_max[i, :],
                        self.vgrid/self.v_th,
                        self.dvc/self.v_th,
                        ex_trans.sigma
                    )
                    eps = ex_trans.thresh * spf.el_charge
                    if per_z:
                        z = ex_trans.from_state.iz
                        ex_E_rates[i, z] += eps * self.n_norm * K_ex * \
                            self.dens[i, ex_trans.from_loc] / self.t_norm
                        ex_E_rates_max[i, z] += eps * self.n_norm * K_ex_max * \
                            self.dens_max[i, ex_trans.from_loc] / self.t_norm
                    else:
                        ex_E_rates[i] += eps * self.n_norm * K_ex * \
                            self.dens[i, ex_trans.from_loc] / self.t_norm
                        ex_E_rates_max[i] += eps * self.n_norm * K_ex_max * \
                            self.dens_max[i, ex_trans.from_loc] / self.t_norm

        return ex_E_rates, ex_E_rates_max

    def get_deex_E_rates(self, per_z=False):

        collrate_const = self.n_norm * self.v_th * \
            self.sigma_0 * self.t_norm

        if per_z:
            deex_E_rates = np.zeros((self.num_x, self.num_z))
            deex_E_rates_max = np.zeros((self.num_x, self.num_z))
        else:
            deex_E_rates = np.zeros(self.num_x)
            deex_E_rates_max = np.zeros(self.num_x)

        # Calculate collisional deexcitation rates
        for i in range(self.num_x):
            if self.opts['COLL_EX_DEEX']:
                for ex_trans in self.ex_transitions:
                    eps = ex_trans.thresh / self.T_norm
                    statw_ratio = ex_trans.from_state.statw / ex_trans.to_state.statw
                    K_deex = collrate_const * rates.deex_rate(
                        self.f0[i, :],
                        self.vgrid/self.v_th,
                        self.dvc/self.v_th,
                        eps,
                        statw_ratio,
                        ex_trans.sigma
                    )
                    K_deex_max = collrate_const * rates.deex_rate(
                        self.f0_max[i, :],
                        self.vgrid/self.v_th,
                        self.dvc/self.v_th,
                        eps,
                        statw_ratio,
                        ex_trans.sigma
                    )
                    eps = ex_trans.thresh * spf.el_charge
                    if per_z:
                        z = ex_trans.from_state.iz
                        deex_E_rates[i, z] += eps * self.n_norm * K_deex * \
                            self.dens[i, ex_trans.to_loc] / self.t_norm
                        deex_E_rates_max[i, z] += eps * self.n_norm * K_deex_max * \
                            self.dens_max[i, ex_trans.to_loc] / \
                            self.t_norm
                    else:
                        deex_E_rates[i] += eps * self.n_norm * K_deex * \
                            self.dens[i, ex_trans.to_loc] / self.t_norm
                        deex_E_rates_max[i] += eps * self.n_norm * K_deex_max * \
                            self.dens_max[i, ex_trans.to_loc] / \
                            self.t_norm

        return deex_E_rates, deex_E_rates_max

    def get_spontem_E_rates(self, per_z=False):

        if per_z:
            spontem_E_rates = np.zeros((self.num_x, self.num_z))
            spontem_E_rates_max = np.zeros((self.num_x, self.num_z))
        else:
            spontem_E_rates = np.zeros(self.num_x)
            spontem_E_rates_max = np.zeros(self.num_x)

        # Calculate spontaneous emission rates
        for i in range(self.num_x):
            if self.opts['SPONT_EM']:
                for em_trans in self.spontem_transitions:
                    beta_spontem = em_trans.spontem_rate
                    eps = (em_trans.from_state.energy -
                           em_trans.to_state.energy) * spf.el_charge
                    rate = eps * self.n_norm * beta_spontem * \
                        self.dens[i, em_trans.from_loc] / self.t_norm
                    rate_max = eps * self.n_norm * beta_spontem * \
                        self.dens_max[i, em_trans.from_loc] / \
                        self.t_norm
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
            adas_ex_E_rates = np.zeros((self.num_x, self.num_z))
            adas_ex_E_rates_max = np.zeros((self.num_x, self.num_z))
            adas_ex_E_rates_aib = np.zeros((self.num_x, self.num_z))
        else:
            adas_ex_E_rates = np.zeros(self.num_x)
            adas_ex_E_rates_max = np.zeros(self.num_x)
            adas_ex_E_rates_aib = np.zeros(self.num_x)

        adas_plt, adas_eff_plt_max, adas_eff_plt, adas_eff_plt_aib = self.get_adas_PLT()

        # Calculate ADAS excitation radiation rates
        ne = self.ne
        if per_z is False:
            adas_ex_E_rates[:] = adas_eff_plt * \
                ne * np.sum(self.dens, 1)
            adas_ex_E_rates_max[:] = adas_eff_plt_max * \
                ne * np.sum(self.dens_max, 1)
            adas_ex_E_rates_aib[:] = adas_eff_plt_aib * \
                ne * np.sum(self.dens_adas, 1)
        else:
            z_dens, z_dens_max, _ = self.get_z_dens()
            z_dens_aib = self.get_z_dens_adas()
            for z in range(self.num_z-1):
                for i in range(self.num_x):
                    adas_ex_E_rates[i, z] += adas_plt[i, z] * \
                        z_dens[i, z] * ne[i]
                    adas_ex_E_rates[i, z] += adas_plt[i, z] * \
                        z_dens_max[i, z] * ne[i]
                    adas_ex_E_rates_aib[i, z] += adas_plt[i, z] * \
                        z_dens_aib[i, z] * ne[i]

        adas_ex_E_rates *= (self.n_norm ** 2)
        adas_ex_E_rates_max *= (self.n_norm ** 2)
        adas_ex_E_rates_aib *= (self.n_norm ** 2)

        return adas_ex_E_rates, adas_ex_E_rates_max, adas_ex_E_rates_aib

    def get_ion_rates(self, per_z=True):
        # Todo: make this an effective rate coefficient
        collrate_const = self.n_norm * self.v_th * \
            self.sigma_0 * self.t_norm

        if per_z:
            ion_rates = np.zeros((self.num_x, self.num_z))
            ion_rates_max = np.zeros((self.num_x, self.num_z))
        else:
            ion_rates = np.zeros(self.num_x)
            ion_rates_max = np.zeros(self.num_x)

        # Calculate collisional ionization rates
        for i in range(self.num_x):
            if self.opts['COLL_ION_REC']:
                for iz_trans in self.iz_transitions:
                    K_ion = collrate_const * rates.ion_rate(
                        self.f0[i, :],
                        self.vgrid/self.v_th,
                        self.dvc/self.v_th,
                        iz_trans.sigma
                    )
                    K_ion_max = collrate_const * rates.ion_rate(
                        self.f0_max[i, :],
                        self.vgrid/self.v_th,
                        self.dvc/self.v_th,
                        iz_trans.sigma
                    )
                    rate = K_ion / self.ne[i]
                    rate_max = K_ion_max / self.ne[i]
                    if per_z:
                        z = iz_trans.from_state.iz
                        ion_rates[i, z] += rate
                        ion_rates_max[i, z] += rate_max
                    else:
                        dens_ratio = self.dens[i, z] / np.sum(self.dens[i, :])
                        dens_ratio_max = self.dens[i,
                                                   z] / np.sum(self.dens[i, :])
                        ion_rates[i] += rate * dens_ratio
                        ion_rates_max[i] += rate_max(dens_ratio_max)

        # Denormalise
        ion_rates /= (self.t_norm * self.n_norm)
        ion_rates_max /= (self.t_norm * self.n_norm)

        return ion_rates, ion_rates_max

    def get_tbrec_rates(self, per_z=True):
        # Todo: make this an effective rate coefficient
        collrate_const = self.n_norm * self.v_th * \
            self.sigma_0 * self.t_norm
        tbrec_norm = self.n_norm * \
            np.sqrt((spf.planck_h ** 2) / (2 * np.pi *
                    spf.el_mass * self.T_norm * spf.el_charge)) ** 3

        if per_z:
            tbrec_rates = np.zeros((self.num_x, self.num_z))
            tbrec_rates_max = np.zeros((self.num_x, self.num_z))
        else:
            tbrec_rates = np.zeros(self.num_x)
            tbrec_rates_max = np.zeros(self.num_x)

        # Calculate three-body recombination rates
        for i in range(self.num_x):
            if self.opts['COLL_ION_REC']:
                for iz_trans in self.iz_transitions:
                    eps = iz_trans.thresh / self.T_norm
                    statw_ratio = iz_trans.to_state.statw / iz_trans.from_state.statw
                    K_rec = self.ne[i] * collrate_const * tbrec_norm * rates.tbrec_rate(
                        self.f0[i, :],
                        self.Te[i],
                        self.vgrid/self.v_th,
                        self.dvc/self.v_th,
                        eps,
                        statw_ratio,
                        iz_trans.sigma
                    )
                    K_rec_max = self.ne[i] * collrate_const * tbrec_norm * rates.tbrec_rate(
                        self.f0_max[i, :],
                        self.Te[i],
                        self.vgrid/self.v_th,
                        self.dvc/self.v_th,
                        eps,
                        statw_ratio,
                        iz_trans.sigma
                    )
                    rate = K_rec / self.ne[i]
                    rate_max = K_rec_max / self.ne[i]
                    if per_z:
                        z = iz_trans.from_state.iz
                        tbrec_rates[i, z] += rate
                        tbrec_rates_max[i, z] += rate_max
                    else:
                        dens_ratio = self.dens[i, z] / np.sum(self.dens[i, :])
                        dens_ratio_max = self.dens[i,
                                                   z] / np.sum(self.dens[i, :])
                        tbrec_rates[i] += rate * dens_ratio
                        tbrec_rates_max[i] += rate_max(dens_ratio_max)

        # Denormalise
        tbrec_rates /= (self.t_norm * self.n_norm)
        tbrec_rates_max /= (self.t_norm * self.n_norm)

        return tbrec_rates, tbrec_rates_max

    def get_ion_rates_adas(self, per_z=True):
        # Assume ground states only here
        if per_z:
            ion_rates = np.zeros((self.num_x, self.num_z))
        else:
            ion_rates = np.zeros(self.num_x)

        # Calculate collisional excitation rates
        adas_iz_coeffs = self.get_adas_iz_coeffs()
        for i in range(self.num_x):
            if self.opts['COLL_ION_REC']:
                for iz_trans in self.iz_transitions:
                    K_ion = adas_iz_coeffs[i, iz_trans.from_state.iz]
                    if per_z:
                        rate = K_ion
                        z = iz_trans.from_state.iz
                        ion_rates[i, z] += rate
                    else:
                        rate = K_ion * self.dens[i, iz_trans.from_loc]
                        ion_rates[i] += rate / np.sum(self.dens[i, :])

        return ion_rates

    def get_rec_rates_adas(self, per_z=True):
        # Assume ground states only here
        if per_z:
            rec_rates = np.zeros((self.num_x, self.num_z))
        else:
            rec_rates = np.zeros(self.num_x)

        # Calculate collisional excitation rates
        adas_rec_coeffs = self.get_adas_rec_coeffs()
        for i in range(self.num_x):
            if self.opts['COLL_ION_REC']:
                for iz_trans in self.iz_transitions:
                    K_rec = adas_rec_coeffs[i, iz_trans.from_state.iz]
                    if per_z:
                        rate = K_rec
                        z = iz_trans.from_state.iz
                        rec_rates[i, z] += rate
                    else:
                        rate = K_rec * self.dens[i, iz_trans.from_loc]
                        rec_rates[i] += rate / np.sum(self.dens[i, :])

        return rec_rates

    def get_PLT(self):
        spontem_E_rates, spontem_E_rates_max = self.get_spontem_E_rates(
            per_z=True)

        z_dens, z_dens_max, z_dens_saha = self.get_z_dens()
        PLT = np.zeros((self.num_x, self.num_z))
        PLT_max = np.zeros((self.num_x, self.num_z))
        eff_PLT = np.zeros((self.num_x))
        eff_PLT_max = np.zeros((self.num_x))

        for z in range(self.num_z-1):
            PLT[:, z] = (spontem_E_rates[:, z]) / (z_dens[:, z] *
                                                   self.ne * self.n_norm * self.n_norm)
            PLT_max[:, z] = (spontem_E_rates_max[:, z]) / (z_dens_max[:, z] *
                                                           self.ne * self.n_norm * self.n_norm)
            eff_PLT[:] += (PLT[:, z] * z_dens[:, z]) / np.sum(z_dens, 1)
            eff_PLT_max[:] += (PLT_max[:, z] *
                               z_dens_max[:, z]) / np.sum(z_dens, 1)
            # PLT[z_dens[:, z] < 1e-16] = np.nan
            # PLT_max[z_dens_max[:, z] < 1e-16] = np.nan

        return PLT, PLT_max, eff_PLT, eff_PLT_max

    def get_adas_iz_coeffs(self):
        if self.name == 'C':
            adas_iz_file = aurora.adas_file(os.path.join(
                os.path.dirname(__file__), 'imp_data/Carbon/scd96_c.dat'))

            # Interpolate adas data to sktrun profile
            Te = self.Te * self.T_norm
            ne = self.ne * self.n_norm
            adas_iz_interp = interpolate_adf11_data(
                adas_iz_file, Te, ne, self.num_z)

        return adas_iz_interp

    def get_adas_rec_coeffs(self):
        if self.name == 'C':
            adas_rec_file = aurora.adas_file(os.path.join(
                os.path.dirname(__file__), 'imp_data/Carbon/acd96_c.dat'))

            # Interpolate adas data to sktrun profile
            Te = self.Te * self.T_norm
            ne = self.ne * self.n_norm
            adas_rec_interp = interpolate_adf11_data(
                adas_rec_file, Te, ne, self.num_z)

        return adas_rec_interp

    def get_adas_PLT(self):
        if self.name == 'C':
            adas_plt_file = aurora.adas_file(os.path.join(
                os.path.dirname(__file__), 'imp_data/Carbon/plt96_c.dat'))

        # Interpolate adas data to sktrun profile
        Te = self.Te * self.T_norm
        ne = self.ne * self.n_norm
        plt_interp = interpolate_adf11_data(adas_plt_file, Te, ne, self.num_z)

        # Caluclate effective PLT
        z_dens, z_dens_max, _ = self.get_z_dens()
        z_dens_aib = self.get_z_dens_adas()
        adas_eff_plt_max = np.zeros(self.num_x)
        adas_eff_plt = np.zeros(self.num_x)
        adas_eff_plt_aib = np.zeros(self.num_x)
        for z in range(self.num_z-1):
            adas_eff_plt_max[:] += (plt_interp[:, z] *
                                    z_dens_max[:, z]) / np.sum(z_dens, 1)
            adas_eff_plt[:] += (plt_interp[:, z] *
                                z_dens[:, z]) / np.sum(z_dens, 1)
            adas_eff_plt_aib[:] += (plt_interp[:, z] *
                                    z_dens_aib[:, z]) / np.sum(z_dens_aib, 1)

        return plt_interp, adas_eff_plt_max, adas_eff_plt, adas_eff_plt_aib

    def plot_ion_rates(self, logx=True, plot_sk=True, plot_max=True, compare_adas=False, savepath=None):

        colours = ['orange', 'green', 'blue', 'cyan', 'brown', 'pink']
        legend_lines = []
        legend_labels = []
        # Todo: Make these effective rate coefficients
        ion_rates, ion_rates_max = self.get_ion_rates(per_z=True)
        if compare_adas:
            ion_rates_adas = self.get_ion_rates_adas(per_z=True)
        # fig, ax = plt.subplots(1)
        fig, ax = plt.subplots(1, figsize=(4, 3.5))
        ax.set_yscale('log')
        if logx:
            ax.set_xscale('log')
        # ax.minorticks_on()
        ax.grid(which='both')
        for i in range(0, 6):
            legend_labels.append(
                'C$^{' + str(i) + r'+}\rightarrow$ C$^{' + str(i+1) + '+}$')
            legend_lines.append(
                Line2D([0], [0], linestyle='-', color=colours[i]))
            if plot_sk:
                ax.plot(self.Te*self.T_norm, 1e6 *
                        ion_rates[:, i], '--', color=colours[i])
            if plot_max:
                ax.plot(self.Te*self.T_norm, 1e6 *
                        ion_rates_max[:, i], color=colours[i])
            if compare_adas:
                ax.plot(self.Te*self.T_norm, 1e6 *
                        ion_rates_adas[:, i], '--', color=colours[i])

        if plot_max:
            legend_labels.append('Maxwellian')
            legend_lines.append(
                Line2D([0], [0], linestyle='-', color='black'))
        if plot_sk:
            legend_labels.append('Kinetic')
            legend_lines.append(
                Line2D([0], [0], linestyle='--', color='black'))

        # ax.legend(legend_lines, legend_labels, loc='lower right')
        ax.legend(legend_lines, legend_labels, bbox_to_anchor=(0.6, 0.65))
        ax.set_xlabel('$T_e$ [eV]', size=14)
        ax.set_ylim([1e-32, 1e-7])
        # ax.set_ylabel('Ionization rate coefficient [cm$^{3}$s$^{-1}$]',size=14)
        ax.set_ylabel('Rate coefficient [cm$^{3}$s$^{-1}$]', size=14)
        fig.tight_layout(pad=2.0)
        if savepath is not None:
            fig.savefig(savepath)

    def plot_rec_rates(self, compare_adas=False):

        colours = ['orange', 'green', 'blue', 'cyan', 'brown', 'pink']
        # Todo: Make these effective rate coefficients
        tbrec_rates, _ = self.get_tbrec_rates(per_z=True)
        if compare_adas:
            rec_rates_adas = self.get_rec_rates_adas(per_z=True)
        fig, ax = plt.subplots(1)
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.grid()
        for i in range(0, 6):
            ax.plot(self.Te*self.T_norm, 1e6 *
                    tbrec_rates[:, i], color=colours[i], label=str(i))
            if compare_adas:
                ax.plot(self.Te*self.T_norm, 1e6 *
                        rec_rates_adas[:, i], '--', color=colours[i], label=str(i))
        ax.legend()
        ax.set_xlabel('$T_e$ [eV]')
        ax.set_ylabel(
            'Recombination (three-body) rate coefficient [cm$^{3}$s$^{-1}$]')

    def plot_PLT(self, xaxis='Te', compare_adas=False, plot_eff=True, plot_sk=True, plot_max=True, plot_stages=False, logx=True):
        fig, ax = plt.subplots(1)
        if xaxis == 'Te':
            x = self.Te * self.T_norm
            ax.set_xlabel('$T_e$ [eV]')
        elif xaxis == 'x':
            x = self.xgrid
            ax.set_xlabel('x [m]')
        PLT, PLT_max, eff_PLT, eff_PLT_max = self.get_PLT()
        if compare_adas:
            adas_PLT, adas_eff_PLT_max, adas_eff_PLT, adas_eff_plt_aib = self.get_adas_PLT()

        colours = ['orange', 'green', 'blue', 'cyan', 'brown', 'pink']
        if plot_stages:
            for z in range(self.num_z-1):
                if plot_max:
                    ax.plot(x, PLT_max[:, z], color=colours[z],
                            label=self.name + '$^{' + str(z) + '+}$')
                if plot_sk:
                    ax.plot(x, PLT[:, z], '--',
                            color=colours[z])
        if plot_sk and plot_eff:
            ax.plot(x, eff_PLT[:], '--',
                    color='black', label='Effective excitation radiation per ion (SOL-KiT $f_0$)')
        if plot_max and plot_eff:
            ax.plot(x, eff_PLT_max[:], '-',
                    color='black', label='(Maxwellian $f_0$)')

        if compare_adas:
            if plot_stages:
                for z in range(self.num_z-1):
                    if plot_max:
                        if z == 0:
                            ax.plot(x, adas_PLT[:, z], linestyle=(
                                0, (1, 1)), color=colours[z], label='ADAS')
                        else:
                            ax.plot(x, adas_PLT[:, z], linestyle=(
                                0, (1, 1)), color=colours[z])
            if plot_sk and plot_eff:
                ax.plot(x, adas_eff_PLT[:], '--',
                        color='grey', label='ADAS effective PLT (SK iz balance)')
            if plot_max and plot_eff:
                ax.plot(x, adas_eff_PLT_max[:], linestyle=(0, (1, 1)),
                        color='grey', label='ADAS effective PLT (Max iz balance)')
        if plot_eff:
            ax.legend(loc='lower left', framealpha=0.1)
        else:
            ax.legend(loc='lower left')
        if logx:
            ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylabel('Carbon excitation radiation [Wm$^{3}$]')
        ax.grid()
        # ax.set_ylim([2e-32, None])
        # ax.set_xlim([1, 100])

    def get_q_rad(self, ex_only=False, compare_adas=False):

        spontem_E_rates, spontem_E_rates_max = self.get_spontem_E_rates()
        q_spontem = np.sum(self.dxc[::2] * 1e-6*spontem_E_rates[::2])
        q_spontem_max = np.sum(
            self.dxc[::2] * 1e-6*spontem_E_rates_max[::2])

        if ex_only is False:
            radrec_E_rates, radrec_E_rates_max = self.get_radrec_E_rates()
            q_radrec = np.sum(self.dxc[::2] * 1e-6*radrec_E_rates[::2])
            q_radrec_max = np.sum(
                self.dxc[::2] * 1e-6*radrec_E_rates_max[::2])

            q_rad = q_radrec + q_spontem
            q_rad_max = q_radrec_max + q_spontem_max

        else:
            q_rad = q_spontem
            q_rad_max = q_spontem_max

        if compare_adas:
            adas_ex_E_rates, adas_ex_E_rates_max, _ = self.get_adas_ex_E_rates()
            q_ex_adas = np.sum(self.dxc[::2] * 1e-6*adas_ex_E_rates[::2])
            q_ex_adas_max = np.sum(
                self.dxc[::2] * 1e-6*adas_ex_E_rates_max[::2])
            return q_rad, q_rad_max, q_ex_adas, q_ex_adas_max
        else:
            return q_rad, q_rad_max

    def plot_radiation(self, logy=False, xaxis='x', compare_adas=False, plot_sk=True, plot_max=True):

        radrec_E_rates, radrec_E_rates_max = self.get_radrec_E_rates()
        spontem_E_rates, spontem_E_rates_max = self.get_spontem_E_rates()
        if compare_adas:
            adas_ex_E_rates, adas_ex_E_rates_max, adas_ex_E_rates_aib = self.get_adas_ex_E_rates()

        fig, ax = plt.subplots(1)

        if xaxis == 'x':
            x = self.xgrid
        elif xaxis == 'Te':
            x = self.Te * self.T_norm

        if compare_adas:
            ax.plot(x, 1e-6*adas_ex_E_rates, linestyle='dotted',
                    color='red', label='spont-em (SK iz balance)')
            ax.plot(x, 1e-6*adas_ex_E_rates_max, linestyle=(0, (5, 10)),
                    color='red', label='spont-em (Max iz balance)')
            ax.plot(x, 1e-6*adas_ex_E_rates_aib, '-.',
                    color='grey', label='spont-em (ADAS iz balance)')
        if plot_sk:
            # ax.plot(x, 1e-6*radrec_E_rates, '--',
            #         color='blue', label='Rad-rec (SOL-KiT $f_0$)')
            ax.plot(x, 1e-6*(spontem_E_rates+radrec_E_rates), '--',
                    color='red', label='$Q_{rad}$ (SOL-KiT $f_0$)')
        if plot_max:
            # ax.plot(x, 1e-6*radrec_E_rates_max, '-',
            #         color='blue', label='Rad-rec (Maxwellian $f_0$)')

            ax.plot(x, 1e-6*(spontem_E_rates_max+radrec_E_rates_max), '-',
                    color='red', label='$Q_{rad}$ (Maxwellian $f_0$)')

        q_radrec = np.sum(self.dxc[::2] * 1e-6*radrec_E_rates[::2])
        q_radrec_max = np.sum(
            self.dxc[::2] * 1e-6*radrec_E_rates_max[::2])
        q_spontem = np.sum(self.dxc[::2] * 1e-6*spontem_E_rates[::2])
        q_spontem_max = np.sum(
            self.dxc[::2] * 1e-6*spontem_E_rates_max[::2])

        print('q_imp (SOL-KiT $f_0$) = ',
              '{:.2f}MW/m^2'.format(q_radrec + q_spontem))
        print('q_imp (Maxwellian $f_0$) = ',
              '{:.2f}MW/m^2'.format(q_radrec_max + q_spontem_max))

        ax.legend()
        ax.set_xlabel('x [m]')
        if logy:
            ax.set_yscale('log')
        ax.set_ylabel('Radiatiative power loss [MWm$^{-3}$]')

    def plot_Zdist(self, cells=-1, plot_saha=False):
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
        z_dens, z_dens_max, z_dens_saha = self.get_z_dens()
        for i, cell in enumerate(cells):
            ax.plot(zs, z_dens_max[cell, min_z:max_z],
                    label=cell_pref[i] + 'Maxwellian $f_0$', color='black', alpha=1.0-0.2*i)
            ax.plot(zs, z_dens[cell, min_z:max_z],
                    '--', label=cell_pref[i] + 'SOL-KiT $f_0$', color='red', alpha=1.0-0.2*i)
            if plot_saha:
                ax.plot(zs, z_dens_saha[cell, min_z:max_z],
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


def interpolate_adf11_data(adas_file, Te, ne, num_z):
    num_x = len(Te)
    interp_data = np.zeros([num_x, num_z-1])
    for z in range(num_z-1):
        adas_file_interp = interpolate.interp2d(
            adas_file.logNe, adas_file.logT, adas_file.data[z], kind='linear')
        for i in range(num_x):
            log_ne = np.log10(1e-6 * ne[i])
            log_Te = np.log10(Te[i])
            interp_result = adas_file_interp(log_ne, log_Te)
            interp_data[i, z] = 1e-6 * \
                (10 ** interp_result[0])

    return interp_data
