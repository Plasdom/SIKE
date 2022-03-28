import numpy as np
import input
import rates
import sk_plotting_functions as spf
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


class Impurity:
    def __init__(self, name, skrun):
        self.name = name
        self.skrun = skrun
        if self.name == 'C':
            self.num_z = 7
            self.longname = 'Carbon'
        elif self.name == 'W':
            self.num_z = 11
            self.longname = 'Tungsten'

        self.load_cross_sections()
        self.init_dens()

    def init_dens(self):
        self.dens = np.zeros((self.skrun.num_x, self.num_z))
        self.dens_max = np.zeros((self.skrun.num_x, self.num_z))
        self.dens_saha = np.zeros((self.skrun.num_x, self.num_z))
        self.dens[:, 0] = input.FRAC_IMP_DENS * self.skrun.data['DENSITY']
        self.dens_max[:, 0] = input.FRAC_IMP_DENS * \
            self.skrun.data['DENSITY']
        self.tmp_dens = np.zeros((self.skrun.num_x, self.num_z))

    def load_cross_sections(self):
        if self.name == 'C':
            self.sigma_ion, self.ion_eps = input.load_carbon_cross_sections(
                self.skrun.vgrid / self.skrun.v_th,
                self.skrun.T_norm,
                self.skrun.sigma_0,
                self.num_z)
        elif self.name == 'W':
            self.sigma_ion, self.ion_eps = input.load_tungsten_cross_sections(
                self.skrun.vgrid / self.skrun.v_th,
                self.skrun.T_norm,
                self.skrun.sigma_0,
                self.num_z)

        self.statw = np.ones(self.num_z)

    def get_saha_eq(self):

        for i in range(self.skrun.num_x):
            de_broglie_l = np.sqrt(
                (spf.planck_h ** 2) / (2 * np.pi * spf.el_mass * spf.el_charge * self.skrun.T_norm * self.skrun.data['TEMPERATURE'][i]))

            # Compute ratios
            dens_ratios = np.zeros(self.num_z - 1)
            for z in range(1, self.num_z):
                eps = self.ion_eps[z-1]
                dens_ratios[z-1] = (2 * np.exp(-eps / (self.skrun.data['TEMPERATURE'][i]*self.skrun.T_norm))) / (
                    (self.skrun.data['DENSITY'][i] * self.skrun.n_norm) * (de_broglie_l ** 3))
            # Fill densities
            imp_dens_tot = np.sum(self.dens[i, :])
            denom_sum = 1.0 + np.sum([np.prod(dens_ratios[:z+1])
                                      for z in range(self.num_z-1)])
            self.dens_saha[i, 0] = imp_dens_tot / denom_sum
            for z in range(1, self.num_z):
                self.dens_saha[i, z] = self.dens_saha[i,
                                                      z-1] * dens_ratios[z-1]

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
        op_mat = [np.zeros((self.num_z, self.num_z))
                  for _ in range(self.skrun.num_x)]
        rate_mat = [np.zeros((self.num_z, self.num_z))
                    for _ in range(self.skrun.num_x)]
        op_mat_max = [np.zeros((self.num_z, self.num_z))
                      for _ in range(self.skrun.num_x)]
        rate_mat_max = [np.zeros((self.num_z, self.num_z))
                        for _ in range(self.skrun.num_x)]
        tbrec_norm = self.skrun.n_norm * np.sqrt((spf.planck_h ** 2) / (2 * np.pi *
                                                                        spf.el_mass * self.skrun.T_norm * spf.el_charge)) ** 3

        # Build kinetic rate matrices
        for i in range(self.skrun.num_x):

            # Compute ionisation rate coeffs
            for z in range(self.num_z-1):
                K_ion = collrate_const * rates.ion_rate(
                    f0[i, :],
                    self.skrun.vgrid/self.skrun.v_th,
                    self.skrun.dvc/self.skrun.v_th,
                    self.sigma_ion[z, :])
                rate_mat[i][z, z] -= K_ion
                rate_mat[i][z+1, z] += K_ion

            # Compute recombination rate coeffs
            for z in range(1, self.num_z):
                eps = self.ion_eps[z-1] / self.skrun.T_norm
                statw_ratio = self.statw[z] / self.statw[z-1]
                K_rec = ne[i] * collrate_const * tbrec_norm * rates.tbrec_rate(
                    f0[i, :],
                    Te[i],
                    self.skrun.vgrid/self.skrun.v_th,
                    self.skrun.dvc/self.skrun.v_th,
                    eps,
                    statw_ratio,
                    self.sigma_ion[z-1, :])
                rate_mat[i][z-1, z] += K_rec
                rate_mat[i][z, z] -= K_rec

            op_mat[i] = np.linalg.inv(np.identity(
                self.num_z) - input.DELTA_T * rate_mat[i])

        self.op_mat = op_mat
        self.rate_mat = rate_mat

        # Build Maxwellian rate matrices
        for i in range(self.skrun.num_x):

            # Compute ionisation rate coeffs
            for z in range(self.num_z-1):
                K_ion = collrate_const * rates.ion_rate(
                    f0_max[i, :],
                    self.skrun.vgrid/self.skrun.v_th,
                    self.skrun.dvc/self.skrun.v_th,
                    self.sigma_ion[z, :])
                rate_mat_max[i][z, z] -= K_ion
                rate_mat_max[i][z+1, z] += K_ion

            # Compute recombination rate coeffs
            for z in range(1, self.num_z):
                eps = self.ion_eps[z-1] / self.skrun.T_norm
                statw_ratio = self.statw[z] / self.statw[z-1]
                K_rec = ne[i] * collrate_const * tbrec_norm * rates.tbrec_rate(
                    f0_max[i, :],
                    Te[i],
                    self.skrun.vgrid/self.skrun.v_th,
                    self.skrun.dvc/self.skrun.v_th,
                    eps,
                    statw_ratio,
                    self.sigma_ion[z-1, :])
                rate_mat_max[i][z-1, z] += K_rec
                rate_mat_max[i][z, z] -= K_rec

            op_mat_max[i] = np.linalg.inv(np.identity(
                self.num_z) - input.DELTA_T * rate_mat_max[i])

        self.op_mat_max = op_mat_max
        self.rate_mat_max = rate_mat_max

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
            for z in range(1, self.num_z):
                self.Zeff[i] += self.dens[i, z] * (float(z) ** 2)
                self.Zeff_max[i] += self.dens_max[i, z] * (float(z) ** 2)
                self.Zeff_saha[i] += self.dens_saha[i, z] * (float(z) ** 2)
            if dens_tot[i] > 0:
                self.Zeff[i] = self.Zeff[i] / self.skrun.data['DENSITY'][i]
            if dens_tot_max[i] > 0:
                self.Zeff_max[i] = self.Zeff_max[i] / \
                    self.skrun.data['DENSITY'][i]
            if dens_tot_saha[i] > 0:
                self.Zeff_saha[i] = self.Zeff_saha[i] / \
                    self.skrun.data['DENSITY'][i]

    def plot_Zeff(self):
        self.get_Zeff()
        fig, ax = plt.subplots(1)
        ax.plot(self.skrun.xgrid[:-1], self.Zeff_max[:-1],
                label='Maxwellian $f_0$', color='black')
        ax.plot(self.skrun.xgrid[:-1], self.Zeff_saha[:-1], '--',
                label='Saha equilibrium', color='blue')
        ax.plot(self.skrun.xgrid[:-1], self.Zeff[:-1], '--',
                label='SOL-KiT $f_0$', color='red')
        ax.grid()
        ax.legend()
        ax.set_xlabel('x [m]')
        ax.set_ylabel('$Z_{eff}=\Sigma_i Z^2_i n_{Z}^i / n_e$')
        ax.set_title('$Z_{eff}$ profile')
        # ax.set_xlim([10.4, 11.8])

    def plot_dens(self, xaxis='normal', plot_kin=True, plot_max=True, plot_saha=False):
        fig, ax = plt.subplots(1)
        cmap = plt.cm.get_cmap('plasma')
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
                ax.plot(x, self.dens[:-1, z] / np.sum(self.dens[:-1, :], 1),
                        '--', color=sm.to_rgba(z))
            if plot_max:
                ax.plot(x, self.dens_max[:-1, z] / np.sum(self.dens_max[:-1, :], 1),
                        color=sm.to_rgba(z))
            if plot_saha:
                ax.plot(
                    x, self.dens_saha[:-1, z] / np.sum(self.dens_saha[:-1, :], 1), '-.', color=sm.to_rgba(z))
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
        ax.set_ylabel('$n_Z / n_Z^{tot}$')
        ax.set_title('$n_Z$ profiles')

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
        ax.set_title('Impurity state densities')


def get_maxwellians(num_x, ne, Te, vgrid, v_th, num_v):
    f0_max = np.zeros([num_x, num_v])
    vgrid = vgrid / v_th
    for i in range(num_x):
        f0_max[i, :] = spf.maxwellian(Te[i], ne[i], vgrid)
    return f0_max
