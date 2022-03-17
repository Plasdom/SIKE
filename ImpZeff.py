from lib2to3.pgen2.token import STAR
import sk_plotting_functions as spf
import sys
import os
import numpy
import rates
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from numba import jit

NUM_Z = 11          # Number of tracked ionization states of tungsten
DELTA_T = 1E-5      # Timestep in seconds
RES_THRESH = 1E-10  # Residual threshold at which to stop evolving
MAX_STEPS = 1e6     # Max number of timesteps to evolve if equilibrium not reached
T_SAVE = 1e5        # Timestep multiple on which to save output
START_CELL = 150


def run(skrun_dir):

    # Load the SOL-KiT run
    skrun = spf.SKRun(skrun_dir)
    skrun.load_dist()
    f0 = np.transpose(skrun.data['DIST_F'][0])
    f0 = f0[START_CELL:, :]
    f0_max = get_maxwellians(skrun)

    # Load the tungsten cross-sections and interpolate onto the SOL-KiT velocity grid
    sigma_ion_W = load_cross_sections(
        skrun.vgrid / skrun.v_th, skrun.T_norm, skrun.sigma_0, 'W')

    # Initialise relative impurity densities (start in ground state)
    frac_imp_dens = 0.01
    start_cell = 150
    n_W = np.zeros([skrun.num_x - START_CELL, NUM_Z])
    n_W_max = np.zeros([skrun.num_x - START_CELL, NUM_Z])
    n_W[:, 0] = frac_imp_dens * skrun.data['DENSITY'][START_CELL:]
    n_W_max[:, 0] = frac_imp_dens * skrun.data['DENSITY'][START_CELL:]
    # n_W = np.loadtxt('output/kinetic/imp_dens/n_W_100000.txt')
    # n_W_max = np.loadtxt('output/fluid/imp_dens/n_W_100000.txt')

    # Evolve until equilibrium reached
    res = 1.0
    res_max = 1.0
    step = 0

    while max(res, res_max) > RES_THRESH:

        if step % T_SAVE == 0:
            # Output kinetic data
            np.savetxt('output/kinetic/imp_dens/n_W_' +
                       str(step) + '.txt', n_W)
            Zeff, n_W_tot = get_Zeff(n_W, skrun.num_x)
            np.savetxt('output/kinetic/Z_eff/Z_eff_' + str(step) +
                       '.txt', Zeff)
            np.savetxt('output/kinetic/tot_imp_dens/n_W_tot_' + str(step) +
                       '.txt', n_W_tot)

            # Ouptut fluid data
            np.savetxt('output/fluid/imp_dens/n_W_' +
                       str(step) + '.txt', n_W_max)
            Zeff_max, n_W_tot_max = get_Zeff(n_W_max, skrun.num_x)
            np.savetxt('output/fluid/Z_eff/Z_eff_' + str(step) +
                       '.txt', Zeff_max)
            np.savetxt('output/fluid/tot_imp_dens/n_W_tot_' + str(step) +
                       '.txt', n_W_tot_max)

        n_W, res = evolve(
            n_W,
            f0,
            sigma_ion_W,
            skrun.T_norm,
            skrun.sigma_0,
            skrun.n_norm,
            skrun.v_th,
            skrun.data['TEMPERATURE'],
            skrun.data['DENSITY'],
            skrun.vgrid/skrun.v_th,
            skrun.dvc/skrun.v_th,
            skrun.num_x)

        n_W_max, res_max = evolve(
            n_W_max,
            f0_max,
            sigma_ion_W,
            skrun.T_norm,
            skrun.sigma_0,
            skrun.n_norm,
            skrun.v_th,
            skrun.data['TEMPERATURE'],
            skrun.data['DENSITY'],
            skrun.vgrid/skrun.v_th,
            skrun.dvc/skrun.v_th,
            skrun.num_x)

        print(step, max(res, res_max))
        step += 1
        if step > MAX_STEPS:
            break


@jit(nopython=True)
def evolve(imp_dens, f0, sigma_ion, T_norm, sigma_0, n_norm, v_th, Te, ne, vgrid, dvc, num_x):

    # Load relevant data from SOL-KiT run
    tbrec_norm = np.sqrt((spf.planck_h ** 2) / (2*np.pi *
                                                spf.el_mass * T_norm * spf.el_charge)) ** 3
    statw = np.ones(NUM_Z)
    ion_eps = np.array([7.86403, 16.37, 26.0, 38.2, 51.6,
                       64.77, 122.01, 141.2, 160.2, 179.0, 208.9, 231.6])

    # Compute the particle source for each ionization state
    part_source = [np.zeros(num_x-START_CELL) for _ in range(NUM_Z)]
    for z in range(NUM_Z):
        for i in range(num_x-START_CELL):
            if i % 2 == 0:

                S_ion = 0
                S_rec = 0

                # Ion gain
                if z > 0:
                    ion_rate = rates.ion_rate(
                        f0[i, :],
                        vgrid,
                        dvc,
                        sigma_ion[z-1])
                    S_ion += (ion_rate * n_norm *
                              v_th * sigma_0) * imp_dens[i, z-1]

                # Ion loss
                if z < NUM_Z-1:
                    ion_rate = rates.ion_rate(
                        f0[i, :],
                        vgrid,
                        dvc,
                        sigma_ion[z])
                    S_ion -= (ion_rate * n_norm *
                              v_th * sigma_0) * imp_dens[i, z]

                # 3b-rec gain
                if z < NUM_Z-1:
                    eps = ion_eps[z+1] - ion_eps[z]
                    statw_ratio = statw[z+1] / statw[z]
                    rec_rate = rates.tbrec_rate(
                        f0[i, :],
                        Te[i],
                        vgrid,
                        dvc,
                        eps,
                        statw_ratio,
                        sigma_ion[z])
                    S_rec += rec_rate * (n_norm ** 2) * v_th * \
                        sigma_0 * tbrec_norm * ne[i] * imp_dens[i, z+1]

                # 3b-rec loss
                if z > 0:
                    eps = ion_eps[z] - ion_eps[z-1]
                    statw_ratio = statw[z] / statw[z-1]
                    rec_rate = rates.tbrec_rate(
                        f0[i, :],
                        Te[i],
                        vgrid,
                        dvc,
                        eps,
                        statw_ratio,
                        sigma_ion[z-1])
                    S_rec -= rec_rate * (n_norm ** 2) * v_th * \
                        sigma_0 * tbrec_norm * ne[i] * imp_dens[i, z]

                part_source[z][i] = S_ion + S_rec

    # Update the impurity densities
    imp_dens_new = np.zeros(np.shape(imp_dens))
    for z in range(NUM_Z):
        imp_dens_new[:, z] = imp_dens[:, z] + \
            (part_source[z][:] / n_norm) * DELTA_T

    residual = np.max(np.abs(imp_dens_new - imp_dens))

    return imp_dens_new, residual


def load_cross_sections(vgrid, T_norm, sigma_0, impurity='W'):
    # Read in raw cross-sections
    W_ion_raw = [None] * NUM_Z
    for i in range(NUM_Z):
        dat_file = os.path.join(
            'imp_data', impurity + str(i) + '+->' + impurity + str(i+1) + '+.dat')
        W_ion_raw[i] = np.loadtxt(dat_file, skiprows=1)
        W_ion_raw[i][:, 1] = W_ion_raw[i][:, 1] / (1e4 * sigma_0)

    # Interpolate to the provided velocity grid
    E_grid = [vgrid[i]**2 * T_norm
              for i in range(len(vgrid))]
    W_ion_interp = [None] * len(W_ion_raw)
    for i in range(len(W_ion_raw)):
        f = interp1d(W_ion_raw[i][:, 0], W_ion_raw[i]
                     [:, 1], bounds_error=False, fill_value=0)
        W_ion_interp[i] = f(E_grid)

    return W_ion_interp


def get_maxwellians(skrun):
    f0_max = np.zeros([skrun.num_x-START_CELL, skrun.num_v])
    vgrid = skrun.vgrid / skrun.v_th
    Te = skrun.data['TEMPERATURE'][START_CELL:]
    ne = skrun.data['DENSITY'][START_CELL:]
    for i in range(skrun.num_x-START_CELL):
        f0_max[i, :] = spf.maxwellian(Te[i], ne[i], vgrid)
    return f0_max


@jit(nopython=True)
def get_Zeff(imp_dens, num_x):
    Zeff = np.zeros(num_x-START_CELL)
    imp_dens_tot = np.zeros(num_x-START_CELL)

    for i in range(num_x-START_CELL):
        if i % 2 == 0:
            for z in range(1, NUM_Z):
                imp_dens_tot[i] += imp_dens[i, z]

    for i in range(num_x-START_CELL):
        if i % 2 == 0:
            for z in range(1, NUM_Z):
                Zeff[i] += imp_dens[i, z] * float(z)
            if imp_dens_tot[i] > 0:
                Zeff[i] = Zeff[i] / imp_dens_tot[i]

    return Zeff, imp_dens_tot


# run('/Users/dpower/Documents/01 - PhD/14 - ELM investigation/01 - Runs/01 - Equilibria/02 - Kinetic/P_in = 4MW/Output_job_EQ_K4_10e19/Run_9')
run('/Users/dpower/Documents/01 - PhD/14 - ELM investigation/01 - Runs/01 - Equilibria/02 - Kinetic/P_in = 64MW/Output_job_EQ_K64_3e19/Run_20')
