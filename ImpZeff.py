from lib2to3.pgen2.token import STAR
import sk_plotting_functions as spf
import os
import input
import rates
import output
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from numba import jit


def run(skrun_dir):

    # Load the SOL-KiT run
    skrun = spf.SKRun(skrun_dir)
    skrun.load_dist()
    f0 = np.transpose(skrun.data['DIST_F'][0])
    f0 = f0[input.START_CELL:, :]
    f0_max = get_maxwellians(skrun)

    # Load the tungsten cross-sections and interpolate onto the SOL-KiT velocity grid
    sigma_ion_W = input.load_cross_sections(
        skrun.vgrid / skrun.v_th, skrun.T_norm, skrun.sigma_0, 'W')

    # Initialise relative impurity densities (start in ground state)
    frac_imp_dens = 0.01
    start_cell = 150
    n_W = np.zeros([skrun.num_x - input.START_CELL, input.NUM_Z])
    n_W_max = np.zeros([skrun.num_x - input.START_CELL, input.NUM_Z])
    n_W[:, 0] = frac_imp_dens * skrun.data['DENSITY'][input.START_CELL:]
    n_W_max[:, 0] = frac_imp_dens * skrun.data['DENSITY'][input.START_CELL:]
    # n_W = np.loadtxt('output/kinetic/imp_dens/n_W_100000.txt')
    # n_W_max = np.loadtxt('output/fluid/imp_dens/n_W_100000.txt')

    # Evolve until equilibrium reached
    res = 1.0
    res_max = 1.0
    step = 0

    # Build the rate matrix
    op_mat = build_matrices(skrun.num_x,
                            skrun.v_th,
                            skrun.n_norm,
                            skrun.T_norm,
                            skrun.sigma_0,
                            sigma_ion_W,
                            f0,
                            skrun.data['TEMPERATURE'],
                            skrun.vgrid/skrun.v_th,
                            skrun.dvc/skrun.v_th)

    op_mat_max = [np.zeros((input.NUM_Z, input. NUM_Z))
                  for _ in range(skrun.num_x - input.START_CELL)]
    op_mat_max = build_matrices(skrun.num_x,
                                skrun.v_th,
                                skrun.n_norm,
                                skrun.T_norm,
                                skrun.sigma_0,
                                sigma_ion_W,
                                f0_max,
                                skrun.data['TEMPERATURE'],
                                skrun.vgrid/skrun.v_th,
                                skrun.dvc/skrun.v_th)

    # Evolve densities
    while max(res, res_max) > input.RES_THRESH:

        if step % input.T_SAVE == 0:
            output.save_output(n_W, n_W_max, step, skrun.num_x)

        n_W, res = evolve(
            op_mat,
            n_W,
            skrun.num_x)

        n_W_max, res_max = evolve(
            op_mat_max,
            n_W_max,
            skrun.num_x)

        # if step % 100 == 0:
        print(step, max(res, res_max))
        step += 1
        if step > input.MAX_STEPS:
            break

    # Print converged output
    output.save_output(n_W, n_W_max, step, skrun.num_x)


def build_matrices(num_x, v_th, n_norm, T_norm, sigma_0, sigma_ion, f0, Te, vgrid, dvc):

    op_mat = [np.zeros((input.NUM_Z, input.NUM_Z))
              for _ in range(num_x - input.START_CELL)]
    tbrec_norm = n_norm * np.sqrt((spf.planck_h ** 2) / (2 * np.pi *
                                                         spf.el_mass * T_norm * spf.el_charge)) ** 3
    statw = np.ones(input.NUM_Z)
    ion_eps = np.array([7.86403, 17.98685, 35.3633, 54.99565428, 89.150121267,
                        130.884623956, 152.97199570, 187.1264626, 272.73443534, 373.556998, 533.6939364, 693.8308741]) / T_norm

    for i in range(num_x-input.START_CELL):

        op_mat[i]

        rate_mat = np.zeros((input.NUM_Z, input.NUM_Z))

        # Compute ionisation rate coeffs
        for z in range(input.NUM_Z-1):
            K_ion = v_th * sigma_0 * rates.ion_rate(
                f0[i, :],
                vgrid,
                dvc,
                sigma_ion[z, :])
            rate_mat[z, z] -= K_ion
            rate_mat[z+1, z] += K_ion

        # Compute recombination rate coeffs
        for z in range(1, input. NUM_Z):
            eps = ion_eps[z] - ion_eps[z-1]
            statw_ratio = statw[z] / statw[z-1]
            K_rec = v_th * sigma_0 * tbrec_norm * rates.tbrec_rate(
                f0[i, :],
                Te[i],
                vgrid,
                dvc,
                eps,
                statw_ratio,
                sigma_ion[z-1, :])
            rate_mat[z-1, z] += K_rec
            rate_mat[z, z] -= K_rec

        op_mat[i] = np.linalg.inv(np.identity(
            input.NUM_Z) - input.DELTA_T * rate_mat)

    return op_mat


def evolve(op_mat, imp_dens, num_x):

    imp_dens_new = np.zeros(np.shape(imp_dens))

    # Compute the particle source for each ionization state
    for i in range(num_x-input.START_CELL):

        # Calculate new densities
        imp_dens_new[i, :] = op_mat[i].dot(imp_dens[i, :])

    residual = np.max(np.abs(imp_dens_new - imp_dens))

    return imp_dens_new, residual


def get_maxwellians(skrun):
    f0_max = np.zeros([skrun.num_x-input.START_CELL, skrun.num_v])
    vgrid = skrun.vgrid / skrun.v_th
    Te = skrun.data['TEMPERATURE'][input.START_CELL:]
    ne = skrun.data['DENSITY'][input.START_CELL:]
    for i in range(skrun.num_x-input.START_CELL):
        f0_max[i, :] = spf.maxwellian(Te[i], ne[i], vgrid)
    return f0_max


run('/Users/dpower/Documents/01 - PhD/14 - ELM investigation/01 - Runs/01 - Equilibria/02 - Kinetic/P_in = 4MW/Output_job_EQ_K4_10e19/Run_9')
# run('/Users/dpower/Documents/01 - PhD/14 - ELM investigation/01 - Runs/01 - Equilibria/02 - Kinetic/P_in = 64MW/Output_job_EQ_K64_3e19/Run_20')
