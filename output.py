import imp
import numpy as np
from numba import jit
import input
import sk_plotting_functions as spf


def save_output(n_W, n_W_max, step, num_x, Te, T_norm, ne, n_norm):

    # Output kinetic data
    np.savetxt('output/kinetic/imp_dens/n_W_' +
               str(step) + '.txt', n_W)
    Zeff = get_Zeff(n_W, num_x)
    np.savetxt('output/kinetic/Z_eff/Z_eff_' + str(step) +
               '.txt', Zeff)

    # Ouptut fluid data
    np.savetxt('output/fluid/imp_dens/n_W_' +
               str(step) + '.txt', n_W_max)
    Zeff_max = get_Zeff(n_W_max, num_x)
    np.savetxt('output/fluid/Z_eff/Z_eff_' + str(step) +
               '.txt', Zeff_max)

    # Output theoretical densities
    n_W_saha = np.zeros((num_x-input.START_CELL, input.NUM_Z))
    for i in range(num_x-input.START_CELL):
        n_W_saha[i, :] = get_saha_dens(
            n_W_max[i, :], Te[i], T_norm, ne[i], n_norm)
    np.savetxt('output/saha/imp_dens/n_W_' + str(step) +
               '.txt', n_W_saha)
    Zeff_saha = get_Zeff(n_W_saha, num_x)
    np.savetxt('output/saha/Z_eff/Z_eff_' + str(step) +
               '.txt', Zeff_saha)


def get_Zeff(imp_dens, num_x):
    Zeff = np.zeros(num_x-input.START_CELL)
    imp_dens_tot = np.sum(imp_dens, 1)

    for i in range(num_x-input.START_CELL):
        for z in range(1, input.NUM_Z):
            Zeff[i] += imp_dens[i, z] * float(z)
        if imp_dens_tot[i] > 0:
            Zeff[i] = Zeff[i] / imp_dens_tot[i]

    return Zeff


def get_saha_dens(imp_dens, Te, T_norm, ne, n_norm):

    de_broglie_l = np.sqrt(
        (spf.planck_h ** 2) / (2 * np.pi * spf.el_mass * spf.el_charge * T_norm * Te))
    dens_ratios = np.zeros(input.NUM_Z - 1)
    for z in range(1, input.NUM_Z):
        eps = input.ION_EPS[z-1]
        dens_ratios[z-1] = (2 * np.exp(-eps / (Te*T_norm))) / ((ne * n_norm) * (de_broglie_l **
                                                                                3))
    # Fill densities
    saha_dens = np.zeros(input.NUM_Z)
    imp_dens_tot = np.sum(imp_dens)
    denom_sum = 1.0 + np.sum([np.prod(dens_ratios[:z+1])
                             for z in range(input.NUM_Z-1)])
    saha_dens[0] = imp_dens_tot / denom_sum
    for z in range(1, input.NUM_Z):
        saha_dens[z] = saha_dens[z-1] * dens_ratios[z-1]

    return saha_dens
