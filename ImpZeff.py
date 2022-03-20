from lib2to3.pgen2.token import STAR
import sk_plotting_functions as spf
import input
import rates
import output
import numpy as np

TEMP_FAC = 1.0


def run(skrun_dir):

    # Load the SOL-KiT run
    skrun = spf.SKRun(skrun_dir)
    skrun.load_dist()
    f0 = np.transpose(skrun.data['DIST_F'][0])
    f0 = f0[input.START_CELL:, :]
    # ne = skrun.data['DENSITY']
    ne = np.ones(skrun.num_x)
    Te = skrun.data['TEMPERATURE']*TEMP_FAC
    f0_max = get_maxwellians(
        skrun.num_x, ne, Te, skrun.vgrid, skrun.v_th, skrun.num_v)

    # Load the tungsten cross-sections and interpolate onto the SOL-KiT velocity grid
    sigma_ion_W = input.load_cross_sections(
        skrun.vgrid / skrun.v_th, skrun.T_norm, skrun.sigma_0, 'W')

    # Initialise relative impurity densities (start in ground state)
    frac_imp_dens = 1.0
    n_W = np.zeros([skrun.num_x - input.START_CELL, input.NUM_Z])
    n_W_max = np.zeros([skrun.num_x - input.START_CELL, input.NUM_Z])
    n_W[:, 0] = frac_imp_dens * ne
    n_W_max[:, 0] = frac_imp_dens * ne

    # Evolve until equilibrium reached
    res = 1.0
    res_max = 1.0
    step = 0

    # Build the rate matrix
    op_mat, rate_mat = build_matrices(skrun.num_x,
                                      skrun.v_th,
                                      skrun.n_norm,
                                      skrun.T_norm,
                                      skrun.t_norm,
                                      skrun.sigma_0,
                                      sigma_ion_W,
                                      f0,
                                      Te,
                                      ne,
                                      skrun.vgrid/skrun.v_th,
                                      skrun.dvc/skrun.v_th)

    op_mat_max, rate_mat_max = build_matrices(skrun.num_x,
                                              skrun.v_th,
                                              skrun.n_norm,
                                              skrun.T_norm,
                                              skrun.t_norm,
                                              skrun.sigma_0,
                                              sigma_ion_W,
                                              f0_max,
                                              Te,
                                              ne,
                                              skrun.vgrid/skrun.v_th,
                                              skrun.dvc/skrun.v_th)

    # Evolve densities
    while max(res, res_max) > input.RES_THRESH:

        n_W, res = evolve(
            op_mat,
            n_W,
            skrun.num_x,
            skrun.n_norm)

        n_W_max, res_max = evolve(
            op_mat_max,
            n_W_max,
            skrun.num_x,
            skrun.n_norm)

        # if step % 100 == 0:
        print(step, max(res, res_max))
        step += 1
        if step > input.MAX_STEPS:
            break

    # Print converged output
    output.save_output(n_W, n_W_max, step, skrun.num_x,
                       Te, skrun.T_norm, ne, skrun.n_norm)


def build_matrices(num_x, v_th, n_norm, T_norm, t_norm, sigma_0, sigma_ion, f0, Te, ne, vgrid, dvc):

    collrate_const = n_norm * v_th * sigma_0 * t_norm
    op_mat = [np.zeros((input.NUM_Z, input.NUM_Z))
              for _ in range(num_x - input.START_CELL)]
    rate_mat = [np.zeros((input.NUM_Z, input.NUM_Z))
                for _ in range(num_x - input.START_CELL)]
    tbrec_norm = n_norm * np.sqrt((spf.planck_h ** 2) / (2 * np.pi *
                                                         spf.el_mass * T_norm * spf.el_charge)) ** 3

    for i in range(num_x-input.START_CELL):

        # Compute ionisation rate coeffs
        for z in range(input.NUM_Z-1):
            K_ion = collrate_const * rates.ion_rate(
                f0[i, :],
                vgrid,
                dvc,
                sigma_ion[z, :])
            rate_mat[i][z, z] -= K_ion
            rate_mat[i][z+1, z] += K_ion

        # Compute recombination rate coeffs
        for z in range(1, input. NUM_Z):
            eps = input.ION_EPS[z-1] / T_norm
            statw_ratio = input.STATW[z] / input.STATW[z-1]
            K_rec = ne[i] * collrate_const * tbrec_norm * rates.tbrec_rate(
                f0[i, :],
                Te[i],
                vgrid,
                dvc,
                eps,
                statw_ratio,
                sigma_ion[z-1, :])
            rate_mat[i][z-1, z] += K_rec
            rate_mat[i][z, z] -= K_rec

        op_mat[i] = np.linalg.inv(np.identity(
            input.NUM_Z) - input.DELTA_T * rate_mat[i])

    return op_mat, rate_mat


def evolve(op_mat, imp_dens, num_x, n_norm):

    imp_dens_new = np.zeros(np.shape(imp_dens))

    # Compute the particle source for each ionization state
    for i in range(num_x-input.START_CELL):

        # Calculate new densities
        imp_dens_new[i, :] = op_mat[i].dot(imp_dens[i, :])

    residual = np.max(np.abs(imp_dens_new - imp_dens))

    return imp_dens_new, residual


def get_maxwellians(num_x, ne, Te, vgrid, v_th, num_v):
    f0_max = np.zeros([num_x-input.START_CELL, num_v])
    vgrid = vgrid / v_th
    for i in range(num_x-input.START_CELL):
        f0_max[i, :] = spf.maxwellian(Te[i], ne[i], vgrid)
    return f0_max


run(input.RUN)
