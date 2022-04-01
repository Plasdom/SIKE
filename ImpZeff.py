from lib2to3.pgen2.token import STAR
import matplotlib.pyplot as plt
import sk_plotting_functions as spf
import input
import rates
import output
import numpy as np
from impurity import Impurity
from numba import jit

TEMP_FAC = 1.0


def run(skrun_dir, save=False, modelled_species=['C'], evolve=False):

    # Load the SOL-KiT run
    skrun = spf.SKRun(skrun_dir)
    ne = skrun.data['DENSITY']
    # Load the tungsten cross-sections and interpolate onto the SOL-KiT velocity grid
    species = {}
    for imp in modelled_species:
        species[imp] = Impurity(imp, skrun)

    # Build the rate matrices
    for imp in modelled_species:
        species[imp].build_rate_matrices()
        species[imp].get_saha_eq()

    # Calculate densities
    res = 1.0
    step = 0
    if evolve:
        while res > input.RES_THRESH:
            res = 0
            for imp in modelled_species:
                imp_res = species[imp].evolve()
                if imp_res > res:
                    res = imp_res

            # if step % 100 == 0:
            print('TIMESTEP ' + str(step) +
                  ' | RESIDUAL {:.2e}'.format(res), end='\r')
            step += 1
            if step > input.MAX_STEPS:
                break

    else:

        for imp in modelled_species:
            species[imp].solve()

    # Print converged output
    if save:
        pass
        # output.save_output(n_W, n_W_max, step, skrun.num_x,
        #    Te, skrun.T_norm, ne, skrun.n_norm, input.NUM_W, ion_eps_W)
    else:
        return species


def get_maxwellians(num_x, ne, Te, vgrid, v_th, num_v):
    f0_max = np.zeros([num_x, num_v])
    vgrid = vgrid / v_th
    for i in range(num_x):
        f0_max[i, :] = spf.maxwellian(Te[i], ne[i], vgrid)
    return f0_max


# run('/Users/dpower/Documents/01 - PhD/14 - ELM investigation/01 - Runs/01 - Equilibria/02 - Kinetic/P_in = 4MW/Output_job_EQ_K4_10e19/Run_9')
