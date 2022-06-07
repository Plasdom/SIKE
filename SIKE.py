from lib2to3.pgen2.token import STAR
import matplotlib.pyplot as plt
import sk_plotting_functions as spf  # TODO: Remove spf dependency?
import input
import rates
import output
import numpy as np
from impurity import Impurity
from numba import jit
import pickle

default_opts = {'EVOLVE': True,
                'SAVE': False,
                'MODELLED SPECIES': ['C'],
                'MODELLED_STATES': 'all',
                'DELTA_T': 1.0e5,
                'RES_THRESH': 1e-5,
                'MAX_STEPS': 1e3,
                'T_SAVE': 1e6,
                'FRAC_IMP_DENS': 0.05,
                'COLL_ION_REC': True,
                'RAD_REC': True,
                'COLL_EX_DEEX': True,
                'SPONT_EM': True,
                'COMPARE_ADAS': True}


def run(sktrun=None, sk_timestep=-1, Te=None, ne=None, opts=default_opts):

    # Load the tungsten cross-sections and interpolate onto the SOL-KiT velocity grid
    print('\nInitialising...')
    species = {}
    for imp in opts['MODELLED SPECIES']:
        if sktrun is not None:
            species[imp] = Impurity(
                imp, opts, sktrun=sktrun, sk_timestep=sk_timestep)
        elif Te is not None and ne is not None:
            species[imp] = Impurity(imp, opts, Te=Te, ne=ne)

    # Build the rate matrices
    print('Building rate matrices...')
    for imp in opts['MODELLED SPECIES']:
        species[imp].build_rate_matrices()
        if opts['COLL_ION_REC']:
            species[imp].get_saha_eq()

    # Calculate densities
    print('Solving state densities...')
    res = 1.0
    step = 0
    if opts['EVOLVE']:
        while res > opts['RES_THRESH']:
            res = 0
            for imp in opts['MODELLED SPECIES']:
                imp_res = species[imp].evolve()
                if imp_res > res:
                    res = imp_res

            # if step % 100 == 0:
            print('TIMESTEP ' + str(step) +
                  ' | RESIDUAL {:.2e}'.format(res), end='\r')
            step += 1
            if step > opts['MAX_STEPS']:
                break

    else:

        for imp in opts['MODELLED SPECIES']:
            species[imp].solve()

    # Do some tidying up
    for imp in opts['MODELLED SPECIES']:
        del species[imp].tmp_dens
        del species[imp].op_mat
        del species[imp].op_mat_max
        del species[imp].rate_mat
        del species[imp].rate_mat_max
        if opts['COMPARE_ADAS']:
            del species[imp].op_mat_adas

    # Print converged output
    if opts['SAVE']:

        # Save pickle
        with open(opts['SAVE_PATH'] + '.pickle', 'wb') as f:
            pickle.dump(species, f)
    else:
        return species


def get_maxwellians(num_x, ne, Te, vgrid, v_th, num_v):
    f0_max = np.zeros([num_x, num_v])
    vgrid = vgrid / v_th
    for i in range(num_x):
        f0_max[i, :] = spf.maxwellian(Te[i], ne[i], vgrid)
    return f0_max
