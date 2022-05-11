from lib2to3.pgen2.token import STAR
import matplotlib.pyplot as plt
import sk_plotting_functions as spf
import input
import rates
import output
import numpy as np
from impurity import Impurity
from numba import jit
import pickle

TEMP_FAC = 1.0
default_opts = {'EVOLVE': True,
                'SAVE': False,
                'MODELLED SPECIES': ['C'],
                'DELTA_T': 1.0e5,
                'RES_THRESH': 1e-12,
                'MAX_STEPS': 1e4,
                'T_SAVE': 1e6,
                'FRAC_IMP_DENS': 0.05,
                'COLL_ION_REC': True,
                'RAD_REC': True,
                'COLL_EX_DEEX': True,
                'SPONT_EM': True,
                'GS_ONLY': False,
                'GS_ONLY_RADREC': False}


def run(skrun_dir, opts=default_opts):

    # Load the SOL-KiT run
    skrun = spf.SKRun(skrun_dir)
    ne = skrun.data['DENSITY']
    # Load the tungsten cross-sections and interpolate onto the SOL-KiT velocity grid
    species = {}
    for imp in opts['MODELLED SPECIES']:
        species[imp] = Impurity(imp, skrun, opts)

    # Build the rate matrices
    for imp in opts['MODELLED SPECIES']:
        species[imp].build_rate_matrices()
        if opts['COLL_ION_REC']:
            species[imp].get_saha_eq()

    # Calculate densities
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

    # Print converged output
    if opts['SAVE']:
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
