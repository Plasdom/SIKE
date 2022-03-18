import numpy as np
import os
from scipy.interpolate import interp1d

NUM_Z = 11           # Number of tracked ionization states of tungsten
DELTA_T = 1E22       # Timestep in seconds
RES_THRESH = 1E-16  # Residual threshold at which to stop evolving
MAX_STEPS = 1e6     # Max number of timesteps to evolve if equilibrium not reached
T_SAVE = 1e6        # Timestep multiple on which to save output
START_CELL = 0


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
    W_ion_interp = np.zeros([len(W_ion_raw), len(vgrid)])
    for i in range(len(W_ion_raw)):
        f = interp1d(W_ion_raw[i][:, 0], W_ion_raw[i]
                     [:, 1], bounds_error=False, fill_value=0)
        W_ion_interp[i, :] = f(E_grid)

    return W_ion_interp
