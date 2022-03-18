import numpy as np
import os
from scipy.interpolate import interp1d

NUM_Z = 11
DELTA_T = 100.0
RES_THRESH = 1E-13
MAX_STEPS = 1e6
T_SAVE = 1e6
START_CELL = 0
ION_EPS = np.array([7.86403, 17.98685, 35.3633, 54.99565428, 89.150121267,
                    130.884623956, 152.97199570, 187.1264626, 272.73443534, 373.556998, 533.6939364, 693.8308741])
STATW = np.ones(NUM_Z)
RUN = '/Users/dpower/Documents/01 - PhD/14 - ELM investigation/01 - Runs/01 - Equilibria/02 - Kinetic/P_in = 4MW/Output_job_EQ_K4_5e19/Run_6'
# RUN = '/Users/dpower/Documents/01 - PhD/14 - ELM investigation/01 - Runs/01 - Equilibria/02 - Kinetic/P_in = 64MW/Output_job_EQ_K64_3e19/Run_20'


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
