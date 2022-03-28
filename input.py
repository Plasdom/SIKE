import numpy as np
import os
from scipy.interpolate import interp1d

DELTA_T = 1.0e13
RES_THRESH = 1E-13
MAX_STEPS = 5e4
T_SAVE = 1e6
FRAC_IMP_DENS = 0.04
STATW_W = np.ones(11)
C_ION_COEFFS = [
    [
        [1.829, -1.975, 1.149, -3.583, 2.451]
    ],
    [
        [8.39e-1, -7.95e-1, 3.263, -5.382, 3.476]
    ],
    [
        [4.009e-1, -3.518e-1, 2.375, -3.992, 2.794]
    ],
    [
        [1.35, -8.748e-1, -1.444, 2.33, -2.73],
        [-2.777, 5.376, -8.748, 1.766e1, -9.086]
    ],
    [
        [9.205e-1, -6.297e-1, 1.316, -9.156e-2, 0.0]
    ],
    [
        [2.489e-1, 1.847e-1, 4.475e-2, -9.432e-2, 5.122e-1]
    ]
]
C_ION_COEFFS_I = [[10.6], [24.4], [41.4], [64.5, 285], [392.0], [490.0]]


def load_tungsten_cross_sections(vgrid, T_norm, sigma_0, num_z):
    # Read in raw cross-sections
    impurity = 'W'
    W_ion_raw = [None] * num_z
    for i in range(num_z):
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

    levels = np.array([7.86403, 17.98685, 35.3633, 54.99565428,
                       89.150121267, 130.884623956, 152.97199570, 187.1264626,
                       272.73443534, 373.556998, 533.6939364, 693.8308741])
    eps = [levels[z] - levels[z-1] for z in range(1, num_z)]

    return W_ion_interp, eps


def load_carbon_cross_sections(vgrid, T_norm, sigma_0, num_z):

    # Generate cross-sections on input grid
    sigma = np.zeros((num_z, len(vgrid)))
    for z in range(num_z-1):
        for i in range(len(vgrid)):
            v = vgrid[i]
            E = T_norm * (v ** 2)

            sigma[z, i] = 0.0
            I = C_ION_COEFFS_I[z][0]
            A_1 = C_ION_COEFFS[z][0][0]

            for s in range(len(C_ION_COEFFS[z])):
                I = C_ION_COEFFS_I[z][s]
                A_1 = C_ION_COEFFS[z][s][0]
                sigma[z, i] += (1e-13 / (I * E)) * (A_1 * np.log(E / I))
                for k in range(1, len(C_ION_COEFFS[z][s])):
                    A_k = C_ION_COEFFS[z][s][k]
                    sigma[z, i] += (1e-13 / (I * E)) * (A_k *
                                                        (1.0 - (I / E)) ** k)

    # Apply threshold energy
    eps = [13.492605497, 26.368833629, 41.35397821,
           221.38500660, 391.9647306, 489.8226743]
    for z in range(num_z-1):
        for i in range(len(vgrid)):
            E = vgrid[i]**2 * T_norm
            if E > eps[z]:
                sigma[z, :i] = 0.0
                break

    sigma = sigma / (1e4 * sigma_0)

    return sigma, eps
