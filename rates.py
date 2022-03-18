import numpy as np
from numba import jit


@jit(nopython=True)
def ion_rate(f0, vgrid, dv, sigma_ion):

    # Compute the ionisation rate
    rate = 0
    for i in range(len(dv)):
        rate += rate + 4 * np.pi * dv[i] * \
            (vgrid[i] ** 3) * f0[i] * sigma_ion[i]
    return rate


@ jit(nopython=True)
def tbrec_rate(f0, Te, vgrid, dv, eps, statw_ratio, sigma_ion):

    # Find 3b-rec cross-section from detailed balance
    sigma_tbrec = np.zeros(len(sigma_ion))
    for i in range(len(dv)):

        vprime = vgrid[i] * np.sqrt(1.0 + ((2 * eps) / (vgrid[i] ** 2)))

        sigma_tbrec[i] = 0.5 * statw_ratio * \
            (Te ** (-3/2)) * sigma_ion[i] * \
            ((vgrid[i] / vprime) **
             2)  # Note sigma_tbrec is implicitly divided by n_e here

    # Compute the 3b-rec rate
    rate = 0
    for i in range(len(dv)):
        rate += rate + 4 * np.pi * dv[i] * \
            (vgrid[i] ** 3) * f0[i] * sigma_tbrec[i]

    return rate
