import numpy as np

from numba import jit


@jit(nopython=True)
def ion_rate(f0, vgrid, dv, sigma_ion):
    f0[np.where(f0 < 0.0)] = 0.0
    # Compute the ionisation rate
    rate = 0
    for i in range(len(dv)):
        rate = rate + 4 * np.pi * dv[i] * \
            (vgrid[i] ** 3) * f0[i] * sigma_ion[i]
    return rate


@jit(nopython=True)
def tbrec_rate(f0, Te, vgrid, dv, eps, statw_ratio, sigma_ion):
    f0[np.where(f0 < 0.0)] = 0.0
    # Find 3b-rec cross-section from detailed balance
    sigma = sigma_tbrec(sigma_ion, statw_ratio, vgrid, Te, eps)

    # Compute the 3b-rec rate
    rate = 0
    for i, v in enumerate(vgrid):
        rate = rate + 4 * np.pi * dv[i] * \
            (vgrid[i] ** 3) * f0[i] * sigma[i]

    return rate


@jit(nopython=True)
def sigma_tbrec(sigma_ion, statw_ratio, vgrid, Te, eps):
    # Find 3b-rec cross-section from detailed balance
    sigma_tbrec = np.zeros(len(sigma_ion))
    for i, vprime in enumerate(vgrid):

        # if vprime ** 2 > eps:
        v = np.sqrt(vprime ** 2 + eps)

        # Interpolate sigma_ion to v
        sigma_interp = interp_val(sigma_ion, vgrid, v)

        sigma_tbrec[i] = 0.5 * statw_ratio * (1 / (np.sqrt(Te) ** 3)) * sigma_interp * \
            ((v / vprime) ** 2)  # Note sigma_tbrec is implicitly divided by n_e here

    return sigma_tbrec


@jit(nopython=True)
def interp_val(a, x, val):
    x_idx = bisect_left(x, val) - 1
    if x_idx == len(x) - 1:
        return a[x_idx]
    else:
        dx1 = val - x[x_idx]
        dx2 = x[x_idx + 1] - val
        val_interp = (a[x_idx] * dx2 + a[x_idx + 1] * dx1) / (dx1 + dx2)
        return val_interp


@jit(nopython=True)
def bisect_left(x, val):
    for i in range(len(x)):
        if x[i] > val:
            return i
    return len(x)
