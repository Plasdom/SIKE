import numpy as np
import os
from scipy.interpolate import interp1d
import re

DELTA_T = 1.0e10
RES_THRESH = 1E-12
MAX_STEPS = 5e4
T_SAVE = 1e6
FRAC_IMP_DENS = 0.01
COLL_ION_REC = True
RAD_REC = True
COLL_EX_DEEX = False
SPONT_EM = False
GS_ONLY = True
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


def load_sunokato_iz_sigma(vgrid, from_state, to_state, T_norm, sigma_0):

    sigma = np.zeros(len(vgrid))

    cs_file = os.path.join(
        'imp_data', 'Carbon', 'sunokato_iz_cs.txt')
    with open(cs_file) as f:
        lines = f.readlines()
        for l in lines[1:]:
            line_data = l.split('\t')
            from_iz = int(line_data[0])
            from_statename = line_data[1]
            to_iz = int(line_data[2])
            to_statename = line_data[3]
            if from_state.iz == from_iz and \
                    from_state.statename == from_statename and \
                    to_state.iz == to_iz and \
                    to_state.statename == to_statename:
                coeffs = [float(x) for x in line_data[5:-1]]
                coeffs_I = float(line_data[4])
                try:
                    thresh = float(line_data[-1])
                except:
                    pass
                sigma += sunokato_iz_fit(coeffs_I, coeffs,
                                         vgrid, T_norm, sigma_0, thresh)

    return sigma, thresh


def sunokato_iz_fit(I, coeffs, vgrid, T_norm, sigma_0, thresh):
    sigma = np.zeros(len(vgrid))
    A_1 = coeffs[0]
    for i in range(len(vgrid)):
        v = vgrid[i]
        E = T_norm * (v ** 2)

        sigma[i] += (1e-13 / (I * E)) * (A_1 * np.log(E / I))
        for k in range(1, len(coeffs)):
            A_k = coeffs[k]
            sigma[i] += (1e-13 / (I * E)) * (A_k * (1.0 - (I / E)) ** k)

    for i in range(len(vgrid)):
        E = vgrid[i]**2 * T_norm
        if E > thresh:
            sigma[:i] = 0.0
            break

    return sigma / (1e4 * sigma_0)


def adas_rm1(a):
    return a[0][:-1]


def lmom(l):
    if l == '0':
        return 'S'
    if l == '1':
        return 'P'
    if l == '2':
        return 'D'
    if l == '3':
        return 'F'
    if l == '4':
        return 'G'


def get_adas_statename(line):
    fields = line.split('    ')
    shell = fields[2][1:].lower().replace(' ', '-')
    shell = re.sub('[spdfg]1', adas_rm1, shell)
    mom = re.search('\(\d\)\d\(', line)
    s = mom[0][1]
    l = lmom(mom[0][3])
    return shell + ' ' + s + l


def load_adas_radrec_rates(imp_name, from_state, to_state, T_norm, n_norm, t_norm):
    adas_file = get_adas_file('radrec', imp_name, from_state.iz)
    with open(adas_file) as f:
        lines = f.readlines()

    # Get parent and child states
    started_to = False
    for i, l in enumerate(lines):
        if 'PARENT TERM INDEXING' in l:
            from_start = i+4
        if 'LS RESOLVED TERM INDEXING' in l:
            from_end = i-1
            started_to = True
            to_start = i+4
        if started_to:
            if l == ' \n' or l == '\n':
                to_end = i
                break
    from_states = []
    to_states = []
    for l in lines[from_start:from_end]:
        from_states.append(get_adas_statename(l))
    for l in lines[to_start:to_end]:
        to_states.append(get_adas_statename(l))

    # Get Te index
    for i, l in enumerate(lines):
        if 'INDX TE=' in l and 'PRTI=' in lines[i-2]:
            Te = np.array([float(T.replace('D', 'E')) for T in l.split()[2:]])
            break
    K2eV = 11603.247217
    Te = Te / (K2eV * T_norm)  # Convert to eV and normalise

    # Get transitions
    started_trans = False
    from_trans = []
    to_trans = []
    for i, l in enumerate(lines):
        if i > 4:
            if 'INDX TE=' in l and 'PRTI=' in lines[i-2]:
                started_trans = True
                from_trans.append(i+2)
            if started_trans:
                if l == ' \n' or l == '\n':
                    to_trans.append(i)
                    started_trans = False
    searching = True
    for i, from_idx in enumerate(from_trans):
        to_idx = to_trans[i]
        parent_idx = i
        if searching:
            for l in lines[from_idx:to_idx]:
                child_idx = int(l.split()[0]) - 1
                if from_states[parent_idx] == from_state.statename and to_states[child_idx] == to_state.statename:
                    rates = np.array([float(r.replace('D', 'E'))
                                     for r in l.split()[1:]])
                    searching = False
                    break
    rates = rates * t_norm * n_norm  # Normalise

    return rates, Te


def get_adas_file(trans_type, imp_name, z):
    pref = os.path.join('imp_data', imp_name)
    if trans_type == 'radrec':
        if imp_name == 'Carbon':
            if z == 1:
                adas_file = os.path.join(pref, 'rrc93#b_c1ls.dat')
            elif z == 2:
                adas_file = os.path.join(pref, 'rrc96#be_c2ls.dat')
            elif z == 3:
                adas_file = os.path.join(pref, 'rrc93#li_c3ls.dat')
            elif z == 4:
                adas_file = os.path.join(pref, 'rrc96#he_c4ls.dat')
            elif z == 5:
                adas_file = os.path.join(pref, 'rrc96#h_c5ls.dat')
            elif z == 6:
                adas_file = os.path.join(pref, 'rrc93##_c6ls.dat')
    return adas_file


def load_tungsten_cross_sections(vgrid, T_norm, sigma_0, num_z):
    # Read in raw cross-sections
    W_ion_raw = [None] * num_z
    for i in range(num_z):
        dat_file = os.path.join(
            'imp_data', 'Tungsten', 'W' + str(i) + '+->' + 'W' + str(i+1) + '+.dat')
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
