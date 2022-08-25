import numpy as np
import os
import scipy
from scipy.interpolate import interp1d
import re

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


def get_lotz_iz_cs(vgrid, T_norm, from_state, to_state, sigma_0):
    z = from_state.iz
    I_H = 13.6058
    a_0 = 5.29177e-11
    cs = np.zeros(len(vgrid))
    nu = 0.25 * (np.sqrt((100*z + 91) / (4*z + 3)) - 1)
    C = 2.3

    zeta = from_state.shell_occupation
    I = from_state.shell_iz_energies
    E_f = to_state.energy

    for i in range(len(vgrid)):
        E_0 = T_norm * vgrid[i] ** 2
        # for j in range(len(zeta)):
        for j in range(len(zeta)-1, len(zeta)):
            I_j = I[j] + E_f
            x_j = E_0 / I_j
            if E_0 > I_j:
                cs[i] += 4.5e-18 * zeta[j] * np.log(x_j) / (E_0 * I_j)

    return cs / sigma_0, I[-1] + E_f


def get_BC_iz_cs(vgrid, T_norm, from_state, to_state, sigma_0):
    # Note
    z = from_state.iz
    I_H = 13.6058
    a_0 = 5.29177e-11
    cs = np.zeros(len(vgrid))
    nu = 0.25 * (np.sqrt((100*z + 91) / (4*z + 3)) - 1)
    C = 2.3

    zeta = from_state.shell_occupation
    I = from_state.shell_iz_energies
    E_f = to_state.energy

    for i in range(len(vgrid)):
        E_0 = T_norm * vgrid[i] ** 2
        # for j in range(len(zeta)):
        for j in range(len(zeta)-1, len(zeta)):
            I_j = I[j] + E_f
            x_j = E_0 / I_j
            if x_j <= 0.0:
                print('hey')
            if E_0 > I_j:
                w = np.log(x_j) ** (nu / x_j)
                cs[i] += C * zeta[j] * ((I_H / I_j) ** 2) * \
                    (np.log(x_j) / x_j) * w * np.pi * (a_0 ** 2)

    return cs / sigma_0, I[-1] + E_f


def load_nifs_ex_sigma(vgrid, from_state, to_state, sigma_0, T_norm, extrapolate=True):
    # Note: Care should be taken when extrapolating from NIFS data. Check with a given transition beforehand.

    nifs_file = os.path.join(os.path.dirname(__file__),
                             'imp_data', 'Carbon', 'nifs_ex_cs.txt')

    Egrid = vgrid ** 2 * T_norm

    # Get the NIFS energy and cross-section data
    with open(nifs_file) as f:
        lines = f.readlines()
        for i, line in enumerate(lines[4::4]):
            iz, from_statename, to_statename, I_0 = line.split('\t')
            if from_state.statename == from_statename and to_state.statename == to_statename and from_state.iz == int(iz):
                nifs_Egrid = np.array([float(v)
                                      for v in lines[4+4*i+1].split('\t')])
                nifs_sigma = np.array([float(v)
                                      for v in lines[4+4*i+2].split('\t')])
                break

    # Interpolate to the SOL-KiT energy grid
    if extrapolate:
        f = scipy.interpolate.interp1d(nifs_Egrid, np.log(
            nifs_sigma), fill_value="extrapolate", bounds_error=False, kind='cubic')
        log_sigma = f(Egrid)
        sigma = np.exp(log_sigma)
    else:
        f = scipy.interpolate.interp1d(
            nifs_Egrid, nifs_sigma, fill_value=0.0, bounds_error=False, kind='cubic')
        sigma = f(Egrid)
    # Compute collision strength and tidy
    sigma = sigma * 1e-4
    coll_strength = sigma * (from_state.statw * Egrid) / 1.1969e-15
    sigma = sigma / sigma_0
    sigma[np.where(Egrid < float(I_0))] = 0.0

    return sigma, float(I_0), coll_strength


def load_sunokato_ex_sigma(vgrid, from_state, to_state, T_norm, sigma_0, g_i):

    cs_file = os.path.join(os.path.dirname(__file__),
                           'imp_data', 'Carbon', 'sunokato_ex_cs.txt')
    with open(cs_file) as f:
        lines = f.readlines()
        for l in lines[1:]:
            line_data = l.split('\t')
            line_data[-1] = line_data[-1].strip('\n')
            from_iz = int(line_data[0])
            from_statename = line_data[1]
            to_iz = from_iz
            to_statename = line_data[2]
            if from_state.iz == from_iz and \
                    from_state.statename == from_statename and \
                    to_state.iz == to_iz and \
                    to_state.statename == to_statename:
                A, B, C, D, E, F, P, Q, X_1 = [
                    float(x) for x in line_data[3:12]]
                V_if = float(line_data[12])
                fit_eqn = int(line_data[13])
    V_if = to_state.energy - from_state.energy
    fit_data = {'A': A, 'B': B, 'C': C,
                'D': D, 'E': E, 'F': F,
                'P': P, 'Q': Q, 'X_1': X_1,
                'V_if': V_if}

    if fit_eqn == 6:
        sigma, coll_strength = sunokato_ex_fit6(
            fit_data, vgrid, T_norm, sigma_0, g_i)
    elif fit_eqn == 7:
        sigma, coll_strength = sunokato_ex_fit7(
            fit_data, vgrid, T_norm, sigma_0, g_i)
    elif fit_eqn == 10:
        sigma, coll_strength = sunokato_ex_fit10(
            fit_data, vgrid, T_norm, sigma_0, g_i)
    elif fit_eqn == 11:
        sigma, coll_strength = sunokato_ex_fit7(
            fit_data, vgrid, T_norm, sigma_0, g_i)

    return sigma, V_if, coll_strength


def sunokato_ex_fit6(fit_data, vgrid, T_norm, sigma_0, g_i):

    A = fit_data['A']
    B = fit_data['B']
    C = fit_data['C']
    D = fit_data['D']
    E = fit_data['E']
    V_if = fit_data['V_if']

    sigma = np.zeros(len(vgrid))
    coll_strength = np.zeros(len(vgrid))
    Egrid = vgrid**2 * T_norm
    Xgrid = Egrid / V_if
    for i in range(len(vgrid)):
        X = Xgrid[i]
        coll_strength[i] = A + (B / X) + (C / (X**2)) + \
            (D / (X**3)) + E * np.log(X)
        sigma[i] = 1.1969e-15 * coll_strength[i] / (g_i * X * V_if)

    # Tidy up and normalise
    sigma[np.where(Egrid < V_if)] = 0.0
    coll_strength[np.where(Egrid < V_if)] = 0.0
    sigma[np.where(sigma < 0.0)] = 0.0
    # coll_strength[np.where(coll_strength < 0.0)] = 0.0
    sigma = sigma / (1e4 * sigma_0)

    return sigma, coll_strength


def sunokato_ex_fit7(fit_data, vgrid, T_norm, sigma_0, g_i):

    A = fit_data['A']
    B = fit_data['B']
    C = fit_data['C']
    D = fit_data['D']
    E = fit_data['E']
    F = fit_data['F']
    V_if = fit_data['V_if']

    sigma = np.zeros(len(vgrid))
    coll_strength = np.zeros(len(vgrid))
    Egrid = vgrid**2 * T_norm
    Xgrid = Egrid / V_if
    for i in range(len(vgrid)):
        X = Xgrid[i]
        coll_strength[i] = (A / (X**2)) + (B * np.exp(-F*X)) + (C * np.exp(-2*F*X)
                                                                ) + (D * np.exp(-3*F*X)) + (E * np.exp(-4*F*X))
        sigma[i] = 1.1969e-15 * coll_strength[i] / (g_i * X * V_if)

    # Tidy up and normalise
    sigma[np.where(Egrid < V_if)] = 0.0
    coll_strength[np.where(Egrid < V_if)] = 0.0
    sigma[np.where(sigma < 0.0)] = 0.0
    # coll_strength[np.where(coll_strength < 0.0)] = 0.0
    sigma = sigma / (1e4 * sigma_0)

    return sigma, coll_strength


def sunokato_ex_fit10(fit_data, vgrid, T_norm, sigma_0, g_i):

    A = fit_data['A']
    B = fit_data['B']
    C = fit_data['C']
    D = fit_data['D']
    E = fit_data['E']
    F = fit_data['F']
    V_if = fit_data['V_if']

    sigma = np.zeros(len(vgrid))
    coll_strength = np.zeros(len(vgrid))
    Egrid = vgrid**2 * T_norm
    Ygrid = V_if / Egrid
    for i in range(len(vgrid)):
        y = Ygrid[i]
        coll_strength[i] = y * (((A / y) + C) + (0.5 * D * (1 - y)) +
                                (np.exp(y) * scipy.special.exp1(y)) * (B - (C * y) + (0.5 * D * y * y) + (E/y)))
        sigma[i] = 1.1969e-15 * coll_strength[i] / (g_i * y * V_if)

    sigma[np.where(Egrid < V_if)] = 0.0
    coll_strength[np.where(Egrid < V_if)] = 0.0
    sigma[np.where(sigma < 0.0)] = 0.0
    # coll_strength[np.where(coll_strength < 0.0)] = 0.0
    sigma = sigma / (1e4 * sigma_0)

    return sigma, coll_strength


def sunokato_ex_fit11(fit_data, vgrid, T_norm, sigma_0, g_i):

    A = fit_data['A']
    B = fit_data['B']
    C = fit_data['C']
    D = fit_data['D']
    E = fit_data['E']
    F = fit_data['F']
    V_if = fit_data['V_if']

    sigma = np.zeros(len(vgrid))
    coll_strength = np.zeros(len(vgrid))
    Egrid = vgrid**2 * T_norm
    Ygrid = V_if / Egrid
    for i in range(len(vgrid)):
        y = Ygrid[i]
        coll_strength[i] = A * y * \
            (1 - (np.exy(y) * scipy.special.exp1(y) * y)) + (((B * np.exp(-F)) / (F + y)) + ((C * np.exp(-2*F)
                                                                                              ) / (2*F + y)) + ((D * np.exp(-3*F)) / (3*F + y)) + ((E * np.exp(-4*F)) / (4*F + y))) * y
        sigma[i] = 1.1969e-15 * coll_strength[i] / (g_i * y * V_if)

    sigma[np.where(Egrid < V_if)] = 0.0
    coll_strength[np.where(Egrid < V_if)] = 0.0
    sigma[np.where(sigma < 0.0)] = 0.0
    # coll_strength[np.where(coll_strength < 0.0)] = 0.0
    sigma = sigma / (1e4 * sigma_0)

    return sigma, coll_strength


def load_sunokato_iz_sigma(vgrid, from_state, to_state, T_norm, sigma_0):

    sigma = np.zeros(len(vgrid))

    cs_file = os.path.join(os.path.dirname(__file__),
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
                    thresh = float(line_data[-1].strip('\n'))
                except:
                    pass
                sigma += sunokato_iz_fit(coeffs_I, coeffs,
                                         vgrid, T_norm, sigma_0, coeffs_I)

    return sigma, thresh


def sunokato_iz_fit(I, coeffs, vgrid, T_norm, sigma_0, thresh):

    Egrid = vgrid**2 * T_norm

    sigma = np.zeros(len(vgrid))
    A_1 = coeffs[0]
    for i in range(len(vgrid)):
        E = Egrid[i]

        sigma[i] += (1e-13 / (I * E)) * (A_1 * np.log(E / I))
        for k in range(1, len(coeffs)):
            A_k = coeffs[k]
            sigma[i] += (1e-13 / (I * E)) * (A_k * (1.0 - (I / E)) ** k)

    sigma[np.where(Egrid < thresh)] = 0.0

    return sigma / (1e4 * sigma_0)


def adas_rm1(a):
    return a[0][:-1]


def lmom(l):
    if l == '0':
        return 'S'
    elif l == '1':
        return 'P'
    elif l == '2':
        return 'D'
    elif l == '3':
        return 'F'
    elif l == '4':
        return 'G'
    elif l == '5':
        return 'H'
    elif l == '6':
        return 'I'
    elif l == '7':
        return 'J'
    elif l == '8':
        return 'K'
    elif l == '9':
        return 'L'
    elif l == '10':
        return 'M'
    elif l == '11':
        return 'N'
    elif l == '12':
        return 'O'


def get_adas_statename(line):
    shell_re = re.findall('([123456789][SPDFGHIJKLMNOP][123456789])', line)
    shell = (','.join(shell_re)).lower()
    mom = re.search('\(\d\)\d\(', line)
    s = mom[0][1]
    l = lmom(mom[0][3])
    if shell == '':
        return s + l
    else:
        return shell + ' ' + s + l


def load_adas_iz_rates(imp_name, from_state, to_state, T_norm, n_norm, t_norm):
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
    rates = rates * t_norm * n_norm * 1e-6  # Normalise

    return rates, Te


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

    # Find the number of lines over which index / data points are written
    for i, l in enumerate(lines):
        if 'INDX TE=' in l and 'PRTI=' in lines[i-2]:
            idx_line = i
            break
    for i, l in enumerate(lines[idx_line:]):
        if l == '\n' or l == ' \n':
            idx_end = i + idx_line
            break
    block = ''.join(lines[idx_line:idx_end])
    ch_idx_re = re.findall('( \d+ )', block)
    num_ch = len(ch_idx_re)
    lines_per_entry = int((idx_end - idx_line - 2) / num_ch)

    # Get Te index
    # for i, l in enumerate(lines):
    #     if 'INDX TE=' in l and 'PRTI=' in lines[i-2]:
    #         Te = np.array([float(T.replace('D', 'E')) for T in l.split()[2:]])
    #         break
    for i, l in enumerate(lines):
        if 'INDX TE=' in l and 'PRTI=' in lines[i-2]:
            # TODO: What happens if the index is over 3 lines? Need to see an example for format
            T_idx_lines = ' '.join(lines[i:i+lines_per_entry])
            T_idx_re = re.findall('(\d.\d+[ED][+-]\d+)', T_idx_lines)
            Te = np.array([float(T.replace('D', 'E')) for T in T_idx_re])
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
            for j, l in enumerate(lines[from_idx:to_idx:lines_per_entry]):
                child_idx = int(l.split()[0]) - 1
                if from_states[parent_idx] == from_state.statename and to_states[child_idx] == to_state.statename:
                    rate_data = (' '.join(lines[from_idx + j *
                                                lines_per_entry:from_idx + (j+1)*lines_per_entry])).split()
                    rates = np.array([float(r.replace('D', 'E'))
                                     for r in rate_data[1:]])
                    searching = False
                    break
    try:
        rates = rates * t_norm * n_norm * 1e-6  # Normalise
    except:
        print('hey')

    return rates, Te


def get_adas_file(trans_type, imp_name, z):
    pref = os.path.join(os.path.dirname(__file__), 'imp_data', imp_name)

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

        elif imp_name == 'Neon':
            if z == 1:
                adas_file = os.path.join(pref, 'nrb05#f_ne1ls.dat')
            elif z == 2:
                adas_file = os.path.join(pref, 'nrb05#o_ne2ls.dat')
            elif z == 3:
                adas_file = os.path.join(pref, 'nrb05#n_ne3ls.dat')
            elif z == 4:
                adas_file = os.path.join(pref, 'nrb05#c_ne4ls.dat')
            elif z == 5:
                adas_file = os.path.join(pref, 'nrb05#b_ne5ls.dat')
            elif z == 6:
                adas_file = os.path.join(pref, 'nrb05#be_ne6ls.dat')
            elif z == 7:
                adas_file = os.path.join(pref, 'nrb05#li_ne7ls.dat')
            elif z == 8:
                adas_file = os.path.join(pref, 'nrb05#he_ne8ls.dat')
            elif z == 9:
                adas_file = os.path.join(pref, 'nrb05#h_ne9ls.dat')
            elif z == 10:
                adas_file = os.path.join(pref, 'nrb05##_ne10ls.dat')

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
