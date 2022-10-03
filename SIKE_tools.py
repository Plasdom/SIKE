import numpy as np
from numba import jit

# Define some useful constants
el_mass = 9.10938e-31
ion_mass = 2 * 1.67262e-27
epsilon_0 = 8.854188E-12
el_charge = 1.602189E-19
boltzmann_k = 1.38064852E-23
bohr_radius = 5.291772e-11
planck_h = 6.62607004e-34

# Default velocity grid to use when initialising SIKE run from temperature and density profiles
default_vgrid = np.array([46888.62202694,   141838.0816315,   239161.27772617,
                          338917.55372321,   441167.73662016,   545974.17408956,
                          653400.77249567,   763513.03586195,   876378.10581238,
                          992064.80251157,  1110643.66662824,  1232187.00234783,
                          1356768.92146041,  1484465.38855081,  1615354.26731847,
                          1749515.3680553,  1887030.4963105,  2027983.50277225,
                          2172460.33439546,  2320549.08680925,  2472340.05803338,
                          2627925.80353809,  2787401.19268034,  2950863.46655125,
                          3118412.29726881,  3290149.84875457,  3466180.8390272,
                          3646612.60405677,  3831555.16321203,  4021121.28634625,
                          4215426.56255879,  4414589.47067662,  4618731.45149743,
                          4827976.98183885,  5042453.65043867,  5262292.23575344,
                          5487626.78570115,  5718594.69939772,  5955336.8109364,
                          6197997.47526378,  6446724.65619926,  6701670.01665814,
                          6962989.01112837,  7230840.9804606,  7505389.24902588,
                          7786801.22430541,  8075248.49896692,  8370906.95549506,
                          8673956.87343626,  8984583.03932604,  9302974.85936316,
                          9629326.47490113,  9963836.88082742, 10306710.04690213,
                          10658155.04212843, 11018386.16223557, 11387623.0603453,
                          11766090.88090788, 12154020.39698447, 12551648.1509629,
                          12959216.59879099, 13376974.25781453, 13805175.85831377,
                          14244082.49882556, 14693961.80535009, 15155088.09453776,
                          15627742.54095496, 16112213.34853285, 16608795.92629994,
                          17117793.06851126, 17639515.13927802, 18174280.26181391,
                          18722414.51241306, 19284252.11927816, 19860135.66631236,
                          20450416.30202546, 21055453.95362871, 21675617.54652439,
                          22311285.22924176, 22962844.60402593])


@jit(nopython=True)
def lambda_ei(n, T, T_0, n_0, Z_0):
    """e-i Coulomb logarithm

    Args:
        n (float): density
        T (float): temperature
        T_0 (float): temperature normalisation
        n_0 (float): density normalisation
        Z_0 (float): Ion charge

    Returns:
        float: lambda_ei
    """
    if T * T_0 < 10.00 * Z_0 ** 2:
        return 23.00 - np.log(np.sqrt(n * n_0 * 1.00E-6) * Z_0 * (T * T_0) ** (-3.00/2.00))
    else:
        return 24.00 - np.log(np.sqrt(n * n_0 * 1.00E-6) / (T * T_0))


@jit(nopython=True)
def maxwellian(T, n, vgrid):
    """Return a normalised (to n_0 / v_th,0 ** 3) Maxwellian electron distribution (isotropic, as function of velocity magnitude).

    Args:
        T (float): Normalised electron temperature
        n (float): Normalised electron density
        vgrid (np.array, optional): Normalised velocity grid on which to define Maxwellian distribution. If None, create using vgrid = np.arange(0.00001, 10, 1. / 1000.)

    Returns:
        np.array(num_v): numpy array of Maxwellian
    """

    f = [0.0 for i in range(len(vgrid))]
    for i, v in enumerate(vgrid):
        f[i] = (n * (np.pi * T) ** (-3/2) * np.exp(-(v**2) / T))
    f = np.array(f)

    return f

@jit(nopython=True)
def bimaxwellian(T1, n1, T2, n2, vgrid):
    """Return a normalised (to n_0 / v_th,0 ** 3) Maxwellian electron distribution (isotropic, as function of velocity magnitude).

    Args:
        T1 (float): First population electron temperature
        n1 (float): First population electron density
        T2 (float): Second population electron temperature
        n2 (float): Second population electron density
        vgrid (np.array, optional): Velocity grid on which to define Maxwellian distribution

    Returns:
        np.array(num_v): numpy array of Maxwellian
    """

    f = [0.0 for i in range(len(vgrid))]
    for i, v in enumerate(vgrid):
        f[i] = (n1 * (np.pi * T1) ** (-3/2) * np.exp(-(v**2) / T1)) + \
        (n2 * (np.pi * T2) ** (-3/2) * np.exp(-(v**2) / T2))
    f = np.array(f)

    return f


@jit(nopython=True)
def get_maxwellians(ne, Te, vgrid=None, normalised=True):
    """Return an array of Maxwellian electron distributions with the given densities and temperatures.

    Args:
        ne (np.array): Normalised electron densities
        Te (np.array): Normalised electron temperatures
        vgrid (np.array): Normalised velocity grid on which to calculate Maxwellians

    Returns:
        np.array(num_v, num_x): 2d numpy array of Maxwellians at each location in x
    """
    
    if normalised is False:
        T_norm = 10
        n_norm = 1e19
        v_th = np.sqrt(2 * el_charge * T_norm / el_mass)
        ne /= n_norm
        Te /= T_norm
    
    if vgrid is None:
        vgrid = np.geomspace(0.025,12,100)
    elif vgrid is not None and normalised is False:
        vgrid /= v_th

    f0_max = [[0.0 for i in range(len(ne))]
              for j in range(len(vgrid))]
    for i in range(len(ne)):
        f0_max_loc = maxwellian(Te[i], ne[i], vgrid)
        for j in range(len(vgrid)):
            f0_max[j][i] = f0_max_loc[j]
    f0_max = np.array(f0_max)
    
    if normalised is False:
        f0_bimax *= n_norm / v_th ** 3
    
    return f0_max

@jit(nopython=True)
def get_bimaxwellians(n1, n2, T1, T2, vgrid=None, normalised=True):
    """Return an array of bi-Maxwellian electron distributions with the given densities and temperatures.

    Args:
        T1 (np.ndarray): First population electron temperatures
        n1 (np.ndarray): First population electron densities
        T2 (np.ndarray): Second population electron temperatures
        n2 (np.ndarray): Second population electron densities
        vgrid (np.array): Velocity grid on which to calculate bi-Maxwellians
        normalised (bool): 

    Returns:
        np.array(num_v, num_x): 2d numpy array of Maxwellians at each location in x
    """
    
    if normalised is False:
        T_norm = 10
        n_norm = 1e19
        v_th = np.sqrt(2 * el_charge * T_norm / el_mass)
        n1 /= n_norm; n2 /= n_norm
        T1 /= T_norm; T2 /= T_norm
    
    if vgrid is None:
        no_vgrid_given = True
        vgrid = np.geomspace(0.025,12,100)
    elif vgrid is not None and normalised is False:
        vgrid /= v_th

    f0_bimax = [[0.0 for i in range(len(ne))]
              for j in range(len(vgrid))]
    for i in range(len(ne)):
        f0_bimax_loc = bimaxwellian(T1[i], n1[i], T2[i], n2[i], vgrid)
        for j in range(len(vgrid)):
            f0_bimax[j][i] = f0_bimax_loc[j]
    f0_bimax = np.array(f0_bimax)
    
    if normalised is False:
        f0_bimax *= n_norm / v_th ** 3
        
    if no_vgrid_given:
        return f0_bimax, vgrid
    else:
        return f0_bimax


@jit(nopython=True)
def density_moment(f0, vgrid, dvc):
    """Calculate density moment of input electron distribution

    Args:
        f0 (np.array): Electron distribution
        vgrid (_type_): Velocity grid
        dvc (_type_): Velocity grid widths
        normalised (bool, optional): Specify if inputs and output are normalised. Defaults to False.

    Returns:
        float: density. Units are normalised or m**-3 depending on whether inputs are normalised. 
    """
    n = 4 * np.pi * np.sum(f0 * vgrid ** 2 * dvc)
    return n


@jit(nopython=True)
def temperature_moment(f0, vgrid, dvc, normalised=True):
    """_summary_

    Args:
        f0 (_type_): _description_
        vgrid (_type_): _description_
        dvc (_type_): _description_
        normalised (bool, optional): _description_. Defaults to True.

    Returns:
        float: temperature. Units are dimensionless or eV depending on normalised argument
    """

    n = density_moment(f0, vgrid, dvc)
    if normalised:

        T = (2/3) * 4 * np.pi * np.sum(f0 * vgrid ** 4 * dvc) / n
    else:
        T = (2/3) * 4 * np.pi * 0.5 * el_mass * \
            np.sum(f0 * vgrid ** 4 * dvc) / n
        T /= el_charge

    return T


@jit(nopython=True)
def interp_val(a, x, val):
    """interpolate a single value on an array of values at a given set of coordinates 

    Args:
        a (np.array): dependent variable array
        x (np.array): independent variable array
        val (float): the independent value on which to interpolate a new value of the dependent variable 

    Returns:
        float: interpolate point
    """
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
    """Get the array index of the closest value to the input value

    Args:
        x (np.array): ordered array
        val (float): value

    Returns:
        int: array index
    """
    for i in range(len(x)):
        if x[i] > val:
            return i
    return len(x)


@jit(nopython=True)
def calc_rate(vgrid, dvc, fe, sigma, const=1.0):
    """Efficiently compute the collisional rate for a given process

    Args:
        vgrid (nd.ndarray): velocity grid
        dvc (np.ndarray): velocity grid widths
        fe (np.ndarray): local electron velocity distribution
        sigma (np.ndarray): cross-section
        const (float): normalisation cross-section (defaults to 1)

    Returns:
        _type_: _description_
    """
    rate = 0.0
    for i in range(len(vgrid)):
        rate += vgrid[i] ** 3 * dvc[i] * fe[i] * sigma[i]
    rate *= const * 4.0 * np.pi
    return rate


@jit(nopython=True)
def get_sigma_tbr(vgrid, vgrid_inv, sigma_interp, g_ratio, Te):
    """Calculate the three-body recombination cross-section

    Args:
        vgrid (nd.ndarray): velocity grid
        vgrid_inv (nd.ndarray): post-collision velocity grid
        sigma_interp (np.ndarray): Ionization cross-section interpolated to vgrid_inv
        g_ratio (float): ratio of statistical weights
        Te (float): local electron temperature

    Returns:
        _type_: _description_
    """
    sigma_tbrec = 0.5 * g_ratio * \
        (1 / (np.sqrt(Te) ** 3)) * sigma_interp * ((vgrid_inv / vgrid) ** 2)
    return sigma_tbrec


@jit(nopython=True)
def get_sigma_deex(vgrid, vgrid_inv, sigma_interp, g_ratio):
    """Calculate the deexcitation cross-section

    Args:
        vgrid (nd.ndarray): velocity grid
        vgrid_inv (nd.ndarray): post-collision velocity grid
        sigma_interp (np.ndarray): Excitation cross-section interpolated to vgrid_inv
        g_ratio (float): ratio of statistical weights

    Returns:
        nd.ndarray: deexcitation cross-section
    """
    sigma_deex = g_ratio * sigma_interp * ((vgrid_inv / vgrid) ** 2)
    return sigma_deex


@jit(nopython=True)
def get_associated_transitions(state_id, from_ids, to_ids):
    """Efficiently find the positions of all transitions associated with a given state ID

    Args:
        state_id (int): ID for a given state
        from_ids (np.ndarray): list of all from IDs for each transition
        to_ids (np.ndarray): list of all to IDs for each transition

    Returns:
        list: a list of the indices of all associated transitions
    """
    associated_transition_indices = []
    for i in range(len(from_ids)):
        if from_ids[i] == state_id or to_ids[i] == state_id:
            associated_transition_indices.append(i)
    return associated_transition_indices
