import numpy as np
from numba import jit
from scipy import interpolate

# Define some useful constants
el_mass = 9.10938e-31
ion_mass = 2 * 1.67262e-27
epsilon_0 = 8.854188e-12
el_charge = 1.602189e-19
boltzmann_k = 1.38064852e-23
bohr_radius = 5.291772e-11
planck_h = 6.62607004e-34

# Default velocity grid to use when initialising SIKE run from temperature and density profiles
default_vgrid = np.array(
    [
        46888.62202694,
        141838.0816315,
        239161.27772617,
        338917.55372321,
        441167.73662016,
        545974.17408956,
        653400.77249567,
        763513.03586195,
        876378.10581238,
        992064.80251157,
        1110643.66662824,
        1232187.00234783,
        1356768.92146041,
        1484465.38855081,
        1615354.26731847,
        1749515.3680553,
        1887030.4963105,
        2027983.50277225,
        2172460.33439546,
        2320549.08680925,
        2472340.05803338,
        2627925.80353809,
        2787401.19268034,
        2950863.46655125,
        3118412.29726881,
        3290149.84875457,
        3466180.8390272,
        3646612.60405677,
        3831555.16321203,
        4021121.28634625,
        4215426.56255879,
        4414589.47067662,
        4618731.45149743,
        4827976.98183885,
        5042453.65043867,
        5262292.23575344,
        5487626.78570115,
        5718594.69939772,
        5955336.8109364,
        6197997.47526378,
        6446724.65619926,
        6701670.01665814,
        6962989.01112837,
        7230840.9804606,
        7505389.24902588,
        7786801.22430541,
        8075248.49896692,
        8370906.95549506,
        8673956.87343626,
        8984583.03932604,
        9302974.85936316,
        9629326.47490113,
        9963836.88082742,
        10306710.04690213,
        10658155.04212843,
        11018386.16223557,
        11387623.0603453,
        11766090.88090788,
        12154020.39698447,
        12551648.1509629,
        12959216.59879099,
        13376974.25781453,
        13805175.85831377,
        14244082.49882556,
        14693961.80535009,
        15155088.09453776,
        15627742.54095496,
        16112213.34853285,
        16608795.92629994,
        17117793.06851126,
        17639515.13927802,
        18174280.26181391,
        18722414.51241306,
        19284252.11927816,
        19860135.66631236,
        20450416.30202546,
        21055453.95362871,
        21675617.54652439,
        22311285.22924176,
        22962844.60402593,
    ]
)

extended_Egrid = np.array(
    [
        0.00024999999999999995,
        0.0022816102500000006,
        0.006462885070620249,
        0.012928390636902438,
        0.021819314224404203,
        0.03328376432485325,
        0.047477083900309906,
        0.06456217734187718,
        0.08470985172319684,
        0.10809917296420193,
        0.1349178375469009,
        0.16536256045239564,
        0.19963948001692802,
        0.2379645804345624,
        0.28056413266519264,
        0.3276751545389635,
        0.3795458908819851,
        0.43643631452343473,
        0.49861864908086817,
        0.5663779144588411,
        0.6400124960358596,
        0.7198347385562874,
        0.8061715657872239,
        0.8993651270455983,
        0.9997734717478698,
        1.107771253183902,
        1.2237504627678149,
        1.3481211960720558,
        1.481312452006655,
        1.6237729665636684,
        1.7759720826074048,
        1.93840065725412,
        2.11157200845068,
        2.2960229024303165,
        2.4923145837950917,
        2.701033850049286,
        2.9227941724856454,
        3.158236865407461,
        3.408032305753964,
        3.672881205284592,
        3.9535159375694815,
        4.250701922129355,
        4.565239068167659,
        4.897963280441957,
        5.249748029930017,
        5.621505992059057,
        6.014190755384623,
        6.428798603728337,
        6.866370374911989,
        7.327993399358945,
        7.814803521973129,
        8.32798721085097,
        8.868783756533118,
        9.43848756566035,
        10.038450553062793,
        10.670084636482775,
        11.334864338310634,
        12.034329498898959,
        12.770088106215141,
        13.54381924679455,
        14.357276183167821,
        15.212289563155831,
        16.110770766655374,
        17.05471539577779,
        18.046206914452103,
        19.087420443864257,
        20.180626720375084,
        21.328196222841814,
        22.5326034765631,
        23.79643154137395,
        25.122376691737347,
        26.513253297012945,
        27.971998910431175,
        29.501679575663527,
        31.105495360257972,
        32.7867861256026,
        34.54903754349117,
        36.39588736979291,
        38.331131986175016,
        40.35873322129194,
        42.482825463340774,
        44.707723076387886,
        47.03792813339903,
        49.47813847945525,
        52.0332561392095,
        54.7083960832369,
        57.50889536855344,
        60.44032266922822,
        63.50848821369001,
        66.71945414603553,
        70.07954532938118,
        73.59536061006779,
        77.2737845623266,
        81.12199973384801,
        85.14749941356305,
        89.35810094385415,
        93.76195960035359,
        98.36758306347473,
        103.18384650684432,
        108.220008328876,
        113.48572655483767,
        118.9910759379294,
        124.74656578909878,
        130.76315856658476,
        137.0522892574954,
        143.6258855851009,
        150.49638907695012,
        157.67677703041326,
        165.1805854138069,
        173.0219327428786,
        181.21554497411742,
        189.77678145811987,
        198.72166199807484,
        208.06689506034587,
        217.82990718612638,
        228.02887365522017,
        238.68275045517112,
        249.8113076112249,
        261.4351639349625,
        273.5758232519014,
        286.2557121709225,
        299.49821946104777,
        313.3277371038808,
        327.76970309291767,
        342.85064605396616,
        358.59823176405814,
        375.04131164953145,
        392.2099733473787,
        410.13559341753694,
        428.85089229751077,
        448.38999159460394,
        468.78847381508393,
        490.0834446338151,
        512.3135978122962,
        535.5192828776256,
        559.7425756796848,
        585.0273519488229,
        611.4193639815081,
        638.9663205868338,
        667.7179704323966,
        697.726188933959,
        729.0450688394335,
        761.7310146641156,
        795.8428411407625,
        831.4418758550541,
        868.5920662442169,
        907.3600911441422,
        947.8154770781906,
        990.0307194890879,
        1034.0814091238585,
        1080.0463637906662,
        1128.0077657157137,
        1178.0513047380496,
        1230.2663275902223,
        1284.7459935232473,
        1341.5874365453349,
        1400.891934555253,
        1462.7650856631365,
        1527.3169920039707,
        1594.6624513619606,
        1664.9211569374706,
        1738.2179056023324,
        1814.682815003989,
        1894.451549894231,
        1977.665558074262,
        2064.4723163644317,
        2155.0255870243327,
        2249.4856850670135,
        2348.0197569299035,
        2450.8020709846887,
        2558.014320388842,
        2669.8459388028564,
        2786.4944295194714,
        2908.1657085743745,
        3035.0744624320523,
        3167.4445208656243,
        3305.5092456758202,
        3449.5119359215873,
        3599.706250363426,
        3756.3566478502485,
        3919.738846411626,
        4090.140301849616,
        4267.860706658058,
        4453.212510132385,
        4646.521460569648,
        4848.127170496609,
        5058.3837059035795,
        5277.660200503198,
        5506.341496076603,
        5744.828810014521,
        5993.54043120784,
        6252.91244549124,
        6523.399491894537,
        6805.475551009623,
        7099.634766836484,
        7406.392303529549,
        7726.285238526034,
        8059.873493600792,
        8407.740805457774,
        8770.4957375365,
    ]
)
extended_vgrid = np.sqrt(2.0 * el_charge * extended_Egrid / el_mass)


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
    if T * T_0 < 10.00 * Z_0**2:
        return 23.00 - np.log(
            np.sqrt(n * n_0 * 1.00e-6) * Z_0 * (T * T_0) ** (-3.00 / 2.00)
        )
    else:
        return 24.00 - np.log(np.sqrt(n * n_0 * 1.00e-6) / (T * T_0))


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
        f[i] = n * (np.pi * T) ** (-3 / 2) * np.exp(-(v**2) / T)
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
        f[i] = (n1 * (np.pi * T1) ** (-3 / 2) * np.exp(-(v**2) / T1)) + (
            n2 * (np.pi * T2) ** (-3 / 2) * np.exp(-(v**2) / T2)
        )
    f = np.array(f)

    return f


def boltzmann_dist(Te, energies, stat_weights, gnormalise=False):
    """Generate a boltzmann distribution for the given set of energies and statistical weights

    Args:
        Te (np.ndarray): Electron temperature array [eV]
        energies (np.ndarray): Atomic state energies [eV]
        stat_weights (np.ndarray): Atomic state staistical weights
        gnormalise (bool, optional): Option to normalise output densities by their statistical weights. Defaults to False.

    Returns:
        np.ndarray: Boltzmann-distributed densities, relative to ground state
    """
    rel_dens = np.zeros(len(energies))
    for i in range(len(energies)):
        rel_dens[i] = (stat_weights[i] / stat_weights[0]) * np.exp(
            -(energies[i] - energies[0]) / Te
        )
        if gnormalise:
            rel_dens[i] /= stat_weights[i]
    return rel_dens


def saha_dist(Te, ne, imp_dens_tot, impurity):
    """Generate a Saha distribution of ionization stage densities for the given electron temperature

    Args:
        Te (_type_): _description_
        ne (_type_): _description_
        imp_dens_tot (_type_): _description_
        r (_type_): _description_
        el (_type_): _description_
    """
    el_mass = 9.10938e-31
    el_charge = 1.602189e-19
    planck_h = 6.62607004e-34

    ground_states = [s for s in impurity.states if s.ground is True]
    ground_states = list(reversed(sorted(ground_states, key=lambda x: x.num_el)))

    de_broglie_l = np.sqrt((planck_h**2) / (2 * np.pi * el_mass * el_charge * Te))

    # Compute ratios
    dens_ratios = np.zeros(impurity.num_Z - 1)
    for z in range(1, impurity.num_Z):
        eps = -(ground_states[z - 1].energy - ground_states[z].energy)
        stat_weight_zm1 = ground_states[z - 1].stat_weight
        stat_weight = ground_states[z].stat_weight

        dens_ratios[z - 1] = (
            2 * (stat_weight / stat_weight_zm1) * np.exp(-eps / Te)
        ) / (ne * (de_broglie_l**3))

    # Fill densities
    denom_sum = 1.0 + np.sum(
        [np.prod(dens_ratios[: z + 1]) for z in range(impurity.num_Z - 1)]
    )
    dens_saha = np.zeros(impurity.num_Z)
    dens_saha[0] = imp_dens_tot / denom_sum
    for z in range(1, impurity.num_Z):
        dens_saha[z] = dens_saha[z - 1] * dens_ratios[z - 1]

    return dens_saha


def get_maxwellians(ne, Te, vgrid, normalised=True):
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
        vgrid = vgrid.copy()
        vgrid /= v_th

    f0_max = [[0.0 for i in range(len(ne))] for j in range(len(vgrid))]
    for i in range(len(ne)):
        f0_max_loc = maxwellian(Te[i], ne[i], vgrid)
        for j in range(len(vgrid)):
            f0_max[j][i] = f0_max_loc[j]
    f0_max = np.array(f0_max)

    if normalised is False:
        f0_max *= n_norm / v_th**3

    return f0_max


def get_bimaxwellians(n1, n2, T1, T2, vgrid, normalised=True):
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
        n1 = n1.copy()
        n2 = n2.copy()
        T1 = T1.copy()
        T2 = T2.copy()
        n1 /= n_norm
        n2 /= n_norm
        T1 /= T_norm
        T2 /= T_norm
        vgrid = vgrid.copy()
        vgrid /= v_th

    f0_bimax = np.zeros([len(vgrid), len(n1)])
    for i in range(len(n1)):
        f0_bimax_loc = bimaxwellian(T1[i], n1[i], T2[i], n2[i], vgrid)
        for j in range(len(vgrid)):
            f0_bimax[j, i] = f0_bimax_loc[j]
    f0_bimax = np.array(f0_bimax)

    if normalised is False:
        f0_bimax *= n_norm / v_th**3

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
    n = 4 * np.pi * np.sum(f0 * vgrid**2 * dvc)
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
        T = (2 / 3) * 4 * np.pi * np.sum(f0 * vgrid**4 * dvc) / n
    else:
        T = (2 / 3) * 4 * np.pi * 0.5 * el_mass * np.sum(f0 * vgrid**4 * dvc) / n
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
    sigma_tbrec = (
        0.5
        * g_ratio
        * (1 / (np.sqrt(Te) ** 3))
        * sigma_interp
        * ((vgrid_inv / vgrid) ** 2)
    )
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


def interpolate_adf11_data(adas_file, Te, ne, num_z):
    num_x = len(Te)
    interp_data = np.zeros([num_x, num_z - 1])
    for z in range(num_z - 1):
        adas_file_interp = interpolate.interp2d(
            adas_file.logNe, adas_file.logT, adas_file.data[z], kind="linear"
        )
        for i in range(num_x):
            log_ne = np.log10(1e-6 * ne[i])
            log_Te = np.log10(Te[i])
            interp_result = adas_file_interp(log_ne, log_Te)
            interp_data[i, z] = 1e-6 * (10 ** interp_result[0])

    return interp_data
