from pathlib import Path
import numpy as np

# from sike.plasma_utils import energy2velocity

ATOMIC_DATA_LOCATION = "sike_atomic_data"
ATOMIC_DATA_BASE_URL = "https://zenodo.org/records/14205937/files/"
SYMBOL2ELEMENT = {
    "H": "Hydrogen",
    "He": "Helium",
    "Li": "Lithium",
    "Be": "Beryllium",
    "B": "Boron",
    "C": "Carbon",
    "O": "Oxygen",
    "N": "Nitrogen",
    "Ne": "Neon",
    "Al": "Aluminium",
    "Ar": "Argon",
    "Fe": "Iron",
    "Kr": "Krypton",
    "Mo": "Molybdenum",
    "W": "Tungsten",
    "Xe": "Xenon",
}
ELEMENT2SYMBOL = {v: k for k, v in SYMBOL2ELEMENT.items()}
CONFIG_FILENAME = ".sike_config"
NUCLEAR_CHARGE_DICT = {
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "Ne": 10,
    "Al": 13,
    "Ar": 18,
    "Fe": 26,
    "Kr": 36,
    "Mo": 42,
    "Xe": 54,
    "W": 74,
}


# Physical constants
EL_MASS = 9.10938e-31
ION_MASS = 2 * 1.67262e-27
EPSILON_0 = 8.854188e-12
EL_CHARGE = 1.602189e-19
BOLTZMANN_K = 1.38064852e-23
BOHR_RADIUS = 5.291772e-11
PLANCK_H = 6.62607004e-34
LIGHT_SPEED = 299792458.0

# Default velocity grid to use when initialising SIKE run from temperature and density profiles
DEFAULT_VGRID = np.array(
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

MARCHAND_SCREENING_COEFFS = np.array(
    [
        [0.5966, 0.8597, 0.9923, 0.9800, 0.9725, 0.9970, 0.9990, 0.999, 0.9999, 0.9999],
        [
            0.2345,
            0.6888,
            0.8877,
            0.9640,
            1.0000,
            0.9880,
            0.9900,
            0.9990,
            0.9999,
            0.9999,
        ],
        [
            0.1093,
            0.4018,
            0.7322,
            0.9415,
            0.9897,
            0.9820,
            0.9860,
            0.9900,
            0.9920,
            0.9999,
        ],
        [
            0.0622,
            0.2430,
            0.5150,
            0.6986,
            0.8590,
            0.9600,
            0.9750,
            0.9830,
            0.9860,
            0.9900,
        ],
        [
            0.0399,
            0.1597,
            0.3527,
            0.5888,
            0.8502,
            0.8300,
            0.9000,
            0.9500,
            0.9700,
            0.9800,
        ],
        [
            0.0277,
            0.1098,
            0.2455,
            0.4267,
            0.5774,
            0.7248,
            0.8300,
            0.9000,
            0.9500,
            0.9700,
        ],
        [
            0.0204,
            0.0808,
            0.1811,
            0.3184,
            0.4592,
            0.6098,
            0.7374,
            0.8300,
            0.9000,
            0.9500,
        ],
        [
            0.0156,
            0.0624,
            0.1392,
            0.2457,
            0.3711,
            0.5062,
            0.6355,
            0.7441,
            0.8300,
            0.9000,
        ],
        [
            0.0123,
            0.0493,
            0.1102,
            0.1948,
            0.2994,
            0.4222,
            0.5444,
            0.6558,
            0.7553,
            0.8300,
        ],
        [
            0.0100,
            0.0400,
            0.0900,
            0.1584,
            0.2450,
            0.3492,
            0.4655,
            0.5760,
            0.6723,
            0.7612,
        ],
    ]
).T
