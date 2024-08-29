import numpy as np
import os
import json
import copy

from sike.core import *
from sike.analysis.impurity_utils import *


def load_sikerun_from_dir(rdir, el, check_for_cr_coeffs=True):
    """Create and return a SIKERun object from text files in a given input directory.

    Args:
        rdir (str): Directory containing SIKERun data
        el (str): element
        check_for_cr_coeffs (bool, optional): whether to look for iz and rec coefficients

    Returns:
        SIKERun: SIKERun object of the given element with densities set to those in the input directory
    """
    with open(os.path.join(rdir, el + "_opts.json")) as f:
        opts = json.load(f)

    if opts["kinetic_electrons"] and opts["maxwellian_electrons"]:
        with open(os.path.join(rdir, el + "_dens.txt")) as f:
            dens = np.loadtxt(f)
        with open(os.path.join(rdir, el + "_dens_Max.txt")) as f:
            dens_Max = np.loadtxt(f)
        with open(os.path.join(rdir, "fe.txt")) as f:
            fe = np.loadtxt(f)
        with open(os.path.join(rdir, "vgrid.txt")) as f:
            vgrid = np.loadtxt(f)
        r = core.SIKERun(fe=fe, vgrid=vgrid, opts=opts)
        r.impurities[el].dens = dens
        r.impurities[el].dens_Max = dens_Max

    if opts["maxwellian_electrons"] and not opts["kinetic_electrons"]:
        with open(os.path.join(rdir, el + "_dens_Max.txt")) as f:
            dens_Max = np.loadtxt(f)
        with open(os.path.join(rdir, "Te.txt")) as f:
            Te = np.loadtxt(f)
        with open(os.path.join(rdir, "ne.txt")) as f:
            ne = np.loadtxt(f)
        try:
            with open(os.path.join(rdir, "vgrid.txt")) as f:
                vgrid = np.loadtxt(f)
        except:
            vgrid = None
        r = core.SIKERun(ne=ne, Te=Te, vgrid=vgrid, opts=opts)
        r.impurities[el].dens_Max = dens_Max

    if check_for_cr_coeffs:
        if opts["kinetic_electrons"] and opts["maxwellian_electrons"]:
            if os.path.isfile(os.path.join(rdir, el + "_iz_coeffs.txt")):
                r.impurities[el].iz_coeffs = np.loadtxt(
                    os.path.join(rdir, el + "_iz_coeffs.txt")
                )
                r.impurities[el].iz_coeffs_Max = np.loadtxt(
                    os.path.join(rdir, el + "_iz_coeffs_Max.txt")
                )
                r.impurities[el].rec_coeffs = np.loadtxt(
                    os.path.join(rdir, el + "_rec_coeffs.txt")
                )
                r.impurities[el].rec_coeffs_Max = np.loadtxt(
                    os.path.join(rdir, el + "_rec_coeffs_Max.txt")
                )
        if opts["maxwellian_electrons"] and not opts["kinetic_electrons"]:
            if os.path.isfile(os.path.join(rdir, el + "_iz_coeffs_Max.txt")):
                r.impurities[el].iz_coeffs_Max = np.loadtxt(
                    os.path.join(rdir, el + "_iz_coeffs_Max.txt")
                )
                r.impurities[el].rec_coeffs_Max = np.loadtxt(
                    os.path.join(rdir, el + "_rec_coeffs_Max.txt")
                )

    return r


def update_sikerun_from_dir(r, rdir, el, check_for_cr_coeffs=True):
    """Create and return a SIKERun object from text files in a given input directory.

    Args:
        rdir (str): Directory containing SIKERun data
        el (str): element
        check_for_cr_coeffs (bool, optional): whether to look for iz and rec coefficients

    Returns:
        SIKERun: SIKERun object of the given element with densities set to those in the input directory
    """
    with open(os.path.join(rdir, el + "_opts.json")) as f:
        opts = json.load(f)

    if opts["kinetic_electrons"] and opts["maxwellian_electrons"]:
        with open(os.path.join(rdir, el + "_dens.txt")) as f:
            dens = np.loadtxt(f)
        with open(os.path.join(rdir, el + "_dens_Max.txt")) as f:
            dens_Max = np.loadtxt(f)
        with open(os.path.join(rdir, "fe.txt")) as f:
            fe = np.loadtxt(f)
        with open(os.path.join(rdir, "vgrid.txt")) as f:
            vgrid = np.loadtxt(f)
        r.fe = fe
        r.vgrid = vgrid
        r.init_from_dist()
        r.impurities[el].dens = dens
        r.impurities[el].dens_Max = dens_Max

    if opts["maxwellian_electrons"] and not opts["kinetic_electrons"]:
        with open(os.path.join(rdir, el + "_dens_Max.txt")) as f:
            dens_Max = np.loadtxt(f)
        with open(os.path.join(rdir, "Te.txt")) as f:
            Te = np.loadtxt(f)
        with open(os.path.join(rdir, "ne.txt")) as f:
            ne = np.loadtxt(f)
        r.ne = ne
        r.Te = Te
        r.init_from_profiles()
        r.impurities[el].dens_Max = dens_Max

    if check_for_cr_coeffs:
        if opts["kinetic_electrons"] and opts["maxwellian_electrons"]:
            if os.path.isfile(os.path.join(rdir, el + "_iz_coeffs.txt")):
                r.impurities[el].iz_coeffs = np.loadtxt(
                    os.path.join(rdir, el + "_iz_coeffs.txt")
                )
                r.impurities[el].iz_coeffs_Max = np.loadtxt(
                    os.path.join(rdir, el + "_iz_coeffs_Max.txt")
                )
                r.impurities[el].rec_coeffs = np.loadtxt(
                    os.path.join(rdir, el + "_rec_coeffs.txt")
                )
                r.impurities[el].rec_coeffs_Max = np.loadtxt(
                    os.path.join(rdir, el + "_rec_coeffs_Max.txt")
                )
        if opts["maxwellian_electrons"] and not opts["kinetic_electrons"]:
            if os.path.isfile(os.path.join(rdir, el + "_iz_coeffs_Max.txt")):
                r.impurities[el].iz_coeffs_Max = np.loadtxt(
                    os.path.join(rdir, el + "_iz_coeffs_Max.txt")
                )
                r.impurities[el].rec_coeffs_Max = np.loadtxt(
                    os.path.join(rdir, el + "_rec_coeffs_Max.txt")
                )

    return r


def load_sikerundeck(
    outputdir, el, identifier="Output_", check_for_cr_coeffs=True, full_load=False
):
    """Load a rundeck of SIKERun objects from a directory of the specified element.

    Args:
        outputdir (_type_): _description_
        el (_type_): _description_

    Returns:
        List: List of SIKERun objects
    """
    rdirs = [
        os.path.join(outputdir, d) for d in os.listdir(outputdir) if identifier in d
    ]
    sike_runs = []

    # Load up the first run
    r = load_sikerun_from_dir(rdirs[0], el, check_for_cr_coeffs)
    sike_runs.append(r)

    if full_load:
        for rdir in rdirs[1:]:
            r = load_sikerun_from_dir(rdir, el, check_for_cr_coeffs)
            sike_runs.append(r)

    else:
        # For remaining runs, copy and update the densities and profiles
        for rdir in rdirs[1:]:
            r = copy.copy(sike_runs[0])
            if hasattr(sike_runs[0].impurities[el], "dens"):
                r.impurities[el].dens = copy.deepcopy(sike_runs[0].impurities[el].dens)
            if hasattr(sike_runs[0].impurities[el], "dens_Max"):
                r.impurities[el].dens_Max = copy.deepcopy(
                    sike_runs[0].impurities[el].dens_Max
                )
            update_sikerun_from_dir(r, rdir, el, check_for_cr_coeffs)
            sike_runs.append(r)

    # Sort by density
    densities = [None] * len(sike_runs)
    for i, run in enumerate(sike_runs):
        densities[i] = run.n_norm
    sike_runs = [
        r for _, r in sorted(zip(densities, sike_runs), key=lambda pair: pair[0])
    ]

    return sike_runs


def get_gs_iz_coeffs(r, el, kinetic=False):
    """Calculate the ionization coefficients from/to ground state of each ionization stage.

    Args:
        r (SIKERun): SIKERun object
        el (str): element
        kinetic (bool, optional): whether to calculate kinetic or maxwellian rates. Defaults to False.

    Returns:
        np.ndarray: 2D array of ground state ionization coefficients (num_x, num_Z-1)
    """
    if kinetic:
        fe = r.fe
    else:
        fe = r.fe_Max

    num_Z = r.impurities[el].num_Z
    gs_iz_coeffs = np.zeros([r.num_x, num_Z - 1])

    for Z in range(num_Z - 1):
        Z_states = gather_states(r.impurities[el].states, Z)
        gs = Z_states[0]
        Zplus1_states = gather_states(r.impurities[el].states, Z + 1)
        gs_Zplus1 = Zplus1_states[0]
        for s in r.impurities[el].transitions:
            if (
                s.type == "ionization"
                and s.from_id == gs.id
                and s.to_id == gs_Zplus1.id
            ):
                iz_trans = s
                break

        for k in range(r.num_x):
            iz_rate = iz_trans.get_mat_value(fe[:, k], r.vgrid, r.dvc) / r.ne[k]
            gs_iz_coeffs[k, Z] = iz_rate / (r.n_norm * r.t_norm)

    return gs_iz_coeffs


def get_cooling_curves(run, element, kinetic=True):
    """Calculate the radiation per ion profiles

    Args:
        run (SIKERun): An equilibrated SIKERun object
        element (str): The element to be calculated

    Returns:
        cooling_curves (np.ndarray): A 2D numpy array (num_z, num_x) of cooling curves at each spatial location
        eff_cooling_curve (np.ndarray): A 1D numpy array (num_x) of the effective cooling curve I.e. weighted by density of each ionization stage) at each spatial location
    """
    num_Z = run.impurities[element].num_Z

    cooling_curves = np.zeros([run.num_x, num_Z])

    el = run.impurities[element]

    if kinetic:
        dens = run.impurities[element].dens
        fe = run.fe
    else:
        dens = run.impurities[element].dens_Max
        fe = run.fe_Max

    Z_dens = get_Z_dens(dens, el.states)

    for Z in range(num_Z):
        em_transitions = gather_transitions(
            el.transitions, el.states, type="emission", Z=Z
        )

        for em_trans in em_transitions:
            cooling_curves[:, Z] += (
                em_trans.delta_E * em_trans.get_mat_value() * dens[:, em_trans.from_pos]
            )

        cooling_curves[:, Z] /= Z_dens[:, Z] * run.ne

        # cooling_curves[np.where(Z_dens[:,Z] < 0.0), Z] = 0.0

    cooling_curves *= EL_CHARGE * run.T_norm / (run.t_norm * run.n_norm)

    eff_cooling_curve = np.zeros(run.num_x)
    for Z in range(num_Z):
        eff_cooling_curve += np.nan_to_num(cooling_curves[:, Z]) * np.nan_to_num(
            Z_dens[:, Z]
        )
    tot_dens = np.sum(dens, 1)
    eff_cooling_curve /= tot_dens

    # eff_cooling_curve = np.sum([
    #     cooling_curves[:, Z] * Z_dens[:, Z] for Z in range(num_Z)]) / np.sum(Z_dens, 1)

    return cooling_curves, eff_cooling_curve


def get_cr_iz_coeffs(r, el, kinetic=False):
    # TODO: Delete this function in favour of doing it the sensible way (M_eff)
    """Calculate the collisional-radiative ionization coefficients, as per Summers, P. et al. PPCF (2006)

    Args:
        r (SIKERun): SIKERun object
        el (str): element
        kinetic (bool, optional): whether to calculate kinetic or maxwellian rates. Defaults to False.

    Returns:
        np.ndarray: 2D array of CR ionization coefficients (num_x, num_Z-1)
    """

    r.calc_eff_rate_mats(kinetic=kinetic)
    cr_iz_coeffs = np.zeros([r.loc_num_x, r.impurities[el].num_Z - 1])

    for x_pos in range(r.min_x, r.max_x):
        for Z in range(r.impurities[el].num_Z - 1):
            if kinetic:
                cr_iz_coeffs[x_pos - r.min_x, Z] = -r.eff_rate_mats[el][
                    x_pos - r.min_x
                ][Z + 1, Z] / (r.ne[x_pos] * r.n_norm * r.t_norm)
            else:
                cr_iz_coeffs[x_pos - r.min_x, Z] = -r.eff_rate_mats_Max[el][
                    x_pos - r.min_x
                ][Z + 1, Z] / (r.ne[x_pos] * r.n_norm * r.t_norm)

    return cr_iz_coeffs


def get_cr_rec_coeffs(r, el, kinetic=False):
    """Calculate the collisional-radiative recombination coefficients

    Args:
        r (SIKERun): SIKERun object
        el (str): element
        kinetic (bool, optional): whether to calculate kinetic or maxwellian rates. Defaults to False.

    Returns:
        np.ndarray: 2D array of CR recombination coefficients (num_x, num_Z-1)
    """

    r.calc_eff_rate_mats(kinetic=kinetic)
    cr_rec_coeffs = np.zeros([r.loc_num_x, r.impurities[el].num_Z - 1])

    for x_pos in range(r.min_x, r.max_x):
        for Z in range(r.impurities[el].num_Z - 1):
            if kinetic:
                cr_rec_coeffs[x_pos - r.min_x, Z] = -r.eff_rate_mats[el][
                    x_pos - r.min_x
                ][Z, Z + 1] / (r.ne[x_pos] * r.n_norm * r.t_norm)
            else:
                cr_rec_coeffs[x_pos - r.min_x, Z] = -r.eff_rate_mats_Max[el][
                    x_pos - r.min_x
                ][Z, Z + 1] / (r.ne[x_pos] * r.n_norm * r.t_norm)

    return cr_rec_coeffs
