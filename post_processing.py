import numpy as np
import tools


def get_Zavg(dens, states, num_x):
    """Calculate average ionization

    Args:
        dens (np.ndarray): Impurity densities
        states (_type_): The list of states being modelled
        num_x (_type_): The number of spatial cells

    Returns:
        np.ndarray: 1D array of average ionization at each spatial location
    """

    Zavg = np.zeros(num_x)
    dens_tot = np.sum(dens, 1)

    for i in range(num_x):

        for k, state in enumerate(states):
            z = state.Z
            Zavg[i] += dens[i, k] * float(z)
        if dens_tot[i] > 0:
            Zavg[i] = Zavg[i] / dens_tot[i]

    return Zavg


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
            el.transitions, el.states, type='emission', Z=Z)

        for em_trans in em_transitions:

            cooling_curves[:, Z] += em_trans.delta_E * em_trans.get_mat_value() * \
                dens[:, em_trans.from_pos]

        cooling_curves[:, Z] /= (Z_dens[:, Z] * run.ne)

    cooling_curves *= tools.el_charge * run.T_norm / (run.t_norm * run.n_norm)

    eff_cooling_curve = np.zeros(run.num_x)
    for Z in range(num_Z):
        eff_cooling_curve += np.nan_to_num(
            cooling_curves[:, Z]) * np.nan_to_num(Z_dens[:, Z])
    tot_dens = np.sum(dens, 1)
    eff_cooling_curve /= tot_dens

    # eff_cooling_curve = np.sum([
    #     cooling_curves[:, Z] * Z_dens[:, Z] for Z in range(num_Z)]) / np.sum(Z_dens, 1)

    return cooling_curves, eff_cooling_curve


def get_Z_dens(dens, states):
    """Get the density of each charge stage

    Args:
        dens (np.ndarray): All state densities
        states (list): List of states

    Returns:
        np.ndarray: Array (num_x,num_Z) of densities of each charge stage
    """

    num_Z = states[0].nuc_chg + 1
    num_x = len(dens[:, 0])
    Z_dens = np.zeros([num_x, num_Z])
    for Z in range(num_Z):

        Z_dens_all = gather_dens(dens, states, Z)
        Z_dens[:, Z] = np.sum(Z_dens_all, 1)

    return Z_dens


def gather_dens(dens, states, Z):
    """Gather all densities with a given charge Z

    Args:
        dens (np.ndarray): All state densities
        states (list): List of states
        Z (int): Ion charge Z

    Returns:
        np.ndarray: 2D array of densities with the given charge Z
    """
    num_el_states = gather_states(states, Z)
    state_positions = [s.pos for s in num_el_states]

    gathered_dens = np.zeros([len(dens), len(state_positions)])

    for i, pos in enumerate(state_positions):
        gathered_dens[:, i] = dens[:, pos]

    return gathered_dens


def gather_states(states, Z):
    """Gather states with a given number of electrons

    Args:
        states (list): list of modelled states
        num_el (int): number of electrons

    Returns:
        list: list of states with given number of electrons
    """
    gathered_states = []
    for s in states:
        if s.Z == Z:
            gathered_states.append(s)

    return gathered_states


def gather_transitions(transitions, states=None, type='ionization', Z=None):
    """Return all transitions of a given type and, optionally, for a given initial charge

    Args:
        transitions (list): list of transition objects
        states (list, optional): list of modelled states
        type (str, optional): transition type. Defaults to 'ionization'.
        num_el (int, optional): number of electrons of initial state in transition. If None, gather all

    Returns:
        list: gathered transitions
    """
    gathered_transitions = []

    if Z is None:
        for t in transitions:
            if t.type == type:
                gathered_transitions.append(t)
    else:
        for t in transitions:
            if t.type == type and states[t.from_pos].Z == Z:
                gathered_transitions.append(t)

    return gathered_transitions
