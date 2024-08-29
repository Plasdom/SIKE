import numpy as np

from sike.utils.constants import *
from sike.atomics.atomic_state import State
from sike.atomics.transition import Transition


def boltzmann_dist(
    Te: float, energies: np.ndarray, stat_weights: np.ndarray, gnormalise: bool = False
) -> np.ndarray:
    """Generate a boltzmann distribution for the given set of energies and statistical weights

    :param Te: Electron temperature array [eV]
    :param energies: Atomic state energies [eV]
    :param stat_weights: Atomic state statistical weights
    :param gnormalise: Option to normalise output densities by their statistical weights. Defaults to False
    :return: Boltzmann-distributed densities, relative to ground state
    """
    rel_dens = np.zeros(len(energies))
    for i in range(len(energies)):
        rel_dens[i] = (stat_weights[i] / stat_weights[0]) * np.exp(
            -(energies[i] - energies[0]) / Te
        )
        if gnormalise:
            rel_dens[i] /= stat_weights[i]
    return rel_dens


def saha_dist(
    Te: float, ne: float, imp_dens_tot: float, states: list[State], num_Z: int
) -> np.ndarray:
    """Generate a Saha distribution of ionization stage densities for the given electron temperature

    :param Te: Electron temperature [eV]
    :param ne: Electron density [m^-3]
    :param imp_dens_tot: Total impurity species density [m^-3]
    :param states: Impurity atomic states
    :param num_Z: Number of ionisation stages
    :return: Numpy array of Saha distribution of ionisation stage densities
    """
    ground_states = [s for s in states if s.ground is True]
    ground_states = list(reversed(sorted(ground_states, key=lambda x: x.num_el)))

    de_broglie_l = np.sqrt((PLANCK_H**2) / (2 * np.pi * EL_MASS * EL_CHARGE * Te))

    # Compute ratios
    dens_ratios = np.zeros(num_Z - 1)
    for z in range(1, num_Z):
        eps = -(ground_states[z - 1].energy - ground_states[z].energy)
        stat_weight_zm1 = ground_states[z - 1].stat_weight
        stat_weight = ground_states[z].stat_weight

        dens_ratios[z - 1] = (
            2 * (stat_weight / stat_weight_zm1) * np.exp(-eps / Te)
        ) / (ne * (de_broglie_l**3))

    # Fill densities
    denom_sum = 1.0 + np.sum([np.prod(dens_ratios[: z + 1]) for z in range(num_Z - 1)])
    dens_saha = np.zeros(num_Z)
    dens_saha[0] = imp_dens_tot / denom_sum
    for z in range(1, num_Z):
        dens_saha[z] = dens_saha[z - 1] * dens_ratios[z - 1]

    return dens_saha


def get_Zavg(dens: np.ndarray, states: list[State], num_x: int = None) -> np.ndarray:
    """Calculate average ionization

    :param dens: Impurity densities
    :param states: The list of states being modelled
    :param num_x: The number of spatial cells, defaults to None
    :return: 1D array of average ionization at each spatial location
    """
    if num_x is None:
        num_x = len(dens[:, 0])
    Zavg = np.zeros(num_x)
    dens_tot = np.sum(dens, 1)

    for i in range(num_x):
        for k, state in enumerate(states):
            z = state.Z
            Zavg[i] += dens[i, k] * float(z)
        if dens_tot[i] > 0:
            Zavg[i] = Zavg[i] / dens_tot[i]

    return Zavg


def get_Z_dens(dens: np.ndarray, states: list[State]) -> np.ndarray:
    """Get the density of each charge state

    :param dens: All state densities
    :param states: List of states
    :return: Array (num_x,num_Z) of densities of each charge stage
    """
    num_Z = states[0].nuc_chg + 1
    num_x = len(dens[:, 0])
    Z_dens = np.zeros([num_x, num_Z])
    for Z in range(num_Z):
        Z_dens_all = gather_dens(dens, states, Z)
        Z_dens[:, Z] = np.sum(Z_dens_all, 1)

    return Z_dens


def gather_dens(dens: np.ndarray, states: list[State], Z: int) -> np.ndarray:
    """Gather all densities with a given charge Z

    :param dens: All state densities
    :param states: List of states
    :param Z: Ion charge Z
    :return: 2D array of densities with the given charge Z
    """
    num_el_states = gather_states(states, Z)
    state_positions = [s.pos for s in num_el_states]

    gathered_dens = np.zeros([len(dens), len(state_positions)])

    for i, pos in enumerate(state_positions):
        gathered_dens[:, i] = dens[:, pos]

    return gathered_dens


def gather_states(states, Z) -> list[State]:
    """Gather states with a given number of electrons

    :param states: list of modelled states
    :param Z: Ion charge Z
    :return: list of states with given number of electrons
    """
    gathered_states = []
    for s in states:
        if s.Z == Z:
            gathered_states.append(s)

    return gathered_states


def gather_transitions(
    transitions: list[Transition],
    states: list[State],
    type: str = "ionization",
    Z: int = None,
) -> list[Transition]:
    """Return all transitions of a given type and, optionally, for a given initial charge

    :param transitions: list of transition objects
    :param states: list of modelled states
    :param type: transition type, defaults to "ionization"
    :param Z: ion charge Z in initial state in transition. If None, gather all, defaults to None
    :return: gathered transitions
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
