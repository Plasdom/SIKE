import numpy as np
import os
import json

from sike.atomics.transition import *
from sike.atomics.atomic_state import State
from sike.analysis.impurity_utils import saha_dist, boltzmann_dist, gather_states
import sike.utils.constants as c


class Impurity:
    """Impurity class to hold information on the states and transitions for a given modelled impurity species."""

    def __init__(
        self,
        name: str,
        resolve_l: bool,
        resolve_j: bool,
        state_ids: list[int],
        saha_boltzmann_init: bool,
        fixed_fraction_init: bool,
        frac_imp_dens: float,
        ionization: bool,
        autoionization: bool,
        emission: bool,
        radiative_recombination: bool,
        excitation: bool,
        collrate_const: float,
        tbrec_norm: float,
        sigma_norm: float,
        time_norm: float,
        T_norm: float,
        n_norm: float,
        vgrid: np.ndarray,
        Egrid: np.ndarray,
        ne: np.ndarray,
        Te: np.ndarray,
        atom_data_savedir: Path,
    ):
        """Initialise

        :param name: Name of the impurity
        :param resolve_l: Whether to resolve states by orbital angular momentum quantum number l
        :param resolve_j: Whether to resolve by total angular momentum quantum number j
        :param state_ids: List of state IDs to evolve (default is for all states to be included)
        :param saha_boltzmann_init: Whether to initialise state distribution to Saha-Boltzmann equilibria
        :param fixed_fraction_init: Whether to initialise total impurity density to a fixed fraction of the electron density
        :param frac_imp_dens: Fractional impurity density to initialise (total) impurity densities with, if fixed_fraction_init is True
        :param ionization: Whether to include ionization transitions and inverse
        :param autoionization: Whether to include autoionization transitions
        :param emission: Whether to include spontaneous emission transitions
        :param radiative_recombination: Whether to include radiative recombination transitions
        :param excitation: Whether to include collisional excitation transitions and inverse
        :param collrate_const: Collision rate normalisation constant
        :param tbrec_norm: Three-body recombination rate normalisation constant
        :param sigma_norm: Cross-section normalisation constant
        :param time_norm: Time normalisation constant
        :param T_norm: Temperature normalisation constant
        :param n_norm: Density normalisation constant
        :param vgrid: Electron velocity grid
        :param Egrid: Electron energy grid
        :param ne: Electron densities
        :param Te: Electron temperatures
        """

        # Save settings
        self.name = name
        self.resolve_j = resolve_j
        self.resolve_l = resolve_l
        self.state_ids = state_ids
        self.saha_boltzmann_init = saha_boltzmann_init
        self.fixed_fraction_init = fixed_fraction_init
        self.frac_imp_dens = frac_imp_dens
        self.ionization = ionization
        self.autoionization = autoionization
        self.emission = emission
        self.radiative_recombination = radiative_recombination
        self.excitation = excitation
        self.collrate_const = collrate_const
        self.tbrec_norm = tbrec_norm
        self.sigma_norm = sigma_norm
        self.time_norm = time_norm
        self.T_norm = T_norm
        self.n_norm = n_norm
        self.atom_data_savedir = atom_data_savedir

        # Initialise impurity data
        self.get_element_data()
        self.check_data_exists()
        print(" Initialising states...")
        self.init_states()
        print(" Initialising transitions...")
        self.init_transitions(vgrid=vgrid, Egrid=Egrid)
        print(" Initialising densities...")
        self.init_dens(ne=ne, Te=Te)
        print(" Finalising states...")
        self.set_state_positions()
        self.set_transition_positions()

    def check_data_exists(self):
        """Check that the atomic data for the given species and level of state resolution exists.

        :raises FileNotFoundError: If requested atomic data does not exist.
        :raises Exception: If some other unhandled exception occurs.
        """
        if not (self.atom_data_savedir / self.longname).exists():
            raise FileNotFoundError(
                "No atomic data was found for "
                + self.longname
                + ". Please run setup, including '"
                + self.name
                + "' in the list of elements to ensure atomic data is downloaded (see readme for instructions)."
            )

        # Find what data exists
        levels_filepath_nlj = (
            self.atom_data_savedir / self.longname / (self.name + "_levels_nlj.json")
        )
        # levels_filepath_nl = (
        #     self.atom_data_savedir / self.longname / (self.name + "_levels_nl.json")
        # )
        levels_filepath_n = (
            self.atom_data_savedir / self.longname / (self.name + "_levels_n.json")
        )
        j_resolved_exists = levels_filepath_nlj.exists()
        # l_resolved_exists = levels_filepath_nl.exists()
        n_resolved_exists = levels_filepath_n.exists()

        # Compare with input options
        if (self.resolve_l and self.resolve_j) or (
            self.resolve_l and not self.resolve_j
        ):
            # Resolved in both l and j
            if j_resolved_exists:
                return
            else:
                raise FileNotFoundError(
                    "Data for "
                    + self.longname
                    + " resolved in l or j was not found. Set resolve_j=False and resolve_l=False."
                )
                # if l_resolved_exists and not n_resolved_exists:
                #     # Set resolve_j = False and resolve_l = True
                #     raise FileNotFoundError(
                #         "Data for "
                #         + self.longname
                #         + " resolved in both l and j was not found. Set resolve_j=False."
                #     )
                # elif not l_resolved_exists and n_resolved_exists:
                #     # Set resolve_j = False and resolve_l = False
                #     raise FileNotFoundError(
                #         "Data for "
                #         + self.longname
                #         + " resolved in both l and j was not found. Set resolve_j=False and resolve_l=False."
                #     )
                # else:
                #     raise Exception
        elif self.resolve_j and not self.resolve_l:
            # Resolved in j but not l - this does not make sense
            raise Exception("resolve_j=True is not compatible with resolve_l=False.")
        # elif not self.resolve_j and self.resolve_l:
        #     # Resolved in l but not j
        #     if l_resolved_exists:
        #         return
        #     else:
        #         if n_resolved_exists:
        #             raise FileNotFoundError(
        #                 "Data for "
        #                 + self.longname
        #                 + " resolved in l was not found. Set resolve_l=False."
        #             )
        #         else:
        #             raise Exception
        elif not self.resolve_j and not self.resolve_j:
            # Resolved in n only
            if n_resolved_exists:
                return
            else:
                if j_resolved_exists:
                    raise FileNotFoundError(
                        "Data for "
                        + self.longname
                        + " resolved in only n was not found. Set resolve_l=True and resolve_j=True, or resolve_l=True and resolve_j=False."
                    )
                else:
                    raise Exception

    def get_element_data(self):
        """Set the nuclear charge and number of ionisation stages"""

        self.nuc_chg = c.NUCLEAR_CHARGE_DICT[self.name]
        self.num_Z = self.nuc_chg + 1
        self.longname = c.SYMBOL2ELEMENT[self.name]

    def init_states(self):
        """Initialise the evolved atomic states"""
        if (self.resolve_l and self.resolve_j) or (
            self.resolve_l and not self.resolve_j
        ):
            levels_f = (
                self.atom_data_savedir
                / self.longname
                / (self.name + "_levels_nlj.json")
            )
        else:
            # if self.resolve_l:
            #     levels_f = (
            #         self.atom_data_savedir
            #         / self.longname
            #         / (self.name + "_levels_nl.json")
            #     )
            # else:
            levels_f = (
                self.atom_data_savedir / self.longname / (self.name + "_levels_n.json")
            )
        with open(levels_f) as f:
            levels_dict = json.load(f)
            self.states = [None] * len(levels_dict)
            for i, level_dict in enumerate(levels_dict):
                self.states[i] = State(**level_dict)

        # Keep only user-specified states
        if self.state_ids is not None:
            for i, state in enumerate(self.states):
                if state.id not in self.state_ids:
                    self.states[i] = None
        self.states = [s for s in self.states if s is not None]

        self.tot_states = len(self.states)

        self.init_ionization_energies()

    def init_ionization_energies(self):
        """Set the ground state levels, ionization energies, delta E from ground state for each atomic state"""

        # Find the lowest energy states
        gs_energies = np.zeros(self.num_Z)
        gs_pos = np.zeros(self.num_Z, dtype=int)
        for Z in range(self.num_Z):
            Z_states = [s for s in self.states if s.Z == Z]
            energies = [s.energy for s in Z_states]
            gs = Z_states[np.argmin(energies)]
            for i, s in enumerate(self.states):
                if s.equals(gs):
                    gs_pos[Z] = i
                    gs_energies[Z] = gs.energy
                    break

        # Mark ground states and calculate ionization energy
        for i in range(len(self.states)):
            if i in gs_pos:
                self.states[i].ground = True
            else:
                self.states[i].ground = False

            if self.states[i].Z < self.num_Z - 1:
                iz_energy = gs_energies[self.states[i].Z + 1] - self.states[i].energy
                self.states[i].iz_energy = iz_energy
            else:
                self.states[i].iz_energy = 0.0

            energy_from_gs = self.states[i].energy - gs_energies[self.states[i].Z]
            self.states[i].energy_from_gs = energy_from_gs

    def init_transitions(
        self,
        vgrid: np.ndarray,
        Egrid: np.ndarray,
    ):
        """Initialise all transitions between evolved atomic states

        :param vgrid: Electron velocity grid
        :param Egrid: Electron energy grid
        """
        if (self.resolve_l and self.resolve_j) or (
            self.resolve_l and not self.resolve_j
        ):
            trans_f = (
                self.atom_data_savedir
                / self.longname
                / (self.name + "_transitions_nlj.json")
            )
        else:
            # if self.resolve_l:
            #     trans_f = (
            #         self.atom_data_savedir
            #         / self.longname
            #         / (self.name + "_transitions_nl.json")
            #     )
            # else:
            trans_f = (
                self.atom_data_savedir
                / self.longname
                / (self.name + "_transitions_n.json")
            )
        print("  Loading transitions from json...")
        with open(trans_f) as f:
            trans_dict = json.load(f)
            # trans_Egrid = np.array(trans_dict[0]["E_grid"])

        print("  Creating transition objects...")
        num_transitions = len(trans_dict)
        transitions = [None] * num_transitions

        for i, trans in enumerate(trans_dict[1:]):
            if self.state_ids is not None:
                if (trans["from_id"] not in self.state_ids) or (
                    trans["to_id"] not in self.state_ids
                ):
                    continue
            if trans["type"] == "ionization" and self.ionization:
                transitions[i] = IzTrans(
                    **trans,
                    collrate_const=self.collrate_const,
                    sigma_norm=self.sigma_norm,
                    tbrec_norm=self.tbrec_norm,
                    simulation_E_grid=Egrid,
                    from_state=self.states[trans["from_id"]],
                    to_state=self.states[trans["to_id"]],
                )
            elif trans["type"] == "autoionization" and self.autoionization:
                transitions[i] = AiTrans(**trans, time_norm=self.time_norm)
            elif (
                trans["type"] == "radiative_recombination"
                and self.radiative_recombination
            ):
                transitions[i] = RRTrans(
                    **trans,
                    collrate_const=self.collrate_const,
                    sigma_norm=self.sigma_norm,
                    simulation_E_grid=Egrid,
                    from_state=self.states[trans["from_id"]],
                    to_state=self.states[trans["to_id"]],
                )
            elif trans["type"] == "emission" and self.emission:
                transitions[i] = EmTrans(**trans, time_norm=self.time_norm)
            elif trans["type"] == "excitation" and self.excitation:
                transitions[i] = ExTrans(
                    **trans,
                    collrate_const=self.collrate_const,
                    sigma_norm=self.sigma_norm,
                    simulation_E_grid=Egrid,
                )
        transitions = [t for t in transitions if t is not None]

        self.transitions = transitions

        # # Calculate cross-sections on the given energy grid
        # if (self.resolve_l and self.resolve_j) or (
        #     self.resolve_l and not self.resolve_j
        # ):
        #     self.interpolate_cross_sections(Egrid)
        # else:
        #     self.compute_cross_sections(Egrid)

        # Set the de-excitation cross-sections
        print("  Creating data for inverse transitions...")
        id2pos = {self.states[i].id: i for i in range(len(self.states))}
        if self.excitation:
            for i, t in enumerate(self.transitions):
                if t.type == "excitation":
                    g_ratio = (
                        self.states[id2pos[t.from_id]].stat_weight
                        / self.states[id2pos[t.to_id]].stat_weight
                    )
                    t.set_sigma_deex(g_ratio, vgrid)

        # Set the statistical weight ratios for ionization cross-sections
        if self.ionization:
            for i, t in enumerate(self.transitions):
                if t.type == "ionization":
                    g_ratio = (
                        self.states[id2pos[t.from_id]].stat_weight
                        / self.states[id2pos[t.to_id]].stat_weight
                    )
                    t.set_inv_data(g_ratio, vgrid)

        # Checks
        print("  Performing checks on transition data...")
        self.state_and_transition_checks()

    # def interpolate_cross_sections(self, Egrid: np.ndarray):
    #     # Interpolate the cross sections
    #     for t in self.transitions:
    #         if (
    #             t.type == "excitation"
    #             or t.type == "ionization"
    #             or t.type == "radiative recombination"
    #         ):
    #             t.interpolate_cross_section(new_Egrid=Egrid)

    # def compute_cross_sections(self, Egrid: np.ndarray):
    #     pass

    def state_and_transition_checks(self):
        """Perform some checks on states and transitions belonging to the impurity. Removes orphaned states, transitions where one or more associated states are not evolved, etc"""

        # Check for no orphaned states (i.e. states with either no associated transitions or )
        id2pos = {self.states[i].id: i for i in range(len(self.states))}
        from_ids = np.array([t.from_id for t in self.transitions], dtype=int)
        to_ids = np.array([t.to_id for t in self.transitions], dtype=int)
        for i, state in enumerate(self.states):
            associated_transitions = get_associated_transitions(
                state.id, from_ids, to_ids
            )
            if len(associated_transitions) == 0:
                print("State ID " + str(state.id) + " is an orphaned state, removing.")
                self.states[i] = None
                self.tot_states -= 1
        self.states = [s for s in self.states if s is not None]

        # Remove states above ionization energy if no autoionization
        # TODO: Is this necessary?
        if self.autoionization is False:
            for i, state in enumerate(self.states):
                if state.iz_energy < 0:
                    self.states[i] = None
                    self.tot_states -= 1
        self.states = [s for s in self.states if s is not None]

        # Remove excitation/ionisation transitions with negative transition energy
        for i, trans in enumerate(self.transitions):
            if trans.type == "excitation" or trans.type == "ionization":
                if trans.delta_E < 0.0:
                    self.transitions[i] = None
                    print(
                        "Removing {} transition with transition energy < 0 eV".format(
                            trans.type
                        )
                    )
        self.transitions = [t for t in self.transitions if t is not None]

        # Check for no orphaned transitions (i.e. transitions where either from_id or to_id is not evolved)
        state_ids = [s.id for s in self.states]
        for i, trans in enumerate(self.transitions):
            if trans.from_id not in state_ids or trans.to_id not in state_ids:
                self.transitions[i] = None
        self.transitions = [t for t in self.transitions if t is not None]

    def init_dens(
        self,
        ne: np.ndarray,
        Te: np.ndarray,
    ):
        """Initialise densities of impurity states

        :param ne: Electron densities
        :param Te: Electron temperatures
        """
        self.dens = np.zeros((len(ne), self.tot_states))

        if self.saha_boltzmann_init:
            self.set_state_positions()

            Z_dens = np.zeros([len(ne), self.num_Z])
            for i in range(len(ne)):
                Z_dens[i, :] = (
                    saha_dist(
                        Te[i] * self.T_norm,
                        ne[i] * self.n_norm,
                        self.n_norm,
                        self.states,
                        self.num_Z,
                    )
                    / self.n_norm
                )

            for Z in range(self.num_Z):
                Z_states = gather_states(self.states, Z)

                energies = [s.energy for s in Z_states]
                stat_weights = [s.stat_weight for s in Z_states]
                locs = [s.pos for s in Z_states]
                for i in range(len(ne)):
                    Z_dens_loc = Z_dens[i, Z]

                    rel_dens = boltzmann_dist(
                        Te[i] * self.T_norm, energies, stat_weights, gnormalise=False
                    )

                    self.dens[i, locs] = rel_dens * Z_dens_loc / np.sum(rel_dens)
                    if self.fixed_fraction_init:
                        self.dens[i, locs] *= self.frac_imp_dens * ne[i]

        else:
            if self.fixed_fraction_init:
                self.dens[:, 0] = self.frac_imp_dens * ne

            else:
                self.dens[:, 0] = 1.0

    def set_state_positions(self):
        """Store the positions of each state (which may be different from the state ID)"""
        for i, state in enumerate(self.states):
            self.states[i].pos = i

    def set_transition_positions(self):
        """Store the positions of each from and to state in each transition"""

        id2pos = {}
        for i, state in enumerate(self.states):
            id2pos[state.id] = i

        for i, trans in enumerate(self.transitions):
            self.transitions[i].from_pos = id2pos[self.transitions[i].from_id]
            self.transitions[i].to_pos = id2pos[self.transitions[i].to_id]

    def reorder_PQ_states(self, P_states: str = "ground"):
        """Ensure evolved and non-evolved atomic states are in the correct order

        :param P_states: Which atomic states are evolved, defaults to "ground"
        """
        if P_states == "ground":
            ground_states = [s for s in self.states if s.ground is True]
            other_states = [s for s in self.states if s.ground is False]
            self.num_P_states = len(ground_states)
            self.num_Q_states = len(other_states)
            self.states = ground_states + other_states

        self.set_state_positions()
        self.set_transition_positions()
