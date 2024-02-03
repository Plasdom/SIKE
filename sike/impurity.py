import numpy as np
import os
from transition import *
from atomic_state import State
import json
from mpi4py import MPI
import math
import post_processing
from scipy import interpolate
class Impurity:
    """Impurity class to hold information on the states and transitions for a given modelled impurity species.
    """

    def __init__(self, rank, size, name, opts, vgrid, Egrid, ne, Te, collrate_const, tbrec_norm, sigma_norm, time_norm, T_norm, n_norm):
        """Initialise Impurity object.

        Args:
            name (str): chemical symbol of the impurity
            opts (dict): options dictionary of the SIKERun object
            vgrid (np.ndarray): velocity grid
            Egrid (np.ndarray): energy grid
            ne (np.ndarray): electron densty profile
            collrate_const (float): collisional rate normalisation
            tbrec_norm (float): three-body recombination rate normalisation
            sigma_norm (float): cross-section normalisation (m^2)
            time_norm (float): time normalisation (s)
            T_norm (float): temperature normalisation (eV)
        """
        self.name = name
        self.get_element_data()
        if rank == 0:
            print(' Initialising states...')
        self.init_states(opts)
        if rank == 0:
            print(' Initialising transitions...')
        self.init_transitions(rank, size, vgrid, Egrid, collrate_const, tbrec_norm,
                              sigma_norm, time_norm, T_norm, opts)
        if rank == 0:    
            print(' Initialising densities...')
        self.init_dens(opts, ne, n_norm, Te, T_norm)
        if rank == 0:
            print(' Finalising states...')
        self.set_state_positions()
        self.set_transition_positions()

    def get_element_data(self):
        """Set the nuclear charge and number of ionisation stages
        """
        nuc_chg_dict = {'H': 1,
                        'He': 2,
                        'Li': 3,
                        'Be': 4,
                        'B': 5,
                        'C': 6,
                        'N': 7,
                        'O': 8,
                        'Ne': 10,
                        'Ar': 18,
                        'W': 74}
        longname_dict = {'H': 'Hydrogen',
                         'He': 'Helium',
                         'Li': 'Lithium',
                         'Be': 'Beryllium',
                         'B': 'Boron',
                         'C': 'Carbon',
                         'N': 'Nitrogen',
                         'O': 'Oxygen',
                         'Ne': 'Neon',
                         'Ar': 'Argon',
                         'W': 'Tungsten'}
        self.nuc_chg = nuc_chg_dict[self.name]
        self.num_Z = self.nuc_chg + 1
        self.longname = longname_dict[self.name]

    def init_states(self, opts):
        """Initialise the evolved atomic states

        Args:
            opts (dict): options dictionary from the SIKERun object
        """
        if opts['resolve_j']:
            levels_f = os.path.join(os.path.dirname(__file__), 'atom_data', self.longname,
                                    self.name + '_levels_nlj.json')
        else:
          if opts['resolve_l']:
            levels_f = os.path.join(os.path.dirname(__file__), 'atom_data', self.longname,
                                    self.name + '_levels_nl.json')
          else:
            levels_f = os.path.join(os.path.dirname(__file__), 'atom_data', self.longname,
                                    self.name + '_levels_n.json')
        with open(levels_f) as f:
            levels_dict = json.load(f)
            self.states = [None] * len(levels_dict)
            for i, level_dict in enumerate(levels_dict):
                self.states[i] = State(i, level_dict)

        # Keep only user-specified states
        if opts['state_ids'] is not None:
            for i, state in enumerate(self.states):
                if state.id not in opts['state_ids']:
                    self.states[i] = None
        self.states = [s for s in self.states if s is not None]

        self.tot_states = len(self.states)

        self.init_ionization_energies()

    def init_ionization_energies(self):
        """Set the ground state levels, ionization energies, delta E from ground state for each atomic state
        """

        # Find the lowest energy states        
        gs_energies = np.zeros(self.num_Z)
        gs_pos = np.zeros(self.num_Z, dtype=int)
        for Z in range(self.num_Z):
          Z_states = [s for s in self.states if s.Z == Z]
          energies = [s.energy for s in Z_states]
          gs = Z_states[np.argmin(energies)]
          for i,s in enumerate(self.states):
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

            if self.states[i].Z < self.num_Z-1:
                iz_energy = gs_energies[self.states[i].Z +
                                        1] - self.states[i].energy
                self.states[i].iz_energy = iz_energy
            else:
                self.states[i].iz_energy = 0.0

            energy_from_gs = self.states[i].energy - \
                gs_energies[self.states[i].Z]
            self.states[i].energy_from_gs = energy_from_gs

    def init_transitions(self, rank, size, vgrid, Egrid, collrate_const, tbrec_norm, sigma_norm, time_norm, T_norm, opts):
        """Initialise all transitions between atomic states

        Args:
            vgrid (np.ndarray): velocity grid
            Egrid (np.ndarray): energy grid
            collrate_const (float): collisional rate normalisation
            tbrec_norm (float): three-body recombination rate normalisation
            sigma_norm (float): cross-section normalisation (m^2)
            time_norm (float): time normalisation (s)
            T_norm (float): temperature normalisation (eV)
            opts (dict): options dictionary from the SIKERun object
        """
        if opts['resolve_j']:
            trans_f = os.path.join(os.path.dirname(__file__), 'atom_data', self.longname,
                                   self.name + '_transitions_nlj.json')
        else:
          if opts['resolve_l']:
            trans_f = os.path.join(os.path.dirname(__file__), 'atom_data', self.longname,
                                   self.name + '_transitions_nl.json')
          else:
            trans_f = os.path.join(os.path.dirname(__file__), 'atom_data', self.longname,
                                   self.name + '_transitions_n.json')
        if rank == 0:
            print('  Loading transitions from json...')
        with open(trans_f) as f:
            trans_dict = json.load(f)
            trans_Egrid = trans_dict[0]["E_grid"]

        if rank == 0:
            print('  Creating transition objects...')
        num_transitions = len(trans_dict)
        transitions = [None] * num_transitions
        
        for i, trans in enumerate(trans_dict[1:]):
            # if rank == 0:
            #     print('  {:.1f}%'.format(100*i/num_transitions), end='\r')
            if opts['state_ids'] is not None:
                if (trans['from_id'] not in opts['state_ids']) or (trans['to_id'] not in opts['state_ids']):
                    continue
            if trans['type'] == 'ionization' and opts['ionization']:
                transitions[i] = IzTrans(
                    trans, collrate_const, tbrec_norm, sigma_norm, T_norm)
            elif trans['type'] == 'autoionization' and opts['autoionization']:
                transitions[i] = AiTrans(trans, time_norm, T_norm)
            elif trans['type'] == 'radiative recombination' and opts['radiative recombination']:
                transitions[i] = RRTrans(
                    trans, collrate_const, sigma_norm, T_norm)
            elif trans['type'] == 'emission' and opts['emission']:
                transitions[i] = EmTrans(trans, time_norm, T_norm)
            elif trans['type'] == 'excitation' and opts['excitation']:
                transitions[i] = ExTrans(
                    trans, collrate_const, sigma_norm, T_norm)
        transitions = [t for t in transitions if t is not None]
        
        self.transitions = transitions

        # Set the de-excitation cross-sections
        if rank == 0:
            print('  Creating data for inverse transitions...')
        id2pos = {self.states[i].id: i for i in range(
            len(self.states))}
        if opts['excitation']:
            for i, t in enumerate(self.transitions):
                if t.type == 'excitation':
                    g_ratio = self.states[id2pos[t.from_id]].stat_weight / \
                        self.states[id2pos[t.to_id]].stat_weight
                    t.set_sigma_deex(g_ratio, vgrid)

        # Set the stat weight ratios for ionization cross-sections
        if opts['ionization']:
            for i, t in enumerate(self.transitions):
                if t.type == 'ionization':
                    g_ratio = self.states[id2pos[t.from_id]].stat_weight / \
                        self.states[id2pos[t.to_id]].stat_weight
                    t.set_inv_data(g_ratio, vgrid)

        # Checks
        if rank == 0:
            print('  Performing checks on transition data...')
        self.state_and_transition_checks(rank,
            Egrid, trans_Egrid, opts['autoionization'])

    def state_and_transition_checks(self,rank, Egrid, trans_Egrid, autoionization):
        """Perform some checks on states and transitions belonging to the impurity. Removes orphaned states, transitions where one or more associated states are not evolved, etc

        Args:
            Egrid (_type_): _description_
            trans_Egrid (_type_): _description_
            autoionization (_type_): _description_

        Raises:
            ValueError: _description_
        """

        # Check that simulation E_grid is the same as the transitions E_grid
        if np.max(np.abs(Egrid - trans_Egrid)) > 1e-5:
            # TODO: Handle different energy grids from input to transitions by interpolation
            raise ValueError(
                "Energy grid is different from grid on which transitions evaluated. This will be handled in the future.")

        # Check for no orphaned states (i.e. states with either no associated transitions or )
        id2pos = {self.states[i].id: i for i in range(
            len(self.states))}
        from_ids = np.array([t.from_id for t in self.transitions], dtype=int)
        to_ids = np.array([t.to_id for t in self.transitions], dtype=int)
        for i, state in enumerate(self.states):
            # if rank == 0:
            #     print('  {:.1f}%'.format(100*i/self.tot_states), end='\r')
            associated_transitions = SIKE_tools.get_associated_transitions(
                state.id, from_ids, to_ids)
            if len(associated_transitions) == 0:
                if rank == 0:
                    print('State ID ' + str(state.id) +
                      ' is an orphaned state, removing.')
                self.states[i] = None
                self.tot_states -= 1
        self.states = [s for s in self.states if s is not None]
        # if rank == 0:
        #     print('  {:.1f}%'.format(100), end='\r')

        # Remove states above ionization energy if no autoionization
        # TODO: Is this necessary?
        if autoionization is False:
            for i, state in enumerate(self.states):
                if state.iz_energy < 0:
                    self.states[i] = None
                    self.tot_states -= 1
        self.states = [s for s in self.states if s is not None]

        # Check for no orphaned transitions (i.e. transitions where either from_id or to_id is not evolved)
        state_ids = [s.id for s in self.states]
        for i, trans in enumerate(self.transitions):
            if trans.from_id not in state_ids or trans.to_id not in state_ids:
                self.transitions[i] = None
        self.transitions = [t for t in self.transitions if t is not None]

    def init_dens(self, opts, ne, n_norm, Te, T_norm):
        """Initialise densities of impurity states

        Args:
            opts (dict): options dictionary from the SIKERun object
            ne (np.array): electron density array
        """

        if opts['kinetic_electrons']:
            self.dens = np.zeros((len(ne), self.tot_states))
        if opts['maxwellian_electrons']:
            self.dens_Max = np.zeros((len(ne), self.tot_states))

        if opts['saha_boltzmann_init']:

          self.set_state_positions()
          
          Z_dens = np.zeros([len(ne),self.num_Z])
          for i in range(len(ne)):
              Z_dens[i,:] = SIKE_tools.saha_dist(Te[i]*T_norm, ne[i]*n_norm, n_norm, self) / n_norm
          
          for Z in range(self.num_Z):
              
              Z_states = post_processing.gather_states(self.states,Z)
              
              energies = [s.energy for s in Z_states]
              stat_weights = [s.stat_weight for s in Z_states]
              locs = [s.pos for s in Z_states]
              for i in range(len(ne)):
                
                Z_dens_loc = Z_dens[i,Z]
                
                rel_dens = SIKE_tools.boltzmann_dist(Te[i] * T_norm,energies,stat_weights,gnormalise=False)
                
                if opts['kinetic_electrons']:
                  self.dens[i,locs] = rel_dens * Z_dens_loc / np.sum(rel_dens)
                  if opts['fixed_fraction_init']:
                    self.dens[i,locs] *= opts['frac_imp_dens'] * ne[i]
                if opts['maxwellian_electrons']:
                  self.dens_Max[i,locs] = rel_dens * Z_dens_loc / np.sum(rel_dens)
                  if opts['fixed_fraction_init']:
                    self.dens_Max[i,locs] *= opts['frac_imp_dens'] * ne[i]
        else:
            
            if opts['fixed_fraction_init']:
              if opts['kinetic_electrons']:
                  self.dens[:, 0] = opts['frac_imp_dens'] * ne
              if opts['maxwellian_electrons']:
                  self.dens_Max[:, 0] = opts['frac_imp_dens'] * ne
           
            else:
              if opts['kinetic_electrons']:
                  self.dens[:, 0] = 1.0
              if opts['maxwellian_electrons']:
                  self.dens_Max[:, 0] = 1.0

        
    def set_state_positions(self):
        """Store the positions of each state (which may be different from the state ID)
        """
        for i, state in enumerate(self.states):
            self.states[i].pos = i

    def set_transition_positions(self):
        """Store the positions of each from and to state in each transition
        """

        id2pos = {}
        for i, state in enumerate(self.states):
            id2pos[state.id] = i

        for i, trans in enumerate(self.transitions):
            self.transitions[i].from_pos = id2pos[self.transitions[i].from_id]
            self.transitions[i].to_pos = id2pos[self.transitions[i].to_id]
    
    def reorder_PQ_states(self, P_states="ground"):
        if P_states == "ground":
            ground_states = [s for s in self.states if s.ground is True]
            other_states = [s for s in self.states if s.ground is False]
            self.num_P_states = len(ground_states)
            self.num_Q_states = len(other_states)
            self.states = ground_states + other_states
        
        self.set_state_positions()
        self.set_transition_positions()
