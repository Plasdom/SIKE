import numpy as np
import SIKE_tools


def get_Zavg(dens, states, num_x=None):
    """Calculate average ionization

    Args:
        dens (np.ndarray): Impurity densities
        states (_type_): The list of states being modelled
        num_x (_type_): The number of spatial cells

    Returns:
        np.ndarray: 1D array of average ionization at each spatial location
    """
    if num_x is None:
        num_x = len(dens[:,0])
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

    cooling_curves *= SIKE_tools.el_charge * run.T_norm / (run.t_norm * run.n_norm)

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


def gather_transitions(transitions, states, type='ionization', Z=None):
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


def get_gs_iz_coeffs(r,el,kinetic=False):
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
    gs_iz_coeffs = np.zeros([r.num_x,num_Z-1])
        
    for Z in range(num_Z-1):

        Z_states = gather_states(r.impurities[el].states,Z)
        gs = Z_states[0]
        Zplus1_states = gather_states(r.impurities[el].states,Z+1)
        gs_Zplus1 = Zplus1_states[0]
        for s in r.impurities[el].transitions:
            if s.type == 'ionization' and s.from_id == gs.id and s.to_id == gs_Zplus1.id:
                iz_trans = s
                break
        
        for k in range(r.num_x):
            
            iz_rate = iz_trans.get_mat_value( fe[:,k], r.vgrid, r.dvc) / r.ne[k]
            gs_iz_coeffs[k,Z] = iz_rate / (r.n_norm * r.t_norm)
    
    
    return gs_iz_coeffs

        


def get_cr_iz_coeffs(r, el, kinetic=False):
    """Calculate the collisional-radiative ionization coefficients, as per Summers, P. et al. PPCF (2006)

    Args:
        r (SIKERun): SIKERun object
        el (str): element 
        kinetic (bool, optional): whether to calculate kinetic or maxwellian rates. Defaults to False.

    Returns:
        np.ndarray: 2D array of CR ionization coefficients (num_x, num_Z-1)
    """
    
    if kinetic:
        m = r.rate_mats[el]
        fe = r.fe
    else:
        m = r.rate_mats_Max[el]
        fe = r.fe_Max
    m.assemblyBegin()
    m.assemblyEnd()
    tot_states = r.impurities[el].tot_states
    
    cr_iz_coeffs = np.zeros([r.num_x,r.impurities[el].num_Z-1])
    for x_pos in range(r.num_x):
        
        print('{:.2f}%'.format(100*x_pos/r.num_x),end='\r')
        
        for Z in range(r.impurities[el].num_Z-1):
            
            # Extract relevant part of rate matrix
            offset = x_pos * tot_states
            rows = [s.pos + offset for s in r.impurities[el].states if s.Z == Z] 
            cols = rows
            rate_mat = m.getValues(rows=rows, cols=cols)
            
            # Build ionization array
            Z_states = gather_states(r.impurities[el].states,Z)
            Z_ids = [s.id for s in Z_states]
            gs = Z_states[0]
            Zplus1_states = gather_states(r.impurities[el].states,Z+1)
            Zplus1_ids = [s.id for s in Zplus1_states]
            gs_Zplus1 = Zplus1_states[0]
            iz_trans = [t for t in r.impurities[el].transitions if t.type == 'ionization'
                        and t.from_id in Z_ids and t.to_id == gs_Zplus1.id]
            S = np.zeros(len(Z_states))
            for i in range(len(Z_states)):
                izt = None
                for t in iz_trans:
                    if t.from_id == Z_states[i].id:
                        izt = t
                        break
                if izt is not None:
                    S[i] = izt.get_mat_value(fe[:,x_pos], r.vgrid, r.dvc)
            S_vs = S[0]
            S_vj = S[1:]
            
            # Compute the collisional-radiative ionization coefficients
            C_ij_inv = np.linalg.inv(rate_mat[1:,1:])
            C_is = rate_mat[1:,0]
            iz_coeff = S_vs - np.dot(S_vj, np.dot(C_ij_inv, C_is) )
            cr_iz_coeffs[x_pos,Z] = iz_coeff / (r.ne[x_pos] * r.n_norm * r.t_norm)
    
    print('{:.2f}%'.format(100))
    
    return cr_iz_coeffs
