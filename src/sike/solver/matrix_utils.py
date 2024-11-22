import numpy as np
from numba import jit

from sike.atomics.impurity import Impurity

# Update docstrings to Sphinx format


def build_matrix(min_x, max_x, num_states):
    """Construct an array of local numpy matrices for the problem

    Args:
        num_states (int): number of evolved states
        evolve (bool): whether problem is evolved in time or solved assuming d/dt = 0

    Returns:
        _type_: _description_
    """

    rate_mat = []
    for i in range(min_x, max_x):
        loc_mat = np.zeros([num_states, num_states])
        rate_mat.append(loc_mat)

    return rate_mat


def fill_rate_matrix(
    loc_num_x: int,
    min_x: int,
    max_x: int,
    mat: list,
    impurity: Impurity,
    fe: np.ndarray,
    ne: np.ndarray,
    Te: np.ndarray,
    Egrid: np.ndarray,
    dE: np.ndarray,
):
    """Fill the rate matrix with rates calculated by each transition object

    Args:
        mat (list): the implicit matrix
        impurity (Impurity): the impurity being modelled (contains all transitions)
        fe (np.ndarray): electron distributions in each cell
        ne (np.ndarray): electron density profile
        Te (np.ndarray): electron temperature profile
        vgrid (np.ndarray): velocity grid
        dvc (np.ndarray): velocity grid widths
        :param v_th: Normalisation constant [ms^-1] for electron velocities
    """

    num_states = impurity.tot_states

    # Next, calculate the values
    rank = 0  # TODO: Implement parallelisation
    for i in range(loc_num_x):
        if rank == 0:
            print(" {:.1f}%".format(100 * float(i / loc_num_x)), end="\r")

        for j, trans in enumerate(impurity.transitions):
            from_pos = trans.from_pos
            to_pos = trans.to_pos
            typ = trans.type

            # # Calculate the value to be added to the matrix
            val = trans.get_mat_value(fe[:, i + min_x], Egrid, dE)

            # Add the loss term
            row = from_pos
            col = from_pos
            mat[i][row, col] += -val

            # Add the gain term
            row = to_pos
            col = from_pos
            mat[i][row, col] += val

            # # Calculate inverse process matrix entries (3-body recombination & de-excitation)
            if typ == "excitation":
                val = trans.get_mat_value_inv(fe[:, i + min_x], Egrid, dE)

                # Add the loss term
                row = to_pos
                col = to_pos
                mat[i][row, col] += -val

                # Add the gain term
                row = from_pos
                col = to_pos
                mat[i][row, col] += val

            elif typ == "ionization":
                val = trans.get_mat_value_inv(
                    fe[:, i + min_x], Egrid, dE, ne[i + min_x], Te[i + min_x]
                )

                # Add the loss term
                row = to_pos
                col = to_pos
                mat[i][row, col] += -val

                # Add the gain term
                row = from_pos
                col = to_pos
                mat[i][row, col] += val

    if rank == 0:
        print(" {:.1f}%".format(100))

    return mat
