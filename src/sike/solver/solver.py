import numpy as np


def solve(
    loc_num_x: int,
    min_x: int,
    max_x: int,
    rate_matrix: list[np.ndarray],
    n_init: np.ndarray,
) -> np.ndarray:
    """Solve the matrix equation R * n = b using numpy. R is the rate matrix, n is the density array and b is the right-hand side, which is zero apart from the last element in each spatial cell which is equal to the total impurity density

    :param loc_num_x: Local number of spatial cells
    :param min_x: Minimimum local x index
    :param max_x: Maximum local x index
    :param rate_matrix: List of rate matrices
    :param n_init: Initial densities
    :return: Equilibrium densities
    """

    num_states = len(n_init[0, :])

    # Initialise the rhs and new density vectors
    n_solved = [np.zeros(num_states) for i in range(loc_num_x)]
    rhs = [np.zeros(num_states) for i in range(loc_num_x)]
    for i in range(loc_num_x):
        rhs[i][-1] = np.sum(n_init[i + min_x, :])

    # Set the last row of numpy matrix to ones
    for i in range(loc_num_x):
        for j in range(num_states):
            row = num_states - 1
            col = j
            rate_matrix[i][row, col] = 1.0

    # Solve the matrix equation
    for i in range(loc_num_x):
        n_solved[i] = np.linalg.inv(rate_matrix[i]) @ rhs[i]

    n_solved = np.array(n_solved)

    out = np.sum(n_solved) - np.sum(n_init[min_x:max_x, :])
    print(f"Conservation check: {out:.2e}")

    return n_solved


def evolve(
    loc_num_x,
    min_x,
    max_x,
    rate_matrix,
    n_init,
    dt,
    num_t,
):
    """Evolve densities in time till equilibrium is reached

    :param loc_num_x: Local number of spatial cells
    :param min_x: Minimimum local x index
    :param max_x: Maximum local x index
    :param rate_matrix: List of rate matrices
    :param n_init: Initial densities
    :param dt: Timestep
    :param num_t: Number of timesteps
    :return: Equilibrium densities
    """

    num_states = len(n_init[0, :])
    rank = 0  # TODO: Implement parallelisation

    # Initialise the old and new density vectors
    n_old = [np.zeros(num_states) for i in range(loc_num_x)]
    for i in range(loc_num_x):
        n_old[i][:] = n_init[i + min_x, :]
    n_new = [np.zeros(num_states) for i in range(loc_num_x)]

    # Create an identity matrix
    I = np.diag(np.ones(num_states))  # noqa: E741

    # Create the backwards Euler operator matrix
    be_op_mat = []
    for i in range(loc_num_x):
        be_op_mat.append(I - dt * rate_matrix[i])

    # Find inverse of operator matrix
    for i in range(loc_num_x):
        be_op_mat[i] = np.linalg.inv(be_op_mat[i])
    for _i in range(num_t):
        # Solve the matrix equation
        for j in range(loc_num_x):
            n_new[j] = be_op_mat[j].dot(n_old[j])

        # Update densities
        for j in range(loc_num_x):
            n_old[j] = n_new[j]

    n_solved = np.array(n_new)

    if rank == 0:
        print("")
    out = np.sum(n_solved) - np.sum(n_init[min_x:max_x, :])
    print(f"Conservation check on rank {rank}: {out:.2e}")

    return n_solved
