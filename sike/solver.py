from petsc4py import PETSc
import petsc4py
import numpy as np
import math
import scipy
from mpi4py import MPI


def solve_petsc(loc_num_x, min_x, max_x, rate_matrix, n_init, num_x, ksp_solver, ksp_pc, ksp_tol):
    """Solve the matrix equation R * n = b using PETSc. R is the rate matrix, n is the density array and b is the right-hand side, which is zero apart from the last element in each spatial cell which is equal to the total impurity density

    Args:
        rate_matrix (AIJMat): rate matrix
        n_init (np.array): initial densities
        num_x (int): number of spatial cells

    Returns:
        n_solved (np.array): equilibrium densities
    """

    num_states = len(n_init[0, :])
    num_x = len(n_init[:,0])

    rank = PETSc.COMM_WORLD.Get_rank()
    
    # # Initialise the rhs and new density vectors
    n_solved = PETSc.Vec().createSeq(num_states * loc_num_x, comm=PETSc.COMM_SELF)
    
    rhs_arr = np.zeros(num_states * loc_num_x)
    
    rhs_arr[
        [num_states * (i+1) - 1 for i in range(loc_num_x)]
    ] = [np.sum(n_init[i, :]) for i in range(min_x,max_x)]

    rhs = PETSc.Vec().createSeq(num_states * loc_num_x,comm=PETSc.COMM_SELF)
    rhs.setValuesLocal(range(num_states * loc_num_x), rhs_arr)   
    rhs.assemblyBegin()
    rhs.assemblyEnd()
    

    # Set the last row of petsc matrix to ones
    rate_matrix.assemblyBegin()
    rate_matrix.assemblyEnd()
    for i in range(min_x,max_x):
        offset = (i-min_x)*num_states
        for j in range(num_states):
            row = offset + num_states-1
            col = offset + j
            rate_matrix.setValue(row, col, 1.0)
    rate_matrix.assemblyBegin()
    rate_matrix.assemblyEnd()
    

    # Initialise the KSP solver
    ksp = PETSc.KSP().create(comm=PETSc.COMM_SELF)
    ksp.setOperators(rate_matrix)
    ksp.setType(ksp_solver)
    ksp.setTolerances(ksp_tol*np.sum(n_init[min_x:max_x,:]))
    pc = ksp.getPC()
    if ksp_pc is not None:
        pc.setType(ksp_pc)
    if rank == 0:
        print('Solving with:', ksp.getType(), ', preconditioner:', pc.getType())
    
    ksp.solve(rhs, n_solved)
    if ksp.getConvergedReason() < 0:
        print(rank,'\nKSP solve failed: ' + str(ksp.getConvergedReason()))
        return None
    else:
        print(rank,'Converged in', ksp.getIterationNumber(), 'iterations.')

    n_solved = np.array([[n_solved[i + (j * num_states)]
                        for i in range(num_states)] for j in range(loc_num_x)])

    PETSc.COMM_WORLD.Barrier()
    print("Conservation check on rank " + str(rank) + ": {:.2e}".format(
        np.sum(n_solved) - np.sum(n_init[min_x:max_x,:])))

    return n_solved

def solve_np(loc_num_x, min_x, max_x, rate_matrix, n_init, num_x, ksp_solver, ksp_pc, ksp_tol):
    """Solve the matrix equation R * n = b using numpy. R is the rate matrix, n is the density array and b is the right-hand side, which is zero apart from the last element in each spatial cell which is equal to the total impurity density

    Args:
        rate_matrix (list): rate matrix
        n_init (np.array): initial densities
        num_x (int): number of spatial cells

    Returns:
        n_solved (np.array): equilibrium densities
    """

    num_states = len(n_init[0, :])
    num_x = len(n_init[:,0])

    rank = PETSc.COMM_WORLD.Get_rank()
    
    # Initialise the rhs and new density vectors
    n_solved = [np.zeros(num_states) for i in range(loc_num_x)]
    rhs = [np.zeros(num_states) for i in range(loc_num_x)] 
    for i in range(loc_num_x):
        rhs[i][-1] = np.sum(n_init[i+min_x, :])

    # Set the last row of numpy matrix to ones
    for i in range(loc_num_x):
        for j in range(num_states):
            row = num_states-1
            col = j
            rate_matrix[i][row,col] = 1.0
    
    # Solve the matrix equation
    for i in range(loc_num_x):
        # n_solved[i] = np.linalg.inv(rate_matrix[i]) @ rhs[i]
        n_solved[i] = np.linalg.solve(rate_matrix[i], rhs[i])

    n_solved = np.array(n_solved)

    PETSc.COMM_WORLD.Barrier()
    print("Conservation check on rank " + str(rank) + ": {:.2e}".format(
        np.sum(n_solved) - np.sum(n_init[min_x:max_x,:])))

    return n_solved


def evolve_petsc(loc_num_x, min_x, max_x, rate_matrix, n_init, num_x, dt, num_t, dndt_thresh, n_norm, t_norm, ksp_solver, ksp_pc, ksp_tol):
    """Evolve the matrix equation dn/dt = R * n using PETSc using backwards Euler time-stepping. R is the rate matrix, n is the density array

    Args:
        rate_matrix (AIJMat): rate matrix
        n_init (np.array): initial densities
        num_x (int): number of spatial cells
        dt (float): time step

    Returns:
        n_solved (np.array): equilibrium densities
    """
    
    # dndt_thresh *= (n_norm / t_norm)
    
    num_states = len(n_init[0, :])

    rate_matrix.assemblyBegin()
    rate_matrix.assemblyEnd()
    
    rank = PETSc.COMM_WORLD.Get_rank()

    # Initialise the old and new density vectors
    n_old = PETSc.Vec().createSeq(num_states * loc_num_x, comm=PETSc.COMM_SELF)
    n_old.setValues(range(num_states * loc_num_x), n_init[min_x:max_x,:].flatten())
    n_new = PETSc.Vec().createSeq(num_states * loc_num_x,comm=PETSc.COMM_SELF)

    # Create an identity matrix
    I = PETSc.Mat().createAIJ([num_states * loc_num_x, num_states * loc_num_x], nnz=1,comm=PETSc.COMM_SELF)
    for row in range(num_states * loc_num_x):
        I.setValue(row, row, 1.0)
    I.assemblyBegin()
    I.assemblyEnd()

    # Create the backwards Euler operator matrix
    be_op_mat = PETSc.Mat().createAIJ(
        [num_states*loc_num_x, num_states*loc_num_x], nnz=num_states,comm=PETSc.COMM_SELF)
    be_op_mat = I - dt * rate_matrix
    be_op_mat.assemblyBegin()
    be_op_mat.assemblyEnd()

    # Initialise the KSP solver
    ksp = PETSc.KSP().create(comm=PETSc.COMM_SELF)
    ksp.setOperators(be_op_mat)
    ksp.setType(ksp_solver)
    ksp.setTolerances(ksp_tol*np.sum(n_init[min_x:max_x,:]))
    pc = ksp.getPC()
    if ksp_pc is not None:
        pc.setType(ksp_pc)
    if rank == 0:
        print('Evolving with:', ksp.getType(), ', preconditioner: ', pc.getType())

    prev_residual = 1e20
    for i in range(num_t):
        
        ksp.solve(n_old, n_new)
        if ksp.getConvergedReason() < 0:
            ksp_failed = 1
        else:
            ksp_failed = 0
        num_its = ksp.getIterationNumber()

        dndt = np.max(np.abs(n_old - n_new)) / dt

        # Update densities
        n_old.setValues(range(num_states*loc_num_x),
                        n_new.getValues(range(num_states*loc_num_x)))

        # Do some communication
        all_ksp_failed = MPI.COMM_WORLD.gather(ksp_failed,root=0)
        all_dndts = MPI.COMM_WORLD.gather(dndt,root=0)
        all_its = MPI.COMM_WORLD.gather(num_its,root=0)
        max_dndt = None
        max_its = None
        solver_failed = None
        if rank == 0:
            max_dndt = max(all_dndts)
            max_its = max(all_its)
            solver_failed = sum(all_ksp_failed)
        dndt_global = MPI.COMM_WORLD.bcast(max_dndt, root=0)
        num_its_global = MPI.COMM_WORLD.bcast(max_its, root=0)
        exit_flag = MPI.COMM_WORLD.bcast(solver_failed, root=0)
        
        if exit_flag > 0:
            if rank == 0:
                print('\nKSP solve failed.')
            reason = ksp.getConvergedReason()
            if reason < 0:
                print('\nKSP failed on rank ' + str(rank) + ', reason: ' + str(reason))
            return None
        
        # if dndt_global > prev_residual and i/num_t > 0.01:
        #     print('Finishing time integration because dn/dt has begun to increase.')
        #     break

        prev_residual = dndt_global
        
        if rank == 0:
          print('TIMESTEP ' + str(i+1) +
          ' | max(dn/dt) {:.2e}'.format((n_norm / t_norm) * dndt_global) + ' / {:.2e}'.format((n_norm / t_norm) * dndt_thresh) + ' | NUM_ITS ' + str(num_its_global) + '            ', end='\r')
        
        if dndt_global < dndt_thresh:
            print('Finishing time integration because dn/dt reached threshold.')
            break
        

    n_solved = np.array([[n_new[i + (j * num_states)]
                        for i in range(num_states)] for j in range(loc_num_x)])
    
    PETSc.COMM_WORLD.Barrier()
    
    if rank == 0:
        print('')
    print("Conservation check on rank " + str(rank) + ": {:.2e}".format(
        np.sum(n_solved) - np.sum(n_init[min_x:max_x,:])))

    return n_solved

def evolve_np(loc_num_x, min_x, max_x, rate_matrix, n_init, num_x, dt, num_t, dndt_thresh, n_norm, t_norm, ksp_solver, ksp_pc, ksp_tol):
    """Evolve the matrix equation dn/dt = R * n using PETSc using backwards Euler time-stepping. R is the rate matrix, n is the density array

    Args:
        rate_matrix (AIJMat): rate matrix
        n_init (np.array): initial densities
        num_x (int): number of spatial cells
        dt (float): time step

    Returns:
        n_solved (np.array): equilibrium densities
    """
    
    # dndt_thresh *= (n_norm / t_norm)
    
    num_states = len(n_init[0, :])
    
    rank = PETSc.COMM_WORLD.Get_rank()

    # Initialise the old and new density vectors
    n_old = [np.zeros(num_states) for i in range(loc_num_x)]
    for i in range(loc_num_x):
      n_old[i][:] = n_init[i+min_x,:]
    n_new = [np.zeros(num_states) for i in range(loc_num_x)]

    # Create an identity matrix
    I = np.diag(np.ones(num_states))

    # Create the backwards Euler operator matrix
    be_op_mat = []
    for i in range(loc_num_x):
        be_op_mat.append(I - dt * rate_matrix[i])
    
    # Find inverse of operator matrix
    for i in range(loc_num_x):
        be_op_mat[i] = np.linalg.inv(be_op_mat[i])
    prev_residual = 1e20
    for i in range(num_t):
        
        # Solve the matrix equation
        for j in range(loc_num_x):
            n_new[j] = be_op_mat[j].dot(n_old[j])
            # n_new[j] = np.linalg.solve(be_op_mat[j], n_old[j])

        # Find dn/dt
        dndt = 0
        for j in range(loc_num_x):
            dndt_cur = np.max(np.abs(n_old[j] - n_new[j])) / dt
            if dndt_cur > dndt:
                dndt = dndt_cur

        # Update densities
        for j in range(loc_num_x):
            n_old[j] = n_new[j]

        # Do some communication
        all_dndts = MPI.COMM_WORLD.gather(dndt,root=0)
        max_dndt = None
        if rank == 0:
            max_dndt = max(all_dndts)
        dndt_global = MPI.COMM_WORLD.bcast(max_dndt, root=0)

        # if dndt_global > prev_residual and i/num_t > 0.01:
        #   print('Finishing time integration because dn/dt has begun to increase.')
        #   break

        prev_residual = dndt_global
        
        if rank == 0:
          print('TIMESTEP ' + str(i+1) +
          ' | max(dn/dt) {:.2e}'.format((n_norm / t_norm) * dndt_global) + ' / {:.2e}'.format((n_norm / t_norm) * dndt_thresh) + '            ', end='\r')
        
        if dndt_global < dndt_thresh:
            print('Finishing time integration because dn/dt reached threshold.')
            break
        

    n_solved = np.array(n_new)
    
    PETSc.COMM_WORLD.Barrier()
    
    if rank == 0:
        print('')
    print("Conservation check on rank " + str(rank) + ": {:.2e}".format(
        np.sum(n_solved) - np.sum(n_init[min_x:max_x,:])))

    return n_solved

def evolve_rk4(loc_num_x, min_x, max_x, rate_matrix, n_init, num_x, dt, num_t, dndt_thresh, n_norm, t_norm, ksp_solver, ksp_pc, ksp_tol):
    """Evolve the matrix equation dn/dt = R * n using RK4 explicit time-stepping. R is the rate matrix, n is the density array

    Args:
        rate_matrix (AIJMat): rate matrix
        n_init (np.array): initial densities
        num_x (int): number of spatial cells
        dt (float): time step

    Returns:
        n_solved (np.array): equilibrium densities
    """
    
    num_states = len(n_init[0, :])

    rate_matrix.assemblyBegin()
    rate_matrix.assemblyEnd()
    
    rank = PETSc.COMM_WORLD.Get_rank()

    # Initialise the old and new density vectors
    n_old = PETSc.Vec().createSeq(num_states * loc_num_x, comm=PETSc.COMM_SELF)
    n_old.setValues(range(num_states * loc_num_x), n_init[min_x:max_x,:].flatten())
    n_new = PETSc.Vec().createSeq(num_states * loc_num_x,comm=PETSc.COMM_SELF)
    
    k1 = PETSc.Vec().createSeq(num_states * loc_num_x,comm=PETSc.COMM_SELF)
    k2 = PETSc.Vec().createSeq(num_states * loc_num_x,comm=PETSc.COMM_SELF)
    k3 = PETSc.Vec().createSeq(num_states * loc_num_x,comm=PETSc.COMM_SELF)
    k4 = PETSc.Vec().createSeq(num_states * loc_num_x,comm=PETSc.COMM_SELF)

    prev_residual = 1e20
    for i in range(num_t):
        
        rate_matrix.mult(n_old,k1)
        rate_matrix.mult(n_old + k1 * (dt/2.0),k2)
        rate_matrix.mult(n_old + k2 * (dt/2.0),k3)
        rate_matrix.mult(n_old + k3 * dt,k4)
        
        dt_6 = dt / 6.0
        
        add_vec = k1 * dt_6
        add_vec = add_vec + 2.0 * k2 * dt_6
        add_vec = add_vec + 2.0 * k3 * dt_6
        add_vec = add_vec + k4 * dt_6
        
        n_new = n_old + add_vec

        dndt = np.max(np.abs(n_old - n_new)) / dt

        # Update densities
        n_old.setValues(range(num_states*loc_num_x),
                        n_new.getValues(range(num_states*loc_num_x)))

        # Do some communication
        all_dndts = MPI.COMM_WORLD.gather(dndt,root=0)
        max_dndt = None
        solver_failed = None
        if rank == 0:
            max_dndt = max(all_dndts)
        dndt_global = MPI.COMM_WORLD.bcast(max_dndt, root=0)

        prev_residual = dndt_global
        
        if rank == 0:
            if (i+1) % 10 == 0:
                print('TIMESTEP ' + str(i+1) +
                ' | max(dn/dt) {:.2e}'.format((n_norm / t_norm) * dndt_global), end='\r')
        
        if dndt_global < dndt_thresh:
            break
        

    n_solved = np.array([[n_new[i + (j * num_states)]
                        for i in range(num_states)] for j in range(loc_num_x)])
    
    PETSc.COMM_WORLD.Barrier()
    
    if rank == 0:
        print('')
    print("Conservation check on rank " + str(rank) + ": {:.2e}".format(
        np.sum(n_solved) - np.sum(n_init[min_x:max_x,:])))

    return n_solved
