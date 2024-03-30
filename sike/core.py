import physics_tools
import numpy as np
from impurity import Impurity
import matrix_utils
import solver
from mpi4py import MPI
from petsc4py import PETSc
import petsc4py

default_opts = {
    "modelled_impurities": ["Li"],
    "delta_t": 1.0e-3,
    "evolve": True,
    "use_petsc": True,
    "kinetic_electrons": False,
    "maxwellian_electrons": True,
    "dndt_thresh": 1e-5,
    "max_steps": 1000,
    "frac_imp_dens": 0.05,
    "resolve_l": True,
    "resolve_j": True,
    "ionization": True,
    "radiative recombination": True,
    "excitation": True,
    "emission": True,
    "autoionization": True,
    "fixed_fraction_init": True,
    "saha_boltzmann_init": True,
    "state_ids": None,
    "ksp_solver": "ibcgs",
    "ksp_pc": "bjacobi",
    "ksp_tol": 1e-15,
}

# TODO: Do we ever want to actually evolve all states? Or only build M_eff and get derived coefficients? Opportunity to massively simplify by removing petsc & mpi dependency
# TODO: I guess we should only ever really be evolving the P states, so all that code is still useful, but could probably do it with dense numpy matrices rather than petsc, and probably don't need MPI!


class SIKERun(object):
    """
    A class which stores all relevant data and methods for a SIKE simulation.

    ...

    Initialisation: option 1
    ________________________
    Provide electron distribution functions on an x-grid. Temperatures and densities will be evaluated from electron distribution.
        fe: np.array(num_v, num_x)
            isotropic part of electron distribution function as a function of velocity magnitude (units [m^-6 s^-3])
        vgrid: np.array(num_v)
            velocity grid on which fe is defined (units [m/s])
        xgrid: np.array(num_x)
            x-grid on which to evolve impurity densities (units [m])
        opts: run options (see below for description and defaults)

    ...

    Initialisation: option 2
    ________________________
    Provide electron temperature and density profiles (assuming Maxwellian electrons) on an x-grid.
        Te: np.array(num_x)
            electron temperature profile (units [eV])
        ne: np.array(num_x)
            electron density profile (units [m^-3])
        xgrid: np.array(num_x)
            x-grid on which to evolve impurity densities (units [m])
        opts: run options (see below for description and defaults)

    ...

    Options
    _______
    Simulation options. If not provided defaults will be used.
        opts: dict
            'evolve': boolean(=True)
                Specify whether to evolve the state density equations in time. If false, simply invert the rate matrix (this method sometimes suffers from numerical instabilities)
            'use_petsc': boolean(=True)
                Specify whether to use PETSc matrix solver routines. If False, numpy matrix inversion iwll be used instead
            'modelled_impurities': list(=['C'])
                A list of the impurity species to evolve (use chemical symbols)
            'modelled_states': str(='all')
                Specify whether to evolve 'all', 'ground' or 'metastable' states
            'kinetic_electrons': boolean(=True)
                Solve rate equations for Maxwellian electrons at given density and temperatures
            'maxwellian_electrons': boolean(=True)
                Solve rate equations for kinetic electrons with given distribution functions
            'delta_t': float(=1.0e-3)
                The  to use in seconds if "evolve" option is true
            'dn_dt_thresh': float(=1e-5)
                The threshold density residual between subsequent s which defines whether equilibrium has been reached
            'max_steps': int(=1e3)
                The maximum number of s to evolve if EVOLVE is true
            'frac_imp_dens': 0.05
                The fractional impurity density at initialisation
            'ionization': boolean(=True)
                Include collisional ionisation and three-body recombination processes
            'radiative recombination': boolean(=True)
                Include radiative recombination process
            'excitation': boolean(=True)
                Include collisional excitation and deexcitation processes
            'emission': boolean(=True)
                Include spontaneous emission process
            'autoionization': boolean(=True)
                Include autoionization process
            'fixed_fraction_init': boolean(=True)
                Specify whether to initialise impurity densities to fixed fraction of electron density. If false, use flat impurity density profiles.
            'state_ids': list(=None)
                A specific list of state IDs to evolve. If None then all states in levels.json will be evolved.

    """

    def __init__(
        self,
        fe=None,
        vgrid=None,
        Te=None,
        ne=None,
        xgrid=None,
        opts=default_opts,
        rank=0,
    ):
        # TODO: Change fe so that spatial index comes first (like everywhere else)
        self.opts = opts
        for option in default_opts:
            if option not in list(self.opts.keys()):
                self.opts[option] = default_opts[option]

        if xgrid is not None:
            self.xgrid = xgrid.copy()
        else:
            self.xgrid = None

        if fe is not None and vgrid is not None:
            self.fe = fe.copy()
            self.vgrid = vgrid.copy()
            self.init_from_dist()
        elif Te is not None and ne is not None:
            self.Te = Te.copy()
            self.ne = ne.copy()
            self.init_from_profiles(vgrid)
        else:
            raise ValueError(
                "Must specify either electron distribution and vgrid or electron temperature and density profiles"
            )

        # Initialise species
        if self.rank == 0:
            print("Initialising the impurity species to be modelled...")
        self.impurities = {}
        for el in self.opts["modelled_impurities"]:
            self.impurities[el] = Impurity(
                self.rank,
                self.num_procs,
                el,
                self.opts,
                self.vgrid,
                self.Egrid,
                self.ne,
                self.Te,
                self.collrate_const,
                self.tbrec_norm,
                self.sigma_0,
                self.t_norm,
                self.T_norm,
                self.n_norm,
            )
        if self.rank == 0:
            print("Finished initialising impurity species objects.")

        self.rate_mats = {}
        self.rate_mats_Max = {}

    def init_from_dist(self):
        self.num_x = len(self.fe[0, :])
        if self.xgrid is None:
            self.xgrid = np.linspace(0, 1, self.num_x)
        self.num_v = len(self.vgrid)
        self.generate_grid_widths()

        self.num_procs = PETSc.COMM_WORLD.Get_size()
        self.rank = PETSc.COMM_WORLD.Get_rank()
        loc_x = self.num_x / self.num_procs
        self.min_x = int(self.rank * loc_x)
        if self.rank == self.num_procs - 1:
            self.max_x = self.num_x
        else:
            self.max_x = int((self.rank + 1) * loc_x)
        self.loc_num_x = self.max_x - self.min_x

        # Generate temperature and density profiles
        self.ne = np.array(
            [
                physics_tools.density_moment(self.fe[:, i], self.vgrid, self.dvc)
                for i in range(self.num_x)
            ]
        )
        self.Te = np.array(
            [
                physics_tools.temperature_moment(
                    self.fe[:, i], self.vgrid, self.dvc, normalised=False
                )
                for i in range(self.num_x)
            ]
        )

        # Generate normalisation constants and normalise everything
        self.init_norms()
        self.apply_normalisation()

        # Create the E_grid
        self.Egrid = self.T_norm * self.vgrid**2

        # Generate Maxwellians if required
        if self.opts["maxwellian_electrons"]:
            self.fe_Max = physics_tools.get_maxwellians(self.ne, self.Te, self.vgrid)

    def init_from_profiles(self, vgrid=None):
        self.opts["kinetic_electrons"] = False
        self.opts["maxwellian_electrons"] = True

        if vgrid is None:
            self.vgrid = physics_tools.default_vgrid.copy()
        else:
            self.vgrid = vgrid
        self.num_x = len(self.Te)
        if self.xgrid is None:
            self.xgrid = np.linspace(0, 1, self.num_x)
        self.num_v = len(self.vgrid)
        self.generate_grid_widths()

        self.num_procs = PETSc.COMM_WORLD.Get_size()
        self.rank = PETSc.COMM_WORLD.Get_rank()
        loc_x = self.num_x / self.num_procs
        self.min_x = int(self.rank * loc_x)
        if self.rank == self.num_procs - 1:
            self.max_x = self.num_x
        else:
            self.max_x = int((self.rank + 1) * loc_x)
        self.loc_num_x = self.max_x - self.min_x

        # Generate normalisation constants and normalise everything
        self.init_norms()
        self.apply_normalisation()

        # Create the E_grid
        self.Egrid = self.T_norm * self.vgrid**2

        # Generature Maxwellians
        self.fe_Max = physics_tools.get_maxwellians(self.ne, self.Te, self.vgrid)

    def init_norms(self):
        # self.T_norm = 10.0
        # self.n_norm = 5e19
        self.T_norm = np.mean(self.Te)  # eV
        self.n_norm = np.mean(self.ne) * self.opts["frac_imp_dens"]  # m^-3
        self.v_th = np.sqrt(
            2 * self.T_norm * physics_tools.el_charge / physics_tools.el_mass
        )  # m/s

        Z = 1
        gamma_ee_0 = physics_tools.el_charge**4 / (
            4 * np.pi * (physics_tools.el_mass * physics_tools.epsilon_0) ** 2
        )
        gamma_ei_0 = Z**2 * gamma_ee_0
        self.t_norm = self.v_th**3 / (
            gamma_ei_0
            * self.n_norm
            * physics_tools.lambda_ei(1.0, 1.0, self.T_norm, self.n_norm, Z)
            / Z
        )  # s
        self.x_norm = self.v_th * self.t_norm  # m
        self.sigma_0 = 8.797355066696007e-21  # m^2
        self.collrate_const = self.n_norm * self.v_th * self.sigma_0 * self.t_norm
        self.tbrec_norm = (
            self.n_norm
            * np.sqrt(
                (physics_tools.planck_h**2)
                / (
                    2
                    * np.pi
                    * physics_tools.el_mass
                    * self.T_norm
                    * physics_tools.el_charge
                )
            )
            ** 3
        )

    def apply_normalisation(self):
        # Apply normalisation
        self.Te /= self.T_norm
        self.ne /= self.n_norm
        self.vgrid /= self.v_th
        self.dvc /= self.v_th
        self.xgrid /= self.x_norm
        self.dxc /= self.x_norm
        if self.opts["kinetic_electrons"]:
            self.fe /= self.n_norm / (self.v_th**3)

    def generate_grid_widths(self):
        # Spatial grid widths
        self.dxc = np.zeros(self.num_x)
        self.dxc[0] = 2.0 * self.xgrid[1]
        self.dxc[-1] = 2.0 * (self.xgrid[-1] - self.xgrid[-2])
        for i in range(1, self.num_x - 1):
            self.dxc[i] = self.xgrid[i + 1] - self.xgrid[i - 1]

        # Velocity grid widths
        self.dvc = np.zeros(self.num_v)
        self.dvc[0] = 2 * self.vgrid[0]
        for i in range(1, self.num_v):
            self.dvc[i] = 2 * (self.vgrid[i] - self.vgrid[i - 1]) - self.dvc[i - 1]

    def run(self):
        """Run the program to find equilibrium impurity densities on the provided background plasma."""

        if self.opts["kinetic_electrons"]:
            self.build_matrix(kinetic=True)
            self.compute_densities(
                self.opts["delta_t"] / self.t_norm,
                int(self.opts["max_steps"]),
                self.opts["evolve"],
                kinetic=True,
            )
            # del self.rate_mats
        if self.opts["maxwellian_electrons"]:
            self.build_matrix(kinetic=False)
            self.compute_densities(
                self.opts["delta_t"] / self.t_norm,
                int(self.opts["max_steps"]),
                self.opts["evolve"],
                kinetic=False,
            )
            # del self.rate_mats_Max

    def calc_eff_rate_mats(self, kinetic=False, P_states="ground"):
        """Calculate the effective rate matrix at each spatial location, for the given P states ("metastables")

        Args:
            kinetic (bool, optional): whether to plot for kinetic or Maxwellian electrons. Defaults to False.
            P_states (str, optional): choice of P states (i.e. metastables). Defaults to "ground", meaning ground states of all ionization stages will be treated as evolved states..
        """

        # TODO: Add functions (probably in post_processing) to extract ionization, recombination coeffs etc from M_eff

        if kinetic:
            fe = self.fe
        else:
            fe = self.fe_Max

        eff_rate_mats = {}

        for el in self.opts["modelled_impurities"]:
            # TODO: Implement Greenland P-state validation checker
            self.impurities[el].reorder_PQ_states(P_states)

            eff_rate_mats[el] = [None] * self.loc_num_x

            num_P = self.impurities[el].num_P_states
            num_Q = self.impurities[el].num_Q_states

            for i in range(self.min_x, self.max_x):
                print("{:.1f}%".format(100 * i / self.loc_num_x), end="\r")

                # Build the local matrix
                M = matrix_utils.fill_local_mat(
                    self.impurities[el].transitions,
                    self.impurities[el].tot_states,
                    fe[:, i],
                    self.ne[i],
                    self.Te[i],
                    self.vgrid,
                    self.dvc,
                )

                # Calculate M_eff
                M_P = M[:num_P, :num_P]
                M_Q = M[num_P + 1 :, num_P + 1 :]
                M_PQ = M[:num_P, num_P + 1 :]
                M_QP = M[num_P + 1 :, :num_P]
                M_eff = -(M_P - M_PQ @ np.linalg.inv(M_Q) @ M_QP)

                eff_rate_mats[el][i - self.min_x] = M_eff

            print("{:.1f}%".format(100))

        if kinetic:
            self.eff_rate_mats = eff_rate_mats
        else:
            self.eff_rate_mats_Max = eff_rate_mats

    def build_matrix(self, kinetic=False):
        # Build the rate matrices
        for el in self.opts["modelled_impurities"]:
            if kinetic:
                if self.rank == 0:
                    print("Filling kinetic transition matrix for " + el + "...")
                if self.opts["use_petsc"]:
                    petsc_mat = matrix_utils.build_petsc_matrix(
                        self.loc_num_x,
                        self.min_x,
                        self.max_x,
                        self.impurities[el].tot_states,
                        self.impurities[el].transitions,
                        self.num_x,
                        self.opts["evolve"],
                    )
                    self.rate_mats[el] = matrix_utils.fill_petsc_rate_matrix(
                        self.loc_num_x,
                        self.min_x,
                        self.max_x,
                        petsc_mat,
                        self.impurities[el],
                        self.fe,
                        self.ne,
                        self.Te,
                        self.vgrid,
                        self.dvc,
                    )
                else:
                    np_mat = matrix_utils.build_np_matrix(
                        self.min_x, self.max_x, self.impurities[el].tot_states
                    )
                    self.rate_mats[el] = matrix_utils.fill_np_rate_matrix(
                        self.loc_num_x,
                        self.min_x,
                        self.max_x,
                        np_mat,
                        self.impurities[el],
                        self.fe,
                        self.ne,
                        self.Te,
                        self.vgrid,
                        self.dvc,
                    )
            else:
                if self.rank == 0:
                    print("Filling Maxwellian transition matrix for " + el + "...")
                if self.opts["use_petsc"]:
                    petsc_mat = matrix_utils.build_petsc_matrix(
                        self.loc_num_x,
                        self.min_x,
                        self.max_x,
                        self.impurities[el].tot_states,
                        self.impurities[el].transitions,
                        self.num_x,
                        self.opts["evolve"],
                    )
                    self.rate_mats_Max[el] = matrix_utils.fill_petsc_rate_matrix(
                        self.loc_num_x,
                        self.min_x,
                        self.max_x,
                        petsc_mat,
                        self.impurities[el],
                        self.fe_Max,
                        self.ne,
                        self.Te,
                        self.vgrid,
                        self.dvc,
                    )
                else:
                    np_mat = matrix_utils.build_np_matrix(
                        self.min_x, self.max_x, self.impurities[el].tot_states
                    )
                    self.rate_mats_Max[el] = matrix_utils.fill_np_rate_matrix(
                        self.loc_num_x,
                        self.min_x,
                        self.max_x,
                        np_mat,
                        self.impurities[el],
                        self.fe_Max,
                        self.ne,
                        self.Te,
                        self.vgrid,
                        self.dvc,
                    )

    def compute_densities(self, dt=None, num_t=None, evolve=True, kinetic=False):
        # Solve or evolve the matrix equation to find the equilibrium densities
        if evolve:
            for el in self.opts["modelled_impurities"]:
                if kinetic:
                    if self.rank == 0:
                        print(
                            "Computing densities with kinetic electrons for "
                            + el
                            + "..."
                        )
                    if self.opts["use_petsc"]:
                        n_solved = solver.evolve_petsc(
                            self.loc_num_x,
                            self.min_x,
                            self.max_x,
                            self.rate_mats[el],
                            self.impurities[el].dens,
                            self.num_x,
                            dt,
                            num_t,
                            self.opts["dndt_thresh"],
                            self.n_norm,
                            self.t_norm,
                            self.opts["ksp_solver"],
                            self.opts["ksp_pc"],
                            self.opts["ksp_tol"],
                        )
                    else:
                        n_solved = solver.evolve_np(
                            self.loc_num_x,
                            self.min_x,
                            self.max_x,
                            self.rate_mats[el],
                            self.impurities[el].dens,
                            self.num_x,
                            dt,
                            num_t,
                            self.opts["dndt_thresh"],
                            self.n_norm,
                            self.t_norm,
                            self.opts["ksp_solver"],
                            self.opts["ksp_pc"],
                            self.opts["ksp_tol"],
                        )
                    if n_solved is not None:
                        self.impurities[el].dens = n_solved
                        self.success = True
                    else:
                        self.success = False
                else:
                    if self.rank == 0:
                        print(
                            "Computing densities with Maxwellian electrons for "
                            + el
                            + "..."
                        )
                    if self.opts["use_petsc"]:
                        n_solved = solver.evolve_petsc(
                            self.loc_num_x,
                            self.min_x,
                            self.max_x,
                            self.rate_mats_Max[el],
                            self.impurities[el].dens_Max,
                            self.num_x,
                            dt,
                            num_t,
                            self.opts["dndt_thresh"],
                            self.n_norm,
                            self.t_norm,
                            self.opts["ksp_solver"],
                            self.opts["ksp_pc"],
                            self.opts["ksp_tol"],
                        )
                    else:
                        n_solved = solver.evolve_np(
                            self.loc_num_x,
                            self.min_x,
                            self.max_x,
                            self.rate_mats_Max[el],
                            self.impurities[el].dens_Max,
                            self.num_x,
                            dt,
                            num_t,
                            self.opts["dndt_thresh"],
                            self.n_norm,
                            self.t_norm,
                            self.opts["ksp_solver"],
                            self.opts["ksp_pc"],
                            self.opts["ksp_tol"],
                        )
                    if n_solved is not None:
                        self.impurities[el].dens_Max = n_solved
                        self.success = True
                    else:
                        self.success = False
        else:
            for el in self.opts["modelled_impurities"]:
                if kinetic:
                    if self.rank == 0:
                        print(
                            "Computing densities with kinetic electrons for "
                            + el
                            + "..."
                        )
                    if self.opts["use_petsc"]:
                        n_solved = solver.solve_petsc(
                            self.loc_num_x,
                            self.min_x,
                            self.max_x,
                            self.rate_mats[el],
                            self.impurities[el].dens,
                            self.num_x,
                            self.opts["ksp_solver"],
                            self.opts["ksp_pc"],
                            self.opts["ksp_tol"],
                        )
                    else:
                        n_solved = solver.solve_np(
                            self.loc_num_x,
                            self.min_x,
                            self.max_x,
                            self.rate_mats[el],
                            self.impurities[el].dens,
                            self.num_x,
                            self.opts["ksp_solver"],
                            self.opts["ksp_pc"],
                            self.opts["ksp_tol"],
                        )

                    if n_solved is not None:
                        self.impurities[el].dens = n_solved
                        self.success = True
                    else:
                        self.success = False

                else:
                    if self.rank == 0:
                        print(
                            "Computing densities with Maxwellian electrons for "
                            + el
                            + "..."
                        )
                    if self.opts["use_petsc"]:
                        n_solved = solver.solve_petsc(
                            self.loc_num_x,
                            self.min_x,
                            self.max_x,
                            self.rate_mats_Max[el],
                            self.impurities[el].dens_Max,
                            self.num_x,
                            self.opts["ksp_solver"],
                            self.opts["ksp_pc"],
                            self.opts["ksp_tol"],
                        )
                    else:
                        n_solved = solver.solve_np(
                            self.loc_num_x,
                            self.min_x,
                            self.max_x,
                            self.rate_mats_Max[el],
                            self.impurities[el].dens_Max,
                            self.num_x,
                            self.opts["ksp_solver"],
                            self.opts["ksp_pc"],
                            self.opts["ksp_tol"],
                        )

                    if n_solved is not None:
                        self.impurities[el].dens_Max = n_solved
                        self.success = True
                    else:
                        self.success = False
