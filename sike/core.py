import numpy as np

from impurity import Impurity
import matrix_utils
import solver
from constants import *

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
        **kwargs: run options (see __init__() documentation for details)

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
        **kwargs: run options (see __init__() documentation for details)
    """

    def __init__(
        self,
        fe: np.ndarray | None = None,
        vgrid: np.ndarray | None = None,
        Te: np.ndarray | None = None,
        ne: np.ndarray | None = None,
        xgrid: np.ndarray | None = None,
        modelled_impurities: list[str] = ["Li"],
        delta_t: float = 1.0e-3,
        evolve: bool = True,
        kinetic_electrons: bool = False,
        maxwellian_electrons: bool = True,
        dndt_thresh: float = 1e-5,
        max_steps: int = 1000,
        frac_imp_dens: float = 0.05,
        resolve_l: bool = True,
        resolve_j: bool = True,
        ionization: bool = True,
        radiative_recombination: bool = True,
        excitation: bool = True,
        emission: bool = True,
        autoionization: bool = True,
        fixed_fraction_init: bool = True,
        saha_boltzmann_init: bool = True,
        state_ids: list[int] | None = None,
    ):
        """Initialise

        :param fe: Isotropic part of electron distribution function as a function of velocity magnitude (units [m^-6 s^-3]), defaults to None
        :type fe: np.ndarray | None, optional
        :param vgrid: Velocity grid on which fe is defined (units [m/s]), defaults to None
        :type vgrid: np.ndarray | None, optional
        :param Te: electron temperature profile (units [eV]), defaults to None
        :type Te: np.ndarray | None, optional
        :param ne: electron density profile (units [m^-3]), defaults to None
        :type ne: np.ndarray | None, optional
        :param xgrid: x-grid on which to evolve impurity densities (units [m]), defaults to None
        :type xgrid: np.ndarray | None, optional
        :param modelled_impurities: A list of the impurity species to evolve (use chemical symbols), defaults to ["Li"]
        :type modelled_impurities: list[str], optional
        :param delta_t: The time step to use in seconds if `evolve` option is true, defaults to 1.0e-3
        :type delta_t: float, optional
        :param evolve: Specify whether to evolve the state density equations in time. If false, simply invert the rate matrix (this method sometimes suffers from numerical instabilities), defaults to True
        :type evolve: bool, optional
        :param kinetic_electrons: Solve rate equations for Maxwellian electrons at given density and temperatures, defaults to False
        :type kinetic_electrons: bool, optional
        :param maxwellian_electrons: Solve rate equations for kinetic electrons with given distribution functions, defaults to True
        :type maxwellian_electrons: bool, optional
        :param dndt_thresh: The threshold density residual between subsequent s which defines whether equilibrium has been reached, defaults to 1e-5
        :type dndt_thresh: float, optional
        :param max_steps: The maximum number of s to evolve if `evolve` is true, defaults to 1000
        :type max_steps: int, optional
        :param frac_imp_dens: The fractional impurity density at initialisation, defaults to 0.05
        :type frac_imp_dens: float, optional
        :param resolve_l: _description_, defaults to True
        :type resolve_l: bool, optional
        :param resolve_j: _description_, defaults to True
        :type resolve_j: bool, optional
        :param ionization: Include collisional ionisation and three-body recombination processes, defaults to True
        :type ionization: bool, optional
        :param radiative_recombination: Include radiative recombination process, defaults to True
        :type radiative_recombination: bool, optional
        :param excitation: Include collisional excitation and deexcitation processes, defaults to True
        :type excitation: bool, optional
        :param emission: Include spontaneous emission process, defaults to True
        :type emission: bool, optional
        :param autoionization: Include autoionization process, defaults to True
        :type autoionization: bool, optional
        :param fixed_fraction_init: Specify whether to initialise impurity densities to fixed fraction of electron density. If false, use flat impurity density profiles., defaults to True
        :type fixed_fraction_init: bool, optional
        :param saha_boltzmann_init: Specify whether to initialise impurity state densities to Saha-Boltzmann equilibrium, defaults to True
        :type saha_boltzmann_init: bool, optional
        :param state_ids: A specific list of state IDs to evolve. If None then all states in levels.json will be evolved., defaults to None
        :type state_ids: list[int] | None, optional
        :raises ValueError: If input is incorrectly specified (must specify either electron distribution and vgrid or electron temperature and density profiles)
        """
        # TODO: Change fe so that spatial index comes first (like everywhere else)

        # Save input options
        self.modelled_impurities = modelled_impurities
        self.delta_t = delta_t
        self.evolve = evolve
        self.kinetic_electrons = kinetic_electrons
        self.maxwellian_electrons = maxwellian_electrons
        self.dndt_thresh = dndt_thresh
        self.max_steps = max_steps
        self.frac_imp_dens = frac_imp_dens
        self.resolve_l = resolve_l
        self.resolve_j = resolve_j
        self.ionization = ionization
        self.radiative_recombination = radiative_recombination
        self.excitation = excitation
        self.emission = emission
        self.autoionization = autoionization
        self.fixed_fraction_init = fixed_fraction_init
        self.saha_boltzmann_init = saha_boltzmann_init
        self.state_ids = state_ids

        # Save simulation set-up
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
        print("Initialising the impurity species to be modelled...")
        self.impurities = {}
        for el in self.opts["modelled_impurities"]:
            self.impurities[el] = Impurity(
                name=el,
                resolve_l=self.resolve_l,
                resolve_j=self.resolve_j,
                state_ids=self.state_ids,
                kinetic_electrons=self.kinetic_electrons,
                maxwellian_electrons=self.maxwellian_electrons,
                saha_boltzmann_init=self.saha_boltzmann_init,
                fixed_fraction_init=self.fixed_fraction_init,
                frac_imp_dens=self.frac_imp_dens,
                ionization=self.ionization,
                autoionization=self.autoionization,
                emission=self.emission,
                radiative_recombination=self.radiative_recombination,
                excitation=self.excitation,
                collrate_const=self.collrate_const,
                tbrec_norm=self.tbrec_norm,
                sigma_norm=self.sigma_0,
                time_norm=self.t_norm,
                T_norm=self.T_norm,
                n_norm=self.n_norm,
                vgrid=self.vgrid,
                Egrid=self.Egrid,
                ne=self.ne,
                Te=self.Te,
            )
        print("Finished initialising impurity species objects.")

        self.rate_mats = {}
        self.rate_mats_Max = {}

    def init_from_dist(self):
        """Initialise simulation from electron distributions"""
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

    def init_from_profiles(self, vgrid: np.ndarray | None = None):
        """Initialise simulation from electron temperature and density profiles

        :param vgrid: Electron velocity grid, defaults to None
        :type vgrid: np.ndarray or None, optional
        """
        # Rate equations will be solved for Maxwellian electrons only
        self.kinetic_electrons = False
        self.maxwellian_electrons = True

        # Save/create the velocity grid
        if vgrid is None:
            self.vgrid = physics_tools.default_vgrid.copy()
        else:
            self.vgrid = vgrid
        self.num_x = len(self.Te)
        if self.xgrid is None:
            self.xgrid = np.linspace(0, 1, self.num_x)
        self.num_v = len(self.vgrid)
        self.generate_grid_widths()

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
        """Initialise the normalisation constants for the simulation"""

        self.T_norm = np.mean(self.Te)  # eV
        self.n_norm = np.mean(self.ne) * self.frac_imp_dens  # m^-3
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
        """Normalise all physical values in the simulation"""
        # Apply normalisation
        self.Te /= self.T_norm
        self.ne /= self.n_norm
        self.vgrid /= self.v_th
        self.dvc /= self.v_th
        self.xgrid /= self.x_norm
        self.dxc /= self.x_norm
        self.delta_t /= self.t_norm
        if self.kinetic_electrons:
            self.fe /= self.n_norm / (self.v_th**3)

    def generate_grid_widths(self):
        """Generate spatial and velocity grid widths"""
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

        if self.kinetic_electrons:
            self.build_matrix(kinetic=True)
            self.compute_densities(
                dt=self.delta_t,
                num_t=self.max_steps,
                evolve=self.evolve,
                kinetic=True,
            )
            # del self.rate_mats
        if self.maxwellian_electrons:
            self.build_matrix(kinetic=False)
            self.compute_densities(
                dt=self.delta_t,
                num_t=self.max_steps,
                evolve=self.evolve,
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


@jit(nopython=True)
def lambda_ei(n, T, T_0, n_0, Z_0):
    """e-i Coulomb logarithm

    Args:
        n (float): density
        T (float): temperature
        T_0 (float): temperature normalisation
        n_0 (float): density normalisation
        Z_0 (float): Ion charge

    Returns:
        float: lambda_ei
    """
    if T * T_0 < 10.00 * Z_0**2:
        return 23.00 - np.log(
            np.sqrt(n * n_0 * 1.00e-6) * Z_0 * (T * T_0) ** (-3.00 / 2.00)
        )
    else:
        return 24.00 - np.log(np.sqrt(n * n_0 * 1.00e-6) / (T * T_0))


@jit(nopython=True)
def maxwellian(T, n, vgrid):
    """Return a normalised (to n_0 / v_th,0 ** 3) Maxwellian electron distribution (isotropic, as function of velocity magnitude).

    Args:
        T (float): Normalised electron temperature
        n (float): Normalised electron density
        vgrid (np.array, optional): Normalised velocity grid on which to define Maxwellian distribution. If None, create using vgrid = np.arange(0.00001, 10, 1. / 1000.)

    Returns:
        np.array(num_v): numpy array of Maxwellian
    """

    f = [0.0 for i in range(len(vgrid))]
    for i, v in enumerate(vgrid):
        f[i] = n * (np.pi * T) ** (-3 / 2) * np.exp(-(v**2) / T)
    f = np.array(f)

    return f


@jit(nopython=True)
def bimaxwellian(T1, n1, T2, n2, vgrid):
    """Return a normalised (to n_0 / v_th,0 ** 3) Maxwellian electron distribution (isotropic, as function of velocity magnitude).

    Args:
        T1 (float): First population electron temperature
        n1 (float): First population electron density
        T2 (float): Second population electron temperature
        n2 (float): Second population electron density
        vgrid (np.array, optional): Velocity grid on which to define Maxwellian distribution

    Returns:
        np.array(num_v): numpy array of Maxwellian
    """

    f = [0.0 for i in range(len(vgrid))]
    for i, v in enumerate(vgrid):
        f[i] = (n1 * (np.pi * T1) ** (-3 / 2) * np.exp(-(v**2) / T1)) + (
            n2 * (np.pi * T2) ** (-3 / 2) * np.exp(-(v**2) / T2)
        )
    f = np.array(f)

    return f


def boltzmann_dist(Te, energies, stat_weights, gnormalise=False):
    """Generate a boltzmann distribution for the given set of energies and statistical weights

    Args:
        Te (np.ndarray): Electron temperature array [eV]
        energies (np.ndarray): Atomic state energies [eV]
        stat_weights (np.ndarray): Atomic state staistical weights
        gnormalise (bool, optional): Option to normalise output densities by their statistical weights. Defaults to False.

    Returns:
        np.ndarray: Boltzmann-distributed densities, relative to ground state
    """
    rel_dens = np.zeros(len(energies))
    for i in range(len(energies)):
        rel_dens[i] = (stat_weights[i] / stat_weights[0]) * np.exp(
            -(energies[i] - energies[0]) / Te
        )
        if gnormalise:
            rel_dens[i] /= stat_weights[i]
    return rel_dens


def saha_dist(Te, ne, imp_dens_tot, impurity):
    """Generate a Saha distribution of ionization stage densities for the given electron temperature

    Args:
        Te (_type_): _description_
        ne (_type_): _description_
        imp_dens_tot (_type_): _description_
        r (_type_): _description_
        el (_type_): _description_
    """
    el_mass = 9.10938e-31
    el_charge = 1.602189e-19
    planck_h = 6.62607004e-34

    ground_states = [s for s in impurity.states if s.ground is True]
    ground_states = list(reversed(sorted(ground_states, key=lambda x: x.num_el)))

    de_broglie_l = np.sqrt((planck_h**2) / (2 * np.pi * el_mass * el_charge * Te))

    # Compute ratios
    dens_ratios = np.zeros(impurity.num_Z - 1)
    for z in range(1, impurity.num_Z):
        eps = -(ground_states[z - 1].energy - ground_states[z].energy)
        stat_weight_zm1 = ground_states[z - 1].stat_weight
        stat_weight = ground_states[z].stat_weight

        dens_ratios[z - 1] = (
            2 * (stat_weight / stat_weight_zm1) * np.exp(-eps / Te)
        ) / (ne * (de_broglie_l**3))

    # Fill densities
    denom_sum = 1.0 + np.sum(
        [np.prod(dens_ratios[: z + 1]) for z in range(impurity.num_Z - 1)]
    )
    dens_saha = np.zeros(impurity.num_Z)
    dens_saha[0] = imp_dens_tot / denom_sum
    for z in range(1, impurity.num_Z):
        dens_saha[z] = dens_saha[z - 1] * dens_ratios[z - 1]

    return dens_saha


def get_maxwellians(ne, Te, vgrid, normalised=True):
    """Return an array of Maxwellian electron distributions with the given densities and temperatures.

    Args:
        ne (np.array): Normalised electron densities
        Te (np.array): Normalised electron temperatures
        vgrid (np.array): Normalised velocity grid on which to calculate Maxwellians

    Returns:
        np.array(num_v, num_x): 2d numpy array of Maxwellians at each location in x
    """

    if normalised is False:
        T_norm = 10
        n_norm = 1e19
        v_th = np.sqrt(2 * EL_CHARGE * T_norm / EL_MASS)
        ne /= n_norm
        Te /= T_norm
        vgrid = vgrid.copy()
        vgrid /= v_th

    f0_max = [[0.0 for i in range(len(ne))] for j in range(len(vgrid))]
    for i in range(len(ne)):
        f0_max_loc = maxwellian(Te[i], ne[i], vgrid)
        for j in range(len(vgrid)):
            f0_max[j][i] = f0_max_loc[j]
    f0_max = np.array(f0_max)

    if normalised is False:
        f0_max *= n_norm / v_th**3

    return f0_max


def get_bimaxwellians(n1, n2, T1, T2, vgrid, normalised=True):
    """Return an array of bi-Maxwellian electron distributions with the given densities and temperatures.

    Args:
        T1 (np.ndarray): First population electron temperatures
        n1 (np.ndarray): First population electron densities
        T2 (np.ndarray): Second population electron temperatures
        n2 (np.ndarray): Second population electron densities
        vgrid (np.array): Velocity grid on which to calculate bi-Maxwellians
        normalised (bool):

    Returns:
        np.array(num_v, num_x): 2d numpy array of Maxwellians at each location in x
    """

    if normalised is False:
        T_norm = 10
        n_norm = 1e19
        v_th = np.sqrt(2 * EL_CHARGE * T_norm / EL_MASS)
        n1 = n1.copy()
        n2 = n2.copy()
        T1 = T1.copy()
        T2 = T2.copy()
        n1 /= n_norm
        n2 /= n_norm
        T1 /= T_norm
        T2 /= T_norm
        vgrid = vgrid.copy()
        vgrid /= v_th

    f0_bimax = np.zeros([len(vgrid), len(n1)])
    for i in range(len(n1)):
        f0_bimax_loc = bimaxwellian(T1[i], n1[i], T2[i], n2[i], vgrid)
        for j in range(len(vgrid)):
            f0_bimax[j, i] = f0_bimax_loc[j]
    f0_bimax = np.array(f0_bimax)

    if normalised is False:
        f0_bimax *= n_norm / v_th**3

    return f0_bimax


@jit(nopython=True)
def density_moment(f0, vgrid, dvc):
    """Calculate density moment of input electron distribution

    Args:
        f0 (np.array): Electron distribution
        vgrid (_type_): Velocity grid
        dvc (_type_): Velocity grid widths
        normalised (bool, optional): Specify if inputs and output are normalised. Defaults to False.

    Returns:
        float: density. Units are normalised or m**-3 depending on whether inputs are normalised.
    """
    n = 4 * np.pi * np.sum(f0 * vgrid**2 * dvc)
    return n


@jit(nopython=True)
def temperature_moment(f0, vgrid, dvc, normalised=True):
    """_summary_

    Args:
        f0 (_type_): _description_
        vgrid (_type_): _description_
        dvc (_type_): _description_
        normalised (bool, optional): _description_. Defaults to True.

    Returns:
        float: temperature. Units are dimensionless or eV depending on normalised argument
    """

    n = density_moment(f0, vgrid, dvc)
    if normalised:
        T = (2 / 3) * 4 * np.pi * np.sum(f0 * vgrid**4 * dvc) / n
    else:
        T = (2 / 3) * 4 * np.pi * 0.5 * EL_MASS * np.sum(f0 * vgrid**4 * dvc) / n
        T /= EL_CHARGE

    return T
