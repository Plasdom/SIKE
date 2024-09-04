import numpy as np
from numba import jit
import os
import xarray as xr

from sike.atomics.impurity import Impurity
from sike.utils.constants import *
from sike.solver.matrix_utils import *
from sike.solver import solver
from sike.analysis.plasma_utils import *
from sike.io.generate_output import generate_output

# TODO: Do we ever want to actually evolve all states? Or only build M_eff and get derived coefficients? Opportunity to massively simplify by removing petsc & mpi dependency
# TODO: I guess we should only ever really be evolving the P states, so all that code is still useful, but could probably do it with dense numpy matrices rather than petsc, and probably don't need MPI!


class SIKERun(object):
    """
    A class which stores all relevant data and methods for a SIKE simulation.

    ...

    Initialisation: option 1
    ________________________
    Provide electron distribution functions on an x-grid. Temperatures and densities will be evaluated from electron distribution.
        fe: np.ndarray(num_v, num_x)
            isotropic part of electron distribution function as a function of velocity magnitude (units [m^-6 s^-3])
        vgrid: np.ndarray(num_v)
            velocity grid on which fe is defined (units [m/s])
        xgrid: np.ndarray(num_x)
            x-grid on which to evolve impurity densities (units [m])
        **kwargs: run options (see __init__() documentation for details)

    ...

    Initialisation: option 2
    ________________________
    Provide electron temperature and density profiles (assuming Maxwellian electrons) on an x-grid.
        Te: np.ndarray(num_x)
            electron temperature profile (units [eV])
        ne: np.ndarray(num_x)
            electron density profile (units [m^-3])
        xgrid: np.ndarray(num_x)
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
        element: str = "Li",
        delta_t: float = 1.0e-3,
        evolve: bool = True,
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
        :param vgrid: Velocity grid on which fe is defined (units [m/s]), defaults to None
        :param Te: electron temperature profile (units [eV]), defaults to None
        :param ne: electron density profile (units [m^-3]), defaults to None
        :param xgrid: x-grid on which to evolve impurity densities (units [m]), defaults to None
        :param impurity: The impurity species to evolve (use chemical symbols), defaults to "Li"
        :param delta_t: The time step to use in seconds if `evolve` option is true, defaults to 1.0e-3
        :param evolve: Specify whether to evolve the state density equations in time. If false, simply invert the rate matrix (this method sometimes suffers from numerical instabilities), defaults to True
        :param dndt_thresh: The threshold density residual between subsequent s which defines whether equilibrium has been reached, defaults to 1e-5
        :param max_steps: The maximum number of s to evolve if `evolve` is true, defaults to 1000
        :param frac_imp_dens: The fractional impurity density at initialisation, defaults to 0.05
        :param resolve_l: Reolve states by orbital angular momentum quantum number, defaults to True
        :param resolve_j: Reolve states by total angular momentum quantum number, defaults to True
        :param ionization: Include collisional ionisation and three-body recombination processes, defaults to True
        :param radiative_recombination: Include radiative recombination process, defaults to True
        :param excitation: Include collisional excitation and deexcitation processes, defaults to True
        :param emission: Include spontaneous emission process, defaults to True
        :param autoionization: Include autoionization process, defaults to True
        :param fixed_fraction_init: Specify whether to initialise impurity densities to fixed fraction of electron density. If false, use flat impurity density profiles., defaults to True
        :param saha_boltzmann_init: Specify whether to initialise impurity state densities to Saha-Boltzmann equilibrium, defaults to True
        :param state_ids: A specific list of state IDs to evolve. If None then all states in levels.json will be evolved., defaults to None
        :raises ValueError: If input is incorrectly specified (must specify either electron distribution and vgrid or electron temperature and density profiles)
        """
        # TODO: Change fe so that spatial index comes first (like everywhere else)

        # Save input options
        self.delta_t = delta_t
        self.evolve = evolve
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
        self.atom_data_savedir = self.get_atom_data_savedir()

        self.num_procs = 1  # TODO: Parallelisation
        self.rank = 0  # TODO: Parallelisation

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
        self.impurity = Impurity(
            name=element,
            resolve_l=self.resolve_l,
            resolve_j=self.resolve_j,
            state_ids=self.state_ids,
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
            atom_data_savedir=self.atom_data_savedir,
        )
        print("Finished initialising impurity species objects.")

        self.rate_mats = {}

    def get_atom_data_savedir(self) -> Path:
        """Open the config file to find the location of the saved atomic data

        :return: Path to atomic data savedir
        """
        config_file = Path(os.getenv("HOME")) / CONFIG_FILENAME
        if config_file.exists():
            with open(config_file, "r+") as f:
                l = f.readlines()
            atom_data_savepath = Path(l[0])
            if not atom_data_savepath.exists():
                raise FileNotFoundError(
                    "The atomic data savedir specified in the config file does not appear to exist. Has it been moved? Check the config file ('$HOME/.sike_config') or re-run setup, see readme for instructions."
                )
            return atom_data_savepath
        else:
            raise FileNotFoundError(
                "No config file found. Have you run setup to download the atomic data? See readme ofr instructions."
            )

    def init_from_dist(self) -> None:
        """Initialise simulation from electron distributions"""
        self.num_x = len(self.fe[0, :])
        if self.xgrid is None:
            self.xgrid = np.linspace(0, 1, self.num_x)
        self.num_v = len(self.vgrid)
        self.generate_grid_widths()

        self.rank = 0  # TODO: Implement parallelisation
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
                density_moment(self.fe[:, i], self.vgrid, self.dvc)
                for i in range(self.num_x)
            ]
        )
        self.Te = np.array(
            [
                temperature_moment(
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

    def init_from_profiles(self, vgrid: np.ndarray | None = None):
        """Initialise simulation from electron temperature and density profiles

        :param vgrid: Electron velocity grid, defaults to None
        """

        # Save/create the velocity grid
        if vgrid is None:
            self.vgrid = DEFAULT_VGRID.copy()
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

        # Generature Maxwellians
        self.fe = get_maxwellians(self.ne, self.Te, self.vgrid, normalised=False)

        # Generate normalisation constants and normalise everything
        self.init_norms()
        self.apply_normalisation()

        # Create the E_grid
        self.Egrid = self.T_norm * self.vgrid**2

    def init_norms(self) -> None:
        """Initialise the normalisation constants for the simulation"""

        self.T_norm = np.mean(self.Te)  # eV
        self.n_norm = np.mean(self.ne) * self.frac_imp_dens  # m^-3
        self.v_th = np.sqrt(2 * self.T_norm * EL_CHARGE / EL_MASS)  # m/s

        Z = 1
        gamma_ee_0 = EL_CHARGE**4 / (4 * np.pi * (EL_MASS * EPSILON_0) ** 2)
        gamma_ei_0 = Z**2 * gamma_ee_0
        self.t_norm = self.v_th**3 / (
            gamma_ei_0
            * self.n_norm
            * lambda_ei(1.0, 1.0, self.T_norm, self.n_norm, Z)
            / Z
        )  # s
        self.x_norm = self.v_th * self.t_norm  # m
        self.sigma_0 = 8.797355066696007e-21  # m^2
        self.collrate_const = self.n_norm * self.v_th * self.sigma_0 * self.t_norm
        self.tbrec_norm = (
            self.n_norm
            * np.sqrt((PLANCK_H**2) / (2 * np.pi * EL_MASS * self.T_norm * EL_CHARGE))
            ** 3
        )

    def apply_normalisation(self) -> None:
        """Normalise all physical values in the simulation"""
        # Apply normalisation
        self.Te /= self.T_norm
        self.ne /= self.n_norm
        self.vgrid /= self.v_th
        self.dvc /= self.v_th
        self.xgrid /= self.x_norm
        self.dxc /= self.x_norm
        self.delta_t /= self.t_norm
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

    def run(self) -> xr.Dataset:
        """Run the program to find equilibrium impurity densities on the provided background plasma.

        :return: xarray dataset containing densities, states, transitions and other relevant information for the case
        """

        self.build_matrix()
        self.compute_densities(
            dt=self.delta_t,
            num_t=self.max_steps,
            evolve=self.evolve,
        )
        return generate_output(
            self.impurity,
            self.xgrid,
            self.vgrid,
            self.x_norm,
            self.t_norm,
            self.n_norm,
            self.v_th,
            self.T_norm,
            self.Te,
            self.ne,
            self.fe,
            self.rate_mats,
            self.resolve_l,
            self.resolve_j,
            self.ionization,
            self.radiative_recombination,
            self.excitation,
            self.emission,
            self.autoionization,
            self.atom_data_savedir,
        )

    def calc_eff_rate_mats(self, P_states: str = "ground") -> None:
        """Calculate the effective rate matrix at each spatial location, for the given P states ("metastables")

        :param P_states: choice of P states (i.e. metastables). Defaults to "ground", meaning ground states of all ionization stages will be treated as evolved states.
        """

        # TODO: Add functions (probably in post_processing) to extract ionization, recombination coeffs etc from M_eff

        fe = self.fe

        eff_rate_mats = {}

        # TODO: Implement Greenland P-state validation checker
        self.impurity.reorder_PQ_states(P_states)

        eff_rate_mats = [None] * self.loc_num_x

        num_P = self.impurity.num_P_states
        num_Q = self.impurity.num_Q_states

        for i in range(self.min_x, self.max_x):
            print("{:.1f}%".format(100 * i / self.loc_num_x), end="\r")

            # Build the local matrix
            M = fill_local_mat(
                self.impurity.transitions,
                self.impurity.tot_states,
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

            eff_rate_mats[i - self.min_x] = M_eff

        print("{:.1f}%".format(100))

        self.eff_rate_mats = eff_rate_mats

    def build_matrix(self) -> None:
        """Build the rate matrices"""
        # Build the rate matrices
        if self.rank == 0:
            print("Filling transition matrix for...")
        np_mat = build_matrix(self.min_x, self.max_x, self.impurity.tot_states)
        self.rate_mats = fill_rate_matrix(
            self.loc_num_x,
            self.min_x,
            self.max_x,
            np_mat,
            self.impurity,
            self.fe,
            self.ne,
            self.Te,
            self.vgrid,
            self.dvc,
        )

    def compute_densities(
        self,
        num_t: int | None = None,
        evolve: bool = True,
        dt: float | None = None,
    ) -> None:
        """Solve or evolve the matrix equation to find the equilibrium densities

        :param evolve: Whether to solve densities directly or evolve until equilibrium is reached, defaults to True
        :param dt: Timestep to use if evolve=True, defaults to None
        :param num_t: Number of timesteps to use if evolve=True, defaults to None
        """
        # Solve or evolve the matrix equation to find the equilibrium densities
        if evolve:
            if self.rank == 0:
                print("Computing densities for...")
            n_solved = solver.evolve(
                self.loc_num_x,
                self.min_x,
                self.max_x,
                self.rate_mats,
                self.impurity.dens,
                dt,
                num_t,
                self.dndt_thresh,
                self.n_norm,
                self.t_norm,
            )
            if n_solved is not None:
                self.impurity.dens = n_solved
                self.success = True
            else:
                self.success = False
        else:
            if self.rank == 0:
                print("Computing densities with for...")
            n_solved = solver.solve(
                self.loc_num_x,
                self.min_x,
                self.max_x,
                self.rate_mats,
                self.impurity.dens,
            )

            if n_solved is not None:
                self.impurity.dens = n_solved
                self.success = True
            else:
                self.success = False
