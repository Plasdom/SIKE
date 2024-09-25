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
        frac_imp_dens: float = 0.05,
        resolve_l: bool = False,
        resolve_j: bool = False,
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
        :param frac_imp_dens: The fractional impurity density at initialisation, defaults to 0.05
        :param resolve_l: Reolve states by orbital angular momentum quantum number, defaults to False
        :param resolve_j: Reolve states by total angular momentum quantum number, defaults to False
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
        self.atom_data_savedir = get_atom_data_savedir()

        self.num_procs = 1  # TODO: Parallelisation
        self.rank = 0  # TODO: Parallelisation

        self.matrix_needs_building = True

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
        Te_norm = self.Te.copy() / self.T_norm
        self.Te = Te_norm
        ne_norm = self.ne.copy() / self.n_norm
        self.ne = ne_norm
        vgrid_norm = self.vgrid.copy() / self.v_th
        self.vgrid = vgrid_norm
        dvc_norm = self.dvc.copy() / self.v_th
        self.dvc = dvc_norm
        xgrid_norm = self.xgrid.copy() / self.x_norm
        self.xgrid = xgrid_norm
        dxc_norm = self.dxc.copy() / self.x_norm
        self.dxc = dxc_norm
        fe_norm = self.fe.copy() / (self.n_norm / (self.v_th**3))
        self.fe = fe_norm

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

    def solve(self) -> xr.Dataset:
        """Carry out direct solve on state density evolution equations

        :return: xarray dataset containing densities, states, transitions and other relevant information for the case
        """
        self.build_matrix()

        self.impurity.dens = solver.solve(
            self.loc_num_x,
            self.min_x,
            self.max_x,
            self.rate_mats,
            self.impurity.dens,
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

    def evolve(self, dt: float, num_t: int = 10) -> xr.Dataset:
        """Evolve the rate equations by a set timestep

        :param dt: Timestep in seconds
        :param num_t: Number of timesteps to take, defaults to 1
        :return: xarray dataset containing densities, states, transitions and other relevant information for the case
        """
        self.build_matrix()

        sub_dt = (dt / self.t_norm) / num_t
        print("Evolving evolution equations...")
        self.impurity.dens = solver.evolve(
            self.loc_num_x,
            self.min_x,
            self.max_x,
            self.rate_mats,
            self.impurity.dens,
            sub_dt,
            num_t,
        )
        print("Done.")

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

    def build_matrix(self) -> None:
        """Build the rate matrices"""
        if self.matrix_needs_building:
            # Build the rate matrices
            if self.rank == 0:
                print("Filling transition matrix for " + self.impurity.longname)
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

            self.matrix_needs_building = False


def get_atom_data_savedir() -> Path:
    """Open the config file to find the location of the saved atomic data

    :return: Path to atomic data savedir
    """
    config_file = Path(os.getenv("HOME")) / CONFIG_FILENAME
    if config_file.exists():
        with open(config_file, "r+") as f:
            l = f.readlines()
        atom_data_savepath = Path(l[0].strip("\n"))
        if not atom_data_savepath.exists():
            raise FileNotFoundError(
                "The atomic data savedir specified in the config file does not appear to exist. Has it been moved? Check the config file ('$HOME/.sike_config') or re-run setup, see readme for instructions."
            )
        return atom_data_savepath
    else:
        raise FileNotFoundError(
            "No config file found. Have you run setup to download the atomic data? See readme ofr instructions."
        )
