import xarray as xr
import pandas as pd

from sike.core import SIKERun


def generate_densities_output(case: SIKERun) -> xr.Dataset:
    """Generate an xarray dataset containing states and densities

    :param case: A SIKERun case
    :return: An xarray dataset containing state information and densities
    """
    # Extract relevant information (densities & states) from the SIKERun object
    dens = case.impurity.dens
    x = case.xgrid
    k = [s.id for s in case.impurity.states]
    selected_cols = [
        "id",
        "Z",
        "n",
        "l",
        "j",
        "stat_weight",
        "iz_energy",
        "energy_from_gs",
    ]
    states = [
        {k: v for k, v in s.__dict__.items() if k in selected_cols}
        for s in case.impurity.states
    ]

    # Generate the densities DataArray
    dens_da = xr.DataArray(dens, coords={"x": x, "k": k})
    dens_da.attrs["long_name"] = "Density"
    dens_da.attrs["units"] = "[n_0]"
    dens_da.attrs["description"] = "Atomic state density in normalised units"
    dens_da.x.attrs["long_name"] = "x"
    dens_da.x.attrs["units"] = "[x_0]"
    dens_da.x.attrs["description"] = "Spatial coordinate in normalised units"
    dens_da.k.attrs["long_name"] = "k"
    dens_da.k.attrs["units"] = "N/A"
    dens_da.k.attrs["description"] = "Atomic state index"

    # Generate the states Dataset from a pandas dataframe
    states_df = pd.DataFrame(states).rename(columns={"id": "k"}).set_index("k")
    states_ds = xr.Dataset.from_dataframe(states_df)

    # Add the densities to this Dataset
    states_ds["density"] = dens_da

    # Add Te and ne arrays
    Te_da = xr.DataArray(case.Te, coords={"x": x})
    ne_da = xr.DataArray(case.ne, coords={"x": x})
    states_ds["Te"] = Te_da
    states_ds["ne"] = ne_da

    # Add vgrid and v coordinate?

    # Add metadata
    metadata_dict = {
        "x_norm": case.x_norm,
        "T_norm": case.T_norm,
        "n_norm": case.T_norm,
        "time_norm": case.t_norm,
        "resolve_l": case.resolve_l,
        "resolve_j": case.resolve_j,
        "ionization": case.ionization,
        "radiative_recombination": case.radiative_recombination,
        "excitation": case.excitation,
        "emission": case.emission,
        "autoionization": case.autoionization,
        "atom_data_savedir": case.atom_data_savedir,
        "min_x": 0,
        "max_x": 100,
        "v_th": case.v_th,
        "sigma_0": case.sigma_0,
        "collrate_const": case.collrate_const,
        "tbrec_norm": case.tbrec_norm,
    }
    states_ds.attrs["metadata"] = metadata_dict

    return states_ds
