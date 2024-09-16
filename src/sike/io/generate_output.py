import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path

from sike.atomics.impurity import Impurity


def generate_output(
    impurity: Impurity,
    xgrid: np.ndarray,
    vgrid: np.ndarray,
    x_norm: float,
    t_norm: float,
    n_norm: float,
    v_th: float,
    T_norm: float,
    Te: np.ndarray,
    ne: np.ndarray,
    fe: np.ndarray,
    rate_mats: list[np.ndarray],
    resolve_l: bool,
    resolve_j: bool,
    ionization: bool,
    radiative_recombination: bool,
    excitation: bool,
    emission: bool,
    autoionization: bool,
    atom_data_savedir: Path,
) -> xr.Dataset:
    """Generate an xarray dataset containing all relevant case information (densities, coordinates, states, transitions, rate matrices, etc)

    :param impurity: Impurity object containing information on states modelled, transitions, etc
    :param xgrid: Spatial grid
    :param vgrid: Velocity grid
    :param x_norm: X normalisation
    :param t_norm: Time normalisation
    :param n_norm: Density normalisation
    :param v_th: Velocity normalisation
    :param T_norm: Temperature normalisation
    :param Te: Electron temperature
    :param ne: Electron density
    :param fe: Electron velocity distributions
    :param rate_mats: Rate matrices
    :param resolve_l: Whether l states are resolved
    :param resolve_j: Whether j states are resolved
    :param ionization: Whether ionisation process is included
    :param radiative_recombination: Whether radiative recombination process is included
    :param excitation: Whether excitation process is included
    :param emission: Whether radiative emission process is included
    :param autoionization: Whether autoionisation process is included
    :param atom_data_savedir: Location of atomic data used
    :return: xarray dataset
    """
    # Extract relevant information (densities & states) from the SIKERun object
    dens = impurity.dens
    x = xgrid * x_norm
    k = [s.id for s in impurity.states]
    v = vgrid * v_th
    selected_state_cols = [
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
        {"state_" + k: v for k, v in s.__dict__.items() if k in selected_state_cols}
        for s in impurity.states
    ]
    selected_trans_cols = ["from_id", "to_id", "type", "rate", "rate_inv", "delta_E"]
    transitions = [
        {
            "transition_" + k: v
            for k, v in s.__dict__.items()
            if k in selected_trans_cols
        }
        for s in impurity.transitions
    ]
    transitions_df = pd.DataFrame(transitions)
    transitions_df.index.name = "i"
    transitions_df.rename(
        columns={
            "transition_from_id": "transition_from_k",
            "transition_to_id": "transition_to_k",
        }
    )

    # Generate Dataarrays
    dens_da = xr.DataArray(dens * n_norm, coords={"x": x, "k": k})
    Te_da = xr.DataArray(Te * T_norm, coords={"x": x})
    ne_da = xr.DataArray(ne * n_norm, coords={"x": x})
    fe_da = xr.DataArray(fe * n_norm / (v_th**3), coords={"v": v, "x": x})
    rate_mats_da = xr.DataArray(
        [mat * t_norm for mat in rate_mats],
        coords={"x": x, "j": k, "k": k},
    )  # TODO: Check normalisation here

    # Generate the states dataset from a pandas dataframe
    states_df = pd.DataFrame(states).rename(columns={"state_id": "k"}).set_index("k")
    output_ds = xr.Dataset.from_dataframe(states_df)

    # Generate transitions dataset and merge with the states dataset
    transitions_df["transition_delta_E"] *= T_norm
    transitions_df["transition_rate"] /= t_norm
    transitions_ds = xr.Dataset.from_dataframe(transitions_df)
    output_ds = output_ds.merge(transitions_ds)

    # Combine with other dataarrays
    output_ds["nk"] = dens_da
    output_ds["Te"] = Te_da
    output_ds["ne"] = ne_da
    output_ds["fe"] = fe_da
    output_ds["M"] = rate_mats_da

    # Add metadata and other info
    output_ds.attrs["metadata"] = get_metadata(
        impurity,
        resolve_l,
        resolve_j,
        ionization,
        radiative_recombination,
        excitation,
        emission,
        autoionization,
        atom_data_savedir,
    )
    output_ds = add_coordinate_info(output_ds)
    output_ds = add_data_info(output_ds)

    return output_ds


def get_metadata(
    impurity: Impurity,
    resolve_l: bool,
    resolve_j: bool,
    ionization: bool,
    radiative_recombination: bool,
    excitation: bool,
    emission: bool,
    autoionization: bool,
    atom_data_savedir: Path,
) -> dict:
    """Generate metadata dictionary

    :param impurity: Impurity object containing states modelled, transitions, etc
    :param resolve_l: Whether l states are resolved
    :param resolve_j: Whether j states are resolved
    :param ionization: Whether ionisation process is included
    :param radiative_recombination: Whether radiative recombination process is included
    :param excitation: Whether excitation process is included
    :param emission: Whether radiative emission process is included
    :param autoionization: Whether autoionisation process is included
    :param atom_data_savedir: Location of atomic data used
    :return: Metadata dictionary
    """
    # Generate dictionary
    metadata_dict = {
        "element": impurity.name,
        "longname": impurity.longname,
        "resolve_l_states": resolve_l,
        "resolve_j_states": resolve_j,
        "ionization": ionization,
        "radiative_recombination": radiative_recombination,
        "excitation": excitation,
        "emission": emission,
        "autoionization": autoionization,
        "atom_data_savedir": atom_data_savedir,
    }
    metadata_dict

    return metadata_dict


def add_data_info(ds: xr.Dataset) -> xr.Dataset:
    """Add information on the dataarrays to the dataset

    :param ds: Xarray dataset built from SIKERun
    :return: Modified xarray dataset
    """
    # Densities
    ds.nk.attrs["long_name"] = "nk"
    ds.nk.attrs["units"] = "[m^-3]"
    ds.nk.attrs["description"] = "Impurity atomic state density"
    ds.ne.attrs["long_name"] = "ne"
    ds.ne.attrs["units"] = "[m^-3]"
    ds.ne.attrs["description"] = "Electron density"

    # Te
    ds.Te.attrs["long_name"] = "Te"
    ds.Te.attrs["units"] = "[eV]"
    ds.Te.attrs["description"] = "Electron temperature"

    # fe
    ds.fe.attrs["long_name"] = "fe"
    ds.fe.attrs["units"] = "[m^-6 s^-3]"
    ds.fe.attrs["description"] = "Electron velocity distribution (isotropic part)"

    # M
    ds.M.attrs["long_name"] = "M"
    ds.M.attrs["units"] = "[s^-1]"
    ds.M.attrs["description"] = "Rate matrices"

    return ds


def add_coordinate_info(ds: xr.Dataset) -> xr.Dataset:
    """Add information on coordinates (x, v, k) to output dataset

    :param ds: Xarray dataset built from SIKERun
    :return: Modified xarray dataset
    """

    ds.x.attrs["long_name"] = "x"
    ds.x.attrs["units"] = "[m]"
    ds.x.attrs["description"] = "Spatial coordinate"

    ds.k.attrs["long_name"] = "k"
    ds.k.attrs["units"] = "N/A"
    ds.k.attrs["description"] = (
        "Atomic state index (vertical index in the case of the rate matrices)"
    )

    ds.j.attrs["long_name"] = "j"
    ds.j.attrs["units"] = "N/A"
    ds.j.attrs["description"] = "Horizontal atomic state index in the rate matrices"

    ds.v.attrs["long_name"] = "v"
    ds.v.attrs["units"] = "[m/s]"
    ds.v.attrs["description"] = "Velocity coordinate"

    ds.i.attrs["long_name"] = "i"
    ds.i.attrs["units"] = "N/A"
    ds.i.attrs["description"] = "Transition ID"

    return ds