import xarray as xr
import numpy as np

from sike.utils import constants as c


def get_Zavg(ds: xr.Dataset) -> xr.DataArray:
    """Get mean charge state from dataset

    :param ds: xarray dataset from SIKERun
    :return: Zavg (coords = x)
    """
    Zavg = ((ds.nk * ds.state_Z).sum(dim="k")) / ds.nk.sum(dim="k")
    return Zavg


def get_nz(ds: xr.Dataset) -> xr.DataArray:
    """Get the density of each charge state [m^-3] from dataset

    :param ds: xarray dataset from SIKERun
    :return: Z_dens (coords = [x, state_Z])
    """
    Z_dens = ds.nk.groupby(ds.state_Z).sum(dim="k")
    return Z_dens


def get_Qz(ds: xr.Dataset) -> xr.DataArray:
    Zs = sorted(list(set(ds.state_Z.values)))
    Qz = np.zeros((len(ds.x), len(Zs)))
    for Z in Zs:
        Z_states = ds.k[ds.state_Z == Z]
        emission_ds = ds.sel(
            i=(ds["transition_type"] == "emission")
            & ds["transition_from_id"].isin(Z_states)
            & ds["transition_to_id"].isin(Z_states)
        )
        Qz[:, Z] = 1e-6 * (
            emission_ds.transition_delta_E
            * c.EL_CHARGE
            * emission_ds.transition_rate
            * emission_ds.nk.sel(k=emission_ds.transition_from_id)
        ).sum(dim="i")
    Qz = xr.DataArray(Qz, coords={"x": ds.x, "state_Z": Zs})
    return Qz


def get_Qz_tot(ds: xr.Dataset) -> xr.DataArray:
    """Get the total radiation from spontaneous emission [MW/m^3] from dataset

    :param ds: xarray dataset from SIKERun
    :return: Qz_tot (coords = [x, state_Z])
    """
    emission_ds = ds.sel(i=(ds["transition_type"] == "emission"))
    Qz_tot = 1e-6 * (
        emission_ds.transition_delta_E
        * c.EL_CHARGE
        * emission_ds.transition_rate
        * emission_ds.nk.sel(k=emission_ds.transition_from_id)
    ).sum(dim="i")
    return Qz_tot


def get_Lz_avg(ds: xr.Dataset) -> xr.DataArray:
    """Get the total line emission coefficient (i.e. averaged over all charge states) [MWm^3] from dataset

    :param ds: xarray dataset from SIKERun
    :return: Qz_tot (coords = [x, state_Z])
    """
    emission_ds = ds.sel(i=(ds["transition_type"] == "emission"))
    Lz_avg = (
        1e-6
        * (
            emission_ds.transition_delta_E
            * c.EL_CHARGE
            * emission_ds.transition_rate
            * emission_ds.nk.sel(k=emission_ds.transition_from_id)
        ).sum(dim="i")
        / (ds.ne.values * ds.nk.sum(dim="k").values)
    )
    return Lz_avg


def get_Lz(ds: xr.Dataset) -> xr.DataArray:
    """Get the line emission coefficients for each charge state [MWm^3] from dataset

    :param ds: xarray dataset from SIKERun
    :return: Qz_tot (coords = [x, state_Z])
    """
    Zs = sorted(list(set(ds.state_Z.values)))
    nz = get_nz(ds)
    Lz = np.zeros((len(ds.x), len(Zs)))
    for Z in Zs:
        Z_states = ds.k[ds.state_Z == Z]
        emission_ds = ds.sel(
            i=(ds["transition_type"] == "emission")
            & ds["transition_from_id"].isin(Z_states)
            & ds["transition_to_id"].isin(Z_states)
        )
        Qz = 1e-6 * (
            emission_ds.transition_delta_E
            * c.EL_CHARGE
            * emission_ds.transition_rate
            * emission_ds.nk.sel(k=emission_ds.transition_from_id)
        ).sum(dim="i")
        Lz[:, Z] = Qz / (ds.ne.values * nz.sel(state_Z=Z).values)
    Lz = xr.DataArray(Lz, coords={"x": ds.x, "state_Z": Zs})
    return Lz
