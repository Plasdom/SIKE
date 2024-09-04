import xarray as xr


def get_Zavg(ds: xr.Dataset) -> xr.DataArray:
    """Get mean charge state from dataset

    :param ds: xarray dataset from SIKERun
    :return: Zavg (coords = x)
    """
    Zavg = ((ds.nk * ds.state_Z).sum(dim="k")) / ds.nk.sum(dim="k")
    return Zavg


def get_Z_dens(ds: xr.Dataset) -> xr.DataArray:
    """Get the density of each charge state from dataset

    :param ds: xarray dataset from SIKERun
    :return: Z_dens (coords = [x, state_Z])
    """
    Z_dens = ds.nk.groupby(ds.state_Z).sum(dim="k")
    return Z_dens
