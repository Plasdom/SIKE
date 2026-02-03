import numpy as np
import xarray as xr

from sike import constants as c


def get_Zavg(ds: xr.Dataset) -> xr.DataArray:
    """Get mean charge state from dataset

    :param ds: xarray dataset from SIKERun
    :return: Zavg (coords = x)
    """
    return ((ds.nk * ds.state_Z).sum(dim="k")) / ds.nk.sum(dim="k")


def get_nz(ds: xr.Dataset) -> xr.DataArray:
    """Get the density of each charge state [m^-3] from dataset

    :param ds: xarray dataset from SIKERun
    :return: Z_dens (coords = [x, state_Z])
    """
    return ds.nk.groupby(ds.state_Z).sum(dim="k")


def get_Qz(ds: xr.Dataset) -> xr.DataArray:
    Zs = sorted(set(ds.state_Z.values))
    Qz = np.zeros((len(ds.x), len(Zs)))
    for Z in Zs:
        Z_states = ds.k[ds.state_Z == Z]
        emission_ds = ds.sel(
            i=(ds["transition_type"] == "emission")
            & ds["transition_from_k"].isin(Z_states)
            & ds["transition_to_k"].isin(Z_states)
        )
        Qz[:, Z] = 1e-6 * (
            emission_ds.transition_delta_E
            * c.EL_CHARGE
            * emission_ds.transition_rates
            * emission_ds.nk.sel(k=emission_ds.transition_from_k)
        ).sum(dim="i")
    return xr.DataArray(Qz, coords={"x": ds.x, "state_Z": Zs})


def get_Qz_tot(ds: xr.Dataset) -> xr.DataArray:
    """Get the total radiation from spontaneous emission [MW/m^3] from dataset

    :param ds: xarray dataset from SIKERun
    :return: Qz_tot (coords = [x, state_Z])
    """
    emission_ds = ds.sel(i=(ds["transition_type"] == "emission"))
    return 1e-6 * (
        emission_ds.transition_delta_E
        * c.EL_CHARGE
        * emission_ds.transition_rates
        * emission_ds.nk.sel(k=emission_ds.transition_from_k)
    ).sum(dim="i")


def get_Lz_avg(ds: xr.Dataset, include_radrec: bool = False) -> xr.DataArray:
    """Get the total line emission coefficient (i.e. averaged over all charge states) [MWm^3] from dataset

    :param ds: xarray dataset from SIKERun
    :param include_radrec: whether to include radiative recombination contributions, defaults to False
    :return: Lz_avg (coords = [x, state_Z])
    """
    emission_ds = ds.sel(i=(ds["transition_type"] == "emission"))
    Lz_avg = (
        1e-6
        * (
            emission_ds.transition_delta_E
            * c.EL_CHARGE
            * emission_ds.transition_rates
            * emission_ds.nk.sel(k=emission_ds.transition_from_k)
        ).sum(dim="i")
        / (ds.ne.values * ds.nk.sum(dim="k").values)
    )
    if include_radrec:
        Lz_avg += get_Lz_avg_rr(ds)

    return Lz_avg


def get_Lz_avg_rr(ds: xr.Dataset) -> xr.DataArray:
    """Get the total line emission coefficient (i.e. averaged over all charge states) from radiative recombination [MWm^3] from dataset

    :param ds: xarray dataset from SIKERun
    :return: Lz_avg (coords = [x, state_Z])
    """

    rr_ds = ds.sel(i=(ds["transition_type"] == "radiative_recombination"))
    Qz_avg = 1e-6 * (
        rr_ds.transition_delta_E
        * c.EL_CHARGE
        * rr_ds.transition_rates
        * rr_ds.nk.sel(k=rr_ds.transition_from_k)
    ).sum(dim="i")
    Lz_avg = Qz_avg / (ds.ne.values * ds.nk.sum(dim="k").values)

    return xr.DataArray(Lz_avg, coords={"x": ds.x})


def get_Lz(ds: xr.Dataset, include_radrec: bool = False) -> xr.DataArray:
    """Get the line emission coefficients for each charge state [MWm^3] from dataset

    :param ds: xarray dataset from SIKERun
    :param include_radrec: whether to include radiative recombination contributions, defaults to False
    :return: Qz_tot (coords = [x, state_Z])
    """
    Zs = sorted(set(ds.state_Z.values))
    nz = get_nz(ds)
    Lz = np.zeros((len(ds.x), len(Zs)))
    for Z in Zs:
        Z_states = ds.k[ds.state_Z == Z]
        emission_ds = ds.sel(
            i=(ds["transition_type"] == "emission")
            & ds["transition_from_k"].isin(Z_states)
            & ds["transition_to_k"].isin(Z_states)
        )
        Qz = 1e-6 * (
            emission_ds.transition_delta_E
            * c.EL_CHARGE
            * emission_ds.transition_rates
            * emission_ds.nk.sel(k=emission_ds.transition_from_k)
        ).sum(dim="i")

        if include_radrec:
            Qz += get_Lz_rr(ds)

        Lz[:, Z] = Qz / (ds.ne.values * nz.sel(state_Z=Z).values)

    return xr.DataArray(Lz, coords={"x": ds.x, "state_Z": Zs})


def get_Lz_rr(ds: xr.Dataset) -> xr.DataArray:
    """Get the line emission coefficients for each charge state [MWm^3] from radiative recombination from dataset

    :param include_radrec: whether to include radiative recombination contributions, defaults to False
    :return: Qz_tot (coords = [x, state_Z])
    """
    Zs = sorted(set(ds.state_Z.values))
    nz = get_nz(ds)
    Lz = np.zeros((len(ds.x), len(Zs)))
    for Z in Zs:
        Z_states = ds.k[ds.state_Z == Z]
        rr_ds = ds.sel(
            i=(ds["transition_type"] == "radiative_recombination")
            & ds["transition_from_k"].isin(Z_states)
        )
        Qz = 1e-6 * (
            rr_ds.transition_delta_E
            * c.EL_CHARGE
            * rr_ds.transition_rates
            * rr_ds.nk.sel(k=rr_ds.transition_from_k)
        ).sum(dim="i")

        Lz[:, Z] = Qz / (ds.ne.values * nz.sel(state_Z=Z).values)

    return xr.DataArray(Lz, coords={"x": ds.x, "state_Z": Zs})


def get_Meff(ds: xr.Dataset, P_states: list[int] | None = None) -> xr.DataArray:
    """Find the effective rate matrix for a given list of P states (see Greenland, P. T., "Collisional Radiative Models with Molecules" (2001))

    :param ds: xarray dataset from SIKERun
    :param P_states: k indices of states in the input dataset which form the evolved set of states. All remaining states (Q states) are assumed to not be evolved. Defaults to None, in which case the ground states of each charge state are used.
    :return: Effective rate matrix for transitions between P states
    """

    # Use ground states if no list of P states provided
    if P_states is None:
        P_states = get_ground_states(ds)

    # Q states are the remaining ones
    Q_states = ds.sel(k=np.logical_not(ds.k.isin(P_states))).k.values

    # Get matrix components
    M_P = ds.M.sel(j=P_states, k=P_states).values
    M_Q = ds.M.sel(j=Q_states, k=Q_states).values
    M_PQ = ds.M.sel(j=P_states, k=Q_states).values
    M_QP = ds.M.sel(j=Q_states, k=P_states).values

    # Calculate M_eff
    M_eff = -(M_P - M_PQ @ np.linalg.inv(M_Q) @ M_QP)

    return xr.DataArray(M_eff, coords={"x": ds.x, "j": P_states, "k": P_states})


def get_Keff_iz(ds: xr.Dataset, P_states: list[int] | None = None) -> xr.DataArray:
    """Get the effective ionisation rate coefficients between ground states of each charge state (effective rates for transitions between other states can be specified with the P_states argument)

    :param ds: xarray dataset from SIKERun
    :param P_states: k indices of states in the input dataset which form the evolved set of states. All remaining states (Q states) are assumed to not be evolved. Defaults to None, in which case the ground states of each charge state are used.
    :return: Effective ionisation rate coefficients between P states
    """
    # Use ground states if no list of P states provided
    if P_states is None:
        P_states = get_ground_states(ds)
        Zs = np.sort(ds.state_Z.sel(k=P_states).values)[:-1]
    P_states = get_ground_states(ds)
    Meff = get_Meff(ds, P_states=P_states)

    Keff_iz = np.zeros((len(ds.x), len(Zs)))
    for Z in Zs:
        Keff_iz[:, Z] = -Meff.isel(k=Z, j=Z + 1).values / ds.ne.values

    return xr.DataArray(Keff_iz, coords={"x": ds.x, "state_Z": Zs})


def get_Keff_rec(ds: xr.Dataset, P_states: list[int] | None = None) -> xr.DataArray:
    """Get the effective recombination rate coefficients between ground states of each charge state (effective rates for transitions between other states can be specified with the P_states argument)

    :param ds: xarray dataset from SIKERun
    :param P_states: k indices of states in the input dataset which form the evolved set of states. All remaining states (Q states) are assumed to not be evolved. Defaults to None, in which case the ground states of each charge state are used.
    :return: Effective recombination rate coefficients between P states
    """
    # Use ground states if no list of P states provided
    if P_states is None:
        P_states = get_ground_states(ds)
        Zs = np.sort(ds.state_Z.sel(k=P_states).values)[1:]
    P_states = get_ground_states(ds)
    Meff = get_Meff(ds, P_states=P_states)

    Keff_rec = np.zeros((len(ds.x), len(Zs)))
    for Z in Zs:
        Keff_rec[:, Z - 1] = -Meff.isel(k=Z, j=Z - 1).values / ds.ne.values

    return xr.DataArray(Keff_rec, coords={"x": ds.x, "state_Z": Zs})


def get_K_rr(ds: xr.Dataset) -> xr.DataArray:
    """Get the radiative recombination rate coefficients between each charge state [m^3/s]

    :param ds: xarray dataset from SIKERun
    :return: Effective recombination rate coefficients between P states
    """

    Zs = sorted(set(ds.state_Z.values))

    K_rr = np.zeros((len(ds.x), len(Zs)))
    for Z in Zs:
        Z_states = ds.k[ds.state_Z == Z]
        rr_ds = ds.sel(
            i=(ds["transition_type"] == "radiative_recombination")
            & ds["transition_from_k"].isin(Z_states)
        )
        K_rr[:, Z] = rr_ds.transition_rates.sum(dim="i")

    return xr.DataArray(K_rr, coords={"x": ds.x, "state_Z": Zs})


def get_ground_states(ds: xr.Dataset) -> np.ndarray:
    """Get the list of k indices of ground states in the dataset

    :param ds: xarray dataset from SIKERun
    :return: A numpy array of ground states indices
    """
    return ds.sel(k=ds.state_is_ground).k.values


def get_Lz_br(ds: xr.Dataset) -> xr.DataArray:
    """Get the radiation from Bremsstrahlung

    :param ds: xarray dataset from SIKERun
    :return: A DataArray containing Bremsstrahlung radiation from the impurity species
    """

    Zs = sorted(set(ds.state_Z.values))
    Lz = np.zeros((len(ds.x), len(Zs)))
    for Z in Zs:
        Lz[:, Z] = Z**2 * np.sqrt(ds.Te) / (7.69e18**2)

    return xr.DataArray(Lz, coords={"x": ds.x, "state_Z": Zs})


def get_Lz_avg_br(ds: xr.Dataset) -> xr.DataArray:
    """TODO: Docstring

    :param ds: _description_
    :return: _description_
    """

    Zs = sorted(set(ds.state_Z.values))
    nz = get_nz(ds)
    Lz = np.zeros(len(ds.x))
    for Z in Zs:
        Lz += nz.sel(state_Z=Z) * Z**2 * np.sqrt(ds.Te) / (7.69e18**2)

    Lz = 1e-6 * Lz / nz.sum(dim="state_Z")

    return xr.DataArray(Lz, coords={"x": ds.x})
