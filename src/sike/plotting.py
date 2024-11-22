import xarray as xr
import matplotlib.pyplot as plt

import sike.post_processing as spp

# TODO: Provide routines for plotting bremsstrahlung, excitation and recombination radiation together or separately


def plot_Zavg(
    ds: xr.Dataset,
    xaxis: str = "Te",
    logx: bool = False,
    ax: plt.Axes | None = None,
    **mpl_kwargs,
) -> plt.Axes:
    """Plot mean charge state

    :param ds: xarray dataset from SIKERun
    :param xaxis: Variable to use for x-axis ["Te", "ne" or "x"], defaults to "Te"
    :param logx: Whether x-axis scale should be logarithmic, defaults to False
    :param ax: Existing matplotlib axes, defaults to None
    :return: Matplotlib axes
    """

    Zavg = spp.get_Zavg(ds)

    x, xlabel = get_xaxis(ds, xaxis)

    if ax is None:
        _, ax = plt.subplots(1)
    ax.plot(x, Zavg, **mpl_kwargs)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Average ionization")
    ax.set_title("Average ionization: " + ds.metadata["element"])
    ax.grid()
    if logx:
        ax.set_xscale("log")

    return ax


def plot_nz(
    ds: xr.Dataset,
    xaxis: str = "Te",
    logx: bool = False,
    logy: bool = False,
    normalise: bool = False,
    ax: plt.Axes | None = None,
    **mpl_kwargs,
):
    """Plot the density profiles of each charge state

    :param ds: xarray dataset from SIKERun
    :param xaxis: Variable to use for x-axis ["Te", "ne" or "x"], defaults to "Te"
    :param logx: Whether x-axis scale should be logarithmic, defaults to False
    :param logy: Whether y-axis scale should be logarithmic, defaults to False
    :param normalise: Whether chanrge state densities should be normalised so that sum(nz) = 1, defaults to False
    :param ax: Existing matplotlib axes, defaults to None
    :return: Matplotlib axes
    """
    nz = spp.get_nz(ds)
    if normalise:
        nz = nz / nz.sum(dim="state_Z")

    x, xlabel = get_xaxis(ds, xaxis)

    if ax is None:
        _, ax = plt.subplots(1)

    Zs = range(ds.state_Z.data.min(), ds.state_Z.data.max() + 1)
    for Z in Zs:
        (l,) = ax.plot([], [])
        label = ds.metadata["element"] + "$^{" + str(Z) + "{+}}$"
        ax.plot(
            x,
            nz.sel(state_Z=Z),
            color=l.get_color(),
            label=label,
            **mpl_kwargs,
        )
    ax.legend()
    ax.set_xlabel(xlabel)
    if normalise:
        ax.set_ylabel("$n_Z / n_Z^{tot}$")
    else:
        ax.set_ylabel("Density [m$^{-3}$]")
    ax.set_title("Density profiles per ionization stage: " + ds.metadata["element"])
    ax.grid()
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")

    return ax


def plot_Qz(
    ds: xr.Dataset,
    xaxis: str = "Te",
    logx: bool = False,
    logy: bool = False,
    normalise: bool = False,
    ax: plt.Axes | None = None,
    **mpl_kwargs,
):
    """Plot the radiation from spontaneous emission processes by charge state

    :param ds: xarray dataset from SIKERun
    :param xaxis: Variable to use for x-axis ["Te", "ne" or "x"], defaults to "Te"
    :param logx: Whether x-axis scale should be logarithmic, defaults to False
    :param logy: Whether y-axis scale should be logarithmic, defaults to False
    :param normalise: Whether chanrge state densities should be normalised so that sum(nz) = 1, defaults to False
    :param ax: Existing matplotlib axes, defaults to None
    :return: Matplotlib axes
    """
    Qz = spp.get_Qz(ds)
    if normalise:
        Qz = Qz / Qz.sum(dim="state_Z")

    x, xlabel = get_xaxis(ds, xaxis)

    if ax is None:
        _, ax = plt.subplots(1)

    Zs = range(ds.state_Z.data.min(), ds.state_Z.data.max() + 1)
    for Z in Zs:
        (l,) = ax.plot([], [])
        label = ds.metadata["element"] + "$^{" + str(Z) + "{+}}$"
        ax.plot(
            x,
            Qz.sel(state_Z=Z),
            color=l.get_color(),
            label=label,
            **mpl_kwargs,
        )
    ax.legend()
    ax.set_xlabel(xlabel)
    if normalise:
        ax.set_ylabel("$Q_Z / Q_Z^{tot}$")
    else:
        ax.set_ylabel("$Q_z$ [MWm$^{-3}$]")
    ax.set_title("$Q_z$: " + ds.metadata["element"])
    ax.grid()
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")

    return ax


def plot_Lz(
    ds: xr.Dataset,
    xaxis: str = "Te",
    logx: bool = False,
    logy: bool = True,
    normalise: bool = False,
    ax: plt.Axes | None = None,
    **mpl_kwargs,
):
    """Plot the line emission coefficients for each charge state

    :param ds: xarray dataset from SIKERun
    :param xaxis: Variable to use for x-axis ["Te", "ne" or "x"], defaults to "Te"
    :param logx: Whether x-axis scale should be logarithmic, defaults to False
    :param logy: Whether y-axis scale should be logarithmic, defaults to True
    :param normalise: Whether Lz should be normalised so that sum(Lz) = 1, defaults to False
    :param ax: Existing matplotlib axes, defaults to None
    :return: Matplotlib axes
    """
    Lz = spp.get_Lz(ds)
    if normalise:
        Lz = Lz / Lz.sum(dim="state_Z")

    x, xlabel = get_xaxis(ds, xaxis)

    if ax is None:
        _, ax = plt.subplots(1)

    Zs = range(ds.state_Z.data.min(), ds.state_Z.data.max())
    for Z in Zs:
        (l,) = ax.plot([], [])
        label = ds.metadata["element"] + "$^{" + str(Z) + "{+}}$"
        ax.plot(
            x,
            Lz.sel(state_Z=Z),
            color=l.get_color(),
            label=label,
            **mpl_kwargs,
        )
    ax.legend()
    ax.set_xlabel(xlabel)
    if normalise:
        ax.set_ylabel("$L_Z / L_Z^{tot}$")
    else:
        ax.set_ylabel("$L_z$ [MWm$^{3}$]")
    ax.set_title("$L_z$: " + ds.metadata["element"])
    ax.grid()
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")

    return ax


def plot_Qz_tot(
    ds: xr.Dataset,
    xaxis: str = "Te",
    logx: bool = False,
    ax: plt.Axes | None = None,
    **mpl_kwargs,
) -> plt.Axes:
    """Plot the total radiation from spontaneous emission across all charge states

    :param ds: xarray dataset from SIKERun
    :param xaxis: Variable to use for x-axis ["Te", "ne" or "x"], defaults to "Te"
    :param logx: Whether x-axis scale should be logarithmic, defaults to False
    :param ax: Existing matplotlib axes, defaults to None
    :return: Matplotlib axes
    """

    Qz_tot = spp.get_Qz_tot(ds)
    x, xlabel = get_xaxis(ds, xaxis)

    if ax is None:
        _, ax = plt.subplots(1)
    ax.plot(x, Qz_tot, **mpl_kwargs)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("$Q_{z,tot}$ [MWm$^{-3}$]")
    ax.set_title("$Q_{z,tot}$: " + ds.metadata["element"])
    ax.grid()
    if logx:
        ax.set_xscale("log")

    return ax


def plot_Lz_avg(
    ds: xr.Dataset,
    xaxis: str = "Te",
    logx: bool = False,
    logy: bool = True,
    ax: plt.Axes | None = None,
    **mpl_kwargs,
) -> plt.Axes:
    """Plot the total radiation from excitation across all charge states

    :param ds: xarray dataset from SIKERun
    :param xaxis: Variable to use for x-axis ["Te", "ne" or "x"], defaults to "Te"
    :param logx: Whether x-axis scale should be logarithmic, defaults to False
    :param ax: Existing matplotlib axes, defaults to None
    :return: Matplotlib axes
    """

    Lz_tot = spp.get_Lz_avg(ds)
    x, xlabel = get_xaxis(ds, xaxis)

    if ax is None:
        _, ax = plt.subplots(1)
    ax.plot(x, Lz_tot, **mpl_kwargs)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$\bar{L}_{z}$ [MWm$^{3}$]")
    ax.set_title(r"$\bar{L}_{z}$: " + ds.metadata["element"])
    ax.grid()
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")

    return ax


def plot_Keff_iz(
    ds: xr.Dataset,
    P_states: None | list[int] = None,
    xaxis: str = "Te",
    logx: bool = False,
    logy: bool = True,
    ax: plt.Axes | None = None,
    **mpl_kwargs,
):
    """Plot the effective ionisation coefficients between ground states of each charge state, or some other set of P states (see Greenland, P. T., "Collisional Radiative Models with Molecules" (2001))

    :param ds: xarray dataset from SIKERun
    :param P_states: k indices of states in the input dataset which form the evolved set of states. All remaining states (Q states) are assumed to not be evolved. Defaults to None, in which case the ground states of each charge state are used.
    :param xaxis: Variable to use for x-axis ["Te", "ne" or "x"], defaults to "Te"
    :param logx: Whether x-axis scale should be logarithmic, defaults to False
    :param logy: Whether y-axis scale should be logarithmic, defaults to True
    :param ax: Existing matplotlib axes, defaults to None
    :return: Matplotlib axes
    """
    Keff_iz = spp.get_Keff_iz(ds, P_states)

    x, xlabel = get_xaxis(ds, xaxis)

    if ax is None:
        _, ax = plt.subplots(1)

    Zs = Keff_iz.state_Z.values
    for Z in Zs:
        (l,) = ax.plot([], [])
        label = ds.metadata["element"] + "$^{" + str(Z) + "{+}}$"
        ax.plot(
            x,
            Keff_iz.sel(state_Z=Z),
            color=l.get_color(),
            label=label,
            **mpl_kwargs,
        )
    ax.legend()
    ax.set_xlabel(xlabel)
    ax.set_ylabel("$K_{eff}^{ion,z}$ [m$^{3}$s$^{-1}$]")
    ax.set_title("$K_{eff}^{ion,z}$: " + ds.metadata["element"])
    ax.grid()
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")

    return ax


def plot_Keff_rec(
    ds: xr.Dataset,
    P_states: None | list[int] = None,
    xaxis: str = "Te",
    logx: bool = False,
    logy: bool = True,
    ax: plt.Axes | None = None,
    **mpl_kwargs,
):
    """Plot the effective recombination coefficients between ground states of each charge state, or some other set of P states (see Greenland, P. T., "Collisional Radiative Models with Molecules" (2001))

    :param ds: xarray dataset from SIKERun
    :param P_states: k indices of states in the input dataset which form the evolved set of states. All remaining states (Q states) are assumed to not be evolved. Defaults to None, in which case the ground states of each charge state are used.
    :param xaxis: Variable to use for x-axis ["Te", "ne" or "x"], defaults to "Te"
    :param logx: Whether x-axis scale should be logarithmic, defaults to False
    :param logy: Whether y-axis scale should be logarithmic, defaults to True
    :param ax: Existing matplotlib axes, defaults to None
    :return: Matplotlib axes
    """
    Keff_rec = spp.get_Keff_rec(ds, P_states)

    x, xlabel = get_xaxis(ds, xaxis)

    if ax is None:
        _, ax = plt.subplots(1)

    Zs = Keff_rec.state_Z.values
    for Z in Zs:
        (l,) = ax.plot([], [])
        label = ds.metadata["element"] + "$^{" + str(Z) + "{+}}$"
        ax.plot(
            x,
            Keff_rec.sel(state_Z=Z),
            color=l.get_color(),
            label=label,
            **mpl_kwargs,
        )
    ax.legend()
    ax.set_xlabel(xlabel)
    ax.set_ylabel("$K_{eff}^{rec,z}$ [m$^{3}$s$^{-1}$]")
    ax.set_title("$K_{eff}^{rec,z}$: " + ds.metadata["element"])
    ax.grid()
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")

    return ax


def get_xaxis(ds, xaxis):
    """Return an array to use on x-axis of a plot

    Args:
        r (SIKERun): SIKERun object
        xaxis (str): string describing the x-axis option

    Returns:
        np.ndarray: x array
        str: x-axis plot label
    """
    if xaxis == "Te":
        x = ds.Te
        xlabel = "$T_e$ [eV]"
    elif xaxis == "ne":
        x = ds.ne
        xlabel = "$n_e$ [m$^{-3}$]"
    elif xaxis == "x":
        x = ds.xgrid
        xlabel = "x [m]"
    return x, xlabel
