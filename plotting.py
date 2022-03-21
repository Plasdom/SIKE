
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def plot_Zeff(skrun, Zeff, Zeff_max, Zeff_saha):
    fig, ax = plt.subplots(1)
    ax.plot(skrun.xgrid, Zeff_max, label='Maxwellian $f_0$', color='black')
    ax.plot(skrun.xgrid, Zeff_saha, '--',
            label='Saha equilibrium', color='blue')
    ax.plot(skrun.xgrid, Zeff, '--', label='SOL-KiT $f_0$', color='red')
    ax.grid()
    ax.legend()
    ax.set_xlabel('x [m]')
    ax.set_ylabel('$Z_{eff}=\Sigma_i Z^2_i n_{W}^i / n_e$')
    ax.set_xlim([10.4, 11.8])


def plot_Zdist(skrun, imp_dens, imp_dens_max, imp_dens_saha, cell=-1):
    fig, ax = plt.subplots(1)
    min_z = 0
    max_z = len(imp_dens[0, :])
    zs = [i for i in range(min_z, max_z)]
    ax.plot(zs, imp_dens_max[cell, min_z:max_z],
            label='Maxwellian $f_0$', color='black')
    ax.plot(zs, imp_dens[cell, min_z:max_z],
            '--', label='SOL-KiT $f_0$', color='red')
    ax.plot(zs, imp_dens_saha[cell, min_z:max_z],
            '--', label='Saha equilibrium', color='blue')
    ax.grid()
    ax.legend()
    ax.set_yscale('log')
    ax.set_xlabel('$Z$')
    ax.set_ylabel('$n_Z / n_0$')
    ax.set_title('Impurity state densities')


def plot_densities(skrun, imp_dens=None, imp_dens_max=None, imp_dens_saha=None):

    fig, ax = plt.subplots(1)
    cmap = plt.cm.get_cmap('plasma')
    min_z = 0
    try:
        max_z = len(imp_dens[0, :])
    except:
        try:
            max_z = len(imp_dens_max[0, :])
        except:
            max_z = len(imp_dens_saha[0, :])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(
        vmin=min_z, vmax=max_z))
    legend_lines = []
    legend_labels = []
    for z in range(min_z, max_z):
        if imp_dens is not None:
            ax.plot(skrun.xgrid, imp_dens[:, z], '--', color=sm.to_rgba(z))
        if imp_dens_max is not None:
            ax.plot(skrun.xgrid, imp_dens_max[:, z], color=sm.to_rgba(z))
        if imp_dens_saha is not None:
            ax.plot(skrun.xgrid,
                    imp_dens_saha[:, z], '-.', color=sm.to_rgba(z))
        legend_labels.append('$Z=$' + str(z))
        legend_lines.append(Line2D([0], [0], color=sm.to_rgba(z)))
    if imp_dens_max is not None:
        legend_lines.append(Line2D([0], [0], linestyle='-', color='black'))
    if imp_dens is not None:
        legend_lines.append(Line2D([0], [0], linestyle='--', color='black'))
    if imp_dens_saha is not None:
        legend_lines.append(Line2D([0], [0], linestyle='-.', color='black'))
    if imp_dens_max is not None:
        legend_labels.append('Maxwellian $f_0$')
    if imp_dens is not None:
        legend_labels.append('SOL-KiT $f_0$')
    if imp_dens_saha is not None:
        legend_labels.append('Saha equilibrium')
    ax.legend(legend_lines, legend_labels)
    ax.grid()
    # ax.set_yscale('log')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('$n_Z / n_0$')
    ax.set_title('$n_Z$ profiles')
