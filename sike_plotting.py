import matplotlib.pyplot as plt
import numpy as np
from post_processing import *

"""A number of useful plotting functions for SIKERun objects.
"""

def plot_Zavg(r, el, kinetic=True, maxwellian=True, xaxis='Te', logx=False):
    """Plot the average ionization profile

    Args:
        r (SIKERun): SIKERun object
        el (str): the impurity species to plot
        kinetic (bool, optional): whether to plot kinetic Z_avg profile. Defaults to True.
        maxwellian (bool, optional): whether to plot Maxwellian Z_avg profile. Defaults to True.
        xaxis (str,optional): choice of x-axis: 'Te', 'ne', 'x'. Defaults to 'Te'
    """
    Zavg_Max = get_Zavg(r.impurities[el].dens_Max, r.impurities[el].states, r.num_x)
    Zavg_kin = get_Zavg(r.impurities[el].dens, r.impurities[el].states, r.num_x)
    
    x, xlabel = get_xaxis(r,xaxis)
    
    fig,ax = plt.subplots(1)
    if kinetic:
        ax.plot(x,Zavg_kin, '--', color='black',label='Kinetic')
    if maxwellian:
        ax.plot(x,Zavg_Max, color='black', label='Maxwellian')
    ax.legend()
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Average ionization')
    ax.set_title('Average ionization: ' +el )
    ax.grid()
    if logx:
        ax.set_xscale('log')
    
def plot_Z_dens(r, el, kinetic = True, maxwellian = True, xaxis='Te', logx=False):
    """Plot the density profiles of each ionization stage

    Args:
        r (SIKERun): SIKERun object
        el (str):the impurity species to plot
        kinetic (bool, optional): whether to plot kinetic Z_avg profile. Defaults to True.
        maxwellian (bool, optional): whether to plot Maxwellian Z_avg profile. Defaults to True.
        xaxis (str,optional): choice of x-axis: 'Te', 'ne', 'x'. Defaults to 'Te'
    """
    Z_dens_Max = get_Z_dens(r.impurities[el].dens_Max, r.impurities[el].states)
    Z_dens_kin = get_Z_dens(r.impurities[el].dens, r.impurities[el].states)
    num_Z = r.impurities[el].num_Z
    x, xlabel = get_xaxis(r,xaxis)

    fig,ax = plt.subplots(1)
    for Z in range(num_Z):
        l, = ax.plot([],[])
        if kinetic:
            ax.plot(x, Z_dens_Max[:,Z]*r.n_norm, color=l.get_color(), label=el + '$^{' + str(Z) + '+}$')
        if maxwellian:
            ax.plot(x, Z_dens_kin[:,Z]*r.n_norm, '--', color=l.get_color())
    ax.plot([],[],color='black', label='Maxwellian')
    ax.plot([],[],'--',color='black', label='Kinetic')
    ax.legend()
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Density [m$^{-3}$]')
    ax.set_title('Density profiles per ionization stage: ' + el)
    ax.grid()
    if logx:
        ax.set_xscale('log')

def plot_PLTs(r, el, effective = True, kinetic = True, maxwellian = True, xaxis='Te', logx=False):
    """Plot profiles of line emission (i.e. excitation radiation) coefficients per ion

    Args:
        r (SIKERun): SIKERun object
        el (str):the impurity species to plot
        effective (bool, optional): whether to plot whole-element effective line emission coefficients. Defaults to True.
        kinetic (bool, optional): whether to plot kinetic line emission coefficients profile. Defaults to True.
        maxwellian (bool, optional): whether to plot Maxwellian line emission coefficients profile. Defaults to True.
        xaxis (str,optional): choice of x-axis: 'Te', 'ne', 'x'. Defaults to 'Te'
    """
    PLT_Max, PLT_Max_eff = get_cooling_curves(r, el, kinetic=False)
    PLT_kin, PLT_kin_eff = get_cooling_curves(r, el, kinetic=True)
    
    num_Z = r.impurities[el].num_Z
    x, xlabel = get_xaxis(r,xaxis)
    
    fig,ax = plt.subplots(1)
    for Z in range(num_Z-1):
        l, = ax.plot([],[])
        if kinetic:
            ax.plot(x,PLT_kin[:,Z],'--',color=l.get_color())
        if maxwellian:
            ax.plot(x,PLT_Max[:,Z],color=l.get_color(),label=el + '$^{' + str(Z) + '+}$')
    if effective:
        if kinetic:
            ax.plot(x,PLT_kin_eff,'--',color='black', label='Maxwellian')
        if maxwellian:
            ax.plot(x,PLT_Max_eff,'-',color='black', label='Kinetic')
    ax.set_yscale('log')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Excitation radiation per ion [Wm$^3$]')
    ax.legend()
    ax.grid()
    ax.set_title('Cooling curves: ' + el)
    if logx:
        ax.set_xscale('log')
    
def plot_rad_profile(r, el, kinetic = True, maxwellian = True, xaxis='x', logx=False):
    """Plot the radiative emission profile (currently only contributions from spontaneous emission are included)

    Args:
        r (SIKERun): SIKERun object
        el (str):the impurity species to plot
        kinetic (bool, optional): whether to plot kinetic radiation profile. Defaults to True.
        maxwellian (bool, optional): whether to plot Maxwellian radiation  profile. Defaults to True.
        xaxis (str,optional): choice of x-axis: 'Te', 'ne', 'x'. Defaults to 'x'
    """
    
    ne = r.ne * r.n_norm
    
    PLT_Max, PLT_Max_eff = get_cooling_curves(r, el, kinetic=False)
    Q_rad_Max = 1e-6 * PLT_Max_eff * ne * np.sum(r.impurities[el].dens_Max,1) * r.n_norm
    PLT_kin, PLT_kin_eff = get_cooling_curves(r, el, kinetic=True)
    Q_rad_kin = 1e-6 * PLT_kin_eff * ne * np.sum(r.impurities[el].dens,1) * r.n_norm
    
    x, xlabel = get_xaxis(r,xaxis)
    
    fig,ax = plt.subplots(1)
    if kinetic:
        ax.plot(x,Q_rad_kin,'--',color='black', label='Kinetic')
    if maxwellian:
        ax.plot(x,Q_rad_Max,color='black',label='Maxwellian')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Excitation radiation [Wm$^{-3}$]')
    ax.legend()
    ax.grid()
    ax.set_title('Total line radiation profiles: ' + el)
    if logx:
        ax.set_xscale('log')
    
def get_xaxis(r,xaxis):
    """Return an array to use on x-axis of a plot

    Args:
        r (SIKERun): SIKERun object
        xaxis (str): string describing the x-axis option

    Returns:
        np.ndarray: x array
        str: x-axis plot label
    """
    if xaxis == 'Te':
        x = r.Te * r.T_norm
        xlabel = '$T_e$ [eV]'
    elif xaxis == 'ne':
        x = r.ne * r.n_norm 
        xlabel = '$n_e$ [m$^{-3}$]'
    elif xaxis == 'x':
        x = r.xgrid * r.x_norm
        xlabel = 'x [m]'
    return x, xlabel