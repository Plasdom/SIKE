"""
https://www-amdis.iaea.org/FAC/
"""
import sys
from pfac.fac import *
import numpy as np
import os

# enable openmp with 4 cores
InitializeMPI(16)

# atomic number, number of excitation levels
if len(sys.argv) == 2:
    z = int(sys.argv[1])
else:
    Print('USAGE: python (or python3) ' +
          os.path.basename(__file__) + ' [z] [nmax]')
    exit()

a = ATOMICSYMBOL[z]  # atomic symbol (Ne)
Print('Creating ' + a + '...')
p = a + os.sep + a + '_'
pb = a + os.sep + 'bin' + os.sep + a + '_'
SetAtom(a)

# Use configuration-averaged mode (ie. nl-resolved instead of nlj-resolved)
# TODO: Does SetUTA do what I thought it did?
# SetUTA(1)

# set CE grid
# Note: The lowest value here should be smaller than or equal to the lowest value of the E grid it will be mapped onto
E_grid = list(np.geomspace(0.001, 500, 32))

# Define the configuration groups
Print('Defining groups...')
groups = []
# TODO: Find a way, ideally programatically, to rationalise the number of states you create with this method.
for nele in range(z+1):

    groups.append([])

    if nele == 0:
        # Bare nucleus
        group_name = 'bare'
        Config('', group=group_name)
        groups[nele].append(group_name)

    elif nele == 1:
        # H-like configurations
        nmax = 10  # TODO: Should eventually be 10+
        for n in range(1, nmax+1):
            group_name = str(n) + '*1'
            Config(group_name, str(n) + '*1')
            groups[nele].append(group_name)

    elif nele == 2:
        # He-like configurations
        nmax = 2
        for n in range(1, nmax+1):
            group_name = str(n) + '*2'
            Config(group_name, group_name)
            groups[nele].append(group_name)

        nmax = 2  # TODO: Should eventually be 10+
        for n in range(2, nmax+1):
            group_name = '1s ' + str(n) + '*1'
            Config(group_name, group_name)
            groups[nele].append(group_name)

        group_name = '2p ' + '3*1'
        Config(group_name, group_name)
        groups[nele].append(group_name)

    elif nele == 3:
        # Li-like configurations
        nmax = 5  # TODO: Should eventually be 10+
        for n in range(2, nmax+1):
            group_name = '1s2 ' + str(n) + '*1'
            Config(group_name, group_name)
            groups[nele].append(group_name)

        group_name = '1s ' + '2*2'
        Config(group_name, group_name)
        groups[nele].append(group_name)

        nmax = 5  # TODO: Should eventually be 10+
        for n in range(3, nmax+1):
            group_name = '1s ' + '2*1 ' + str(n) + '*1'
            Config(group_name, group_name)
            groups[nele].append(group_name)

print(groups)
groups_flattened = [g for stage_groups in groups for g in stage_groups]

# radial potential
ConfigEnergy(0)
OptimizeRadial(groups_flattened[0])
# OptimizeRadial(groups_flattened)
ConfigEnergy(1)

# atomic structure
Structure(pb+'b.en', groups_flattened)
MemENTable(pb+'b.en')
PrintTable(pb+'b.en', p+'en.txt')

# transition rates
Print('Transition rates...')
for nele in range(1, z+1):
    stage_groups = groups[nele]
    for n in range(len(stage_groups)):
        for m in range(n, len(stage_groups)):
            Print(stage_groups[n] + ' -> ' + stage_groups[m])
            TRTable(pb+'b.tr', stage_groups[n], stage_groups[m])
PrintTable(pb+'b.tr', p+'tr.txt')

# autoionization rates
Print('Autoionization rates...')
for nele in range(1, z+1):
    to_stage_groups = groups[nele-1]
    from_stage_groups = groups[nele]

    SetPEGrid(E_grid)
    SetUsrPEGrid(E_grid)
    Print(str(nele) + 'el -> ' + str(nele-1) + 'el')
    AITable(pb+'b.ai', from_stage_groups, to_stage_groups)
PrintTable(pb+'b.ai', p+'ai.txt')

# ionization
Print('Ionization cross-sections...')
for nele in range(1, z+1):
    to_stage_groups = groups[nele-1]
    from_stage_groups = groups[nele]

    SetCIEGrid(E_grid)
    SetUsrCIEGrid(E_grid)
    Print(str(nele) + 'el -> ' + str(nele-1) + 'el')
    CITable(pb+'b.ci', from_stage_groups, to_stage_groups)
PrintTable(pb+'b.ci', p+'ci.txt')

# radiative recombination
Print('Radiative recombination...')
for nele in range(z):
    to_stage_groups = groups[nele+1]
    from_stage_groups = groups[nele]

    SetPEGrid(E_grid)
    SetUsrPEGrid(E_grid)
    Print(str(nele) + 'el -> ' + str(nele+1) + 'el')
    RRTable(pb+'b.rr', to_stage_groups, from_stage_groups)
PrintTable(pb+'b.rr', p+'rr.txt')

# excitation
Print('Collisional excitation cross-sections...')
for nele in range(1, z+1):
    stage_groups = groups[nele]
    for n in range(len(stage_groups)):
        for m in range(n, len(stage_groups)):
            SetCEGrid(E_grid)
            SetUsrCEGrid(E_grid)
            Print(stage_groups[n] + ' -> ' + stage_groups[m])
            CETable(pb+'b.ce', stage_groups[n], stage_groups[m])
PrintTable(pb+'b.ce', p+'ce.txt')

# TODO: Do double ionization cross-sections?
# TODO: Do Lithium, Beryllium, etc...

FinalizeMPI()
