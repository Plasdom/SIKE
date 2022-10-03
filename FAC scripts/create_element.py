"""
https://www-amdis.iaea.org/FAC/
"""
import sys
from pfac.fac import *
import os

# # enable openmp with 4 cores
# InitializeMPI(16)

# atomic number, number of excitation levels
z = 3

a = ATOMICSYMBOL[z]  # atomic symbol (Ne)
Print('Creating ' + a + '...')
p = a + '_'
pb = 'bin' + os.sep + a + '_'
SetAtom(a)

nele_rates = int(sys.argv[1])


# Use configuration-averaged mode
SetUTA(1)

# set CE grid
E_grid = [0.00356051833255308, 0.012677290796446573, 0.04513772628785446, 0.10524831771785098, 0.5722240820415418, 1.3342635295459964, 3.1111224119143204, 7.2542510887703235, 16.91484676314779, 39.4406035191801, 60.22562199251383, 91.9642505627806, 140.42899187699828, 214.43443119375743, 327.44040006828527, 500.0]

# Define the configuration groups
Print('Defining groups...')
groups = []
for nele in range(z+1):

    groups.append([])

    if nele == 0:
        # Bare nucleus
        group_name = 'bare'
        Config('', group=group_name)
        groups[nele].append(group_name)

    elif nele == 1:
        # H-like configurations
        nmax = 10
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

        nmax = 10 
        for n in range(2, nmax+1):
            group_name = '1s ' + str(n) + '*1'
            Config(group_name, group_name)
            groups[nele].append(group_name)

        group_name = '2p ' + '3*1'
        Config(group_name, group_name)
        groups[nele].append(group_name)
        
    elif nele == 3:
        # Li-like configurations
        nmax = 14
        for n in range(2, nmax+1):
            group_name = '1s2 ' + str(n) + '*1'
            Config(group_name, group_name)
            groups[nele].append(group_name)

        group_name = '1s ' + '2*2'
        Config(group_name, group_name)
        groups[nele].append(group_name)
        
        group_name = '1s ' + '3*2'
        Config(group_name, group_name)
        groups[nele].append(group_name)

        nmax = 5
        for n in range(3, nmax+1):
            group_name = '1s ' + '2*1 ' + str(n) + '*1'
            Config(group_name, group_name)
            groups[nele].append(group_name)

print(groups)

# radial potential
groups_flattened = [g for stage_groups in groups for g in stage_groups]
ConfigEnergy(0)
OptimizeRadial(groups_flattened[0])
ConfigEnergy(1)

# atomic structure
Structure(pb+'b.en', groups_flattened)
MemENTable(pb+'b.en')
PrintTable(pb+'b.en', p+'en.txt')

# transition rates
Print('Transition rates...')
stage_groups = groups[nele_rates]
for n in range(len(stage_groups)):
    for m in range(n, len(stage_groups)):
        Print(stage_groups[n] + ' -> ' + stage_groups[m])
        TRTable(pb+'b.tr', stage_groups[n], stage_groups[m])
PrintTable(pb+'b.tr', p+'tr.txt')

# autoionization rates
Print('Autoionization rates...')
to_stage_groups = groups[nele_rates-1]
from_stage_groups = groups[nele_rates]

SetPEGrid(E_grid)
SetUsrPEGrid(E_grid)
Print(str(nele_rates) + 'el -> ' + str(nele_rates-1) + 'el')
AITable(pb+'b.ai', from_stage_groups, to_stage_groups)
PrintTable(pb+'b.ai', p+'ai.txt')

# ionization
Print('Ionization cross-sections...')
to_stage_groups = groups[nele_rates-1]
from_stage_groups = groups[nele_rates]

SetCIEGrid(E_grid)
SetUsrCIEGrid(E_grid)
Print(str(nele_rates) + 'el -> ' + str(nele_rates-1) + 'el')
CITable(pb+'b.ci', from_stage_groups, to_stage_groups)
PrintTable(pb+'b.ci', p+'ci.txt')

# radiative recombination
Print('Radiative recombination...')
to_stage_groups = groups[nele_rates]
from_stage_groups = groups[nele_rates-1]

SetPEGrid(E_grid)
SetUsrPEGrid(E_grid)
Print(str(nele_rates-1) + 'el -> ' + str(nele_rates) + 'el')
RRTable(pb+'b.rr', to_stage_groups, from_stage_groups)
PrintTable(pb+'b.rr', p+'rr.txt')

# excitation
Print('Collisional excitation cross-sections...')
stage_groups = groups[nele_rates]
for n in range(len(stage_groups)):
    for m in range(n, len(stage_groups)):
        SetCEGrid(E_grid)
        SetUsrCEGrid(E_grid)
        Print(stage_groups[n] + ' -> ' + stage_groups[m])
        CETable(pb+'b.ce', stage_groups[n], stage_groups[m])
PrintTable(pb+'b.ce', p+'ce.txt')

# FinalizeMPI()
