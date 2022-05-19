import input
from scipy import interpolate
import numpy as np
import re
import matplotlib.pyplot as plt


class State:
    def __init__(self, iz_stage, lower_shells, statename, loc, statw=1, energy=0, I_0=0, metastable=0):
        self.iz = iz_stage
        if lower_shells != '0':
            self.lower_shells = lower_shells
        else:
            self.lower_shells = None
        self.statename = statename
        self.loc = loc
        self.statw = statw
        self.energy = energy
        self.I_0 = I_0
        self.metastable = bool(metastable)
        self.get_iz_energy()
        self.get_shells()
        self.get_shell_occupation()

    def equals(self, other):
        if self.iz == other.iz and self.statename == other.statename:
            return True
        else:
            return False

    def get_iz_energy(self):
        self.iz_energy = self.I_0 - self.energy

    def get_shell_occupation(self):

        shell_occupation = np.zeros(len(self.shells))
        for i, shell in enumerate(self.shells):
            rm = re.search('[spdfghijklmno]', shell)
            if rm is not None:
                shell_occupation[i] = int(shell[rm.start()+1:])

        self.shell_occupation = shell_occupation

    def get_shells(self):
        if self.lower_shells is not None:
            full_structure = self.lower_shells + ',' + self.statename
        else:
            full_structure = self.statename
        shells = full_structure.split(',')
        shells[-1] = (shells[-1].split(' '))[0]
        self.shells = shells

    def get_shell_iz_energies(self, states):
        shell_iz_energies = np.zeros(len(self.shells))
        shell_iz_energies[-1] = self.iz_energy
        for i in range(len(self.shells)-1):
            missing_e = int(np.sum(self.shell_occupation[i+1:]))
            for state in states:
                if state.iz == missing_e:
                    shell_iz_energies[i] = state.iz_energy
                    break

        self.shell_iz_energies = shell_iz_energies
