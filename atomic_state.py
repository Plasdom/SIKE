from scipy import interpolate
import numpy as np
import re
import matplotlib.pyplot as plt


class State2:
    def __init__(self, id, dict):
        self.id = id
        self.nuc_chg = dict['nuc_chg']
        self.num_el = dict['num_el']
        self.Z = self.nuc_chg - self.num_el
        self.config = dict['config']
        if 'config_full' in dict.keys():
            self.config_full = dict['config_full']
        self.energy = dict['energy']
        self.n = dict['n']
        self.l = dict['l']
        if 'j' in dict.keys():
            self.j = dict['j']
        else:
            self.j = None
        self.stat_weight = dict['stat_weight']
        self.metastable = True

    def equals(self, other):
        """Check for equality between two State objects

        Args:
            other (State): Other state

        Returns:
            boolean: True is states are equal
        """
        if self.nuc_chg == other.nuc_chg and self.num_el == other.num_el and self.config == other.config:
            if self.j is not None:
                if self.j == other.j:
                    return True
                else:
                    return False
            else:
                return True
        else:
            return False
