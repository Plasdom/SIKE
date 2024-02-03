from scipy import interpolate
import numpy as np
import re
import matplotlib.pyplot as plt


class State:
    """Atomc state class"""

    def __init__(
        self,
        id: int,
        nuc_chg: int,
        num_el: int,
        config: str,
        energy: float,
        stat_weight: int,
        n: int,
        l: int | None = None,
        j: int | None = None,
        config_full: str | None = None,
        metastable: bool = True,
    ):
        """Initialise

        :param id: Unique ID for the state
        :type id: int
        :param nuc_chg: Nuclear charge
        :type nuc_chg: int
        :param num_el: Number of electrons
        :type num_el: int
        :param config: Electronic configuration (valence shell)
        :type config: str
        :param energy: Energy of the state
        :type energy: float
        :param stat_weight: Statistical weight
        :type stat_weight: int
        :param n: Principal quantum number of the state
        :type n: int
        :param l: Orbital angular momentum quantum number of the state, defaults to None
        :type l: int | None, optional
        :param j: Total angular momentum quantum number of the state, defaults to None
        :type j: int | None, optional
        :param config_full: Full electronic configuration (all shells), defaults to None
        :type config_full: str | None, optional
        :param metastable: Whether the state is metastable, defaults to True
        :type metastable: bool, optional
        """
        self.id = id
        self.nuc_chg = nuc_chg
        self.num_el = num_el
        self.Z = self.nuc_chg - self.num_el
        self.config = config
        self.config_full = config_full
        self.energy = energy
        self.n = n
        self.l = l
        self.j = j
        self.stat_weight = stat_weight
        self.metastable = metastable

    def equals(self, other: State) -> bool:
        """Check for equality between this state and another

        :param other: The other atomic state
        :type other: State
        :return: True if states are equal
        :rtype: bool
        """
        if (
            self.nuc_chg == other.nuc_chg
            and self.num_el == other.num_el
            and self.config == other.config
        ):
            if self.j is not None:
                if self.j == other.j:
                    return True
                else:
                    return False
            else:
                return True
        else:
            return False
