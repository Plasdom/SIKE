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
        :param nuc_chg: Nuclear charge
        :param num_el: Number of electrons
        :param config: Electronic configuration (valence shell)
        :param energy: Energy of the state
        :param stat_weight: Statistical weight
        :param n: Principal quantum number of the state
        :param l: Orbital angular momentum quantum number of the state, defaults to None
        :param j: Total angular momentum quantum number of the state, defaults to None
        :param config_full: Full electronic configuration (all shells), defaults to None
        :param metastable: Whether the state is metastable, defaults to True
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

    def equals(self, other) -> bool:
        """Check for equality between this state and another

        :param other: The other atomic state
        :return: True if states are equal
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
