from concurrent.futures import process
import numpy as np
import os
import re
import matplotlib.pyplot as plt
from scipy import interpolate


class LNJLevel:
    def __init__(
        self,
        id,
        element,
        nuc_chg,
        num_el,
        config,
        config_full,
        energy,
        n,
        l,
        j,
        stat_weight,
    ):
        self.id = id
        self.element = element
        self.nuc_chg = nuc_chg
        self.num_el = num_el
        self.config = config
        self.config_full = config_full
        self.energy = energy
        self.n = n
        self.l = l
        self.j = j
        self.stat_weight = stat_weight


class LNLevel:
    def __init__(self, id, lnj_states):

        self.id = id
        self.element = lnj_states[0].element
        self.nuc_chg = lnj_states[0].nuc_chg
        self.num_el = lnj_states[0].num_el
        self.config = lnj_states[0].config
        self.n = lnj_states[0].n
        self.l = lnj_states[0].l
        self.nlj_ids = [s.id for s in lnj_states]

        self.energy = np.mean([s.energy for s in lnj_states])
        self.stat_weight = int(np.sum([s.stat_weight for s in lnj_states]))


class LNTrans:
    def __init__(self, delta_E, lnj_transitions):
        self.type = lnj_transitions[0].type
        self.element = lnj_transitions[0].element
        self.from_id = lnj_transitions[0].from_nl_id
        self.to_id = lnj_transitions[0].to_nl_id
        self.delta_E = delta_E

        self.from_nlj_ids = [t.from_id for t in lnj_transitions]
        self.to_nlj_ids = [t.to_id for t in lnj_transitions]


class LNExTrans(LNTrans):
    def __init__(self, delta_E, lnj_transitions):
        LNTrans.__init__(self, delta_E, lnj_transitions)

        # Combine cross-sections
        from_nlj_ids = list(dict.fromkeys([t.from_id for t in lnj_transitions]))
        sigma = np.zeros(len(lnj_transitions[0].sigma))
        for t in lnj_transitions:
            sigma += t.sigma
        self.sigma = list(sigma / len(from_nlj_ids))

        # # Set statistical weights
        # self.from_stat_weight = int(
        #     np.sum([t.from_stat_weight for t in lnj_transitions]))
        # self.to_stat_weight = int(
        #     np.sum([t.to_stat_weight for t in lnj_transitions]))


class LNIzTrans(LNTrans):
    def __init__(self, delta_E, lnj_transitions):
        LNTrans.__init__(self, delta_E, lnj_transitions)

        # Combine cross-sections
        from_nlj_ids = list(dict.fromkeys([t.from_id for t in lnj_transitions]))
        sigma = np.zeros(len(lnj_transitions[0].sigma))
        for t in lnj_transitions:
            sigma += t.sigma
        self.sigma = list(sigma / len(from_nlj_ids))

        # # Set statistical weights
        # self.from_stat_weight = int(
        #     np.sum([t.from_stat_weight for t in lnj_transitions]))


class LNRRTrans(LNTrans):
    def __init__(self, delta_E, lnj_transitions):
        LNTrans.__init__(self, delta_E, lnj_transitions)

        # Combine cross-sections
        from_nlj_ids = list(dict.fromkeys([t.from_id for t in lnj_transitions]))
        sigma = np.zeros(len(lnj_transitions[0].sigma))
        for t in lnj_transitions:
            sigma += t.sigma
        self.sigma = list(sigma / len(from_nlj_ids))

        # # Set statistical weights
        # self.from_stat_weight = int(
        #     np.sum([t.from_stat_weight for t in lnj_transitions]))
        # self.to_stat_weight = int(
        #     np.sum([t.to_stat_weight for t in lnj_transitions]))


class LNEmTrans(LNTrans):
    def __init__(self, delta_E, lnj_transitions):
        LNTrans.__init__(self, delta_E, lnj_transitions)

        # Combine rates
        from_nlj_ids = list(dict.fromkeys([t.from_id for t in lnj_transitions]))
        rate = 0.0
        for t in lnj_transitions:
            rate += t.rate
        self.rate = rate / len(from_nlj_ids)


class LNAiTrans(LNTrans):
    def __init__(self, delta_E, lnj_transitions):
        LNTrans.__init__(self, delta_E, lnj_transitions)

        # Combine rates
        from_nlj_ids = list(dict.fromkeys([t.from_id for t in lnj_transitions]))
        rate = 0.0
        for t in lnj_transitions:
            rate += t.rate
        self.rate = rate / len(from_nlj_ids)


class LNJTrans:
    def __init__(self, type, element, from_id, to_id, delta_E):
        self.type = type
        self.element = element
        self.from_id = from_id
        self.to_id = to_id
        self.delta_E = delta_E

    def make_jsonable(self):
        pass


class LNJExTrans(LNJTrans):
    def __init__(
        self,
        element,
        from_id,
        to_id,
        delta_E,
        E_grid,
        sigma,
        from_stat_weight=None,
        born_bethe_coeffs=None,
    ):
        type = "excitation"
        LNJTrans.__init__(self, type, element, from_id, to_id, delta_E)
        self.E_grid = E_grid
        self.sigma = sigma
        self.from_stat_weight = from_stat_weight
        self.born_bethe_coeffs = born_bethe_coeffs

    def process_cross_section(self, new_E_grid):
        interp_func = interpolate.interp1d(
            self.E_grid,
            np.log(self.sigma),
            kind="linear",
            bounds_error=False,
            fill_value=np.nan,
        )
        sigma_new = np.exp(interp_func(new_E_grid))

        # Apply Born-Bethe approximation at energies higher than 200 * transition energy
        bb_thresh = min(self.delta_E * 200, self.E_grid[-1])
        b0 = self.born_bethe_coeffs[0]
        b1 = self.born_bethe_coeffs[1]
        E_grid_bb = new_E_grid[np.where(new_E_grid > bb_thresh)]
        sigma_new[np.where(new_E_grid > bb_thresh)] = (
            1.1969e-15
            * (1 / (self.from_stat_weight * E_grid_bb))
            * (b0 * np.log(E_grid_bb / self.delta_E) + b1)
        )

        # Set below-threshold sigma to zero
        sigma_new[np.where(new_E_grid <= self.delta_E)] = 0.0

        # Interpolate values which are nan but above threshold
        isnans = np.isnan(sigma_new)
        if isnans.any():
            nan_locs = np.argwhere(isnans)
            first_nonnan_E = new_E_grid[nan_locs[0][-1] + 1]
            first_nonnan_sigma = sigma_new[nan_locs[0][-1] + 1]
            for nan_loc in nan_locs[0]:
                nan_E = new_E_grid[nan_loc]
                d1 = nan_E - self.delta_E
                d2 = first_nonnan_E - nan_E
                interp_val = (d1 * first_nonnan_sigma + d2 * 0.0) / (d1 + d2)
                sigma_new[nan_loc] = interp_val

        # If nans still exist, use the Born-Bethe approx for all elements
        isnans = np.isnan(sigma_new)
        if isnans.any():
            bb_thresh = self.delta_E
            b0 = self.born_bethe_coeffs[0]
            b1 = self.born_bethe_coeffs[1]
            E_grid_bb = new_E_grid[np.where(new_E_grid > bb_thresh)]
            sigma_new[np.where(new_E_grid > bb_thresh)] = (
                1.1969e-15
                * (1 / (self.from_stat_weight * E_grid_bb))
                * (b0 * np.log(E_grid_bb / self.delta_E) + b1)
            )

        # Update E grid and cross-section
        self.E_grid = new_E_grid
        self.sigma = sigma_new

        # Final nan check
        isnans = np.isnan(sigma_new)
        if isnans.any():
            print("Found some nans!")

    def make_jsonable(self):
        self.sigma = list(self.sigma)
        self.E_grid = list(self.E_grid)
        # del self.E_grid
        self.born_bethe_coeffs = list(self.born_bethe_coeffs)


class LNJIzTrans(LNJTrans):
    def __init__(
        self,
        element,
        from_id,
        to_id,
        delta_E,
        E_grid,
        sigma,
        from_stat_weight=None,
        fit_params=None,
    ):
        type = "ionization"
        LNJTrans.__init__(self, type, element, from_id, to_id, delta_E)
        self.E_grid = E_grid
        self.sigma = sigma
        self.from_stat_weight = from_stat_weight
        self.fit_params = fit_params

    def process_cross_section(self, new_E_grid):
        interp_func = interpolate.interp1d(
            self.E_grid,
            np.log(self.sigma),
            kind="linear",
            bounds_error=False,
            fill_value=np.nan,
        )
        sigma_new = np.exp(interp_func(new_E_grid))

        # Apply fit to energies higher than FAC calculated
        p = self.fit_params
        E_grid_fit = new_E_grid[np.where(new_E_grid > self.E_grid[-1])]
        x = E_grid_fit / self.delta_E
        y = 1 - (1 / x)
        sigma_new[np.where(new_E_grid > self.E_grid[-1])] = (
            1.1969e-15
            * (1 / (np.pi * self.from_stat_weight * E_grid_fit))
            * (p[0] * np.log(x) + p[1] * y**2 + p[2] * (y / x) + p[3] * (y / x**2))
        )

        # Set below-threshold sigma to zero
        sigma_new[np.where(new_E_grid <= self.delta_E)] = 0.0

        # Interpolate values which are nan but above threshold
        isnans = np.isnan(sigma_new)
        if isnans.any():
            nan_locs = np.argwhere(isnans)
            first_nonnan_E = new_E_grid[nan_locs[0][-1] + 1]
            first_nonnan_sigma = sigma_new[nan_locs[0][-1] + 1]
            for nan_loc in nan_locs[0]:
                nan_E = new_E_grid[nan_loc]
                d1 = nan_E - self.delta_E
                d2 = first_nonnan_E - nan_E
                interp_val = (d1 * first_nonnan_sigma + d2 * 0.0) / (d1 + d2)
                sigma_new[nan_loc] = interp_val

        # Update E grid and cross-section
        self.E_grid = new_E_grid
        self.sigma = sigma_new

        # Final nan check
        isnans = np.isnan(sigma_new)
        if isnans.any():
            print("Found some nans!")

    def make_jsonable(self):
        self.sigma = list(self.sigma)
        self.E_grid = list(self.E_grid)
        # del self.E_grid
        self.fit_params = list(self.fit_params)


class LNJRRTrans(LNJTrans):
    def __init__(
        self,
        element,
        from_id,
        to_id,
        delta_E,
        E_grid,
        sigma,
        from_stat_weight=None,
        to_stat_weight=None,
        l=None,
        fit_params=None,
    ):
        type = "radiative recombination"
        LNJTrans.__init__(self, type, element, from_id, to_id, delta_E)
        self.E_grid = E_grid
        self.sigma = sigma
        self.from_stat_weight = from_stat_weight
        self.to_stat_weight = to_stat_weight
        self.l = l
        self.fit_params = fit_params

    def process_cross_section(self, new_E_grid):
        interp_func = interpolate.interp1d(
            self.E_grid,
            np.log(self.sigma),
            kind="linear",
            bounds_error=False,
            fill_value=np.nan,
        )
        sigma_new = np.exp(interp_func(new_E_grid))

        # Apply fit to energies higher than FAC calculated
        p = self.fit_params
        E_grid_fit = new_E_grid[np.where(new_E_grid > self.E_grid[-1])]
        x = E_grid_fit / self.delta_E
        y = 1 - (1 / x)
        sigma_new[np.where(new_E_grid > self.E_grid[-1])] = (
            1.1969e-15
            * (1 / (np.pi * self.from_stat_weight * E_grid_fit))
            * (p[0] * np.log(x) + p[1] * y**2 + p[2] * (y / x) + p[3] * (y / x**2))
        )

        E_h = 27.211
        p = self.fit_params
        E_grid_fit = new_E_grid[np.where(new_E_grid > self.E_grid[-1])]
        eps = E_grid_fit / E_h
        w = E_grid_fit + self.delta_E
        x = (E_grid_fit + p[3]) / p[3]
        y = (1.0 + p[2]) / (np.sqrt(x) + p[2])
        dgf_dE = (
            (w / (E_grid_fit + p[3]))
            * p[0]
            * x ** (-3.5 - self.l + (p[1] / 2))
            * y ** p[1]
        )
        alpha = 1 / 137
        g_i = self.to_stat_weight
        g_f = self.from_stat_weight
        sigma_pi = (
            (2 * np.pi * alpha / g_i)
            * ((1 + alpha**2 * eps) / (1 + 0.5 * alpha**2 * eps))
            * dgf_dE
        )

        arb_const = 4e-20  # TODO: This constant works, but not sure exactly what it
        sigma_new[np.where(new_E_grid > self.E_grid[-1])] = (
            (alpha**2 / 2)
            * (g_i / g_f)
            * (w**2 / (eps * (1.0 + 0.5 * alpha**2 * eps)))
            * sigma_pi
            * arb_const
        )

        # Update E grid and cross-section
        self.E_grid = new_E_grid
        self.sigma = sigma_new

    def make_jsonable(self):
        self.sigma = list(self.sigma)
        self.E_grid = list(self.E_grid)
        # del self.E_grid
        self.fit_params = list(self.fit_params)


class LNJEmTrans(LNJTrans):
    def __init__(self, element, from_id, to_id, delta_E, gf, rate):
        type = "emission"
        LNJTrans.__init__(self, type, element, from_id, to_id, delta_E)
        self.gf = gf
        self.rate = rate


class LNJAiTrans(LNJTrans):
    def __init__(self, element, from_id, to_id, delta_E, rate):
        type = "autoionization"
        LNJTrans.__init__(self, type, element, from_id, to_id, delta_E)
        self.rate = rate


# TODO: Switch from nlj to nl resolved by averaged over cross-sections, summing statistical weights, etc


def get_levels(lev_f):
    # Return a list of energy levels for the given element (element) and charge (Z)

    neles = []
    nlevs = []
    lev_block_idx = []
    with open(lev_f) as f:
        lines = f.readlines()
        for i, l in enumerate(lines):
            if "Z	= " in l:
                element = l.split()[0]
                nuc_chg = int(float(l.split()[-1]))
            if "NELE" in l:
                neles.append(int(l.split()[-1]))
            if "NLEV" in l:
                nlevs.append(int(l.split()[-1]))
                lev_block_idx.append(i)
            if "E0" in l:
                E0 = float(l.split()[-1])

    levels = []
    for i, block_start in enumerate(lev_block_idx):
        for l in lines[block_start + 2 : block_start + 2 + nlevs[i]]:
            dat = l.split()
            id = int(dat[0])
            energy = E0 + float(dat[2])
            num_el = neles[i]
            vnl = int(dat[4])
            l = int(np.mod(vnl, 100))
            n = int((vnl - l) / 100)
            j = int(dat[5]) / 2
            stat_weight = int(dat[5]) + 1
            config = dat[7]
            config_full = dat[8]
            levels.append(
                LNJLevel(
                    id,
                    element,
                    nuc_chg,
                    num_el,
                    config,
                    config_full,
                    energy,
                    n,
                    l,
                    j,
                    stat_weight,
                )
            )

    return levels


def get_ex_cross_sections(ce_f):
    # Return a list of excitation cross-sections (indexed by energy level)

    ntrans = []
    trans_block_idx = []
    with open(ce_f) as f:
        lines = f.readlines()
        for i, l in enumerate(lines):
            if "Z	= " in l:
                element = l.split()[0]
            if "NTRANS" in l:
                ntrans.append(int(l.split()[-1]))
                trans_block_idx.append(i)
            if "NEGRID" in l:
                num_E = int(l.split()[-1])

    ex_transitions = []
    for i, block_start in enumerate(trans_block_idx):

        block_header_part = lines[block_start : block_start + 10]
        for l in block_header_part:
            if "NTEGRID" in l:
                num_TE_grid = int(l.split()[-1])
                break

        header_skip = block_start + (2 * num_E) + 11 + num_TE_grid
        block_lines = lines[header_skip : header_skip + ntrans[i] * (num_E + 2)]
        for j in range(ntrans[i]):

            trans_dat = block_lines[(j * (num_E + 2))].split()
            from_id = int(trans_dat[0])
            to_id = int(trans_dat[2])
            delta_E = float(trans_dat[4])
            from_stat_weight = int(trans_dat[1]) + 1
            born_bethe_coeffs = [
                float(b) for b in block_lines[j * (num_E + 2) + 1].split()
            ]
            if born_bethe_coeffs[0] == -1:
                # TODO: Check this is the right thing to do here!
                born_bethe_coeffs = [0.0 for _ in born_bethe_coeffs]

            trans_lines = block_lines[(j * (num_E + 2)) + 2 : (j + 1) * (num_E + 2)]
            E_grid, _, cross_section = np.loadtxt(trans_lines, unpack=True)

            E_grid += delta_E
            cross_section *= 1e-20
            ex_transitions.append(
                LNJExTrans(
                    element,
                    from_id,
                    to_id,
                    delta_E,
                    E_grid,
                    cross_section,
                    from_stat_weight,
                    born_bethe_coeffs=born_bethe_coeffs,
                )
            )

    return ex_transitions


def get_iz_cross_sections(ci_f):
    # Return a list of ionization cross-sections (indexed by energy level)

    ntrans = []
    trans_block_idx = []
    with open(ci_f) as f:
        lines = f.readlines()
        for i, l in enumerate(lines):
            if "Z	= " in l:
                element = l.split()[0]
            if "NTRANS" in l:
                ntrans.append(int(l.split()[-1]))
                trans_block_idx.append(i)
            if "NEGRID" in l:
                num_E = int(l.split()[-1])

    iz_transitions = []
    for i, block_start in enumerate(trans_block_idx):

        block_header_part = lines[block_start : block_start + 10]
        for l in block_header_part:
            if "NTEGRID" in l:
                num_TE_grid = int(l.split()[-1])
                break

        header_skip = block_start + (2 * num_E) + 9 + num_TE_grid
        block_lines = lines[header_skip : header_skip + ntrans[i] * (num_E + 2)]
        for j in range(ntrans[i]):

            trans_dat = block_lines[(j * (num_E + 2))].split()
            from_id = int(trans_dat[0])
            to_id = int(trans_dat[2])
            delta_E = float(trans_dat[4])
            from_stat_weight = int(trans_dat[1]) + 1
            fit_params = [float(p) for p in block_lines[j * (num_E + 2) + 1].split()]

            trans_lines = block_lines[(j * (num_E + 2)) + 2 : (j + 1) * (num_E + 2)]
            E_grid, _, cross_section = np.loadtxt(trans_lines, unpack=True)

            E_grid += delta_E
            cross_section *= 1e-20
            iz_transitions.append(
                LNJIzTrans(
                    element,
                    from_id,
                    to_id,
                    delta_E,
                    E_grid,
                    cross_section,
                    from_stat_weight,
                    fit_params=fit_params,
                )
            )

    return iz_transitions


def get_rr_cross_sections(ci_f):
    # Return a list of radiative recombination cross-sections (indexed by energy level)

    # TODO: Process the photo-ionization cross-sections

    ntrans = []
    trans_block_idx = []
    with open(ci_f) as f:
        lines = f.readlines()
        for i, l in enumerate(lines):
            if "Z	= " in l:
                element = l.split()[0]
            if "NTRANS" in l:
                ntrans.append(int(l.split()[-1]))
                trans_block_idx.append(i)
            if "NEGRID" in l:
                num_E = int(l.split()[-1])

    rr_transitions = []
    for i, block_start in enumerate(trans_block_idx):

        block_header_part = lines[block_start : block_start + 10]
        for l in block_header_part:
            if "NTEGRID" in l:
                num_TE_grid = int(l.split()[-1])
                break

        header_skip = block_start + (2 * num_E) + 9 + num_TE_grid
        block_lines = lines[header_skip : header_skip + ntrans[i] * (num_E + 2)]
        for j in range(ntrans[i]):

            trans_dat = block_lines[(j * (num_E + 2))].split()
            to_id = int(trans_dat[0])
            to_stat_weight = int(trans_dat[1]) + 1
            from_id = int(trans_dat[2])
            from_stat_weight = int(trans_dat[3]) + 1
            delta_E = float(trans_dat[4])
            l = int(trans_dat[5])
            fit_params = [float(p) for p in block_lines[j * (num_E + 2) + 1].split()]

            trans_lines = block_lines[(j * (num_E + 2)) + 2 : (j + 1) * (num_E + 2)]
            E_grid, cross_section, _, _ = np.loadtxt(trans_lines, unpack=True)

            cross_section *= 1e-20
            rr_transitions.append(
                LNJRRTrans(
                    element,
                    from_id,
                    to_id,
                    delta_E,
                    E_grid,
                    cross_section,
                    from_stat_weight=from_stat_weight,
                    to_stat_weight=to_stat_weight,
                    l=l,
                    fit_params=fit_params,
                )
            )

    return rr_transitions


def get_em_rates(tr_f, uta=False):
    # Return a list of spontaneous emission rates (indexed by energy level)

    ntrans = []
    trans_block_idx = []
    with open(tr_f) as f:
        lines = f.readlines()
        for i, l in enumerate(lines):
            if "Z	= " in l:
                element = l.split()[0]
            if "NTRANS" in l:
                ntrans.append(int(l.split()[-1]))
                trans_block_idx.append(i)

    em_transitions = []
    for i, block_start in enumerate(trans_block_idx):

        header_skip = block_start + 4
        block_lines = lines[header_skip : header_skip + ntrans[i]]
        for j in range(ntrans[i]):

            trans_dat = block_lines[j].split()
            from_id = int(trans_dat[0])
            to_id = int(trans_dat[2])
            delta_E = float(trans_dat[4])
            if uta:
                gf = float(trans_dat[6])
                rate = float(trans_dat[7])
            else:
                gf = float(trans_dat[5])
                rate = float(trans_dat[6])

            em_transitions.append(
                LNJEmTrans(element, from_id, to_id, delta_E, gf, rate)
            )

    return em_transitions


def get_ai_rates(ai_f):
    # Return a list of spontaneous emission rates (indexed by energy level)

    ntrans = []
    trans_block_idx = []
    try:
        with open(ai_f) as f:
            lines = f.readlines()
            for i, l in enumerate(lines):
                if "Z	= " in l:
                    element = l.split()[0]
                if "NTRANS" in l:
                    ntrans.append(int(l.split()[-1]))
                    trans_block_idx.append(i)
                if "NEGRID" in l:
                    num_E = int(l.split()[-1])
    except:
        return []

    ai_transitions = []
    for i, block_start in enumerate(trans_block_idx):

        header_skip = block_start + 3 + num_E
        block_lines = lines[header_skip : header_skip + ntrans[i]]
        for j in range(ntrans[i]):

            trans_dat = block_lines[j].split()
            from_id = int(trans_dat[0])
            to_id = int(trans_dat[2])
            delta_E = float(trans_dat[4])
            rate = float(trans_dat[5])

            ai_transitions.append(LNJAiTrans(element, from_id, to_id, delta_E, rate))

    return ai_transitions


def aggregate_states(lnj_levels):

    aggregated_states = []
    unique_nls = list(
        dict.fromkeys([(s.n, s.l, s.num_el, s.config) for s in lnj_levels])
    )
    for nl in unique_nls:
        nlj_states = []
        for s in lnj_levels:
            state_nl = (s.n, s.l, s.num_el, s.config)
            if nl == state_nl:
                nlj_states.append(s)

        id = len(aggregated_states)
        for nlj_s in nlj_states:
            nlj_s.nl_id = id
        aggregated_states.append(LNLevel(id, nlj_states))

    return aggregated_states


def aggregate_transitions(nl_levels, nlj_levels, transitions):

    for i, t in enumerate(transitions):
        transitions[i].from_nl_id = nlj_levels[transitions[i].from_id].nl_id
        transitions[i].to_nl_id = nlj_levels[transitions[i].to_id].nl_id

    aggregated_transitions = []

    trans_types = ["ionization", "radiative recombination", "autoionization"]
    for trans_type in trans_types:
        print(trans_type)
        type_transitions = [t for t in transitions if t.type == trans_type]
        nl_trans_ids = [(t.from_nl_id, t.to_nl_id) for t in type_transitions]
        unique_nl_trans_ids = list(dict.fromkeys(nl_trans_ids))
        num_ids = len(unique_nl_trans_ids)
        for i, nl_trans_id in enumerate(unique_nl_trans_ids):
            print("{:.1f}%".format(100 * i / num_ids), end="\r")
            nl_transitions = []
            delta_E = 0.0
            for t in type_transitions:
                cur_nl_trans_id = (t.from_nl_id, t.to_nl_id)
                if cur_nl_trans_id == nl_trans_id:
                    if nl_trans_id[0] == nl_trans_id[1]:
                        pass  # Ignore transitions between j levels with same n, l?
                    else:
                        nl_transitions.append(t)
                        delta_E = abs(
                            nl_levels[t.to_nl_id].energy
                            - nl_levels[t.from_nl_id].energy
                        )
            if len(nl_transitions) > 0:
                # delta_E_2 = np.mean([t.delta_E for t in nl_transitions])
                # if abs(delta_E - delta_E_2) > 1.0:
                #     print('hey!')
                if trans_type == "ionization":
                    aggregated_transitions.append(LNIzTrans(delta_E, nl_transitions))
                elif trans_type == "radiative recombination":
                    aggregated_transitions.append(LNRRTrans(delta_E, nl_transitions))
                elif trans_type == "autotionization":
                    aggregated_transitions.append(LNAiTrans(delta_E, nl_transitions))

    trans_types = ["excitation", "emission"]
    max_num_el = max([l.num_el for l in nl_levels])
    for trans_type in trans_types:
        for num_el in range(max_num_el + 1):
            print(trans_type, num_el)
            type_transitions = [
                t
                for t in transitions
                if t.type == trans_type and nlj_levels[t.from_id].num_el == num_el
            ]
            nl_trans_ids = [(t.from_nl_id, t.to_nl_id) for t in type_transitions]
            unique_nl_trans_ids = list(dict.fromkeys(nl_trans_ids))
            num_ids = len(unique_nl_trans_ids)
            for i, nl_trans_id in enumerate(unique_nl_trans_ids):
                print("{:.1f}%".format(100 * i / num_ids), end="\r")
                nl_transitions = []
                delta_E = 0.0
                for t in type_transitions:
                    cur_nl_trans_id = (t.from_nl_id, t.to_nl_id)
                    if cur_nl_trans_id == nl_trans_id:
                        if nl_trans_id[0] == nl_trans_id[1]:
                            pass  # Ignore transitions between j levels with same n, l?
                        else:
                            nl_transitions.append(t)
                            delta_E = abs(
                                nl_levels[t.to_nl_id].energy
                                - nl_levels[t.from_nl_id].energy
                            )
                if len(nl_transitions) > 0:
                    # delta_E_2 = np.mean([t.delta_E for t in nl_transitions])
                    # if abs(delta_E - delta_E_2) > 1.0:
                    #     print('hey!')
                    if trans_type == "excitation":
                        aggregated_transitions.append(
                            LNExTrans(delta_E, nl_transitions)
                        )
                    elif trans_type == "emission":
                        aggregated_transitions.append(
                            LNEmTrans(delta_E, nl_transitions)
                        )
    print("{:.1f}%".format(100), end="\r")

    return aggregated_transitions
