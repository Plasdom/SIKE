import sys
import os
import numpy as np
import json
import sike
import sike.constants as sc


if __name__ == "__main__":
    # Read data
    # atom_data_file = '/Users/dpower/Documents/01 - PhD/10 - Impurities/FLYCHK data/Carbon/atomic.dat'
    for el in sc.SYMBOL2ELEMENT.keys():
        print(el)
        element = sc.SYMBOL2ELEMENT[el]
        atom_data_file = (
            sike.get_atomic_data_savedir() / "FLYCHK data" / element / "atomic.dat"
        )
        with open(atom_data_file) as f:
            atom_data = f.readlines()

        # Divide levels into ionization stages
        stage_boundaries_start = []
        for i, l in enumerate(atom_data):
            if "enot" in l:
                stage_boundaries_start.append(i + 1)
            if "end data" in l:
                last_line = i
                break
        stage_boundaries_end = [
            stage_boundaries_start[i + 1] - 1
            for i in range(len(stage_boundaries_start) - 1)
        ]
        stage_boundaries_end.append(last_line)
        num_z = len(stage_boundaries_start) + 1

        # Create levels object and add the bare nucleus
        levels = []
        levels.append(
            {
                "id": 0,
                "flychk_id": 1,
                "element": el,
                "nuc_chg": num_z - 1,
                "num_el": 0,
                "energy": 0.0,
                "stat_weight": 1,
                "config": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                "n": 1,
            }
        )

        # Extract atomic data
        id_num = 1
        for z in range(num_z - 1):
            for i in range(stage_boundaries_start[z], stage_boundaries_end[z]):
                levels.append({})
                line_data = atom_data[i].split()
                levels[-1]["id"] = id_num
                levels[-1]["flychk_id"] = int(line_data[2])
                levels[-1]["element"] = el
                levels[-1]["nuc_chg"] = num_z - 1
                levels[-1]["num_el"] = int(line_data[1])
                levels[-1]["energy"] = float(line_data[4])
                levels[-1]["stat_weight"] = int(float(line_data[5]))
                num_shells = len(line_data) - 7
                shell_occ = [int(line_data[j]) for j in range(6, 6 + num_shells)]
                # levels[-1]['config'] = occ2config(shell_occ)
                levels[-1]["config"] = shell_occ
                levels[-1]["n"] = int(line_data[-1])
                if levels[-1]["n"] > 10:
                    for j in range(10, levels[-1]["n"] - 1):
                        levels[-1]["config"].append(0)
                    levels[-1]["config"].append(1)
                id_num += 1

        # Convert level energies into absolute energies
        gs_iz_pots = np.zeros(num_z - 1)
        for i, l in enumerate(stage_boundaries_start):
            gs_iz_pots[i] = atom_data[l - 1].split()[2]
        for num_el in range(1, num_z):
            iz_pot = gs_iz_pots[num_el - 1]
            stage_levs = [lev for lev in levels if lev["num_el"] == num_el]
            for lev in stage_levs:
                lev["iz_pot"] = iz_pot - lev["energy"]
        for num_el in range(1, num_z):
            iz_pot = sum(gs_iz_pots[:num_el])
            stage_levs = [lev for lev in levels if lev["num_el"] == num_el]
            for lev in stage_levs:
                lev["energy"] -= iz_pot

        # Sort by energies
        levels = sorted(levels, key=lambda x: x["energy"])
        for i, lev in enumerate(levels):
            lev["id"] = i

        transitions = []

        # Function to return a level from list given num_el and flychk_id
        def get_level(num_el, flychk_id, levels):
            for lev in levels:
                if lev["num_el"] == num_el and lev["flychk_id"] == flychk_id:
                    return lev

        # Read in excitation transition oscillator strengths
        start_found = False
        for i, l in enumerate(atom_data):
            if "data   phxs " in l:
                ex_start = i
                start_found = True
            if "end data" in l and start_found:
                ex_end = i
                break
        for l in atom_data[ex_start:ex_end]:
            line_data = l.split()
            if len(line_data) > 3:
                transitions.append({})
                transitions[-1]["type"] = "excitation"
                transitions[-1]["element"] = el
                num_el1 = int(line_data[1])
                flychk_id1 = int(line_data[2])
                num_el2 = int(line_data[3])
                flychk_id2 = int(line_data[4])
                from_lev = get_level(num_el1, flychk_id1, levels)
                to_lev = get_level(num_el2, flychk_id2, levels)
                transitions[-1]["to_id"] = to_lev["id"]
                transitions[-1]["from_id"] = from_lev["id"]
                transitions[-1]["from_stat_weight"] = from_lev["stat_weight"]
                transitions[-1]["osc_str"] = float(line_data[5])
                transitions[-1]["delta_E"] = float(line_data[7])

        # # Read in excitation transition rates
        # start_found = False
        # for i, l in enumerate(atom_data):
        #     if "  rate type:   collisional excitation" in l:
        #         ex2_start = i
        #         start_found = True
        #     if "  rate type:   collisional ionization" in l and start_found:
        #         ex2_end = i
        #         break
        # for l in atom_data[ex2_start:ex2_end]:
        #     line_data = l.split()
        #     if len(line_data) > 4:
        #         transitions.append({})
        #         transitions[-1]["type"] = "excitation2"
        #         transitions[-1]["element"] = el
        #         num_el1 = int(line_data[1])
        #         flychk_id1 = int(line_data[2])
        #         num_el2 = int(line_data[3])
        #         flychk_id2 = int(line_data[4])
        #         to_lev = get_level(num_el2, flychk_id2, levels)
        #         from_lev = get_level(num_el1, flychk_id1, levels)
        #         transitions[-1]["to_id"] = from_lev["id"]
        #         transitions[-1]["from_id"] = to_lev["id"]
        #         try:
        #             transitions[-1]["maxwellian_rate"] = float(line_data[5])
        #         except:
        #             transitions[-1]["maxwellian_rate"] = 0.0
        #         try:
        #             transitions[-1]["maxwellian_inv_rate"] = float(line_data[6])
        #         except:
        #             transitions[-1]["maxwellian_inv_rate"] = 0.0
        #         # transitions[-1]['osc_str'] = float(line_data[5])
        #         # transitions[-1]['delta_E'] = float(line_data[7])

        # Read in emission transitions
        start_found = False
        for i, l in enumerate(atom_data):
            if "  rate type:   photoexcitation" in l:
                rad_deex_start = i
                start_found = True
            if "  rate type:   photoionization       " in l and start_found:
                rad_deex_end = i
                break
        for l in atom_data[rad_deex_start:rad_deex_end]:
            line_data = l.split()
            if len(line_data) > 3:
                transitions.append({})
                transitions[-1]["type"] = "emission"
                transitions[-1]["element"] = el
                num_el1 = int(line_data[1])
                flychk_id1 = int(line_data[2])
                num_el2 = int(line_data[3])
                flychk_id2 = int(line_data[4])
                from_lev = get_level(num_el1, flychk_id1, levels)
                to_lev = get_level(num_el2, flychk_id2, levels)
                transitions[-1]["to_id"] = to_lev["id"]
                transitions[-1]["from_id"] = from_lev["id"]
                transitions[-1]["delta_E"] = from_lev["energy"] - to_lev["energy"]
                transitions[-1]["rate"] = float(line_data[5])

        # Read in radiative recombination transitions
        start_found = False
        for i, l in enumerate(atom_data):
            if "  rate type:   photoionization       " in l:
                rad_rec_start = i
                start_found = True
            if "  rate type:   collisional excitation" in l and start_found:
                rad_rec_end = i
                break
        for l in atom_data[rad_rec_start:rad_rec_end]:
            line_data = l.split()
            if len(line_data) > 3:
                transitions.append({})
                transitions[-1]["type"] = "radiative_recombination"
                transitions[-1]["element"] = el
                num_el1 = int(line_data[1])
                flychk_id1 = int(line_data[2])
                num_el2 = int(line_data[3])
                flychk_id2 = int(line_data[4])
                to_lev = get_level(num_el1, flychk_id1, levels)
                from_lev = get_level(num_el2, flychk_id2, levels)
                transitions[-1]["to_id"] = to_lev["id"]
                transitions[-1]["from_id"] = from_lev["id"]
                transitions[-1]["delta_E"] = from_lev["energy"] - to_lev["energy"]
                # transitions[-1]["maxwellian_rate"] = float(line_data[5])
                # transitions[-1]["Te"] = float(atom_data[0].split()[2])

        # Read in autoionization transitions
        start_found = False
        for i, l in enumerate(atom_data):
            if "  rate type:   augxs" in l:
                autoiz_start = i
                break
        autoiz_end = len(atom_data)
        for l in atom_data[autoiz_start:autoiz_end]:
            line_data = l.split()
            if len(line_data) > 3:
                transitions.append({})
                transitions[-1]["type"] = "autoionization"
                transitions[-1]["element"] = el
                num_el1 = int(line_data[1])
                flychk_id1 = int(line_data[2])
                num_el2 = int(line_data[3])
                flychk_id2 = int(line_data[4])
                from_lev = get_level(num_el1, flychk_id1, levels)
                to_lev = get_level(num_el2, flychk_id2, levels)
                transitions[-1]["to_id"] = to_lev["id"]
                transitions[-1]["from_id"] = from_lev["id"]
                transitions[-1]["delta_E"] = from_lev["energy"] - to_lev["energy"]
                transitions[-1]["rate"] = float(line_data[5])

        # Read in ionization transitions
        start_found = False
        for i, l in enumerate(atom_data):
            if "  rate type:   collisional ionization" in l:
                iz_start = i
                start_found = True
            if "  rate type:   augxs" in l and start_found:
                iz_end = i
                break
        for l in atom_data[iz_start:iz_end]:
            line_data = l.split()
            if len(line_data) > 4:
                transitions.append({})
                transitions[-1]["type"] = "ionization"
                transitions[-1]["element"] = el
                num_el1 = int(line_data[1])
                flychk_id1 = int(line_data[2])
                num_el2 = int(line_data[3])
                flychk_id2 = int(line_data[4])
                from_lev = get_level(num_el1, flychk_id1, levels)
                to_lev = get_level(num_el2, flychk_id2, levels)
                transitions[-1]["to_id"] = to_lev["id"]
                transitions[-1]["from_id"] = from_lev["id"]
                transitions[-1]["delta_E"] = to_lev["energy"] - from_lev["energy"]
                # try:
                #     transitions[-1]["maxwellian_rate"] = float(line_data[5])
                # except ValueError:
                #     transitions[-1]["maxwellian_rate"] = 0.0

        vgrid, Egrid = sike.generate_vgrid()
        Egrid = 0.5 * sc.EL_MASS * vgrid**2 / sc.EL_CHARGE

        # Screening coefficients test
        # screening_coeffs = np.loadtxt('/Users/dpower/Documents/01 - PhD/10 - Impurities/FLYCHK data/Marchand_screening_coeffs.txt')
        # screening_coeffs = np.loadtxt('Marchand_screening_coeffs.txt')
        # screening_coeffs = screening_coeffs.transpose()
        screening_coeffs = sc.MARCHAND_SCREENING_COEFFS.T

        Ry = 13.60569
        alpha = 1 / 137
        bohr_radius = 5.292e-11

        def get_iz_pot(n, P, Z):
            sum_val = 0
            for m in range(1, n):
                sum_val += screening_coeffs[n - 1, m - 1] * P[m - 1]
            Q_n = (
                Z
                - sum_val
                - 0.5 * screening_coeffs[n - 1, n - 1] * max(P[n - 1] - 1, 0)
            )
            I_n = (
                Ry
                * (Q_n**2)
                / n**2
                * (1 + ((alpha * Q_n / n) ** 2) * (((2 * n) / (n + 1)) - (3 / 4)))
            )
            return I_n

        # # Calculation of excitation cross-section
        # def g(U):
        #     A = 0.15
        #     B = 0.0
        #     C = 0.0
        #     D = 0.28
        #     return A + B / U + C / U**2 + D * np.log(U)

        # def get_ex_cs(vgrid, from_lev, to_lev, ex_trans):

        #     a_0 = 5.29177e-11
        #     I_H = 13.6058

        #     eps = to_lev["energy"] - from_lev["energy"]
        #     f_ij = ex_trans["osc_str"]

        #     # Calculate cross-section
        #     cs = np.zeros(len(vgrid))
        #     for i in range(len(vgrid)):
        #         E = 0.5 * sc.EL_MASS * vgrid[i] ** 2 / sc.EL_CHARGE
        #         U = E / eps
        #         if E >= eps:
        #             cs[i] = (
        #                 8.0
        #                 * np.pi**2
        #                 * a_0**2
        #                 / np.sqrt(3)
        #                 * (I_H / eps) ** 2
        #                 * f_ij
        #                 * g(U)
        #                 / U
        #             )

        #     if any(np.array(cs) < 0.0):
        #         print("cs below zero")
        #     if any(np.isnan(cs)):
        #         print("nans found in excitation cross-section - setting to zero ")

        #     return list(1e4 * cs)

        # ex_transitions = [tr for tr in transitions if tr["type"] == "excitation"]
        # # ex2_transitions = [tr for tr in transitions if tr["type"] == "excitation2"]
        # for i, ex_trans in enumerate(ex_transitions):
        #     from_lev = levels[ex_trans["from_id"]]
        #     to_lev = levels[ex_trans["to_id"]]
        #     # from_lev2 = levels[ex2_transitions[i]["from_id"]]
        #     # to_lev2 = levels[ex2_transitions[i]["to_id"]]
        #     if from_lev["num_el"] != to_lev["num_el"]:
        #         print("from level is different iso stage from to level")
        #     ex_trans["sigma"] = get_ex_cs(vgrid, from_lev, to_lev, ex_trans)
        #     # for sigma in ex_trans["sigma"]:
        #     #     if sigma < 0.0:
        #     #         print("cs below zero")
        #     #     if np.isnan(sigma):
        #     #         print("nans found")

        # # Calculation of ionization cross-sections (Burgess-Chidichimo formula)
        # def get_BC_iz_cs(vgrid, from_lev, to_lev):
        #     # Note
        #     z = from_lev["nuc_chg"] - from_lev["num_el"]
        #     I_H = 13.6058
        #     a_0 = 5.29177e-11
        #     cs = np.zeros(len(vgrid))
        #     nu = 0.25 * (np.sqrt((100 * z + 91) / (4 * z + 3)) - 1)
        #     C = 2.0

        #     zeta = [c for c in from_lev["config"] if c != 0][-1]
        #     I_n = to_lev["energy"] - from_lev["energy"]

        #     for i in range(len(vgrid)):
        #         E = 0.5 * sc.EL_MASS * vgrid[i] ** 2 / sc.EL_CHARGE
        #         if E >= I_n:
        #             beta = 0.25 * (((100 * z + 91) / (4 * z + 3)) ** 0.5 - 5)
        #             W = np.log(E / I_n) ** (beta * I_n / E)
        #             cs[i] = (
        #                 np.pi
        #                 * a_0**2
        #                 * C
        #                 * zeta
        #                 * (I_H / I_n) ** 2
        #                 * (I_n / E)
        #                 * np.log(E / I_n)
        #                 * W
        #             )

        #     return list(1e4 * cs)

        # iz_transitions = [tr for tr in transitions if tr["type"] == "ionization"]
        # for iz_trans in iz_transitions:
        #     from_lev = levels[iz_trans["from_id"]]
        #     to_lev = levels[iz_trans["to_id"]]
        #     iz_trans["sigma"] = get_BC_iz_cs(vgrid, from_lev, to_lev)

        # # Calculation of radiative recombination rate cross-sections
        # def photo_iz_sigma(I_n, Q_n, E):
        #     return (
        #         (64 * np.pi * alpha * bohr_radius**2 / (3**1.5))
        #         * I_n**2.5
        #         * Ry**0.5
        #         / (Q_n * (E**3))
        #     )

        # alpha = 1 / 137
        # bohr_radius = 5.292e-11
        # Ry = 13.60569
        # c = 299792458
        # planck_h = 6.63e-34

        # rr_transitions = [
        #     tr for tr in transitions if tr["type"] == "radiative recombination"
        # ]
        # for rr_trans in rr_transitions:

        #     from_lev = levels[rr_trans["from_id"]]
        #     to_lev = levels[rr_trans["to_id"]]
        #     flychk_rate = rr_trans["maxwellian_rate"]
        #     Te = rr_trans["Te"]
        #     ne = 1e14

        #     # Calculate iz potential
        #     n = to_lev["n"]
        #     P = to_lev["config"]
        #     Z = to_lev["nuc_chg"]
        #     sum_val = 0
        #     for m in range(1, min(n, 10)):
        #         sum_val += screening_coeffs[min(n, 10) - 1, m - 1] * P[m - 1]
        #     Q_n = (
        #         Z
        #         - sum_val
        #         - 0.5
        #         * screening_coeffs[min(n, 10) - 1, min(n, 10) - 1]
        #         * max(P[n - 1] - 1, 0)
        #     )
        #     I_n = (
        #         Ry
        #         * (Q_n**2)
        #         / n**2
        #         * (1 + ((alpha * Q_n / n) ** 2) * (((2 * n) / (n + 1)) - (3 / 4)))
        #     )

        #     # Calculate rad rec sigma
        #     sigma_rr = np.zeros(len(Egrid))
        #     for i, E in enumerate(Egrid):
        #         hnu = I_n + E
        #         sigma_rr[i] = (
        #             (to_lev["stat_weight"] / from_lev["stat_weight"])
        #             * (hnu**2 / (2.0 * sc.EL_MASS * c**2 / sc.EL_CHARGE))
        #             * photo_iz_sigma(I_n, Q_n, hnu)
        #             / E
        #         )
        #     rr_trans["sigma"] = list(1e4 * sigma_rr)

        # Save output
        # data_dir = '/Users/dpower/Documents/01 - PhD/01 - Code/08 - SIKE/atom_data/Carbon'
        data_dir = sike.get_atomic_data_savedir() / element
        with open(
            os.path.join(data_dir, el + "_levels_n.json"), "w", encoding="utf-8"
        ) as f:
            json.dump([l for l in levels], f, ensure_ascii=False, indent=4)
        with open(
            os.path.join(data_dir, el + "_transitions_n.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(
                # [{"E_grid": list(Egrid)}] + [l for l in transitions],
                [l for l in transitions],
                f,
                ensure_ascii=False,
                indent=4,
            )
