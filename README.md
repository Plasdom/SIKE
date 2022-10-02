# SIKE

SIKE (**S**crape-off layer **I**mpurities with **K**inetic **E**lectrons) solves the density evolution equations for a given set of impurity atomic states. The provided atomic data is generated using [FAC](https://github.com/flexible-atomic-code/fac), but data from any source can be used provided it is formatted correctly (this format is described below). Atomic states are resolved in n, l and (optionally) j. The background plasma electrons are fixed. 

d.power19@imperial.ac.uk

## Dependencies
SIKE is written in python 3 and does not require installation. The following modules are used, which can be installed with a package manager such as pip or conda:
- numpy
- scipy
- numba
- matplotlib
- mpi4py
- petsc4py

For petsc4py, you can either install using your package manager or build alongside a local petsc installation. To do this:
1. Install cython python package
2. Configure petsc with `--with-petsc4py=1`
3. Build petsc
4. Add $PETSC_DIR/$PETSC_ARCH/lib to PYTHONPATH environment variable

## Atomic data format

Two json files are expected for a given element to be modelled:
1. SYMBOL_levels_nl[j].json
2. SYMBOL_transitions_nl[j].json

where SYMBOL is the chemical symbol of the impurity species and "j" specifies whether atomic levels are nl-resolved or j-resolved. For example, j-resolved data for lithium would be contained in files called "Li_levels_nlj.json" and "Li_transitions_nl.json" in the atom_data/Lithium local directory.

### Levels file
The levels file is expected to contain a list of dictionaries which describe each atomic level being modelled. Below is an example showing expected fields in a j-resolved levels file for carbon (the level shown is that of the ground state ) 
```
[
    {
        "id": 0,                # A unique ID for the level
        "element": "C",         # Chemical symbol of the element to which this level belongs
        "nuc_chg": 6,           # Nuclear charge of the element
        "num_el": 6,            # Number of electrons in this level
        "config": "2p2",        # Electronic configuration of valence electrons
        "energy": -1025.00152,  # Energy of this level relative to some highest-energy state (for the FAC data provided this is the bare nucleus)
        "n": 2,                 # Principal quantum number of the level
        "l": 1,                 # Orbital angular momentum quantum number
        "j": 3.5,               # Total angular momentum quantum number (only expected in j-resolved levels file)
        "stat_weight": 8        # (2j + 1)/statistical weight of the level
    },
    # All other levels go here
    ... 
]
```

### Transitions file
The transitions file contains all transitions between atomic levels which are to be modelled. Below is an example showing the expected fields.

```

    {
        "E_grid": [
            # The energy grid (in eV) on which collisional cross-sections are evaluated
            ...
        ]
    },
    {
        "type": "ionization",   # The transition type
        "element": "C",         # The element to which this transition belongs
        "from_id": 1427,        # The unique ID of the initial state
        "to_id": 1478,          # The unique ID of the final state
        "delta_E": 7.6545,      # The difference in energy between initial and final states
        "sigma": [
            # The cross-section (in cm^2) for the given collisional process, evaluated at each point on "E_grid"
            ...
        ]
    },
    {
        "type": "emission",     # Example of an emission transition (i.e. spontaneous deexcitation)
        "element": "C",
        "from_id": 551,
        "to_id": 543,
        "delta_E": 367.477326,
        "rate": 148.0718        # Emission rate (in s^-1)
    },
    # All other transitions go here
    ...
```
It's important to note that, currently, possitble transition types are 
- "ionization"
- "excitation"
- "radiative recombination"
- "emission"
- "autoionization"
Of these, ionization, excitation and radiative recombination should include a "sigma" field, while autionization and emission should include a "rate" field. 

For ionization and excitation, SIKE will automatically handly to inverse processes of three-body recombination and collisional de-excitation. 
