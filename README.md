# SIKE

To install, run:
`
pip install -e .
`

## Atomic data
TODO: Check all this works as described

Because the atomic data files are quite large, and not all users will need to download atomic data for all impurity species, it is not bundled along with the package on git/pypi. Instead, run the setup method to retrieve the atomic data from a data repository (zenodo). There are two options:
1. Run the script "scripts/sike_setup.py":
    ```
    python scripts/sike_setup.py --atomic_data_savedir <SAVEDIR> --elements <ELEMENTS>
    ```
    e.g. `python scripts/sike_setup.py --savedir /Users/username/Downloads/ --elements Li C`.

2. or, in a Python script/notebook, run:
    ```python
    import sike 
    sike.setup(elements=<ELEMENTS>, savedir=<SAVEDIR>)
    ```
    e.g. `sike.setup(elements=["Li", "C"], savedir="/Users/username/Downloads/")`

Here,
- `<ELEMENTS>` is a list of elements (specified by symbol) to download data for. If `<ELEMENTS>` is empty or `None`, then all atomic data will be downloaded. The full list of elements for which data is available is:
    - Hydrogen ("H")
    - Helium ("He")
    - Lithium ("Li")
    - Beryllium ("Be")
    - Boron ("B")
    - Carbon ("C")
    - Oxygen ("O")
    - Nitrogen ("N")
    - Neon ("Ne")
    - Argon ("Ar")
- `<SAVEDIR>` is the location of a directory where the atomic data will be saved. A directory called "sike_atomic_data" will be created here, and the downloaded atomic data placed inside. If empty or `None`, then you will be prompted to input the location. 



