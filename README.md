# SIKE

To install, run:
`
pip install -e .
`

Next, install the atomic data:
`
python install_atomic_data.py
`

## Atomic data
TODO: Check all this works as described

Because the atomic data files are quite large, and not all users will need to download atomic data for all impurity species, it is not bundled along with the package on git/pypi. Instead, run the script to retrieve the aotmic data from a data repository (zenodo):

`
python scripts/get_atomic_data --atomic_data_savedir <PATH_TO_STORE_ATOMIC_DATA> --elements <LIST_OF_ELEMENTS>
`

For exmaple, to store atomic data for lithium and carbon in a new directory your downloads folder, you would run:

`
python scripts/get_atomic_data --atomic_data_savedir $HOME/Downloads/sike_atomic_data --elements Li C
`

Elements are specified by their symbol. The full list of elements for which data is available is:
- Helium (He)
- Lithium (Li)
- Beryllium (Be)
- Boron (B)
- Carbon (C)
- Oxygen (O)
- Nitrogen (N)
- Neon (Ne)
- Argon (Ar)

