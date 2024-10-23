# SIKE

> [!NOTE]
> SIKE is a work in progress and documentation will be coming soon. Please contact power8@llnl.gov for any questions running the code in the meantime.

SIKE (**S**crape-off layer **I**mpurities with **K**inetic **E**lectrons) is a simple atomic kinetics solver for impurity species relevant to magnetic fusion plasmas. It is intended to study the effect of non-Maxwellian electron distributions on mean ionisation, radiative loss rates, etc. For a set of atomic state densities $\vec{n}$, it solves the equation

$\frac{d\vec{n}}{dt} = \mathbf{M}\vec{n}$,

where $\mathbf{M}$ is the rate matrix for transitions between states. No collisional radiative assumptions are made, i.e. all states are evolved as opposed to only a few "metastable" states, but effective rate coefficients given a set of evolved and non-evolved states can be computed by SIKE.  

The SIKE model and atomic data is described in more detail in this pre-print: https://arxiv.org/abs/2410.00651. 

> [!NOTE]
> SIKE is a python package, intended to work with python>=3.11. It is recommended to create and activate a virtual environment with conda (e.g. `conda create -n sike python=3.11 && conda activate sike`) prior to following the quickstart steps below in order to avoid any conflicts with existing python environments. 

## Quickstart

1. Clone or download the repository, open a terminal in the top-level directory and run:

    `pip install .`

2. Download and configure the atomic data:

    `python scripts/sike_setup.py`

> [!NOTE]
> If the download fails for any reason, see section on atomic data below for manual setup instructions.

3. In a python script or notebook, run the following code:

    ```python 
    import numpy as np
    import sike

    nx = 100
    Te = np.linspace(1,10,nx)
    ne = 1e20 * np.ones(nx)

    c = sike.SIKERun(ne=ne, Te=Te, element="C")
    ds = c.solve()

    sike.plotting.plot_nz(ds)
    ```
    ![Charge state profiles](https://github.com/Plasdom/SIKE/blob/main/example_plots/C_dist.png)

4. The above example was initialised with plasma temperature and density profiles. To use electron distributions instead:

    ```python
    import numpy as np
    import sike
    nx = 100
    Te = np.linspace(1,10,nx)
    ne = 1e20 * np.ones(nx)

    hot_frac = 0.001
    fe = sike.get_bimaxwellians(n1=hot_frac*ne,
                                n2=(1-hot_frac)*ne,
                                T1 = 50*np.ones(nx),
                                T2 = Te)

    c = sike.SIKERun(fe, element="C")
    ds = c.solve()

    sike.plotting.plot_nz(ds)
    ```
    ![Charge state profiles with bi-Maxwellians](https://github.com/Plasdom/SIKE/blob/main/example_plots/C_dist_bimax.png)

## Atomic data

The atomic data for SIKE has been derived using outputs from the Flexible Atomic Code (M. F. Gu, Canadian Journal of Physics **86**, 2004) and FLYCHK (H. K. Chung et al. High Energy Density Physics **1**, 2005). 

Because the atomic data files are quite large, and not all users will need to download atomic data for all impurity species, it is not bundled along with the package. Instead, run the setup method to retrieve the atomic data from a data repository (zenodo). There are three options:
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

3. If the download fails, atomic data can be downloaded manually from the Zenodo record: https://zenodo.org/records/13864185. Select one or more elements to download, then extract the folders (named "Lithium", "Carbon", etc) to a local directory somewhere. Then pass this directory in the `atomic_data_savedir` argument to the `SIKERun` initialisation, i.e. 
    ```python 
    c = sike.SIKERun(..., atomic_data_savedir="<SAVEDIR>")
    ```

Above,
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
    - Molybdenum ("Mo")
    - Tungsten ("W")
- `<SAVEDIR>` is the location of a directory where the atomic data will be saved. By default, a directory called "sike_atomic_data" will be created here, and the downloaded atomic data placed inside. If empty or `None`, then you will be prompted to input the location. 



