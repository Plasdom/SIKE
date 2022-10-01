# SIKE

SIKE (Scrape-off layer Impurities with Kinetic Electrons) solves the density evolution equations for a given set of impurity atomic states.

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