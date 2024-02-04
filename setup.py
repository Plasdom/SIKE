# from setuptools import setup, find_packages
# from requests import get
# import os

# setup(name="SIKE", version="0.1.0", packages=find_packages("sike"))

# # Create directory which will contain atomic data
# ATOMIC_DATA_LOCATION = "."
# atomic_data_dir = os.path.join(ATOMIC_DATA_LOCATION, "atomic_data")
# os.makedirs(atomic_data_dir, exist_ok=True)

# # Download and unzip the atomic data for each element
# ATOM_DATA_BASE_URL = "https://zenodo.org/records/10614179/files/"
# ELEMENT_LIST = ["Hydrogen", "Helium", "Lithium"]
# for element in ELEMENT_LIST:
#     element_filename = element + ".zip"
#     element_filepath = os.path.join(atomic_data_dir, element_filename)
#     url = ATOM_DATA_BASE_URL + element_filename + "?download=1"
#     r = get(url=url)
#     open(element_filepath, "wb").write(r.content)

from setuptools import setup, find_packages
from requests import get
from pathlib import Path
from zipfile import ZipFile
import shutil

setup(name="SIKE", version="0.1.0", packages=find_packages("sike/"))

# Create directory which will contain atomic data
ATOMIC_DATA_LOCATION = Path(".")
atomic_data_dir = ATOMIC_DATA_LOCATION / "atomic_data"
atomic_data_dir.mkdir(exist_ok=True)

# Download and unzip the atomic data for each element
ATOM_DATA_BASE_URL = "https://zenodo.org/records/10614179/files/"
ELEMENT_LIST = [
    "Hydrogen",
    # "Helium",
    # "Lithium",
    # "Beryllium",
    # "Boron",
    # "Carbon",
    # "Nitrogen",
    # "Neon",
    # "Argon",
]
for element in ELEMENT_LIST:
    element_zip_name = element + ".zip"
    element_zip_path = atomic_data_dir / element_zip_name
    url = ATOM_DATA_BASE_URL + element_zip_name + "?download=1"
    r = get(url=url)
    open(element_zip_path, "wb").write(r.content)
    with ZipFile(element_zip_path, "r") as zip_ref:
        zip_ref.extractall(atomic_data_dir)
        element_zip_path.unlink()
        shutil.rmtree(atomic_data_dir / "__MACOSX")
