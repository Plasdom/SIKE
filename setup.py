from setuptools import setup, find_packages
from requests import get
from pathlib import Path
from zipfile import ZipFile
import shutil

from sike import constants as c

setup(name="SIKE", version="0.1.0", packages=find_packages())

# Create directory which will contain atomic data
atomic_data_dir = c.ATOMIC_DATA_LOCATION / "atomic_data"
atomic_data_dir.mkdir(exist_ok=True)

# Download and unzip the atomic data for each element
for element in c.ELEMENT_LIST:
    element_zip_name = element + ".zip"
    element_zip_path = atomic_data_dir / element_zip_name
    url = c.ATOM_DATA_BASE_URL + element_zip_name + "?download=1"
    r = get(url=url)
    open(element_zip_path, "wb").write(r.content)
    with ZipFile(element_zip_path, "r") as zip_ref:
        zip_ref.extractall(atomic_data_dir)
        element_zip_path.unlink()
        shutil.rmtree(atomic_data_dir / "__MACOSX")
