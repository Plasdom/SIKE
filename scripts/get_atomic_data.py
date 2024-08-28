from setuptools import setup, find_packages
from requests import get
from zipfile import ZipFile
import shutil
import sys
import argparse
from pathlib import Path

from sike import constants as c


parser = argparse.ArgumentParser(description="Download atomic data for SIKE.")
parser.add_argument(
    "--atomic_data_savedir",
    type=str,
    required=False,
    help="Directory where atomic data will be saved.",
    default=".",
)
parser.add_argument(
    "--elements",
    nargs="+",
    required=False,
    default=[
        "Hydrogen",
        "Helium",
        "Lithium",
        "Beryllium",
        "Boron",
        "Carbon",
        "Nitrogen",
        "Oxygen",
        "Neon",
        "Argon",
    ],
    help="List of elements whose atomic data will be downloaded. Options are: 'Hydrogen', 'Helium', 'Lithium', 'Beryllium', 'Boron', 'Carbon', 'Nitrogen', 'Oxygen', 'Neon', 'Argon'. By default, all will be downloaded.",
)


# TODO: Take install location from input argument, set environment variable appropriately
# TODO: Run this script at install?
# TODO: This link might help: https://docs.python.org/2/distutils/setupscript.html#distutils-additional-files, or this: https://stackoverflow.com/questions/55330280/including-folder-and-files-inside-a-python-package
def main(args):
    # Create directory which will contain atomic data
    atomic_data_dir = Path(args.atomic_data_savedir) / "atomic_data"
    atomic_data_dir.mkdir(exist_ok=True)

    # Verify that all provided elements are in the elements list
    for element in args.elements:
        if not element in c.ELEMENT_LIST:
            raise AttributeError(
                f"Element {element} was not found in the list of elements for which atomic data currently exists."
            )

    # Download and unzip the atomic data for each element
    for element in args.elements:
        print(f"Downloading atomic data for {element}...")
        element_zip_name = element + ".zip"
        element_zip_path = atomic_data_dir / element_zip_name
        url = c.ATOM_DATA_BASE_URL + element_zip_name + "?download=1"
        r = get(url=url)
        open(element_zip_path, "wb").write(r.content)
        with ZipFile(element_zip_path, "r") as zip_ref:
            zip_ref.extractall(atomic_data_dir)
            element_zip_path.unlink()
            shutil.rmtree(atomic_data_dir / "__MACOSX")
    print("Finished downloading atomic data.")

    print(f"Atomic data saved to {atomic_data_dir.absolute()}")


if __name__ == "__main__":
    args = parser.parse_args(sys.argv[1:])
    main(args)
