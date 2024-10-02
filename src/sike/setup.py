"""Setup script for obtaining atomic data and telling SIKE where to find it."""

from requests import get
from zipfile import ZipFile
import shutil
import os
from pathlib import Path

from sike import constants as c


def verify_elements(elements: list[str]) -> list[str]:
    """Verify that data exists for a list of elements (also convert full element names to symbols).

    :param elements: List of elements
    :raises AttributeError: If an element is provided for which data does not exist.
    """
    all_elements = list(c.SYMBOL2ELEMENT.keys())

    verified_elements = []
    for el in elements:
        if el in all_elements:
            verified_elements.append(el)
        else:
            raise AttributeError(
                "Element "
                + str(el)
                + " not found in list of species for which atomic data exists (i.e. it is not in the SYMBOL2ELEMENT dict in constants.py). Possible elements are:\n"
                + str(all_elements)
            )

    return verified_elements


def verify_savedir(savedir: str | Path) -> Path:
    """Verify that the specified savedir exists.

    :param savedir: User-speicified directory where atomic data will be saved
    :raises FileNotFoundError: If savedir does not exist
    :return: Verified savedir
    """
    verified_savedir = Path(savedir)
    if not verified_savedir.exists():
        raise FileNotFoundError("The directory " + savedir + " does not exist.")
    return verified_savedir


def setup(elements: list[str] | None = None, savedir: str | Path | None = None) -> None:
    """Setup SIKE by downloading atomic data from https://zenodo.org/records/13864185/.

    :param elements: List of elements (using their chemical symbol) to download data for. If None, then all available data will be downloaded. Currently, the available species are:
        - H
        - He
        - Li
        - Be
        - B
        - C
        - N
        - O
        - Ne
        - Ar
    :param savedir: Where to save the atomic data files. This may be a string or Path object, or None, in which case you will be prompted to enter the save location.
    """

    # Get the list of elements
    if elements is None:
        elements = c.SYMBOL2ELEMENT.keys()
    elif len(elements) == 0:
        print(
            "An empty elements list has been passed, so no atomic data will be downloaded."
        )
    elements = verify_elements(elements)
    elements_full = [c.SYMBOL2ELEMENT[el] for el in elements]

    # Get the savedir, check it exists and create a "sike_atomic_data" folder underneath
    if savedir is None:
        savedir = input(
            "Please enter the absolute filepath of the directory where atomic data will be saved (a folder named 'sike_atomic_data' will be created underneath):",
        )
    savedir = verify_savedir(savedir)
    sike_data_savedir = savedir / c.ATOMIC_DATA_LOCATION
    sike_data_savedir.mkdir(exist_ok=True)
    sike_data_savedir = sike_data_savedir.resolve()

    # Add this location to a config file
    config_filepath = Path(os.getenv("HOME")) / c.CONFIG_FILENAME
    if config_filepath.exists():
        print("Existing config file exists, which will be overwritten.")
    with open(config_filepath, "w+") as f:
        f.write(str(sike_data_savedir))

    # Download data for the specified elements
    for element in elements_full:
        print(f"Downloading atomic data for {element}...")
        element_zip_name = element + ".zip"
        element_zip_path = sike_data_savedir / element_zip_name
        url = c.ATOMIC_DATA_BASE_URL + element_zip_name + "?download=1"
        r = get(url=url)
        open(element_zip_path, "wb").write(r.content)
        with ZipFile(element_zip_path, "r") as zip_ref:
            zip_ref.extractall(sike_data_savedir)
            element_zip_path.unlink()
            shutil.rmtree(sike_data_savedir / "__MACOSX")
    print("Finished downloading atomic data, saved to: {}".format(sike_data_savedir))
    print(
        "A config file containing the location of this directory has been saved to {}".format(
            config_filepath
        )
    )

    return sike_data_savedir
