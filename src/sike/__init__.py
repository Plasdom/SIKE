from sike.core import *
from sike.plasma_utils import get_bimaxwellians, get_maxwellians
from sike.setup import setup
from sike.constants import *
import sike.plotting

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    # Package hasn't been installed
    pass
