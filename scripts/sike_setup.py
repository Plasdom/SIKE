import sys
import argparse

import sike

parser = argparse.ArgumentParser(description="Download atomic data for SIKE.")
parser.add_argument(
    "--savedir",
    type=str,
    required=False,
    help="Directory where atomic data will be saved. Default will be the current directory.",
    default=".",
)
parser.add_argument(
    "--elements",
    nargs="+",
    required=False,
    default=[v for v in sike.constants.ELEMENT2SYMBOL.values()],
    help="List of elements whose atomic data will be downloaded. Options are: "
    + str([v for v in sike.constants.ELEMENT2SYMBOL.values()])
    + ". By default, all will be downloaded.",
)


if __name__ == "__main__":
    args = parser.parse_args(sys.argv[1:])
    sike.setup(savedir=args.savedir, elements=args.elements)
