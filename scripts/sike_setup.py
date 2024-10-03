import sys
import argparse

import sike

parser = argparse.ArgumentParser(description="Download atomic data for SIKE.")
parser.add_argument(
    "--savedir",
    type=str,
    required=False,
    help="Directory where atomic data will be saved.",
    default=".",
)
parser.add_argument(
    "--elements",
    nargs="+",
    required=False,
    default=["H", "He", "Li", "Be", "B", "C", "N", "O", "Ne", "Ar", "Mo", "W"],
    help="List of elements whose atomic data will be downloaded. Options are: 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'Ne', 'Ar', 'Mo', 'W'. By default, all will be downloaded.",
)


if __name__ == "__main__":
    args = parser.parse_args(sys.argv[1:])
    sike.setup(savedir=args.savedir, elements=args.elements)
