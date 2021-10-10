#!/usr/bin/python3

import argparse
import textwrap

from north_west import  NorhtWestMethod
from vogel import  VogelMethod
from russell import RussellMethod

# Argument handler
parser = argparse.ArgumentParser(
    prog='simplex.py',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description=textwrap.dedent('''
    '''))

parser.add_argument('method', metavar='approximation method', type=int,
                    help=textwrap.dedent('''
                    Approximation method used to solve the problem.
                    1 = NORTH WEST APPROXIMATION METHOD
                    2 = VOGEL APPROXIMATION METHOD
                    3 = RUSSELL APPROXIMATION METHOD    
            '''))

parser.add_argument('file', metavar='file.txt', type=argparse.FileType("r"),
                    help='Text file with the optimization problem in the correct format')

args = parser.parse_args()


def solve():
    if args.method == 1:
        method = NorhtWestMethod(file=args.file, method=args.method)
        method.solve()


if __name__ == "__main__":
    solve()
