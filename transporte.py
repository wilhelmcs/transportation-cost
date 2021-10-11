#!/usr/bin/python3

import argparse
import textwrap

from method_type import MethodType
from north_west import NorthWestMethod
from russell import RussellMethod
from vogel import VogelMethod

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
    solving_methods = {
        MethodType.NORTH_WEST_METHOD: NorthWestMethod,
        MethodType.VOGEL_METHOD: VogelMethod,
        MethodType.RUSSELL_METHOD: RussellMethod
    }
    desired_method = MethodType(args.method)
    solver = solving_methods.get(desired_method)(file=args.file, method=desired_method)
    solver.solve()


if __name__ == "__main__":
    solve()
