#!/usr/bin/python3

import argparse
import textwrap

from method_type import MethodType
from north_west import NorthWestMethod
from russell import RussellMethod
from vogel import VogelMethod

# Argument handler
parser = argparse.ArgumentParser(
    prog='transporte.py',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description=textwrap.dedent('''
    NAME
        Python 3 Transportation Cost Optimization Algorithm
    SYNOPSIS
        transporte.py [-h] method file.txt
    DESCRIPTION
        Solves a given transportation problem in a file.
        Currently the program doesn't solve degenerated problems.
    METHOD
        Approximation method used to solve the problem.
        1 = NORTH WEST APPROXIMATION METHOD
        2 = VOGEL APPROXIMATION METHOD
        3 = RUSSELL APPROXIMATION METHOD    
    FILE
        Text file with the transportation problem in the correct format.
        The file has the following structure, separated by commas.
        Supply column, demand row, transportation costs.
        For example if the problem comes with the following form:
           
                  D1  D2  D3  Supply
              S1   8    6	10   2000
              S2   10   4    9   2500
            Demand 1500 2000 1000
        
        The file must come as the next example:
        2000,2500
        1500,2000,1000
        8,6,10
        10,4,9 '''))

parser.add_argument('method', metavar='method', type=int,
                    help=textwrap.dedent('''
                    Approximation method used to solve the problem.'''))

parser.add_argument('file', metavar='file.txt', type=argparse.FileType("r"),
                    help='Text file with the transportation problem in the correct format')

args = parser.parse_args()


def main():
    solving_methods = {
        MethodType.NORTH_WEST_METHOD: NorthWestMethod,
        MethodType.VOGEL_METHOD: VogelMethod,
        MethodType.RUSSELL_METHOD: RussellMethod
    }
    desired_method = MethodType(args.method)
    solver = solving_methods.get(desired_method)(file=args.file)
    solver.solve()


if __name__ == "__main__":
    main()
