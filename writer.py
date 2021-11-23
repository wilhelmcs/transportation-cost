from typing import Tuple, List

import numpy as np
from pandas import DataFrame

Position = Tuple[int, int]


class Writer:

    def __init__(self, filename: str):
        self.filename = self.change(filename)

    @staticmethod
    def change(filename: str) -> str:
        """
        Formats the given filename.txt to filename_solucion.txt

        :param filename:
        :return: formatted filename
        """

        filename_no_extension = filename.rsplit('.')[0]
        return f'{filename_no_extension}_solucion.txt'

    @staticmethod
    def frame_assignment_table(matrix: np.ndarray) -> DataFrame:
        """
        Creates a dataframe of the assignment table into
            D1 D2 D3 Supply
        S1  X  X  -  X
        S2  -  -  X  X

        :param matrix: ndarray with assigment values
        :return: dataframe with replaced values & headers
        """

        matrix[matrix == 0.0] = '-'
        n, m = matrix.shape
        row_headers = [f'S{i}' for i in range(1, n)] + ['Demand']
        columns_headers = [f'D{j}' for j in range(1, m)] + ['Supply']
        return DataFrame(matrix, row_headers, columns_headers)

    @staticmethod
    def frame_transportation_table(matrix: np.ndarray) -> DataFrame:
        """
        Creates a dataframe of the transportation table into
               V1=y V2=y V3=y
        U1=x    X    X    -
        U2=x    -    -    X

        :param matrix: ndarray with dual variables and indicators
        :return: dataframe with replaced values & headers
        """

        # is none doesn't work for comparing values
        matrix[matrix == None] = '-'
        n, m = matrix.shape
        v_row = matrix[-1]
        u_column = matrix[:, -1]
        matrix = np.delete(matrix, -1, axis=1)
        matrix = np.delete(matrix, -1, axis=0)
        column_headers = [f'V{i}={v_row[i - 1]}' for i in range(1, m)]
        row_headers = [f'U{j}={u_column[j - 1]}' for j in range(1, n)]
        return DataFrame(matrix, row_headers, column_headers)

    def write_halting(self, message: str) -> None:
        """
        Writes to console and the file a terminating error message
        :param message: string to write
        """

        print(message)
        self.write_to_file(message)

    def write_to_file(self, text: str) -> None:
        """
        Generic function used to write into the end of the file
        :param text: string to write
        """

        with open(self.filename, 'a') as f:
            f.write(text)
        f.close()

    def write_initial_cost(self, cost: int) -> None:
        """
        Indicates to the user what was the initial transportation cost
        :param cost: assignment sum to write
        """

        print(f'[Initial transportation cost] = {cost}\n\n')
        self.write_to_file(f'[Initial transportation cost] = {cost}\n\n')

    def write_current_cost(self, cost: int) -> None:
        """
        Indicates to the user what is the current transportation cost
        :param cost: assignment sum to write
        """
        self.write_to_file(f'[Current transportation cost] = {cost}\n\n')

    def write_optimal_cost(self, cost: int) -> None:
        """
        Indicates to the user what was the last calculated cost (most optimal)
        :param cost: assignment sum to write
        """
        print(f'[Optimal transportation cost] = {cost}\n\n')
        self.write_to_file(f'[Optimal transportation cost] = {cost}\n\n')

    def write_transportation_iteration(self, iteration: str,
                                       transportation_matrix: np.ndarray,
                                       assignment_matrix: np.ndarray,
                                       final=False):
        """
        Appends to the file the current state of the transportation
        & assigment table

        :param final: boolean in case the last iteration occurs
        :param iteration: string of the current iteration
        :param transportation_matrix: current transportation table
        :param assignment_matrix: current assigment table
        """
        transportation = self.frame_transportation_table(transportation_matrix)
        assignment = self.frame_assignment_table(assignment_matrix)
        state = f'{iteration}\n' + \
                f'Transportation Table\n{transportation}\n\n' + \
                f'Assignment Table\n{assignment}\n'
        if final:
            print(state)
        self.write_to_file(state)

    def write_loop(self, loop: List[Position],
                   entering: Tuple,
                   leaving: Tuple):
        """
        Indicates what's the loop in a legible way:
        start -> loop -> end

        :param loop: list of indices where a loop is formed
        :param entering: first index in the loop
        :param leaving:  index that has the lowest
                        assigment in the entire loop
        """

        loop_path = [f'{pos} -> ' for pos in loop]
        loop_path[-1] = loop_path[-1].replace('->', '')
        format_loop = '[Loop] = ' + ''.join(loop_path)
        format_entering = f'[Entering pos] = {entering}\n'
        format_leaving = f'[Leaving pos] = {leaving}\n'
        self.write_to_file(f'{format_entering}{format_leaving}{format_loop}\n\n')

    def write_initial_solution(self, matrix: np.ndarray,
                               demand: np.ndarray,
                               supply: np.ndarray) -> None:
        """
        Indicates what's the initial assigment table with the demand and supply column added

        :param matrix: assigment table
        :param demand: demand row of the cost table
        :param supply: supply column of the cost table
        """

        matrix[-1] = demand
        matrix[:, -1] = supply
        dataframe = self.frame_assignment_table(matrix)
        print(f'Inititial Assignment table\n{dataframe}\n\n')
        self.write_to_file(f'Assignment table\n{dataframe}\n\n')
