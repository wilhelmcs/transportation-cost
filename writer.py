import numpy as np

from pandas import DataFrame

from method_type import MethodType

from typing import Tuple, List


class Writer:
    method: MethodType

    WRITE_MAP: dict

    def __init__(self, filename: str, method_type: MethodType):
        self.filename = self.change(filename)
        self.WRITE_MAP = {
            MethodType.VOGEL_METHOD: self.write_vogel_solution,
            MethodType.RUSSELL_METHOD: self.write_russell_solution,
            MethodType.NORTH_WEST_METHOD: self.write_nw_solution,
        }
        self.writing_method = self.WRITE_MAP.get(method_type)

    @staticmethod
    def change(filename: str) -> str:
        filename_no_extension = filename.rsplit('.')[0]
        return f'{filename_no_extension}_solucion.txt'

    @staticmethod
    def frame_assignment_table(matrix: np.ndarray) -> DataFrame:
        matrix[matrix == 0.0] = "-"
        n, m = matrix.shape
        row_headers = [f'S{i}' for i in range(1, n)] + ["Demand"]
        columns_headers = [f'D{j}' for j in range(1, m)] + ["Supply"]
        return DataFrame(matrix, row_headers, columns_headers)

    @staticmethod
    def frame_transportation_table(matrix: np.ndarray) -> DataFrame:
        matrix[matrix == None] = "-"
        n, m = matrix.shape
        v_row = matrix[-1]
        u_column = matrix[:, -1]
        matrix = np.delete(matrix, -1, axis=1)
        matrix = np.delete(matrix, -1, axis=0)
        row_headers = [f'V{i}={v_row[i - 1]}' for i in range(1, n)]
        columns_headers = [f'U{j}={u_column[j - 1]}' for j in range(1, m)]
        return DataFrame(matrix, row_headers, columns_headers)

    @staticmethod
    def write_halting(message) -> None:
        print(f'{message}')

    def write_to_file(self, text: str) -> None:
        with open(self.filename, 'a') as f:
            f.write(text)
        f.close()

    def write_initial_cost(self, cost: int) -> None:
        print(f'[Initial transportation cost] = {cost}\n\n')

    def write_current_cost(self, cost: int) -> None:
        print(f'[Current transportation cost] = {cost}\n\n')

    def write_optimal_cost(self, cost: int) -> None:
        print(f'[Optimal transportation cost] = {cost}\n\n')

    def write_transportation_iteration(self, iteration: int,
                                       transportation_matrix: np.ndarray,
                                       assignment_matrix: np.ndarray):
        iteration = f'[Iteration] = {iteration}'
        transportation = self.frame_transportation_table(transportation_matrix)
        assignment = self.frame_assignment_table(assignment_matrix)
        print(f'{iteration}\n'
              f'Transportation Table\n{transportation}\n\n'
              f'Assignment Table\n{assignment}\n')

    def write_loop(self, loop: List[Tuple], entering: Tuple, leaving: Tuple):
        loop = [f'{pos} -> ' for pos in loop]
        loop[-1] = loop[-1].replace('->', '')
        loop = '[Loop] = ' + ''.join(loop)
        entering = f'[Entering pos] = {entering}\n'
        leaving = f'[Leaving pos] = {leaving}\n'
        print(f'{entering}{leaving}{loop}\n\n')

    def write_initial_solution(self, matrix: np.ndarray, demand: np.ndarray, supply: np.ndarray) -> None:
        matrix[-1] = demand
        matrix[:, -1] = supply
        return self.writing_method(matrix)

    def write_russell_solution(self, matrix) -> None:
        print(f'{self.frame_assignment_table(matrix)}\n\n')

    def write_vogel_solution(self, matrix) -> None:
        print(f'{self.frame_assignment_table(matrix)}\n\n')

    def write_nw_solution(self, matrix) -> None:
        print(f'{self.frame_assignment_table(matrix)}\n\n')
