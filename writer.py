from pandas import DataFrame

from method_type import MethodType


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
    def change(filename) -> str:
        filename_no_extension = filename.rsplit('.')[0]
        return f'{filename_no_extension}_solucion.txt'

    @staticmethod
    def frame(matrix) -> DataFrame:
        return DataFrame(matrix)

    @staticmethod
    def write_halting(message):
        print(f'{message}')

    def write_solution(self, matrix):
        return self.writing_method(matrix)

    def write_russell_solution(self, matrix):
        print(f'\n{self.frame(matrix)}\n')

    def write_vogel_solution(self, matrix):
        print(f'\n{self.frame(matrix)}\n')

    def write_nw_solution(self, matrix):
        print(f'\n{self.frame(matrix)}\n')
