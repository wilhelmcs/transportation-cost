from method_type import MethodType
from pandas import DataFrame


def change(filename) -> str:
    filename_no_extension = filename.rsplit('.')[0]
    return f'{filename_no_extension}_solucion.txt'


class Writer:
    method: MethodType

    _write_map: dict

    def __init__(self, filename: str, method: int):
        self.filename = change(filename)
        self.method = MethodType(method)
        self._write_map = {
            MethodType.VOGEL_METHOD: self.write_vogel_solution,
            MethodType.RUSSELL_METHOD: self.write_russell_solution,
            MethodType.NORTH_WEST_METHOD: self.write_nw_solution,
        }

    def write_solution(self, matrix):
        return self._write_map.get(self.method)(matrix)

    def write_russell_solution(self, matrix):
        print("russell")

    def write_vogel_solution(self, matrix):
        print("vogel")

    def write_nw_solution(self, matrix):
        print(f'\n{self.frame(matrix)}\n')

    def frame(self, matrix) -> DataFrame:
        return DataFrame(matrix)
