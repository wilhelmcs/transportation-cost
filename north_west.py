from abc import ABC

from approximation_method import ApproximationMethod, Frac


class NorhtWestMethod(ApproximationMethod, ABC):

    def __init__(self, file, method):
        super().__init__(file=file, method=method)
        # initial north west position
        self.i, self.j = 0, 0

        self.columns_left = self.columns
        self.rows_left = self.rows

    def solve(self):
        while self.__has_rows_and_columns_left():
            self.__choose_cost()
            self.writer.write_solution(self.assign_table)

    def __choose_cost(self) -> None:
        supply_value = self.assign_table[self.i][self.supply_column]
        demand_value = self.assign_table[self.demand_row][self.j]
        if supply_value < demand_value:
            super().assign(supply_value, self.i, self.j)
            self.i += 1
            self.columns_left -= 1
        else:
            super().assign(demand_value, self.i, self.j)
            self.j += 1
            self.rows_left -= 1

    def __has_rows_and_columns_left(self):
        return self.columns_left >= 0 and self.rows_left >= 0
