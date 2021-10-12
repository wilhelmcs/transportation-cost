from abc import ABC

from approximation_method import ApproximationMethod


class NorthWestMethod(ApproximationMethod, ABC):

    # current position for assigning cost
    i: int
    j: int

    def __init__(self, file, method):
        super().__init__(file=file, method=method)
        # initial north west position
        self.i, self.j = 0, 0

    def solve(self):
        while super().has_rows_and_columns_left():
            self.__choose_cost()
            self.writer.write_solution(self.assign_table)

    def __choose_cost(self) -> None:
        # determine lowest value between supply & demand
        supply_value = self.assign_table[self.i][self.supply_column]
        demand_value = self.assign_table[self.demand_row][self.j]
        if supply_value < demand_value:
            super().assign(supply_value, self.i, self.j)
            # mark the row as unavailable from now on
            self.deleted_rows.add(self.i)
            self.i += 1
        elif demand_value < supply_value:
            super().assign(demand_value, self.i, self.j)
            # mark the row as unavailable from now on
            self.deleted_cols.add(self.j)
            self.j += 1
        else:
            super().assign(supply_value, self.i, self.j)
            # mark both the row & column as unavailable from now on
            self.deleted_rows.add(self.i)
            self.deleted_cols.add(self.j)
            self.i += 1
            self.j += 1
