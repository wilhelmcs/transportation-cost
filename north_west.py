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
        while self.has_rows_and_columns_left():
            self.choose_cost()
            self.writer.write_solution(self.assign_table)
        self.improve()

    def choose_cost(self) -> None:

        previous_rows = len(self.deleted_rows)
        previous_columns = len(self.deleted_cols)

        best = self.best_value_at(self.i, self.j)
        self.assign(*best)

        current_rows = len(self.deleted_rows)
        current_columns = len(self.deleted_cols)

        self.j += current_columns - previous_columns
        self.i += current_rows - previous_rows


