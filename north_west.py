from abc import ABC

from approximation_method import ApproximationMethod


class NorthWestMethod(ApproximationMethod, ABC):

    # current position for assigning cost
    i: int
    j: int

    def __init__(self, file):
        super().__init__(file=file)
        # initial north west position
        self.i, self.j = 0, 0

    def solve(self) -> None:
        """
        Finds the minimum value of the north west corner in the
        cost table and continues moving right or down depending
        of the best value

        """

        while self.has_rows_and_columns_left():
            self.choose_cost()
        self.writer.write_initial_solution(self.assign_table,
                                           demand=self.cost_table[self.demand_row],
                                           supply=self.cost_table[:, self.supply_column])
        self.writer.write_initial_cost(self.total_cost())
        self.improve()

    def choose_cost(self) -> None:
        """
        Assigns the current position with the best value &
        updates the current assigning position
        """

        previous_rows = len(self.deleted_rows)
        previous_columns = len(self.deleted_cols)

        best = self.best_value_at(self.i, self.j)
        self.assign(*best)

        current_rows = len(self.deleted_rows)
        current_columns = len(self.deleted_cols)

        # j += 1 or 0 depending if a column was added
        self.j += current_columns - previous_columns
        # i += 1 or 0 depending if a row was added
        self.i += current_rows - previous_rows
