from abc import ABC

from approximation_method import ApproximationMethod
import numpy as np


class RussellMethod(ApproximationMethod, ABC):
    russell_table: np.ndarray

    max_pos: tuple

    def __init__(self, file):
        super().__init__(file=file)
        self.max_pos = (-1, -1)
        self.__create_russell_table()

    def solve(self) -> None:
        """
        Finds the maximum value of the u column and v row
        and updates it with u+v-c while there is at least
        one column and one row left
        """

        while super().has_rows_and_columns_left():
            self.__update_russell_table()
            self.choose_cost()

        # russell table is no longer needed
        del self.russell_table
        self.writer.write_initial_solution(self.assign_table,
                                           demand=self.cost_table[self.demand_row],
                                           supply=self.cost_table[:, self.supply_column])
        self.writer.write_initial_cost(self.total_cost())
        self.improve()

    def __create_russell_table(self) -> None:
        """
        Fills a table of zeros with the same shape as the cost table
        """

        self.russell_table = np.zeros(self.cost_table.shape, dtype=object)

    def __update_russell_table(self) -> None:
        """
        Updates the value of each unassigned index for its
        corresponding value u+v-c
        """

        self.__update_max_u_column()
        self.__update_max_v_row()
        max_value = -np.inf
        for i, j in self.unassigned_indices:
            u = self.russell_table[i][self.u_column]
            v = self.russell_table[self.v_row][j]
            c = self.cost_table[i][j]
            russell_value = u + v - c
            if russell_value > max_value:
                max_value = russell_value
                self.max_pos = (i, j)
            self.russell_table[i][j] = russell_value

    def __update_max_u_column(self) -> None:
        """
        Calculate the greatest value of all the rows and allocates in
        the u column
        """

        suppliers = self.cost_table[:self.demand_row, :self.supply_column]
        # for each row find the greatest value in terms of cost
        for row, costs in enumerate(suppliers):
            if row in self.deleted_rows:
                u = -np.inf
            else:
                u = max(costs)
            self.russell_table[row][self.u_column] = u

    def __update_max_v_row(self) -> None:
        """
        Calculate the greatest value of all the columns and allocates in
        the v row
        """

        consumers = np.transpose(self.cost_table[:self.demand_row, :self.supply_column])
        # for each column find the greatest value in terms of cost
        for col, costs in enumerate(consumers):
            if col in self.deleted_cols:
                v = -np.inf
            else:
                v = max(costs)
            self.russell_table[self.v_row][col] = v

    def choose_cost(self) -> None:
        """
        Discards the current max position and assigns the best
        value (supply or demand) in the same position
        """

        self.russell_table[self.max_pos] = -np.inf
        best = super().best_value_at(*self.max_pos)
        self.assign(*best)
