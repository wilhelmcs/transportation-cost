from abc import ABC
from typing import Tuple, Any

import numpy as np

from approximation_method import ApproximationMethod


class VogelMethod(ApproximationMethod, ABC):

    def __init__(self, file, method):
        super().__init__(file=file, method=method)
        self.__add_diff_column()
        self.__add_diff_row()

    def solve(self) -> None:
        while super().has_rows_and_columns_left():
            self.__update_diff_row()
            self.__update_diff_column()
            self.__choose_cost()
            self.writer.write_solution(self.assign_table)

    def __add_diff_column(self) -> None:
        dfi_column = np.zeros((self.rows, 1))
        self.cost_table = np.append(self.cost_table, values=dfi_column, axis=1)

    def __add_diff_row(self) -> None:
        dcj_row = np.zeros((1, self.columns + 1))
        self.cost_table = np.append(self.cost_table, values=dcj_row, axis=0)

    def __update_diff_column(self) -> None:
        biggest_diff, biggest_col = -np.inf, -1
        consumers = np.transpose(self.cost_table[:self.demand_row, :self.supply_column])
        # for each column find the lowest difference in terms of cost
        for col, cost in enumerate(consumers):
            # flag deleted columns
            if col in self.deleted_cols:
                self.cost_table[self.rows][col] = -np.inf
                continue
            diff = self.minimum_diff(cost, omit=self.deleted_rows)
            if diff > biggest_diff:
                biggest_diff = diff
                biggest_col = col
            self.cost_table[self.rows][col] = diff
        # set the biggest diff and it's  index in corner
        self.cost_table[self.rows][self.supply_column] = (biggest_diff, biggest_col)

    def __update_diff_row(self) -> None:
        biggest_diff, biggest_row = -np.inf, -1
        suppliers = self.cost_table[:self.demand_row, :self.supply_column]
        # for each row find the lowest difference in terms of cost
        for row, costs in enumerate(suppliers):
            # flag deleted rows
            if row in self.deleted_rows:
                self.cost_table[row][self.columns] = -np.inf
                continue
            diff = self.minimum_diff(costs, omit=self.deleted_cols)
            if diff > biggest_diff:
                biggest_diff = diff
                biggest_row = row
            self.cost_table[row][self.columns] = diff
        # set the biggest diff and it's  index in corner
        self.cost_table[self.demand_row][self.columns] = (biggest_diff, biggest_row)

    @staticmethod
    def minimum_diff(costs, omit) -> Any:
        # find diff between two lowest elements
        lowest, second_lowest = np.inf, np.inf
        for i, c in enumerate(costs):
            # don't use omitted rows or columns
            if i in omit:
                continue
            elif c < lowest:
                second_lowest = lowest
                lowest = c
            elif c < second_lowest:
                second_lowest = c
            else:
                continue
        # case in which there was only 1 element left
        if second_lowest == np.inf:
            return lowest
        else:
            return second_lowest - lowest

    def __choose_cost(self) -> None:
        maximum_supply_diff, i = self.cost_table[self.demand_row][self.columns]
        maximum_demand_diff, j = self.cost_table[self.rows][self.supply_column]
        if maximum_supply_diff > maximum_demand_diff:
            j = self.__minimum_index_in_row(i)
            super().assign(*self.best_value_at(i, j))
        elif maximum_demand_diff > maximum_supply_diff:
            i = self.__minimum_index_in_column(j)
            super().assign(*self.best_value_at(i, j))
        else:
            super().assign(*self.best_value_at(i, j))

    def __minimum_index_in_row(self, i: int):
        costs = self.cost_table[i][:self.supply_column]
        costs_left = np.delete(costs, list(self.deleted_cols))
        lowest_cost = np.min(costs_left)
        j = np.where(costs == lowest_cost)[0][0]
        return j

    def __minimum_index_in_column(self, j: int):
        costs = self.cost_table[:, j][:self.demand_row]
        costs_left = np.delete(costs, list(self.deleted_rows))
        lowest_cost = np.min(costs_left)
        i = np.where(costs == lowest_cost)[0][0]
        return i

    def best_value_at(self, i: int, j: int) -> Tuple[Any, Any, Any]:
        # determine lowest value between supply & demand
        demand_value = self.assign_table[self.demand_row][j]
        supply_value = self.assign_table[i][self.supply_column]
        best = min(demand_value, supply_value)
        if demand_value < supply_value:
            # mark the column as unavailable from now on
            self.deleted_cols.add(j)
        elif supply_value < demand_value:
            # mark the row as unavailable from now on
            self.deleted_rows.add(i)
        else:
            # mark both the row & column as unavailable from now on
            self.deleted_rows.add(i)
            self.deleted_cols.add(j)
        return best, i, j
