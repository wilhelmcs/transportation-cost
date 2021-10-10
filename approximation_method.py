from abc import abstractmethod
from fractions import Fraction as Frac
from writer import Writer

import numpy as np


class ApproximationMethod:
    cost_table: np.ndarray
    assign_table: np.ndarray

    writer: Writer

    rows: int
    columns: int

    demand_row: int
    supply_column: int

    def __init__(self, file, method):
        self.writer = Writer(method=method, filename=file.name)
        self.__create_cost_table(file)
        self.__balance_cost_table()
        self.__create_assign_table()

    def __create_cost_table(self, file):
        # obtain first row of txt file and make it supply column
        supply = np.loadtxt(file, max_rows=1, delimiter=",", comments="\\n") + Frac()
        supply = supply.reshape((-1, 1))

        # obtain second row of txt file and make it demand row
        demand = np.loadtxt(file, max_rows=1, delimiter=",", comments="\\n") + Frac()
        demand = np.append(demand, "X").reshape((1, -1))

        # finally generate the cost_table with costs + demand + supply column
        costs = np.loadtxt(file, delimiter=",", comments="\\n") + Frac()
        self.cost_table = np.append(costs, values=supply, axis=1)
        self.cost_table = np.append(self.cost_table, values=demand, axis=0)

        self.rows, self.columns = self.cost_table.shape
        self.demand_row, self.supply_column = self.rows - 1, self.columns - 1

        file.close()

    def __balance_cost_table(self):
        demand_sum = np.sum(self.cost_table[self.demand_row][:-1])
        supply_sum = np.sum(self.cost_table[:, self.supply_column][:-1])
        diff = abs(demand_sum-supply_sum)
        if demand_sum < supply_sum:
            self.__insert_fictional_demand(fictional_value=diff)
        elif supply_sum > demand_sum:
            self.__insert_fictional_supply(fictional_value=diff)
        else:
            pass

    def __insert_fictional_demand(self, fictional_value):
        fictional_demand = [self.demand_row * [0] + [fictional_value]]
        self.cost_table = np.insert(self.cost_table, -1, values=fictional_demand, axis=1)
        self.columns += 1
        self.supply_column += 1

    def __insert_fictional_supply(self, fictional_value):
        fictional_supply = self.supply_column * [0] + [fictional_value]
        self.cost_table = np.insert(self.cost_table, -1, values=fictional_supply, axis=0)
        self.rows += 1
        self.demand_row += 1

    def __create_assign_table(self):
        # fill table with zeros, supply cost column and demand row from cost_table
        self.assign_table = np.zeros((self.rows, self.columns), dtype=object)
        self.assign_table[:, self.supply_column] = self.cost_table[:, self.supply_column]
        self.assign_table[self.demand_row] = self.cost_table[self.demand_row]

    def assign(self, cost: Frac, i: int, j: int):
        self.assign_table[i][self.supply_column] -= cost
        self.assign_table[self.demand_row][j] -= cost
        self.assign_table[i][j] = cost

    def slice(self, n: int, m: int):
        pass

    @abstractmethod
    def solve(self):
        pass
