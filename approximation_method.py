from abc import abstractmethod
from fractions import Fraction as Frac
from typing import Any, Tuple, List, NoReturn

import numpy as np
from sympy import Symbol, linsolve

from writer import Writer, MethodType


class ApproximationMethod:
    cost_table: np.ndarray
    assign_table: np.ndarray
    transportation_table: np.ndarray

    writer: Writer

    improvable: bool
    entrance_indicator: tuple
    loop: list

    rows: int
    columns: int

    demand_row: int
    supply_column: int

    most_assigned_row: int
    most_assigned_column: int

    v_row: int
    u_column: int

    deleted_rows: set
    deleted_cols: set

    assigned_indices: set
    unassigned_indices: set

    assignments_of_row: dict
    assignments_of_column: dict

    def __init__(self, file, method: MethodType):
        self.improvable = True
        self.most_assigned_row = -1
        self.most_assigned_column = -1
        self.entrance_indicator = tuple()
        self.loop = list()
        self.unassigned_indices = set()
        self.assigned_indices = set()
        self.deleted_rows = set()
        self.deleted_cols = set()
        self.assignments_of_row = {self.most_assigned_row: -1}
        self.assignments_of_column = {self.most_assigned_column: -1}
        self.writer = Writer(method_type=method, filename=file.name)
        self.__create_cost_table(file)
        self.__balance_cost_table()
        self.__create_assign_table()
        self.__create_transportation_table()

    @abstractmethod
    def solve(self) -> None:
        pass

    @abstractmethod
    def choose_cost(self) -> None:
        pass

    def assignment(self, pos: tuple) -> float:
        return self.assign_table[pos]

    def assign(self, assignment: Any, i: int, j: int, new_demand_and_supply=True) -> None:

        self.assign_table[i][j] = assignment
        self.assigned_indices.add((i, j))
        self.unassigned_indices.discard((i, j))
        self.increment_assignments_of(i, j)

        if new_demand_and_supply:
            self.assign_table[i][self.supply_column] -= assignment
            self.assign_table[self.demand_row][j] -= assignment

    def best_value_at(self, i: int, j: int) -> Tuple[int, int, int]:
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

    def increment_assignments_of(self, i: int, j: int):
        i, j = i + 1, j + 1
        self.assignments_of_row[i] = self.assignments_of_row.get(i, 0) + 1
        self.assignments_of_column[j] = self.assignments_of_column.get(j, 0) + 1

        if self.assignments_of_row[i] > self.assignments_of_row[self.most_assigned_row]:
            self.most_assigned_row = i

        if self.assignments_of_column[j] > self.assignments_of_column[self.most_assigned_column]:
            self.most_assigned_column = j

    def decrement_assignments_of(self, i: int, j: int):
        i, j = i + 1, j + 1
        self.assignments_of_row[i] = self.assignments_of_row.get(i, 0) - 1
        if self.most_assigned_row == i:
            self.most_assigned_row = max(self.assignments_of_row)

        if self.most_assigned_column:
            self.most_assigned_column = max(self.assignments_of_column)

    def has_rows_and_columns_left(self) -> bool:
        return len(self.deleted_rows) != self.rows - 1 \
               and len(self.deleted_cols) != self.columns - 1

    def halt(self, message) -> NoReturn:
        self.writer.write_halting(message)
        exit(1)

    def improve(self) -> None:
        self.__find_dual_variables()
        self.__find_non_basic_indicators()

        while self.improvable:
            self.__create_loop()
            self.__assign_loop()
            self.writer.write_solution(matrix=self.assign_table)
            self.writer.write_solution(matrix=self.transportation_table)
            self.__find_dual_variables()
            self.__find_non_basic_indicators()

    def total_cost(self) -> int:
        total = 0
        for pos in self.assigned_indices:
            total += self.assign_table[pos] * self.cost_table[pos]
        return total

    def unassign(self, i: int, j: int) -> None:
        self.unassigned_indices.add((i, j))
        self.decrement_assignments_of(i, j)

    def __assign_loop(self):
        even_indices, odd_indices = self.loop[0::2], self.loop[1::2]

        leaving_position = min(odd_indices, key=self.assignment)
        leaving_assignment = self.assignment(leaving_position)

        unassigned = set()
        for pos in self.assigned_indices:
            if pos in even_indices:
                self.assign_table[pos] += leaving_assignment
            if pos in odd_indices:
                self.assign_table[pos] -= leaving_assignment
                can_be_unassigned = self.assign_table[pos] == 0
                if can_be_unassigned:
                    unassigned.add(pos)
                    self.unassign(*pos)
        self.assigned_indices -= unassigned
        self.assign(leaving_assignment, *self.entrance_indicator, new_demand_and_supply=False)

    def __create_cost_table(self, file) -> None:
        # obtain first row of txt file and make it supply column
        supply = np.loadtxt(file, max_rows=1, delimiter=",", comments="\\n") + Frac()
        supply = supply.reshape((-1, 1))

        # obtain second row of txt file and make it demand row
        demand = np.loadtxt(file, max_rows=1, delimiter=",", comments="\\n") + Frac()
        demand = np.append(demand, "*").reshape((1, -1))

        # finally generate the cost_table with costs + demand + supply column
        costs = np.loadtxt(file, delimiter=",", comments="\\n") + Frac()
        self.cost_table = np.append(costs, values=supply, axis=1)
        self.cost_table = np.append(self.cost_table, values=demand, axis=0)

        # update attributes
        self.rows, self.columns = self.cost_table.shape
        self.demand_row, self.supply_column = self.rows - 1, self.columns - 1

        # file is no longer needed
        file.close()
        del file

    def __balance_cost_table(self) -> None:
        # balance the problem in case of different sums
        demand_sum = np.sum(self.cost_table[self.demand_row][:-1])
        supply_sum = np.sum(self.cost_table[:, self.supply_column][:-1])
        diff = int(abs(demand_sum - supply_sum))
        if demand_sum < supply_sum:
            self.__insert_fictional_demand(fictional_value=diff)
        elif supply_sum < demand_sum:
            self.__insert_fictional_supply(fictional_value=diff)
        else:
            pass

    def __insert_fictional_demand(self, fictional_value: int) -> None:
        # fictional/dummy demand row with zeros
        fictional_demand = [self.demand_row * [0.0] + [fictional_value]]
        self.cost_table = np.insert(self.cost_table, -1, values=fictional_demand, axis=1)
        self.columns += 1
        self.supply_column += 1

    def __insert_fictional_supply(self, fictional_value: int) -> None:
        # fictional/dummy supply column with zeros
        fictional_supply = self.supply_column * [0] + [fictional_value]
        self.cost_table = np.insert(self.cost_table, -1, values=fictional_supply, axis=0)
        self.rows += 1
        self.demand_row += 1

    def __create_assign_table(self) -> None:
        # fill table with zeros, supply cost column and demand row from cost_table
        self.assign_table = np.zeros((self.rows, self.columns), dtype=object)
        self.assign_table[:, self.supply_column] = self.cost_table[:, self.supply_column]
        self.assign_table[self.demand_row] = self.cost_table[self.demand_row]
        # assume all indices are unassigned
        self.unassigned_indices = {(i, j) for i in range(self.demand_row) for j in range(self.supply_column)}

    def __create_transportation_table(self) -> None:
        self.u_column = self.supply_column
        self.v_row = self.demand_row
        self.transportation_table = np.empty((self.rows, self.columns), dtype=object)
        self.transportation_table[self.v_row][self.u_column] = "*"

    def __create_loop(self) -> None:
        start = [self.entrance_indicator]

        def find(loop: List[Tuple]) -> List[Tuple]:
            one_neighbor_left = len(loop) > 3
            if one_neighbor_left:
                not_visited = start
                closable = len(self.find_neighbors(loop, not_visited)) == 1
                if closable:
                    return loop
            not_visited = list(self.assigned_indices - set(loop))
            possible_neighbors = self.find_neighbors(loop, not_visited)
            for neighbor in possible_neighbors:
                new_loop = find(loop + [neighbor])
                if new_loop:
                    return new_loop

        self.loop = find(loop=start)

    @staticmethod
    def find_neighbors(loop: List[Tuple], not_visited: List[Tuple]) -> List[Tuple]:
        last_row, last_column = loop[-1]
        row_neighbors, column_neighbors = list(), list()
        for i, j in not_visited:
            if i == last_row:
                row_neighbors.append((i, j))
            if j == last_column:
                column_neighbors.append((i, j))
        loop_incomplete = len(loop) < 2
        if loop_incomplete:
            return row_neighbors + column_neighbors
        else:
            previous_row, _ = loop[-2]
            is_row_move = previous_row == last_row
            if is_row_move:
                return column_neighbors
            return row_neighbors

    def __find_non_basic_indicators(self) -> None:
        # assume that it isn't improvable from the start
        self.improvable = False
        best_indicator = -np.inf
        for i, j in self.unassigned_indices:
            u = self.transportation_table[i][self.u_column]
            v = self.transportation_table[self.v_row][j]
            c = self.cost_table[i][j]
            nb_indicator = int(u + v - c)
            if nb_indicator > 0:
                self.improvable = True
                if nb_indicator > best_indicator:
                    best_indicator = nb_indicator
                    self.entrance_indicator = (i, j)
            if nb_indicator == 0:
                self.halt(message="Multiple Solutions Found")
            self.transportation_table[i][j] = nb_indicator

    def __find_dual_variables(self) -> None:
        self.__check_degenerated_solutions()
        u_vars, v_vars = self.__find_equation_vars()
        equations = list()
        for i, j in self.assigned_indices:
            u = u_vars[i]
            v = v_vars[j]
            c = self.cost_table[i][j]
            eq = u + v - c
            equations.append(eq)
        solved = linsolve(equations, (u_vars + v_vars)).args[0]
        amount_of_u = self.rows-1
        solved_u = list(map(int, solved[:amount_of_u]))
        solved_v = list(map(int, solved[amount_of_u:]))
        self.transportation_table[-1, :-1] = solved_v
        self.transportation_table[:-1, -1] = solved_u

    def __find_equation_vars(self) -> Tuple:
        if self.assignments_of_column[self.most_assigned_column] >= \
                self.assignments_of_row[self.most_assigned_row]:
            zero_candidate = Symbol(f'V{self.most_assigned_column}')
        else:
            zero_candidate = Symbol(f'U{self.most_assigned_row}')

        v_vars = list()
        for i in range(1, self.columns):
            v = Symbol(f'V{i}')
            if v == zero_candidate:
                v_vars.append(0)
            else:
                v_vars.append(v)

        u_vars = list()
        for j in range(1, self.rows):
            u = Symbol(f'U{j}')
            if u == zero_candidate:
                u_vars.append(0)
            else:
                u_vars.append(u)

        return tuple(u_vars), tuple(v_vars)

    def __check_degenerated_solutions(self) -> None:
        assigned = len(self.assigned_indices)
        if assigned != self.columns + self.rows - 1:
            self.halt(message="Degenerated Solutions Found")
