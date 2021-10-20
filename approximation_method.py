from abc import abstractmethod
from fractions import Fraction as Frac
from typing import Any, Tuple, List, NoReturn

import numpy as np
from sympy import Symbol, linsolve

from writer import Writer


class ApproximationMethod:
    cost_table: np.ndarray
    assign_table: np.ndarray
    transportation_table: np.ndarray

    writer: Writer

    improvable: bool
    entering_variable: tuple
    leaving_variable: tuple
    loop: List[Tuple]

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

    def __init__(self, file):
        self.improvable = True
        self.most_assigned_row = -1
        self.most_assigned_column = -1
        self.entering_variable = tuple()
        self.leaving_variable = tuple()
        self.loop = list()
        self.unassigned_indices = set()
        self.assigned_indices = set()
        self.deleted_rows = set()
        self.deleted_cols = set()
        self.assignments_of_row = {self.most_assigned_row: -1}
        self.assignments_of_column = {self.most_assigned_column: -1}
        self.writer = Writer(filename=file.name)
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
        """
        Sets a given value to the assignment table & updating the problem state

        :param assignment: amount to assign in assignment table
        :param i: row to assign
        :param j: column to assign
        :param new_demand_and_supply: false when assignment comes from loop
        therefore doesn't need update
        """
        self.assign_table[i][j] = assignment
        self.assigned_indices.add((i, j))
        self.unassigned_indices.discard((i, j))
        self.increment_assignments_of(i, j)

        if new_demand_and_supply:
            self.assign_table[i][self.supply_column] -= assignment
            self.assign_table[self.demand_row][j] -= assignment

    def best_value_at(self, i: int, j: int) -> Tuple[int, int, int]:
        """
        Return the minimum value between the demand an supply for a
        certain cell

        :param i: row to check best value
        :param j: column to check best value
        :return: A tuple with the best value, row & column
        """
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

    def increment_assignments_of(self, i: int, j: int) -> None:
        """
        Updates the amount of assignments for the given row
        & column increasing it by 1

        :param i: row in which the assigment was made
        :param j: column in which the assigment was made
        """

        # rows & columns in assignments go from [1,n] & [1,m]
        i, j = i + 1, j + 1
        self.assignments_of_row[i] = self.assignments_of_row.get(i, 0) + 1
        self.assignments_of_column[j] = self.assignments_of_column.get(j, 0) + 1

        # case of new most assigned row
        if self.assignments_of_row[i] > self.assignments_of_row[self.most_assigned_row]:
            self.most_assigned_row = i

        # case of new most assigned column
        if self.assignments_of_column[j] > self.assignments_of_column[self.most_assigned_column]:
            self.most_assigned_column = j

    def decrement_assignments_of(self, i: int, j: int) -> None:
        """
        Updates the amount of assignments for the given row
        & column decreasing it by 1

        :param i: row in which the assigment was removed
        :param j: column in which the assignment was removed
        """
        i, j = i + 1, j + 1
        self.assignments_of_row[i] = self.assignments_of_row.get(i, 0) - 1

        # case in which the most assigned row decremented
        if self.most_assigned_row == i:
            self.most_assigned_row = max(self.assignments_of_row)

        # case in which the most assigned column decremented
        if self.most_assigned_column:
            self.most_assigned_column = max(self.assignments_of_column)

    def has_rows_and_columns_left(self) -> bool:
        """
        Checks if they're columns & rows available for assigment

        :return: false if all rows or columns were deleted/assigned
        """
        return \
            len(self.deleted_rows) != self.rows - 1 \
            and len(self.deleted_cols) != self.columns - 1

    def halt(self, message: str) -> NoReturn:
        """
        Terminates the program with error code 1 with a quick log
        to console/file

        :param message: string to write in console/file with the error name
        """

        self.writer.write_halting(message)
        exit(1)

    def improve(self) -> None:
        """
        Applies the transportation algorithm when the problem has non basic
        indicators greater than 0
        """

        self.__find_dual_variables()
        self.__find_non_basic_indicators()

        i = 0
        while self.improvable:
            self.__create_loop()
            self.__assign_loop()
            self.writer.write_transportation_iteration(transportation_matrix=self.transportation_table,
                                                       assignment_matrix=self.assign_table,
                                                       iteration=f'[Iteration] = {i}')
            self.writer.write_loop(self.loop, entering=self.entering_variable, leaving=self.leaving_variable)
            self.writer.write_current_cost(self.total_cost())
            self.__find_dual_variables()
            self.__find_non_basic_indicators()
            i += 1
        self.writer.write_transportation_iteration(transportation_matrix=self.transportation_table,
                                                   assignment_matrix=self.assign_table,
                                                   iteration=f'[Final Iteration]',
                                                   final=True)
        self.writer.write_optimal_cost(self.total_cost())

    def total_cost(self) -> int:
        """
        Multiplies each assignment for it's cost & sums them
        all up

        :return: integer with the total cost of the assignment table
        """

        total = 0
        for pos in self.assigned_indices:
            total += self.assign_table[pos] * self.cost_table[pos]
        return int(total)

    def unassign(self, i: int, j: int) -> None:
        """
        Updates the unassigned set with the given row & column
        & decrementing occurrences in each of them

        :param i: row to unassign from assignment table
        :param j: column to unassign from assignment table
        """

        self.unassigned_indices.add((i, j))
        self.decrement_assignments_of(i, j)

    def __assign_loop(self) -> None:
        """
        Updates the assignment of each index in the loop
        by decrementing or incrementing its value by the lowest
        of them all
        """

        # even indices are incremented, odd indices are decremented
        even_indices, odd_indices = self.loop[0::2], self.loop[1::2]

        # leaving variable is the lowest of the decremented indices
        # in order to not have negative values
        self.leaving_variable = min(odd_indices, key=self.assignment)
        leaving_assignment = self.assignment(self.leaving_variable)
        # unassigned indices to delete
        unassigned = set()
        for pos in self.assigned_indices:
            if pos in even_indices:
                self.assign_table[pos] += leaving_assignment
            if pos in odd_indices:
                self.assign_table[pos] -= leaving_assignment
                # case in which the decremented value is the lowest of them all
                can_be_unassigned = self.assign_table[pos] == 0
                if can_be_unassigned:
                    unassigned.add(pos)
                    self.unassign(*pos)
        self.assigned_indices -= unassigned
        # finally assign the new variable with the lowest value (demand & supply stays the same)
        self.assign(leaving_assignment, *self.entering_variable, new_demand_and_supply=False)

    def __create_cost_table(self, file) -> None:
        """
        Given a file with a containing valid transportation problem
        creates a cost table with demands, suppliers & cost values

        :param file: file in which the problem values are located
        """
        # obtain first row of txt file & make it supply column
        supply = np.loadtxt(file, max_rows=1, delimiter=",", comments="\\n") + Frac()
        supply = supply.reshape((-1, 1))

        # obtain second row of txt file & make it demand row
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
        """
        Checks if the demand row sum is greater than the supply column
        or viceversa & calculates the new fictional value between the
        difference between them
        """

        demand_sum = np.sum(self.cost_table[self.demand_row][:-1])
        supply_sum = np.sum(self.cost_table[:, self.supply_column][:-1])
        diff = int(abs(demand_sum - supply_sum))
        # balance the problem in case of different demand sum
        if demand_sum < supply_sum:
            self.__insert_fictional_demand(fictional_value=diff)
        # balance the problem in case of different supply sum
        elif supply_sum < demand_sum:
            self.__insert_fictional_supply(fictional_value=diff)
        else:
            pass

    def __insert_fictional_demand(self, fictional_value: int) -> None:
        """
        Inserts a fictional/dummy demand row with zeros
        :param fictional_value: amount to balance for the demand
        """

        fictional_demand = [self.demand_row * [0.0] + [fictional_value]]
        self.cost_table = np.insert(self.cost_table, -1, values=fictional_demand, axis=1)
        self.columns += 1
        self.supply_column += 1

    def __insert_fictional_supply(self, fictional_value: int) -> None:
        """
        Insert a fictional/dummy supply column with zeros
        :param fictional_value: amount to balance for the supply
        """

        fictional_supply = self.supply_column * [0] + [fictional_value]
        self.cost_table = np.insert(self.cost_table, -1, values=fictional_supply, axis=0)
        self.rows += 1
        self.demand_row += 1

    def __create_assign_table(self) -> None:
        """
        Creates a numpy nd.array with the same shape of the cost table for assigning
        indices
        """
        # fill table with zeros, supply cost column & demand row from cost_table
        self.assign_table = np.zeros((self.rows, self.columns), dtype=object)
        self.assign_table[:, self.supply_column] = self.cost_table[:, self.supply_column]
        self.assign_table[self.demand_row] = self.cost_table[self.demand_row]
        # assume all indices are unassigned
        self.unassigned_indices = {(i, j) for i in range(self.demand_row) for j in range(self.supply_column)}

    def __create_transportation_table(self) -> None:
        """
        Creates a numpy nd.array with the same shape of the cost table for non
        basic indicators & dual variables
        """
        self.u_column = self.supply_column
        self.v_row = self.demand_row
        self.transportation_table = np.empty((self.rows, self.columns), dtype=object)

        # bottom right corner is never used
        self.transportation_table[self.v_row][self.u_column] = "*"

    def __create_loop(self) -> None:
        """
        Updates the loop with the neighbors indices of the entrance variable
        """
        start = [self.entering_variable]

        def find(loop: List[Tuple]) -> List[Tuple]:
            """
            Recursively finds the smallest loop the from
            a given list of visited indices

            :param loop: current visited indices
            :return: neighbor indices of the entrance variable (start)
            """
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
        """
        Finds a list of possible indices to visited based on the last index
        of the list

        :param loop: visited indices
        :param not_visited: pending indices that don't have an assignment value
        :return: possible indices that can be assigned some value
        """
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
        """
        Iterates over the unassigned indices & calculates
        each indicator by the form U variable + V variable - C cost
        """
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
                    self.entering_variable = (i, j)
            self.transportation_table[i][j] = nb_indicator

    def __find_dual_variables(self) -> None:
        """
        Iterates over the assigned indices & calculates
        each variable by the form U variable + V variable - C cost
        """
        u_vars, v_vars = self.__find_equation_vars()
        equations = list()
        for i, j in self.assigned_indices:
            u = u_vars[i]
            v = v_vars[j]
            c = self.cost_table[i][j]
            eq = u + v - c
            equations.append(eq)
        solved_v, solved_u = self.__solve_variables(equations, u_vars, v_vars)
        self.transportation_table[-1, :-1] = solved_v
        self.transportation_table[:-1, -1] = solved_u

    def __solve_variables(self, equations: List[Symbol],
                          u_vars: Tuple[Symbol],
                          v_vars: Tuple[Symbol]):
        """
        Finds a value for each U var & V var by having a
        list of equations

        :param equations: list of unsolved equations
        :param u_vars: list of needed u variables to solve
        :param v_vars: list of needed v variables to solve
        :return: tuple with the solved v vars & solved u vars
        """
        try:
            solved = linsolve(equations, (u_vars + v_vars)).args[0]
            amount_of_u = self.rows - 1
            solved_u = list(map(int, solved[:amount_of_u]))
            solved_v = list(map(int, solved[amount_of_u:]))
            return solved_v, solved_u

        # TypeError occurs when a dual variable is not solvable hence
        # it's a degenerated solution
        except TypeError as te:
            self.writer.write_optimal_cost(self.total_cost())
            self.halt(f'Caught exception "{te}"\nDegenerated solutions found, exiting...')

    def __find_equation_vars(self) -> Tuple[Tuple, Tuple]:
        """
        Creates a list of U vars & V vars with the exception
        of the most assigned row or column which takes the value of
        zero
        
        :return: tuple with the v vars & u vars
        """
        if self.assignments_of_column[self.most_assigned_column] >= \
                self.assignments_of_row[self.most_assigned_row]:
            zero_candidate = Symbol(f'V{self.most_assigned_column}')
        else:
            zero_candidate = Symbol(f'U{self.most_assigned_row}')

        # V1, V2, V3 ... Vm
        v_vars = list()
        for i in range(1, self.columns):
            v = Symbol(f'V{i}')
            if v == zero_candidate:
                v_vars.append(0)
            else:
                v_vars.append(v)

        # U1, U2, U3 ... Un
        u_vars = list()
        for j in range(1, self.rows):
            u = Symbol(f'U{j}')
            if u == zero_candidate:
                u_vars.append(0)
            else:
                u_vars.append(u)

        return tuple(u_vars), tuple(v_vars)
