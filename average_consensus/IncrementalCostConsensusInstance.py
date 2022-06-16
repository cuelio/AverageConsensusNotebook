import numpy as np
import setup.graph_creator as graph_creator
import setup.init_edge_weights as init_edge_weights
import random
import math

# a = [0.080, 0.062, 0.075, 0.072, 0.066, 0.070, 0.35, 0.075, 0.060, 0.078, 0.080, 0.085, 0.069, 0.077]
# b = [2.25, 4.20, 3.25, 8.25, 7.20, 4.05, 0, 7.80, 8.05, 8.45, 8.75, 9.00, 7.05, 8.15]
# p_max = [60, 50, 55, -30, -20, 50, 25, -30, -35, -35, -40, -40, -15, -35]
# p_min = [30, 20, 30, -10, -5, 20, -25, -10, -15, -15, -15, -15, -1, -10]
from shared_types.types import EdgeWeightType, TopologyLayout


class IncrementalCostConsensusInstance:
    def __init__(self,
                 num_nodes,
                 topology,
                 num_nbrs=4,
                 edge_weight_type=EdgeWeightType.MEAN_METROPOLIS,
                 ws_rewire_prob=0.5,
                 step_size=0.0001):

        # Input Data
        self.num_nodes = num_nodes
        self.epsilon = step_size * -1
        self.incremental_cost = np.zeros(num_nodes, dtype=float)
        graph = graph_creator.get_graph(topology, self.num_nodes, num_nbrs, ws_rewire_prob)
        self.laplacian = graph_creator.convert_graph_to_laplacian(graph)
        self.edge_weights = init_edge_weights.get_edge_weights(edge_weight_type, self.laplacian, self.num_nodes)
        self.estimated_mismatch = np.zeros(num_nodes, dtype=float)
        self.actual_power = np.zeros(num_nodes, dtype=float)

        # Default test data from paper for 14 nodes
        # self.a = [0.080, 0.062, 0.075, 0.072, 0.066, 0.070, 0.35, 0.075, 0.060, 0.078, 0.080, 0.085, 0.069, 0.077]
        # self.b = [2.25, 4.20, 3.25, 8.25, 7.20, 4.05, 0, 7.80, 8.05, 8.45, 8.75, 9.00, 7.05, 8.15]
        # self.p_max = [60, 50, 55, -30, -20, 50, 25, -30, -35, -35, -40, -40, -15, -35]
        # self.p_min = [30, 20, 30, -10, -5, 20, -25, -10, -15, -15, -15, -15, -1, -10]

        self.a = np.zeros(self.num_nodes, dtype=float)
        self.b = np.zeros(self.num_nodes, dtype=float)
        self.p_max = np.zeros(self.num_nodes, dtype=int)
        self.p_min = np.zeros(self.num_nodes, dtype=int)

        # cost function
        self.b_matrix = np.zeros((num_nodes, num_nodes), dtype=float)
        self.g = np.zeros(num_nodes, dtype=float)

        self.init_starting_values_randomized()

        # Result Data
        self.values_by_round = []
        self.rounds_to_convergence = 0

    def execute_instance(self):
        while not self.is_stopping_condition_satisfied():
            self.rounds_to_convergence += 1
            self.incremental_cost = np.add(np.dot(self.edge_weights, self.incremental_cost),
                                           self.epsilon * self.estimated_mismatch)
            prev_actual_power = self.actual_power
            self.actual_power = np.add(np.dot(self.b_matrix, self.incremental_cost), self.g)
            self.adjust_for_constraints()

            self.estimated_mismatch = np.add(
                np.dot(self.edge_weights, self.estimated_mismatch),
                np.dot(self.edge_weights, np.add(self.actual_power, -1 * prev_actual_power))
            )
            self.values_by_round.append(self.incremental_cost)

    def is_stopping_condition_satisfied(self, epsilon=0.1):
        estimated_power_spread = round(np.max(self.estimated_mismatch) - np.min(self.estimated_mismatch), 4)
        if not math.isclose(estimated_power_spread, 0, abs_tol=epsilon):
            # Only need to check other stopping conditions if all nodes have a near-0 estimates of power mismatch
            return False

        # Both should converge to 0
        estimated_power_mismatch = round(np.average(self.estimated_mismatch), 4)
        ic_spread = round(np.max(self.incremental_cost) - np.min(self.incremental_cost), 4)

        return math.isclose(estimated_power_mismatch, 0, abs_tol=epsilon) and math.isclose(ic_spread, 0, abs_tol=epsilon)

    def print_convergence_status(self, value_mismatch):
        # print_interval = 10
        # if self.rounds_to_convergence >= 1000:
        #     print_interval = 100
        # if self.rounds_to_convergence >= 10000:
        #     print_interval = 1000

        if self.rounds_to_convergence >= 10000 and self.rounds_to_convergence % 10000 == 0:
            print("Min/max mismatch in round " + str(self.rounds_to_convergence) + ": " + str(value_mismatch))
            print("estimated mismatch: " + str(np.mean(self.estimated_mismatch)))
            # print("Average cost list: " + str(self.incremental_cost))
            # print("power mismatch: " + str(self.estimated_mismatch))
            # print("Eigenvalues: " + str(np.linalg.eigvals(self.laplacian)))

    def adjust_for_constraints(self):
        for i in range(0, self.num_nodes):
            if self.p_max[i] < 0 and self.p_min[i] < 0:
                self.adjust_for_load(i)
            elif self.p_max[i] > 0 and self.p_min[i] > 0:
                self.adjust_for_generation(i)

    def adjust_for_generation(self, i):
        if self.actual_power[i] > self.p_max[i]:
            self.actual_power[i] = self.p_max[i]
        elif self.actual_power[i] < self.p_min[i]:
            self.actual_power[i] = self.p_min[i]

    def adjust_for_load(self, i):
        if self.actual_power[i] < self.p_max[i]:
            self.actual_power[i] = self.p_max[i]
        elif self.actual_power[i] > self.p_min[i]:
            self.actual_power[i] = self.p_min[i]

    def init_starting_values(self):
        self.init_actual_power()
        self.init_incremental_cost()
        self.init_estimated_mismatch()
        self.init_g()
        self.init_b()

    def init_actual_power(self):
        for i in range(0, self.num_nodes):
            self.actual_power[i] = self.p_min[i]

            # diff = self.p_max[i] - self.p_min[i]
            # self.actual_power[i] = round(diff * random.random() + self.p_min[i], 1)
            # print("setup: ")
            # print(self.p_min[i], self.p_max[i], self.actual_power[i])

    def init_incremental_cost(self):
        for i in range(0, self.num_nodes):
            self.incremental_cost[i] = self.a[i] * self.actual_power[i] + self.b[i]

    def init_estimated_mismatch(self):
        for i in range(0, self.num_nodes):
            self.estimated_mismatch[i] = self.actual_power[i]

    def init_g(self):
        for i in range(0, self.num_nodes):
            self.g[i] = -1 * (self.b[i] / self.a[i])

    def init_b(self):
        for i in range(0, self.num_nodes):
            self.b_matrix[i][i] = float(1 / self.a[i])

    def init_starting_values_randomized(self):
        attempt_number = 0
        min_load, max_load, min_gen, max_gen = 0, 0, 0, 0
        while attempt_number < 15:
            if attempt_number > 5:
                print("Taking too long to generate...")
                print(min_gen, min_load, max_load, max_gen)
            attempt_number += 1

            min_load, max_load, min_gen, max_gen = 0, 0, 0, 0

            generation_positions = []
            for node_index in range(0, self.num_nodes):
                is_generation = random.random() > 0.6
                generation_positions.append(is_generation)

            for node_index in range(0, self.num_nodes):
                if generation_positions[node_index]:
                    self.a[node_index] = random.uniform(0.05, 0.08)
                    self.b[node_index] = random.uniform(2.0, 5)

                    self.p_min[node_index] = random.randint(10, 15)
                    self.p_max[node_index] = random.randint(self.p_min[node_index] + 30, self.p_min[node_index] + 61)
                    min_gen += self.p_min[node_index]
                    max_gen += self.p_max[node_index]
                else:
                    self.a[node_index] = random.uniform(0.05, 0.08)
                    self.b[node_index] = random.uniform(7.0, 9.0)

                    self.p_min[node_index] = random.randint(10, 21)
                    self.p_max[node_index] = random.randint(self.p_min[node_index] + 10, self.p_min[node_index] + 21)

                    self.p_min[node_index] = self.p_min[node_index] * -1
                    self.p_max[node_index] = self.p_max[node_index] * -1
                    min_load += self.p_min[node_index]
                    max_load += self.p_max[node_index]

            # print(min_gen, min_load, max_load, max_gen)
            if abs(min_gen) < abs(min_load) and abs(max_load) < abs(max_gen):
                self.init_starting_values()
                return 1

        print("Unable to generate initial experiment setup after " + str(attempt_number) + " attempts")
        return 0
