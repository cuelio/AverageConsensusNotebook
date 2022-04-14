import numpy as np
import setup.graph_creator as graph_creator
import setup.init_edge_weights as init_edge_weights
import random

# a = [0.080, 0.062, 0.075, 0.072, 0.066, 0.070, 0.35, 0.075, 0.060, 0.078, 0.080, 0.085, 0.069, 0.077]
# b = [2.25, 4.20, 3.25, 8.25, 7.20, 4.05, 0, 7.80, 8.05, 8.45, 8.75, 9.00, 7.05, 8.15]
# p_max = [60, 50, 55, -30, -20, 50, 25, -30, -35, -35, -40, -40, -15, -35]
# p_min = [30, 20, 30, -10, -5, 20, -25, -10, -15, -15, -15, -15, -1, -10]
from shared_types.types import TopologyLayout, EdgeWeightType


class IncrementalCostConsensusInstance:
    def __init__(self, num_nodes, topology, edge_weight_type):
        # Input Data
        self.num_nodes = num_nodes
        self.incremental_cost = np.zeros(num_nodes, dtype=float)
        self.laplacian = graph_creator.get_graph(topology, self.num_nodes, 8)
        self.edge_weights = init_edge_weights.get_edge_weights(edge_weight_type, self.laplacian, self.num_nodes)
        self.estimated_mismatch = np.zeros(num_nodes, dtype=float)
        self.actual_power = np.zeros(num_nodes, dtype=float)

        self.a = [0.080, 0.062, 0.075, 0.072, 0.066, 0.070, 0.35, 0.075, 0.060, 0.078, 0.080, 0.085, 0.069, 0.077]
        self.b = [2.25, 4.20, 3.25, 8.25, 7.20, 4.05, 0, 7.80, 8.05, 8.45, 8.75, 9.00, 7.05, 8.15]
        self.p_max = [60, 50, 55, -30, -20, 50, 25, -30, -35, -35, -40, -40, -15, -35]
        self.p_min = [30, 20, 30, -10, -5, 20, -25, -10, -15, -15, -15, -15, -1, -10]

        # cost function
        self.b_matrix = np.zeros((num_nodes, num_nodes), dtype=float)
        self.g = np.zeros(num_nodes, dtype=float)

        # Result Data
        self.values_by_round = []
        self.rounds_to_convergence = 0

    def execute_instance(self):
        epsilon = -0.01
        while not self.is_stopping_condition_satisfied():
            self.rounds_to_convergence += 1
            self.incremental_cost = np.add(np.dot(self.edge_weights, self.incremental_cost),
                                           epsilon * self.estimated_mismatch)
            prev_actual_power = self.actual_power
            self.actual_power = np.add(np.dot(self.b_matrix, self.incremental_cost), self.g)
            self.adjust_for_constraints()

            self.estimated_mismatch = np.add(
                np.dot(self.edge_weights, self.estimated_mismatch),
                np.dot(self.edge_weights, np.add(self.actual_power, -1 * prev_actual_power))
            )
            # self.values_by_round.append(self.incremental_cost)

    def is_stopping_condition_satisfied(self, epsilon=0.05):
        max_value = np.max(self.incremental_cost)
        min_value = np.min(self.incremental_cost)
        value_mismatch = abs(max_value - min_value)
        if self.rounds_to_convergence % 10 == 0:
            print("Min/max mismatch in round " + str(self.rounds_to_convergence) + ": " + str(value_mismatch))

        if value_mismatch <= epsilon:
            print("Reached convergence in " + str(self.rounds_to_convergence) + " rounds.")
            return True
        else:
            return False

    def adjust_for_constraints(self):
        for i in range(0, self.num_nodes):
            if self.p_max[i] < 0 and self.p_min[i] < 0:
                self.adjust_for_load(i)
            elif self.p_max[i] > 0 and self.p_min[i] > 0:
                self.adjust_for_generation(i)
            elif self.p_max[i] > 0 and self.p_min[i] < 0:
                self.adjust_for_battery(i)

    def adjust_for_battery(self, i):
        if self.actual_power[i] > self.p_max[i]:
            self.actual_power[i] = self.p_max[i]
        elif self.actual_power[i] < self.p_min[i]:
            self.actual_power[i] = self.p_min[i]

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
        min_load = 0
        max_load = 1
        min_gen = 1
        max_gen = 0

        attempt_number = 0
        while not (max_gen > abs(max_load) and min_gen < abs(min_load)):
            if attempt_number > 10:
                print("Unable to generate initial experiment setup after " + str(attempt_number) + " attempts")
                return 0
            attempt_number += 1

            self.a = np.zeros(self.num_nodes, dtype=float)
            self.b = np.zeros(self.num_nodes, dtype=float)
            self.p_max = np.zeros(self.num_nodes, dtype=int)
            self.p_min = np.zeros(self.num_nodes, dtype=int)

            for node_index in range(0, self.num_nodes):
                self.a[node_index] = random.uniform(0.05, 0.08)
                self.b[node_index] = random.uniform(3.0, 10.0)

                is_generation = random.random() > 0.6
                if is_generation:
                    self.p_min[node_index] = random.randint(15, 26)
                    self.p_max[node_index] = random.randint(self.p_min[node_index] + 20, self.p_min[node_index] + 46)
                    min_gen += self.p_min[node_index]
                    max_gen += self.p_max[node_index]
                else:
                    self.p_min[node_index] = random.randint(10, 21)
                    self.p_max[node_index] = random.randint(self.p_min[node_index] + 10, self.p_min[node_index] + 21)

                    self.p_min[node_index] = self.p_min[node_index] * -1
                    self.p_max[node_index] = self.p_max[node_index] * -1
                    min_load += self.p_min[node_index]
                    max_load += self.p_max[node_index]

            # print(min_load)
            # print(max_load)
            # print(min_gen)
            # print(max_gen)

        self.init_starting_values()
        return 1


if __name__ == '__main__':
    # avg_consensus_instance = IncrementalCostConsensusInstance(50, TopologyLayout.WATTS_STROGATZ,
    # EdgeWeightType.OPTIMAL_CONSTANT)
    avg_consensus_instance = IncrementalCostConsensusInstance(1000, TopologyLayout.LATTICE_RING,
                                                              EdgeWeightType.MEAN_METROPOLIS)
    setup_success = avg_consensus_instance.init_starting_values_randomized()
    if setup_success == 1:
        avg_consensus_instance.execute_instance()
        # print(avg_consensus_instance.incremental_cost)
        # print(avg_consensus_instance.estimated_mismatch)
        # print(avg_consensus_instance.actual_power)
    else:
        print("setup failed")
