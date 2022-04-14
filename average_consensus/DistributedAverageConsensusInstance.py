import numpy as np
import time
import setup.graph_creator as graph_creator
import setup.init_edge_weights as init_edge_weights
import setup.init_starting_values as init_starting_values

from shared_types.types import InitialValueSetup
from shared_types.types import EdgeWeightType
from shared_types.types import TopologyLayout


class DistributedAverageConsensusInstance:
    def __init__(self, num_nodes, initial_value_type, topology, edge_weight_type):
        # Input Data
        self.num_nodes = num_nodes
        self.values = init_starting_values.get_values(initial_value_type, self.num_nodes)
        self.laplacian = graph_creator.get_graph(topology, self.num_nodes)
        self.edge_weights = init_edge_weights.get_edge_weights(edge_weight_type, self.laplacian, self.num_nodes)

        # Result Data
        self.values_by_round = []
        self.rounds_to_convergence = -1

    def execute_instance(self):
        rounds = 0
        while not self.is_stopping_condition_satisfied():
            rounds += 1
            self.values = np.dot(self.edge_weights, self.values)
            self.values_by_round.append(self.values)

        self.rounds_to_convergence = rounds

    def is_stopping_condition_satisfied(self, epsilon=0.01):
        max_value = np.max(self.values)
        min_value = np.min(self.values)
        diff = abs(max_value - min_value)
        # print(diff)
        if diff <= epsilon:
            return True
        else:
            return False


# laplacian = graph_creator.get_lattice_ring(num_nodes, num_neighbors)
# values = init_starting_values.grouped(num_nodes, 0, 10)
# edge_weights = init_edge_weights.optimal_constant(laplacian, num_nodes)
# values_by_round = simulation.run_avg_consensus(values, edge_weights)

if __name__ == '__main__':
    start = time.time()
    for n in range(10, 200, 5):
        avg_consensus_instance = DistributedAverageConsensusInstance(
            n, InitialValueSetup.GROUPED, TopologyLayout.LATTICE_RING, EdgeWeightType.OPTIMAL_CONSTANT)
        avg_consensus_instance.execute_instance()

        print("Finished in " + str(avg_consensus_instance.rounds_to_convergence) + " rounds")

    end = time.time()
    print(end - start)
