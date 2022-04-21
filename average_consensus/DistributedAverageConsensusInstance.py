import numpy as np
import setup.graph_creator as graph_creator
import setup.init_edge_weights as init_edge_weights
import setup.init_starting_values as init_starting_values

from shared_types.types import InitialValueSetup
from shared_types.types import EdgeWeightType
from shared_types.types import TopologyLayout


def run_instance(instance_size):
    instance = DistributedAverageConsensusInstance(instance_size, InitialValueSetup.GROUPED,
                                                   TopologyLayout.LATTICE_RING)
    instance.execute_instance()


def run_ac_batch(instance_sizes, initial_value_type, topology, num_neighbors):
    rounds_to_convergence = []
    # print("Running experiment for " + str(instance_sizes) + " nodes")
    for instance_size in instance_sizes:
        if instance_size % 50 == 0:
            print("Running experiment for " + str(instance_size) + " nodes")

        instance = DistributedAverageConsensusInstance(instance_size, initial_value_type, num_neighbors, topology)
        instance.execute_instance()
        rounds_to_convergence.append(instance.rounds_to_convergence)

    return rounds_to_convergence


class DistributedAverageConsensusInstance:
    def __init__(self, num_nodes, initial_value_type, topology, num_neighbors=4,
                 edge_weight_type=EdgeWeightType.MEAN_METROPOLIS):
        # Input Data
        self.num_nodes = num_nodes
        self.values = init_starting_values.get_values(initial_value_type, self.num_nodes)
        self.laplacian = graph_creator.get_graph(topology, self.num_nodes, num_neighbors)
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
