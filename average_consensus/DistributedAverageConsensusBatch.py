import time
import numpy as np

from shared_types.types import InitialValueSetup
from shared_types.types import EdgeWeightType
from shared_types.types import TopologyLayout
from average_consensus.DistributedAverageConsensusInstance import DistributedAverageConsensusInstance


class DistributedAverageConsensusBatch:
    def __init__(self, initial_value_type, topology, edge_weight_type, min_nodes=10, max_nodes=200, offset=5):
        # self.number_of_instances = int((max_nodes - min_nodes) / offset)
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.instances = []
        self.rounds_to_convergence = []

    def create_and_run_instances(self):
        for instance_size in range(self.min_nodes, self.max_nodes, 5):
            print("Creating instance of size: " + str(instance_size))
            sim_instance = DistributedAverageConsensusInstance(
                instance_size, InitialValueSetup.GROUPED, TopologyLayout.LATTICE_RING, EdgeWeightType.OPTIMAL_CONSTANT)
            sim_instance.execute_instance()
            self.instances.append(sim_instance)
            self.rounds_to_convergence.append(sim_instance.rounds_to_convergence)


if __name__ == '__main__':
    start = time.time()
    batch = DistributedAverageConsensusBatch(InitialValueSetup.GROUPED, TopologyLayout.LATTICE_RING,
                                             EdgeWeightType.OPTIMAL_CONSTANT)

    batch.create_and_run_instances()

    end = time.time()
    print(end - start)
