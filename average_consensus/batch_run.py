from average_consensus.IncrementalCostConsensusInstance import IncrementalCostConsensusInstance
from average_consensus.DistributedAverageConsensusInstance import DistributedAverageConsensusInstance
from shared_types.types import ConsensusAlgorithm
from shared_types.types import InitialValueSetup
import numpy as np


def run_ac_batch(instance_sizes, initial_value_type, topology, num_nbrs, ws_prob=0.2, num_samples=10):
    rounds_to_convergence = []
    for instance_size in instance_sizes:
        if instance_size % 50 == 0:
            print("Running experiment for " + str(instance_size) + " nodes")

        avg_rounds = 0.0
        for i in range(0, num_samples):
            instance = DistributedAverageConsensusInstance(instance_size, initial_value_type, topology, num_nbrs)
            instance.execute_instance()
            avg_rounds += instance.rounds_to_convergence

        # print("Took an average of " + str(avg_rounds) + " for " + str(num_samples) + " samples")
        avg_rounds = avg_rounds / num_samples
        rounds_to_convergence.append(avg_rounds)

    return rounds_to_convergence


def run_batch(alg_type, instance_sizes, topology, num_nbrs, ws_prob=0.2, num_samples=10):
    if alg_type == ConsensusAlgorithm.ICC:
        run_icc_batch(instance_sizes, topology, num_nbrs, ws_prob, num_samples)
    elif alg_type == ConsensusAlgorithm.AC:
        run_ac_batch(instance_sizes, InitialValueSetup.RANDOM, topology, num_nbrs)


def run_icc_batch(instance_sizes, topology, num_nbrs, ws_prob=0.2, num_samples=10):
    rounds_to_convergence = []
    for instance_size in instance_sizes:
        if instance_size % 10 == 0:
            print("Running experiment for " + str(instance_size) + " nodes")

        avg_rounds = run_icc_instance(instance_size, topology, num_nbrs, ws_prob, num_samples)
        rounds_to_convergence.append(avg_rounds)
    return rounds_to_convergence


def run_icc_instance(instance_size, topology, num_nbrs, ws_rewire_prob=0.2, num_samples=10, step_size=0.0001):
    ws_rewire_prob = round(ws_rewire_prob, 2)
    avg_rounds = 0

    if step_size is None:
        step_size = np.round((2 * num_nbrs) / (instance_size ** 2), 5)
    for i in range(0, num_samples):
        instance = IncrementalCostConsensusInstance(instance_size, topology, num_nbrs, ws_rewire_prob=ws_rewire_prob,
                                                    step_size=step_size)
        setup_successful = instance.init_starting_values_randomized()
        # instance.init_starting_values()
        if not setup_successful:
            raise Exception("Unable to successfully create random starting values")

        instance.execute_instance()
        avg_rounds += instance.rounds_to_convergence

    avg_rounds = avg_rounds / num_samples
    print("Took " + str(avg_rounds) + " rounds to converge for " + str(instance_size) + " nodes")
    return avg_rounds
