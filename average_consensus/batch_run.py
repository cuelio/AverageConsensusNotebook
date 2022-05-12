from average_consensus.IncrementalCostConsensusInstance import IncrementalCostConsensusInstance
from average_consensus.DistributedAverageConsensusInstance import DistributedAverageConsensusInstance


def run_ac_and_icc_batches(instance_sizes, initial_value_type, topology, num_neighbors=4, num_samples=5):
    run_ac_batch(instance_sizes, initial_value_type, topology, num_neighbors, num_samples)
    run_icc_batch(instance_sizes, topology, num_neighbors, num_samples)


def run_ac_batch(instance_sizes, initial_value_type, topology, max_offset=2, num_samples=10):
    rounds_to_convergence = []
    print("Running experiment with " + str(num_samples) + " samples")
    for instance_size in instance_sizes:
        if instance_size % 50 == 0:
            print("Running experiment for " + str(instance_size) + " nodes")

        avg_rounds = 0.0
        for i in range(0, num_samples):
            instance = DistributedAverageConsensusInstance(instance_size, initial_value_type, topology, max_offset)
            instance.execute_instance()
            avg_rounds += instance.rounds_to_convergence

        # print("Took an average of " + str(avg_rounds) + " for " + str(num_samples) + " samples")
        avg_rounds = avg_rounds / num_samples
        rounds_to_convergence.append(avg_rounds)

    return rounds_to_convergence


def run_icc_batch(instance_sizes, topology, max_offset=2, num_samples=10):
    rounds_to_convergence = []
    for instance_size in instance_sizes:
        if instance_size % 50 == 0:
            print("Running experiment for " + str(instance_size) + " nodes")
            # num_neighbors += 4
            # print("increasing number of neighbors to : " + str(num_neighbors))

        avg_rounds = 0
        for i in range(0, num_samples):
            instance = IncrementalCostConsensusInstance(instance_size, topology, max_offset)
            setup_successful = instance.init_starting_values_randomized()
            if not setup_successful:
                raise Exception("Unable to successfully create random starting values")

            instance.execute_instance()
            avg_rounds += instance.rounds_to_convergence

        # print("Took an average of " + str(avg_rounds) + " for " + str(num_samples) + " samples")
        avg_rounds = float(avg_rounds / num_samples)
        rounds_to_convergence.append(avg_rounds)

    return rounds_to_convergence


def run_icc_instance(instance_size, topology, num_neighbors=4, ws_prob=0.05, num_samples=10):
    ws_prob = round(ws_prob, 2)
    avg_rounds = 0
    for i in range(0, num_samples):
        instance = IncrementalCostConsensusInstance(instance_size, topology, num_neighbors, rewire_probability=ws_prob)
        setup_successful = instance.init_starting_values_randomized()
        # instance.init_starting_values()
        if not setup_successful:
            raise Exception("Unable to successfully create random starting values")

        instance.execute_instance()
        avg_rounds += instance.rounds_to_convergence

    avg_rounds = avg_rounds / num_samples
    print("Took " + str(avg_rounds) + " rounds to converge")
    return avg_rounds
