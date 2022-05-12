import time

from shared_types.types import InitialValueSetup, TopologyLayout
from average_consensus.batch_run import run_icc_batch, run_ac_batch, run_ac_and_icc_batches, run_icc_instance
import numpy as np


def main():
    probabilities = np.arange(0.05, 0.06, 0.01)
    rounds_to_convergence = []

    instance_size = 10
    offset = 1

    for probability in probabilities:
        print("Probability: " + str(probability))
        avg_rounds_to_converge = run_icc_instance(instance_size, TopologyLayout.LATTICE_RING, offset,
                                                  ws_prob=probability, num_samples=1)
        print(avg_rounds_to_converge)
        rounds_to_convergence.append(avg_rounds_to_converge)


if __name__ == "__main__":
    main()
