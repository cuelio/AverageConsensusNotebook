from shared_types.types import TopologyLayout
from average_consensus.batch_run import run_icc_batch, run_ac_batch, run_icc_instance
import numpy as np
import setup.graph_creator as graph_creator
import matplotlib.pyplot as plt
import setup.step_size_generator as step_size_generator


def main():
    rounds_to_convergence = []

    instance_size = 14
    num_nbrs = 4
    # step_sizes = np.arange(0.0001, 0.0002, 0.00005)

    step_sizes = [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01]
    # step_sizes = step_size_generator.get_step_sizes(0.00001, 0.01, 5)
    for step_size in step_sizes:
        step_size = np.round(step_size, 5)
        print("Running for step size: " + str(step_size))
        avg_rounds_to_converge = run_icc_instance(instance_size, TopologyLayout.LATTICE_RING, num_nbrs,
                                                  num_samples=5, step_size=step_size)
        rounds_to_convergence.append(avg_rounds_to_converge)

    print(rounds_to_convergence)
    plt.plot(step_sizes, rounds_to_convergence, label="rounds")
    plt.show()


if __name__ == "__main__":
    main()
