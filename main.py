from shared_types.types import TopologyLayout
from average_consensus.batch_run import run_icc_batch, run_ac_batch, run_ac_and_icc_batches, run_icc_instance


def main():
    rounds_to_convergence = []

    instance_size = 14
    offset = 2
    avg_rounds_to_converge = run_icc_instance(instance_size, TopologyLayout.LATTICE_RING, offset,
                                              ws_prob=0.1, num_samples=1)
    rounds_to_convergence.append(avg_rounds_to_converge)


if __name__ == "__main__":
    main()
