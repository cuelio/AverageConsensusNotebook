from enum import Enum


class ConsensusAlgorithm:
    AC = "AVERAGE_CONSENSUS"
    ICC = "INCREMENTAL_COST_CONSENSUS"


class EdgeWeightType(Enum):
    OPTIMAL_CONSTANT = "OPTIMAL_CONSTANT"
    MAX_DEGREE = "MAX_DEGREE"
    LOCAL_DEGREE = "LOCAL_DEGREE"
    MEAN_METROPOLIS = "MEAN_METROPOLIS"


class InitialValueSetup(Enum):
    ALTERNATED = "ALTERNATED"
    GROUPED = "GROUPED"
    RANDOM = "RANDOM"


class TopologyLayout(Enum):
    LATTICE_RING = "LATTICE_RING"
    WATTS_STROGATZ = "WATTS_STROGATZ"
    RANDOM_REGULAR = "RANDOM_REGULAR"
    RANDOM_TREE = "RANDOM_TREE"
    FULL_RARY_TREE = "FULL_RARY_TREE"
    K_REGULAR_EVEN_SPACED = "K_REGULAR_EVEN_SPACED"

