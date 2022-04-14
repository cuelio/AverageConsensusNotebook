from enum import Enum


class EdgeWeightType(Enum):
    OPTIMAL_CONSTANT = "OPTIMAL_CONSTANT"
    MAX_DEGREE = "MAX_DEGREE"
    LOCAL_DEGREE = "LOCAL_DEGREE"
    MEAN_METROPOLIS = "MEAN_METROPOLIS"


class InitialValueSetup(Enum):
    ALTERNATED = "ALTERNATED"
    GROUPED = "GROUPED"


class TopologyLayout(Enum):
    RING = "RING"
    LATTICE_RING = "LATTICE_RING"
    K_REGULAR_EVEN_SPACED = "K_REGULAR_EVEN_SPACED"
    WATTS_STROGATZ = "WATTS_STROGATZ"