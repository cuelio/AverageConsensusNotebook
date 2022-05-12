import numpy as np
from shared_types.types import EdgeWeightType


def get_edge_weights(edge_wt_type, laplacian, n):
    if edge_wt_type == EdgeWeightType.OPTIMAL_CONSTANT:
        return optimal_constant(laplacian, n)
    elif edge_wt_type == EdgeWeightType.MEAN_METROPOLIS:
        return mean_metropolis(laplacian, n)
    elif edge_wt_type == EdgeWeightType.LOCAL_DEGREE:
        return local_degree(laplacian, n)
    elif edge_wt_type == EdgeWeightType.MAX_DEGREE:
        return max_degree(laplacian, n)
    else:
        print("Unsupported edge weight type: " + str(edge_wt_type))


def optimal_constant(laplacian, n):
    weights = np.zeros((n, n))
    eigenvalues = np.sort(np.linalg.eigvals(laplacian))
    edge_wt = 2 / (eigenvalues[1] + eigenvalues[-1])
    for i in range(0, n):
        for j in range(i + 1, n):
            if laplacian[i][j] == -1:
                weights[i][j] = edge_wt
                weights[j][i] = edge_wt
        weights[i][i] = (1 - laplacian[i][i] * edge_wt)
    return weights


def max_degree(laplacian, n):
    weights = np.zeros((n, n))
    max_deg = -1
    for i in range(0, n):
        if laplacian[i][i] > max_deg:
            max_deg = laplacian[i][i]
    for i in range(0, n):
        for j in range(i + 1, n):
            if laplacian[i][j] == -1:
                weights[i][j] = 1 / max_deg
                weights[j][i] = 1 / max_deg
        weights[i][i] = (1 - laplacian[i][i] * (1 / max_deg))
    return weights


def local_degree(laplacian, n):
    weights = np.zeros((n, n))
    for i in range(0, n):
        sum_of_weights = 0
        for j in range(0, n):
            if i == j:
                continue
            if laplacian[i][j] == -1:
                local_max_deg = max(laplacian[i][i], laplacian[j][j])
                weights[i][j] = 1 / local_max_deg
                sum_of_weights += 1 / local_max_deg
        weights[i][i] = (1 - sum_of_weights)
    return weights


def mean_metropolis(laplacian, n, delta=0.01):
    weights = np.zeros((n, n))
    for i in range(0, n):
        sum_of_weights = 0
        for j in range(0, n):
            if i == j:
                continue
            if laplacian[i, j] == -1:
                weights[i, j] = 2 / (laplacian[i, i] + laplacian[j, j] + delta)
                sum_of_weights += weights[i, j]
        weights[i, i] = 1 - sum_of_weights
    return weights
