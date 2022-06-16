import numpy as np
import networkx as nx
from shared_types.types import TopologyLayout


# Round to next highest even number > 4
def get_number_of_nbrs(n, nbr_ratio, min_nbrs=4):
    num_nbrs = round(n * nbr_ratio)

    if num_nbrs <= min_nbrs:
        return min_nbrs
    else:
        if num_nbrs % 2 == 0:
            return num_nbrs
        else:
            return num_nbrs + 1


def get_graph(graph_type, n, num_nbrs=4, ws_rewire_prob=0.5):
    attempts = 0
    is_connected = False
    graph = None
    while not is_connected:
        attempts += 1
        if attempts > 10:
            raise Exception("Unable to generate a connected graph")

        if graph_type == TopologyLayout.LATTICE_RING:
            graph = nx.connected_watts_strogatz_graph(n, num_nbrs, 0.0)
        elif graph_type == TopologyLayout.WATTS_STROGATZ:
            graph = nx.connected_watts_strogatz_graph(n, num_nbrs, ws_rewire_prob)
        elif graph_type == TopologyLayout.RANDOM_REGULAR:
            graph = nx.random_regular_graph(num_nbrs, n)
        elif graph_type == TopologyLayout.RANDOM_TREE:
            graph = nx.random_tree(n)
        elif graph_type == TopologyLayout.FULL_RARY_TREE:
            graph = nx.full_rary_tree(num_nbrs, n)
        else:
            print("Unsupported topology passed as input: " + str(graph_type))

        is_connected = nx.is_connected(graph)
    return graph


# TODO: Not sure if this is necessary or if it works well
def get_k_regular_ring_evenly_spaced(n, k):
    spacing = k - 2 + 1
    if n % spacing != 0:
        print("NOT A VALID GRAPH\n")
        exit()
    laplacian = np.zeros((n, n))
    for i in range(0, n):
        laplacian[i][i - 1] = -1
        laplacian[i][i] = k
        laplacian[i][i + 1] = -1
        nbr_index = i + spacing
        while nbr_index < n:
            laplacian[i][nbr_index] = -1
            nbr_index += spacing
    return laplacian


def convert_graph_to_laplacian(graph):
    matrix = nx.to_numpy_matrix(graph, dtype=int) * -1
    for i in range(0, int(graph.number_of_nodes())):
        matrix[i, i] = abs(np.sum(matrix[i]))

    return matrix
