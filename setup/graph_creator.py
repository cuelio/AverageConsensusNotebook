import numpy as np
import networkx as nx
from shared_types.types import TopologyLayout


def get_graph(graph_type, n, max_lattice_offset=1, rewire_probability=0.2):
    if graph_type == TopologyLayout.RING:
        graph = nx.circulant_graph(n, [1])
        return convert_graph_to_laplacian(graph)
    elif graph_type == TopologyLayout.LATTICE_RING:
        offset = list(range(1, max_lattice_offset + 1))
        graph = nx.circulant_graph(n, offset)
        return convert_graph_to_laplacian(graph)
    elif graph_type == TopologyLayout.K_REGULAR_EVEN_SPACED:
        return get_k_regular_ring_evenly_spaced(n, max_lattice_offset)
    elif graph_type == TopologyLayout.WATTS_STROGATZ:
        graph = nx.connected_watts_strogatz_graph(n, max_lattice_offset, rewire_probability)
        return convert_graph_to_laplacian(graph)
    else:
        print("Unsupported topology passed as input: " + str(graph_type))


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
