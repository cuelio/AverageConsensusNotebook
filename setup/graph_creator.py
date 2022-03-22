import numpy as np
import random


def get_ring(n):
    """
    :param n: Number of nodes in graph
    :return: N x N laplacian matrix that represents a ring where each node is connected to its direct neighbors
    i.e. 2-regular ring
    """
    laplacian = np.zeros((n, n), dtype=int)
    for i in range(0, n):
        for offset in range(-1, 2):
            nbr_index = i + offset
            if nbr_index >= n:
                nbr_index = nbr_index % n
            if offset == 0:
                laplacian[i][i] = 2
            else:
                laplacian[i][nbr_index] = -1
    return laplacian


def get_lattice_ring(n, k):
    """
    :param n:
    :param k: Number of neighbors for each node, i.e. a k-regular graph (if k is odd, attach to opposite node)
    :return: k-regular lattice ring, i.e. each node is connected to its k/2 closest neighbors on each side,
             unless k is odd. In that case one neighbor is the opposite node
    """
    if n % 2 == 1 and k % 2 == 1:
        print("NOT A VALID GRAPH\n")
        exit()
    laplacian = np.zeros((n, n), dtype=int)
    # if k is odd
    if k % 2 == 1:
        # if k is odd, must attach to opposite neighbor and (k-1)/2 closest on either side
        for i in range(0, n):
            for offset in range(int(-(k - 1) / 2), int((k - 1) / 2) + 1):
                nbr_index = i + offset
                if nbr_index >= n:
                    nbr_index = nbr_index % n
                if offset == 0:
                    laplacian[i][i] = k
                else:
                    laplacian[i][nbr_index] = -1
                laplacian[i][int(i - n / 2)] = -1
    # if k is even
    else:
        for i in range(0, n):
            for offset in range(int(-k / 2), int(k / 2) + 1):
                nbr_index = i + offset
                if nbr_index >= n:
                    nbr_index = nbr_index % n
                if offset == 0:
                    laplacian[i][i] = k
                else:
                    laplacian[i][nbr_index] = -1
    return laplacian


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


def get_watts_strogatz(n, k):
    random.seed(112)
    beta = 0.4
    laplacian = get_lattice_ring(n, k)

    for i in range(0, n):
        for j in range(int(-k / 2), int(k / 2) + 1 % n):
            if j == -1 or j == 0 or j == 1:
                continue

            nbr_index = get_index_from_offset(i, j, n)

            if random.random() < beta:
                rewire_to = random.randint(0, n - 1)
                while not valid_rewire(laplacian, i, rewire_to, n):
                    rewire_to = random.randint(0, n - 1)

                laplacian[i][nbr_index] = 0
                laplacian[nbr_index][i] = 0
                laplacian[nbr_index][nbr_index] -= 1

                laplacian[i][rewire_to] = -1
                laplacian[rewire_to][i] = -1
                laplacian[rewire_to][rewire_to] += 1
    return laplacian


def valid_rewire(laplacian, index, rewire_to, n):
    if rewire_to < 0 or rewire_to >= n:
        print("Not a valid index")
        return False
    elif abs(rewire_to - index) <= 1:
        print("Self loops and ring neighbors not allowed")
        return False
    elif laplacian[index][rewire_to] == -1:
        print("Edge already exists")
        return False
    else:
        return True


def get_index_from_offset(index, offset, n):
    nbr_index = index + offset
    if nbr_index >= n:
        return nbr_index % n
    elif nbr_index < 0:
        return n - nbr_index
    else:
        return nbr_index
