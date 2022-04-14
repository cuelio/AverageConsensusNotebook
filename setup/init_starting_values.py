import numpy as np
import random
from shared_types.types import InitialValueSetup


def get_values(value_setup_type, n, first_value=0, second_value=1):
    if value_setup_type == InitialValueSetup.GROUPED:
        return grouped(n, first_value, second_value)
    elif value_setup_type == InitialValueSetup.ALTERNATED:
        return alternated(n, first_value, second_value)
    else:
        print("Unsupported value setup type: " + str(value_setup_type))


def alternated(n, value1, value2):
    """
    :param n: Number of nodes in graph
    :param value1: Value to be assigned
    :param value2: Values to be assigned
    :return: List of alternating values, e.g. [0, 1, 0, 1]
    """
    values = np.zeros(n)
    for i in range(0, n):
        if i % 2 == 0:
            values[i] = value1
        else:
            values[i] = value2
    return values


def grouped(n, value1, value2):
    """
    :param n: Number of nodes in graph
    :param value1: Value to be assigned
    :param value2: Value to be assigned
    :return: List of grouped values, e.g. [0, 0, 1, 1]
    """
    values = np.zeros(n, dtype=float)
    for i in range(0, n):
        if i < n / 2:
            values[i] = value1
        else:
            values[i] = value2
    return values


def alternated_and_grouped(n, value1, value2, k):
    """
    :param n: Number of nodes in graph
    :param value1: Value to be assigned
    :param value2: Value to be assigned
    :param k: The index at which we switch from alternated to grouped
    :return: A list of alternated values up to index k, appended with a list of grouped values from k to n
    e.g. (k = 4, n = 10): [0, 1, 0, 1] + [0, 0, 0, 1, 1, 1] = [0, 1, 0, 1, 0, 0, 0, 1, 1, 1]
    """
    if k == 0:
        return grouped(n, value1, value2)
    elif k == n:
        return alternated(n, value1, value2)
    else:
        a1 = alternated(k, value1, value2)
        a2 = grouped(n - k, value1, value2)
        return np.append(a2, a1)


def random_floats(n, max_value):
    """
    :param n: Number of nodes in graph
    :param max_value: Upper ceiling on random values
    :return: A list of n random floats between 0 and max_value
    """
    values = np.zeros(n)
    for i in range(0, n):
        values[i] = random.random() * max_value

    return values

def random_ints(n, max_value):
    """
    :param n: Number of nodes in graph
    :param max_value:  Upper ceiling on random values
    :return: A list of n random integers between 0 and max_value
    """
    values = np.zeros(n)
    for i in range(0, n):
        values[i] = random.randint(0, max_value)

    return values


def normal_dist(n, mu, sigma):
    """
    :param n: Number of nodes in graph
    :param mu: Mean
    :param sigma: Standard deviation
    :return: A list of n values sampled from a normal distribution
    """
    return np.random.normal(mu, sigma, n)


def normal_dist_alternated(n, mu1, mu2, sigma):
    """
    :param n: Number of nodes in graph
    :param mu1: The mean of the first normal distribution
    :param mu2: The mean of the second normal distribution
    :param sigma: The standard deviation of the normal distributions
    :return: A list of alternated values sampled from two different normal distributions
    """
    values = np.zeros(n)
    for i in range(0, n):
        if i % 2 == 0:
            values[i] = np.random.normal(mu1, sigma)
        else:
            values[i] = np.random.normal(mu2, sigma)
    return values


def normal_dist_grouped(n, mu1, mu2, sigma):
    """
    :param n: Number of nodes in graph
    :param mu1: The mean of the first normal distribution
    :param mu2: The mean of the second normal distribution
    :param sigma: The standard deviation of the normal distributions
    :return: A list of grouped values sampled from two different normal distributions
    """
    values = np.zeros(n)
    for i in range(0, n):
        if i < n / 2:
            values[i] = np.random.normal(mu1, sigma)
        else:
            values[i] = np.random.normal(mu2, sigma)
    return values


def normal_dist_alt_and_grouped(n, mu1, mu2, k, sigma):
    """
    :param n: Number of nodes in graph
    :param mu1: The mean of the first normal distribution
    :param mu2: The mean of the second normal distribution
    :param k: The index at which we switch from alternated to grouped
    :param sigma: The standard deviation of the normal distributions
    :return: A list of k alternated values appended with n-k grouped values,
             each taken from two different normal distributions
    """
    if k == 0:
        return normal_dist_grouped(n, mu1, mu2, sigma)
    elif k == n:
        return normal_dist_alternated(n, mu1, mu2, sigma)
    else:
        a1 = normal_dist_alternated(k, mu1, mu2, sigma)
        a2 = normal_dist_grouped(n - k, mu1, mu2, sigma)
        return np.append(a2, a1)
