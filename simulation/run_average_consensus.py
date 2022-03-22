import numpy as np


def run_avg_consensus(values, edge_wts, epsilon=0.01):
    true_avg = np.mean(values)
    # print(true_avg)
    rounds = 0
    values_by_round = [list(values)]
    while not is_stopping_condition_satisfied(values, epsilon):
        values = np.dot(edge_wts, values)
        values_by_round.append(list(values))
        rounds += 1

    # print(values_by_round)
    # print(len(values_by_round))

    values_by_round = np.array(values_by_round)

    return values_by_round


def is_stopping_condition_satisfied(values, epsilon):
    max_value = np.max(values)
    min_value = np.min(values)
    diff = abs(max_value - min_value)
    if diff <= epsilon:
        # if(max_value < true_avg or min_value > true_avg):
        #     print("Distributed Average does not overlap true average")
        return True
    else:
        return False
