import numpy as np


class Node:
    def __init__(self, node_id, starting_value):
        self.id = node_id
        self.current_value = starting_value
        self.neighbor_ids = []
        self.neighbor_values = []
        self.neighbor_weights = []
        self.value_history = []
        self.self_weight = 0

    def update_value(self):
        self.value_history.append(self.current_value)
        self.current_value = np.multiply(self.neighbor_values, self.neighbor_weights)

    def get_value_history(self):
        return self.value_history

    def get_current_value(self):
        return self.current_value


