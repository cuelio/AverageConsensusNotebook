import numpy as np


def get_step_sizes(min_step_size, max_step_size, num_decimals):
    current_step = min_step_size
    step_interval = np.round(1 / (10 ** num_decimals), num_decimals)
    increase_step_interval_at = np.round(1 / (10 ** (num_decimals - 1)), num_decimals - 1)
    step_sizes = []
    while current_step <= max_step_size:
        step_sizes.append(current_step)
        current_step = np.round(current_step + step_interval, num_decimals)
        if current_step >= increase_step_interval_at:
            num_decimals = num_decimals - 1
            step_interval = np.round(1 / (10 ** num_decimals), num_decimals)
            increase_step_interval_at = np.round(1 / (10 ** (num_decimals - 1)), num_decimals - 1)
    return step_sizes
