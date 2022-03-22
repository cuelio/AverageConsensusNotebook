import matplotlib.pyplot as plt


def plot_values_by_round(values_by_round):
    plt.figure(figsize=(20, 12))
    for i in range(0, len(values_by_round[0])):
        plt.plot(values_by_round[:, i])
    plt.show()
