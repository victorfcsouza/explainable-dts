from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t, sem


def plot_factor_graphics(csf_file, output_file, col, y_label):
    data = pd.read_csv(csf_file)
    data_algo = data[data['algorithm'] == 'AlgoWithGini']
    accuracy_trains_by_factor = defaultdict(list)
    for index, row in data_algo.iterrows():
        factor = row["factor"]
        accuracy = row[col]
        accuracy_trains_by_factor[factor].append(accuracy)

    x = []
    average = []
    err = []
    for factor, value in accuracy_trains_by_factor.items():
        x.append(factor)
        average.append(sum(value) / len(value))
        error_interval = t.interval(0.95, len(value) - 1, loc=np.mean(value), scale=sem(value))
        err.append((error_interval[1] - error_interval[0]) / 2)

    f = plt.figure(figsize=(12, 4), dpi=300)
    ax = f.add_subplot(111)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    plt.errorbar(x, average, yerr=err, color="blue")
    plt.xticks(np.arange(min(x), max(x) + 0.1, 0.1))
    plt.xlabel("Factor")
    plt.ylabel(y_label)
    plt.margins(x=0.05, y=0.05)
    plt.savefig(output_file, bbox_inches='tight')


if __name__ == "__main__":
    plot_factor_graphics("results/raw.csv", "results/accuracy_factors.jpg", "test_accuracy", "Test Accuracy")
    plot_factor_graphics("results/raw.csv", "results/wad_factors.jpg", "wapl", "WAD")
    plot_factor_graphics("results/raw.csv", "results/waes_factor.jpg", "wapl_redundant", "WAES")
