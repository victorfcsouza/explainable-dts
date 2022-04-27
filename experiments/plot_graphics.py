from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from os import listdir
from os.path import join, isfile
import pandas as pd
import pickle
from scipy.stats import t, sem

from algorithms.algo_with_gini import AlgoWithGini
from algorithms.cart import Cart
from experiments.test import Test
from tree.tree import Node


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
    # plt.plot(x, average, color="blue")
    plt.xticks(np.arange(min(x), max(x) + 0.1, 0.1))
    plt.xlabel("Factor")
    plt.ylabel(y_label)
    plt.margins(x=0.05, y=0.05)
    plt.savefig(output_file, bbox_inches='tight')

#
# def plot_trees(pickle_dir):
#     files = [f for f in listdir(pickle_dir) if isfile(join(pickle_dir, f))]
#     pickle_files = [f for f in files if 'pickle' in f]
#     pickle_files = sorted(pickle_files)
#     rows = []
#     for pickle_filename in pickle_files:
#         pickle_info = pickle_filename.replace(".pickle", "").split("_")
#         dataset = pickle_info[0]
#         algorithm = pickle_info[1]
#         depth_index = pickle_info.index("depth")
#         max_depth_stop = int(pickle_info[depth_index + 1])
#         min_samples_stop_index = int(pickle_info.index("samples"))
#         min_samples = pickle_info[min_samples_stop_index + 1]
#         gini_factor_index = pickle_info.index("gini-factor")
#         gini_factor = float(pickle_info[gini_factor_index + 1])
#         gamma_factor_index = pickle_info.index("gamma")
#         try:
#             gamma = float(pickle_info[gamma_factor_index + 1])
#         except ValueError:
#             gamma = None
#         # Get Train and test accuracy
#         if algorithm == "AlgoWithGini":
#             clf = AlgoWithGini
#         elif algorithm == "Cart":
#             clf = Cart
#         else:
#             raise ValueError("Classifier not found!")
#         print(f"### Generating results for {dataset} with {algorithm}, "
#               f"max_depth {max_depth_stop}, min_samples_stop {min_samples}, gini factor {gini_factor} "
#               f"and gamma {gamma}")
#
#         with open(f"{pickle_dir}/{pickle_filename}", "rb") as f:
#             tree_obj: Node = pickle.load(f)
#             num_features = len(tree_obj.feature_index_occurrences)
#             clf_obj = clf()
#             clf_obj.tree_ = tree_obj
#             test = Test(classifier=clf, dataset_name=dataset, max_depth_stop=max_depth_stop,
#                         min_samples_stop=min_samples, csv_file=None)
#             tree_img_file = test._get_filename(extension="png", gini_factor=gini_factor,
#                                        gamma_factor=gamma, sub_folder="img")
#             dt.tree_.debug_pydot(tree_img_file)

if __name__ == "__main__":
    # plot_factor_graphics("results/consolidated/algo_gini_experiments.csv", "results/consolidated/accuracy_factors.jpg",
    #                      "test_accuracy", "Test Accuracy")
    # plot_factor_graphics("results/consolidated/algo_gini_experiments.csv", "results/consolidated/wad_factors.jpg",
    #                      "wad", "WAD")
    # plot_factor_graphics("results/consolidated/algo_gini_experiments.csv", "results/consolidated/waes_factors.jpg",
    #                      "waes", "WAES")
    plot_factor_graphics("results/consolidated/Experiments - CART modification - algo_gini.csv", "results/consolidated/accuracy_factors_old.jpg",
                         "test_accuracy", "Test Accuracy")
    plot_factor_graphics("results/consolidated/Experiments - CART modification - algo_gini.csv", "results/consolidated/wad_factors_old.jpg",
                         "wapl", "WAD")
    plot_factor_graphics("results/consolidated/Experiments - CART modification - algo_gini.csv", "results/consolidated/waes_factors_old.jpg",
                         "wapl_redundant", "WAES")
