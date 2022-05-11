from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from os import listdir
from os.path import join, isfile
import pandas as pd
import seaborn as sns
from scipy.stats import t, sem

from experiments.test import Test


def plot_factor_graphics(csf_file, output_file, col, y_label):
    data = pd.read_csv(csf_file)
    data_algo = data[data['algorithm'] == 'AlgoWithGini']
    values_by_factor = defaultdict(list)
    for index, row in data_algo.iterrows():
        factor = row["gini_factor"]
        accuracy = row[col]
        values_by_factor[factor].append(accuracy)

    x = []
    average = []
    err = []
    for factor, value in values_by_factor.items():
        x.append(factor)
        average.append(sum(value) / len(value))
        error_interval = t.interval(0.95, len(value) - 1, loc=np.mean(value), scale=sem(value))
        err.append((error_interval[1] - error_interval[0]) / 2)

    f = plt.figure(figsize=(12, 4), dpi=300)
    plt.rcParams.update({
        'text.usetex': True,
        'font.family': 'monospace',
        'font.size': 14,
        'font.monospace': ['Computer Modern Typewriter']})

    ax = f.add_subplot(111)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    plt.plot(x, average, color="blue")
    # plt.errorbar(x, average, yerr=err, color="blue")
    plt.xlabel("FactorExpl")
    plt.ylabel(y_label)
    plt.xticks(np.arange(min(x), max(x) + 0.1, 0.1))
    # plt.yticks(np.arange(0.625, 1, 0.025))
    plt.margins(x=0.05, y=0.05)
    plt.savefig(output_file, bbox_inches='tight')


def plot_trees(results_dir, pickle_filename=None, pickle_dir=None, pruned=False):
    if pickle_filename:
        pickle_files = [pickle_filename]
    else:
        files = [f for f in listdir(pickle_dir) if isfile(join(pickle_dir, f))]
        pickle_files = [f for f in files if 'pickle' in f]
        pickle_files = [f"{pickle_dir}/{pickle_file}" for pickle_file in pickle_files]
        pickle_files = sorted(pickle_files)
    for pickle_file in pickle_files:
        print(f"### Generate image for tree {pickle_file}")
        test = Test.load_test_from_pickle(pickle_file, results_folder=results_dir)
        sub_folder = "img" if not pruned else "img_pruned"
        tree_img_file = test._get_filename(extension="png", gini_factor=test.gini_factors[0],
                                           gamma_factor=test.gamma_factors[0], sub_folder=sub_folder)
        test.clf_obj.tree_.debug_pydot(tree_img_file)
        print("---------")


def plot_boxplot(csf_file, output_file, column, column_label, use_tex=False):
    data = pd.read_csv(csf_file)
    values_by_dataset = defaultdict(list)

    if use_tex:
        plt.rcParams.update({
            'text.usetex': True,
            # 'font.family': 'monospace',
            'font.size': 16,
            # 'font.monospace': ['Computer Modern Typewriter']})
        })

    for index, row in data.iterrows():
        dataset = row['dataset']
        accuracy = row[column]
        values_by_dataset[dataset].append(accuracy)

    # Accuracy
    # datasets_1 = [key for key, values in values_by_dataset.items() if 0.92 < sum(values) / len(values)]
    # datasets_2 = [key for key, values in values_by_dataset.items() if 0.8 < sum(values) / len(values) <= 0.92]
    # datasets_3 = [key for key, values in values_by_dataset.items() if 0.5 < sum(values) / len(values) <= 0.8]
    # datasets_4 = [key for key, values in values_by_dataset.items() if sum(values) / len(values) <= 0.5]

    # WAES
    datasets_1 = [key for key, values in values_by_dataset.items() if 4 < sum(values) / len(values)]
    datasets_2 = [key for key, values in values_by_dataset.items() if 2 < sum(values) / len(values) <= 4]
    datasets_3 = [key for key, values in values_by_dataset.items() if sum(values) / len(values) <= 2]

    datasets = [datasets_1, datasets_2, datasets_3]

    for count, dataset_list in enumerate(datasets):
        df = data[data['dataset'].isin(dataset_list)]
        plt.figure(figsize=(12, 8), dpi=300)
        sns.boxplot(x="dataset", y=column, hue="algorithm", data=df)
        plt.xticks(rotation='50', ha='right', fontsize=16)
        plt.ylabel(column_label, fontsize=16)
        plt.xlabel("Dataset", fontsize=16)
        output_file = output_file.replace(".jpg", "")
        plt.savefig(f"{output_file}_{count}.jpg", bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    # plot_factor_graphics("results/consolidated/algo_gini_experiments.csv", "results/consolidated/accuracy_factors.jpg",
    #                      "test_accuracy", "Test Accuracy")
    # plot_factor_graphics("results/consolidated/algo_gini_experiments.csv", "results/consolidated/wad_factors.jpg",
    #                      "wad", "WAD")
    # plot_factor_graphics("results/consolidated/algo_gini_experiments.csv", "results/consolidated/waes_factors.jpg",
    #                      "waes", "expl\\textsubscript{avg}")
    # plot_factor_graphics("results/consolidated/algo_gini_experiments.csv", "results/consolidated/nodes_factors.jpg",
    #                      "nodes", "Nodes")
    # plot_factor_graphics("results/consolidated/algo_gini_experiments.csv", "results/consolidated/features_factors.jpg",
    #                      "features", "Features")

    plot_boxplot("results/consolidated/experiments.csv", "results/consolidated/boxplot_waes.jpg", 'waes',
                 "\\texttt{expl\\textsubscript{avg}}",
                 use_tex=True)

    # plot_trees(
    #     pickle_filename="results/algo_gini/pickle_pruned/sensorless_AlgoWithGini_depth_4_samples_0_gini-factor_0.97_gamma_0.9.pickle",
    #     results_dir="results/algo_gini", pruned=True)
    # plot_trees(
    #     pickle_filename="results/cart/pickle_pruned/sensorless_Cart_depth_4_samples_0_gini-factor_1.pickle",
    #     results_dir="results/cart", pruned=True)
