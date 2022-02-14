from experiments.test import MetricType, Test
from algorithms.cart import DecisionTreeClassifier
from algorithms.algo import AlgoClassifier

import csv
import json
import matplotlib.pyplot as plt
import numpy as np
from os import listdir
from os.path import isfile, join
import pandas as pd
from pathlib import Path

RESULTS_FOLDER = "results"


def change_all_jsons(results_folder):
    files = [f for f in listdir(results_folder) if isfile(join(results_folder, f))]
    json_files = [f for f in files if 'json' in f]
    for file in json_files:
        with open(results_folder + "/" + file, "r+") as json_file:
            file_data = json.load(json_file)

            # Change json
            # insert code here

            json_file.seek(0)
            json.dump(file_data, json_file, indent=2, sort_keys=True)
            json_file.truncate()
            print("Changed file:", file)


def plot_opt_table(bins=False):
    """
    Compares metrics between datasets
    """
    files = [f for f in listdir(RESULTS_FOLDER) if isfile(join(RESULTS_FOLDER, f))]
    json_files = [f for f in files if 'json' in f]

    diff_data = []
    cols = [MetricType.factor.value, MetricType.train_accuracy.value, MetricType.test_accuracy.value,
            MetricType.max_depth.value, MetricType.max_depth_redundant.value, MetricType.wapl.value,
            MetricType.wapl_redundant.value]
    rows = []
    for file in json_files:
        with open(f"{RESULTS_FOLDER}/{file}") as json_file:
            file_data = json.load(json_file)
            if file_data['bins'] is not bins:
                # Wrong file regarding bins
                continue

            # Get the only result
            opt_dict = file_data['opt_factor']
            rows.append(file_data['dataset'])
            diff_data.append([opt_dict[col] for col in cols])

    # Sort by wapl_redundant
    # int(x[1][-1][:-2]) converts wapl_redundant to int without the % symbol
    rows, diff_data = zip(*sorted(zip(rows, diff_data), key=lambda x: int(x[1][-1][:-1])))

    # Plot Table
    plt.rcParams["figure.figsize"] = [14, 12]
    plt.rcParams["figure.autolayout"] = True
    fig, ax = plt.subplots()
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(
        cellText=diff_data,
        rowLabels=rows,
        colLabels=cols,
        rowColours=["palegreen"] * 10,
        colColours=["palegreen"] * 10,
        cellLoc='center',
        loc='upper left')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    ax.set_title(f'Metric diffs with bins = {bins}', fontsize=14)
    plt.savefig(f"{RESULTS_FOLDER}/metric_diffs_bins_{bins}.png", dpi=300)


def create_bins(csv_file, bins=10, cols_to_remove=None):
    data = pd.read_csv(csv_file)
    cols = data.columns
    cols = [col for col in cols if col not in cols_to_remove]
    for col in cols:
        possible_values = set(data[col].tolist())
        if len(possible_values) > 10:
            min_value = min(possible_values)
            max_value = max(possible_values)
            limits = np.linspace(min_value, max_value, num=bins + 1)
            conditions = [data[col] <= choice for choice in limits[1:]]
            choices = [(limits[i] + limits[i + 1]) / 2 for i in range(len(limits) - 1)]
            data[col] = np.select(conditions, choices)

    index_str = csv_file.find(".csv")
    outfile = csv_file[:index_str] + "_bins" + ".csv"
    data.to_csv(outfile, encoding='utf-8', index=None)


def generate_consolidates_csv(csv_file, result_dir):
    files = [f for f in listdir(result_dir) if isfile(join(result_dir, f))]
    json_files = [f for f in files if 'json' in f]

    header = ['dataset', 'n_samples', 'n_features', 'n_classes', 'max_depth_stop', 'min_samples_stop', 'discrete',
              'factor', 'train_accuracy', 'test_accuracy', 'max_depth', 'max_depth_redundant', 'wapl', 'wapl_redundant']
    rows = []
    for file in json_files:
        with open(result_dir + "/" + file) as json_file:
            file_data = json.load(json_file)
            for result in file_data['results']:
                row = [file_data['dataset'], file_data['n_samples'], file_data['n_features'], file_data['n_classes'],
                       file_data['max_depth_stop'], file_data['min_samples_stop'], file_data['bins']]
                for it in header[7:]:
                    row.append(result[it])
                rows.append(row)

    rows = sorted(rows, key=lambda x: (x[0], x[4], x[5], x[6], x[7]))
    # Create dir if not exists
    dir_index = csv_file.rfind("/")
    dir_name = csv_file[:dir_index]
    Path(dir_name).mkdir(parents=True, exist_ok=True)
    with open(csv_file, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        # write the header
        writer.writerow(header)
        # write multiple rows
        writer.writerows(rows)


def one_hot_encoding(csv_file, categorical_cols=None, cols_to_remove=None):
    """
    For the categorical (and not numerical) columns, apply one hot encoding
    """
    data = pd.read_csv(csv_file)
    if cols_to_remove:
        data = data.drop(columns=cols_to_remove)
    if categorical_cols:
        prefix_names = {col: f"dummy_{col}" for col in categorical_cols}
        data = pd.get_dummies(data, columns=categorical_cols, prefix=prefix_names)

    index_str = csv_file.find(".csv")
    outfile = csv_file[:index_str] + "_formatted.csv"
    data.to_csv(outfile, encoding='utf-8', index=False)


if __name__ == "__main__":
    datasets = [
        # name, path, bin, col_class_name, categorical_cols, cols_to_delete
        ['avila', '../data/avila/avila_formatted.csv', False, 'Class', [], []],

        ['cardiotocography', '../data/cardiotocography/CTG_formatted.csv', False, 'CLASS', [],
         ['b', 'e', 'LBE', 'DR', 'Tendency', 'A', 'B', 'C', 'D', 'E', 'AD', 'DE', 'LD', 'FS', 'SUSP']],

        ['defaults_credit_card', "../data/defaults_credit_card/defaults_credit_card_formatted.csv", False,
         'default payment next month', ['SEX', 'EDUCATION', 'MARRIAGE'], ['ID']],

        ['dry_bean', "../data/dry_bean/Dry_Bean_Dataset_formatted.csv", False, 'Class', [], []],

        ['eeg_eye_state', '../data/eeg_eye_state/eeg_eye_state_formatted.csv', False, 'eyeDetection', [], []],

         # deu erro max_depth 8
        # ['letter_recognition', '../data/letter_recognition/letter-recognition_formatted.csv', False, 'lettr', [], []],

        # deu erro
        # ['obs_network', '../data/obs_network/obs_network_dataset_formatted.csv', False, 'Class', ['Node', 'NodeStatus'],
        #  ['id']],

        ['occupancy_room', '../data/occupancy_room/Occupancy_Estimation_formatted.csv', False, 'Room_Occupancy_Count',
         [], ['Date', 'Time']],

        # Should also include 'TrafficType' for categorial col, but has many values
        ['online_shoppers_intention', '../data/online_shoppers_intention/online_shoppers_intention_formatted.csv',
         False, 'Revenue', ['Month', 'OperatingSystems', 'Browser', 'Region', 'Weekend'], []],

        ['pen_digits', "../data/pen_digits/pendigits_formatted.csv", False, 'digit', [], []]

    ]

    # new_datasets = datasets.copy()
    # Add bins versions to dataset list:
    # for ds in datasets:
    #     # create_bins(ds[1], cols_to_remove=ds[-1] + [ds[-2]])
    #     extension_index = ds[1].find(".csv")
    #     bins_filename = ds[1][:extension_index] + "_bins.csv"
    #     new_ds = ds.copy()
    #     new_ds[1] = bins_filename
    #     new_ds[2] = True
    #     new_datasets.append(new_ds)

    # Apply one hot encoding to datasets:
    # for ds in datasets:
    #     one_hot_encoding(ds[1], ds[4], ds[5])

    all_datasets = sorted(datasets, key=lambda x: (x[0], x[2]))
    depths = [6, 7, 8]
    min_samples_list = [0]
    # min_samples_list = [0, 30, 100]

    for ds in all_datasets:
        for depth in depths:
            for min_samples_stop in min_samples_list:
                test1 = Test(AlgoClassifier, ds[0], ds[1], depth, ds[2], ds[3],
                             min_samples_stop=min_samples_stop, factors=[1], results_folder="results/algo")
                test2 = Test(DecisionTreeClassifier, ds[0], ds[1], depth, ds[2], ds[3],
                             min_samples_stop=min_samples_stop, factors=[0.8, 0.9, 0.95, 1],
                             results_folder="results/cart")
                test1.run()
                test2.run()

    generate_consolidates_csv("results/algo/consolidated/algo_experiments.csv", "results/algo")
    generate_consolidates_csv("results/cart/consolidated/cart_experiments.csv", "results/cart")

    # plot_opt_table(bins=False)
    # plot_opt_table(bins=True)
