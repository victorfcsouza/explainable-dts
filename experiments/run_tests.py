import csv
import json
import matplotlib.pyplot as plt
import numpy as np
from os import listdir
from os.path import isfile, join
import pandas as pd
from pathlib import Path
import pickle

from experiments.test import MetricType, Test, ResultJson, Metrics
from algorithms.cart import Cart
from algorithms.algo_with_gini import AlgoWithGini
from tree.tree import Node

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
    cols = [MetricType.gini_factor.value, MetricType.train_accuracy.value, MetricType.test_accuracy.value,
            MetricType.max_depth.value, MetricType.max_depth_redundant.value, MetricType.wad.value,
            MetricType.waes.value]
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

    # Sort by waes
    # int(x[1][-1][:-2]) converts waes to int without the % symbol
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


def generate_consolidates_csv(csv_file, result_dir, load_from="json", ds_by_name: dict = None):
    files = [f for f in listdir(result_dir) if isfile(join(result_dir, f))]
    test_info_header = ResultJson.get_cols()
    metrics_header = Metrics.get_cols()
    header = test_info_header + metrics_header

    if load_from == "json":
        json_files = [f for f in files if 'json' in f]
        rows = []
        for file in json_files:
            with open(result_dir + "/" + file) as json_file:
                file_data = json.load(json_file)
                for result in file_data['results']:
                    row = [file_data[info_header] for info_header in test_info_header]
                    row += [result[metric] for metric in metrics_header]
                    rows.append(row)
    # Load from pickles
    elif load_from == "pickle":
        print("####################")
        print("Generating consolidated results:")
        pickle_files = [f for f in files if 'pickle' in f]
        pickle_files = sorted(pickle_files)
        rows = []
        for pickle_filename in pickle_files:
            pickle_info = pickle_filename.replace(".pickle", "").split("_")
            dataset = pickle_info[0]
            algorithm = pickle_info[1]
            depth_index = pickle_info.index("depth")
            max_depth_stop = int(pickle_info[depth_index + 1])
            min_samples_stop_index = int(pickle_info.index("samples"))
            min_samples = pickle_info[min_samples_stop_index + 1]
            gini_factor_index = pickle_info.index("gini-factor")
            gini_factor = float(pickle_info[gini_factor_index + 1])
            gamma_factor_index = pickle_info.index("gamma")
            try:
                gamma = float(pickle_info[gamma_factor_index + 1])
            except ValueError:
                gamma = None
            # Get Train and test accuracy
            if algorithm == "AlgoWithGini":
                clf = AlgoWithGini
            elif algorithm == "Cart":
                clf = Cart
            else:
                raise ValueError("Classifier not found!")
            print(f"### Generating results for {dataset} with {algorithm}, "
                  f"max_depth {max_depth_stop}, min_samples_stop {min_samples}, gini factor {gini_factor} "
                  f"and gamma {gamma}")

            with open(f"{result_dir}/{pickle_filename}", "rb") as f:
                tree_obj: Node = pickle.load(f)
                num_features = len(tree_obj.feature_index_occurrences)
                clf_obj = clf()
                clf_obj.tree_ = tree_obj
                test = Test(classifier=clf, dataset_name=dataset, csv_file=ds_by_name[dataset]["csv"],
                            max_depth_stop=max_depth_stop, col_class_name=ds_by_name[dataset]["col_class"],
                            min_samples_stop=min_samples_stop)
                X, y = test.parse_dataset()
                X_train, X_test, y_train, y_test = test.split_train_test(X, y)
                score_train, score_test = clf_obj.get_score_without_fit(X_train, y_train, X_test, y_test)
                metrics = tree_obj.get_explainability_metrics(num_features)
                row = [dataset, algorithm, max_depth_stop, min_samples, gini_factor, gamma, score_train, score_test,
                       *metrics]
                rows.append(row)

    else:
        raise ValueError("load_from can be json or pickle")

    rows = sorted(rows, key=lambda x: (x[0], x[4], x[5], x[6]))
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
        # name, path, col_class_name, categorical_cols, cols_to_delete
        # ['anuran', '../data/anuran/anuran_formatted.csv', 'Family', [], ['Genus', 'Species', 'RecordID']],
        #
        # ['audit risk', '../data/audit_data/audit_risk_formatted.csv', 'Risk', [], []],
        #
        # ['avila', '../data/avila/avila_formatted.csv', 'Class', [], []],
        #
        # ['banknote', '../data/banknote/data_banknote_authentication.csv', 'class', [], []],
        #
        # ['bankruptcy polish', '../data/bankruptcy_polish/3year.csv', 'class', [], []],
        #
        # ['cardiotocography', '../data/cardiotocography/CTG_formatted.csv', 'CLASS', [],
        #  ['b', 'e', 'LBE', 'DR', 'Tendency', 'A', 'B', 'C', 'D', 'E', 'AD', 'DE', 'LD', 'FS', 'SUSP']],
        #
        # ['collins', "../data/collins/collins_formatted.csv", 'Corp.Genre', [],
        #  ["Text", "Genre", "Counter", "Corpus"]],
        #
        # ['default credit card', "../data/defaults_credit_card/defaults_credit_card_formatted.csv",
        #  'default payment next month', ['SEX', 'EDUCATION', 'MARRIAGE'], ['ID']],
        #
        # ['dry bean', "../data/dry_bean/Dry_Bean_Dataset_formatted.csv", 'Class', [], []],
        #
        # ['eeg eye state', '../data/eeg_eye_state/eeg_eye_state_formatted.csv', 'eyeDetection', [], []],
        #
        # ['htru2', '../data/HTRU2/HTRU_2.csv', 'class', [], []],
        #
        # ['iris', '../data/iris/iris.csv', 'class', [], []],
        #
        # # deu erro max_depth 8
        # ['letter recognition', '../data/letter_recognition/letter-recognition_formatted.csv', 'lettr', [], []],
        #
        # ['mice', '../data/mice/mice_formatted.csv', 'class', ["Genotype", "Treatment", "Behavior"], ["MouseID"]],
        #
        # ['obs network', '../data/obs_network/obs_network_dataset_formatted.csv', 'Class',
        #  ['Node', 'NodeStatus'], ['id']],
        #
        # ['occupancy room', '../data/occupancy_room/Occupancy_Estimation_formatted.csv', 'Room_Occupancy_Count',
        #  [], ['Date', 'Time']],
        #
        # # Should also include 'TrafficType' for categorial col, but has many values
        # ['online shoppers intention', '../data/online_shoppers_intention/online_shoppers_intention_formatted.csv',
        #  'Revenue', ['Month', 'OperatingSystems', 'Browser', 'Region', 'Weekend'], []],
        #
        # ['pen digits', "../data/pen_digits/pendigits_formatted.csv", 'digit', [], []],
        #
        # ['poker hand', "../data/poker_hand/poker_hand.csv", 'class', [], []],

        ['sensorless', "../data/sensorless/sensorless_drive_diagnosis.csv", 'class', [], []]

    ]

    # Add discrete versions to dataset list:
    # new_datasets = datasets.copy()
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
    depths = [4]
    min_samples_list = [0]

    # Run tests
    for ds in all_datasets:
        for depth in depths:
            for min_samples_stop in min_samples_list:
                # algo_name, path, col_class_name, categorical_cols, cols_to_delete
                # Cols to deleted was already deleted
                test1 = Test(classifier=Cart, dataset_name=ds[0], csv_file=ds[1], max_depth_stop=depth,
                             col_class_name=ds[2], cols_to_delete=[], min_samples_stop=min_samples_stop,
                             results_folder="results/cart", gini_factors=[1], gamma_factors=[None])
                test2 = Test(classifier=AlgoWithGini, dataset_name=ds[0], csv_file=ds[1], max_depth_stop=depth,
                             col_class_name=ds[2], cols_to_delete=[], min_samples_stop=min_samples_stop,
                             results_folder="results/algo_gini", gini_factors=[0.97],
                             gamma_factors=[0.9])
                test1.run(debug=True)
                test2.run(debug=True)

    datasets_by_name = {
        ds[0]: {
            "csv": ds[1],
            "col_class": ds[2],
        }
        for ds in datasets}

    # generate_consolidates_csv("results/consolidated/cart_experiments.csv", "results/cart/json",
    #                           load_from="json")
    # generate_consolidates_csv("results/consolidated/algo_gini_experiments.csv", "results/algo_gini/json",
    #                           load_from="json", ds_by_name=datasets_by_name)
    # generate_consolidates_csv("results/consolidated/cart_experiments.csv", "results/cart/pickle",
    #                           load_from="pickle", ds_by_name=datasets_by_name)
    # generate_consolidates_csv("results/consolidated/algo_gini_experiments.csv", "results/algo_gini/pickle",
    #                           load_from="pickle", ds_by_name=datasets_by_name)

    # plot_opt_table(bins=False)
    # plot_opt_table(bins=True)
