import csv
import json
import matplotlib.pyplot as plt
import numpy as np
from os import listdir
from os.path import isfile, join
import pandas as pd
import sklearn.tree as sklearn_tree
from test_cart import MetricType, Test

RESULTS_FOLDER = "results"


def print_score_sklearn(X_train, y_train, X_test, y_test):
    clf = sklearn_tree.DecisionTreeClassifier(criterion="gini", max_depth=5)
    clf.fit(X_train, y_train)
    print("Train accuracy:", round(clf.score(X_train, y_train), 3))
    print("Test accuracy:", round(clf.score(X_test, y_test), 3))


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

    rows = sorted(rows, key=lambda x: (x[0], x[4], x[5], x[6]))
    with open(csv_file, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        # write the header
        writer.writerow(header)
        # write multiple rows
        writer.writerows(rows)


if __name__ == "__main__":
    datasets = [
        # name, path, bin, col_name, cols_to_delete
        ['avila', '../data/avila/avila.csv', False, 'Class', []],

        ['cardiotocography', '../data/cardiotocography/CTG.csv', False, 'CLASS',
         ['b', 'e', 'LBE', 'DR', 'Tendency', 'A', 'B', 'C', 'D', 'E', 'AD', 'DE', 'LD', 'FS', 'SUSP']],

        ['defaults_credit_card', "../data/defaults_credit_card/defaults_credit_card.csv", False,
         'default payment next month', ['ID', 'SEX', 'EDUCATION', 'MARRIAGE', 'PAY_0', 'PAY_2', 'PAY_3',
                                        'PAY_4', 'PAY_5', 'PAY_6']],

        ['dry_bean', "../data/dry_bean/Dry_Bean_Dataset.csv", False, 'Class', []],

        ['eeg_eye_state', '../data/eeg_eye_state/eeg_eye_state.csv', False, 'eyeDetection', []],

        ['letter_recognition', '../data/letter_recognition/letter-recognition.csv', False, 'lettr', []],

        ['obs_network', '../data/obs_network/obs_network_dataset.csv', False, 'Class', ['id', 'Node', 'NodeStatus']],

        ['occupancy_room', '../data/occupancy_room/Occupancy_Estimation.csv', False, 'Room_Occupancy_Count',
         ['Date', 'Time', 'S6_PIR', 'S7_PIR']],

        ['online_shoppers_intention', '../data/online_shoppers_intention/online_shoppers_intention.csv', False,
         'Revenue', ['Month', 'OperatingSystems', 'Browser', 'Region', 'TrafficType', 'VisitorType', 'Weekend']],

        ['pen_digits', "../data/pen_digits/pendigits.csv", False, 'digit', []]

    ]

    # Add bins versions to dataset list:
    new_datasets = datasets.copy()
    for ds in datasets:
        # create_bins(ds[1], cols_to_remove=ds[-1] + [ds[-2]])
        extension_index = ds[1].find(".csv")
        bins_filename = ds[1][:extension_index] + "_bins.csv"
        new_ds = ds.copy()
        new_ds[1] = bins_filename
        new_ds[2] = True
        new_datasets.append(new_ds)

    all_datasets = sorted(new_datasets, key=lambda x: (x[0], x[2]))

    depths = [6, 7, 8]

    for ds in all_datasets:
        for depth in depths:
            test = Test(ds[0], ds[1], depth, ds[2], ds[3], cols_to_delete=ds[4], min_samples_stop=30)
            # test.run()

    generate_consolidates_csv("results/consolidated/cart_experiments.csv", "results")

    # plot_opt_table(bins=False)
    # plot_opt_table(bins=True)
