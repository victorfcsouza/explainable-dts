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


if __name__ == "__main__":
    # test_dataset("dry_bean", "../data/dry_bean/Dry_Bean_Dataset.csv", "Class", max_depth_tree=8)
    # test_dataset("dry_bean_bins", "../data/dry_bean/Dry_Bean_Dataset_bins.csv", "Class", max_depth_tree=8, bins=True)
    #
    # test_dataset("avila", "../data/avila/avila-tr.csv", "Class", max_depth_tree=10)
    # test_dataset("avila_bins", "../data/avila/avila-tr_bins.csv", "Class", max_depth_tree=10, bins=True)
    #
    # test_dataset("obs_network", "../data/obs_network/obs_network_dataset.csv", "Class",
    #              columns_to_delete=['NodeStatus', 'id'], max_depth_tree=10)
    # test_dataset("obs_network_bins", "../data/obs_network/obs_network_dataset_bins.csv", "Class",
    #              columns_to_delete=['NodeStatus', 'id'], max_depth_tree=10, bins=True)
    #
    # test_dataset("cardiotocography", "../data/cardiotocography/CTG.csv", "CLASS",
    #              columns_to_delete=['Tendency', 'A', 'B', 'C', 'D', 'E', 'AD', 'DE', 'LD', 'FS', 'SUSP'],
    #              max_depth_tree=5)
    # test_dataset("cardiotocography_bins", "../data/cardiotocography/CTG_bins.csv", "CLASS",
    #              columns_to_delete=['Tendency', 'A', 'B', 'C', 'D', 'E', 'AD', 'DE', 'LD', 'FS', 'SUSP'],
    #              max_depth_tree=5, bins=True)

    # test_dataset("default_credit_card", "../data/default_credit_card/defaults_credit_card.csv",
    #              "default payment next month", columns_to_delete=['ID'], max_depth_tree=10)
    # test_dataset("default_credit_card_bins", "../data/default_credit_card/defaults_credit_card_bins.csv",
    #              "default payment next month", columns_to_delete=['ID'], max_depth_tree=10, bins=True)
    #
    # test_dataset("eeg_eye_state", "../data/eeg_eye_state/eeg_eye_state.csv", "eyeDetection", max_depth_tree=10)
    # test_dataset("eeg_eye_state_bins", "../data/eeg_eye_state/eeg_eye_state_bins.csv", "eyeDetection",
    #              max_depth_tree=10, bins=True)
    #
    # test_dataset("letter_recognition", "../data/letter_recognition/letter-recognition.csv", "lettr", max_depth_tree=13)
    # test_dataset("letter_recognition_bins", "../data/letter_recognition/letter-recognition_bins.csv", "lettr",
    #              max_depth_tree=13, bins=True)

    # test_dataset("online_shopers_intention", "../data/online_shoppers_intention/online_shoppers_intention.csv",
    #              "Revenue",
    #              columns_to_delete=['Month'], max_depth_tree=9)
    # test_dataset("online_shopers_intention_bins",
    #              "../data/online_shoppers_intention/online_shoppers_intention_bins.csv", "Revenue",
    #              columns_to_delete=['Month'], max_depth_tree=9, bins=True)
    #
    # test_dataset("pen_based_digit_recognition", "../data/pen_digits/pendigits.csv", "digit", max_depth_tree=10)
    # test_dataset("pen_based_digit_recognition_bins", "../data/pen_digits/pendigits_bins.csv", "digit",
    #              max_depth_tree=10, bins=True)
    #
    # test_dataset("room_occupation", "../data/occupancy_room/Occupancy_Estimation.csv", "Room_Occupancy_Count",
    #              columns_to_delete=['Date', 'Time'], max_depth_tree=10)
    # test_dataset("room_occupation_bins", "../data/occupancy_room/Occupancy_Estimation_bins.csv", "Room_Occupancy_Count",
    #              columns_to_delete=['Date', 'Time'], max_depth_tree=10, bins=True)


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
            test = Test(ds[0], ds[1], depth, ds[2],  ds[3], cols_to_delete=ds[4])
            test.run()

    # plot_opt_table(bins=False)
    # plot_opt_table(bins=True)
