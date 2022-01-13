import json
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
import numpy as np
from os import listdir
from os.path import isfile, join
import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn.tree as sklearn_tree

from cart.cart import DecisionTreeClassifier

RESULTS_FOLDER = "results"


def print_score_sklearn(X_train, y_train, X_test, y_test):
    clf = sklearn_tree.DecisionTreeClassifier(criterion="gini", max_depth=5)
    clf.fit(X_train, y_train)
    print("Train accuracy:", round(clf.score(X_train, y_train), 3))
    print("Test accuracy:", round(clf.score(X_test, y_test), 3))


def plot_graphic(factors, train_accuracy_list, test_accuracy_list, max_depth_list, max_depth_redundant_list, wapl_list,
                 wapl_redundant_list, dataset_name, filename='test.png'):
    figure(figsize=(12, 10), dpi=300)
    plt.subplot(2, 1, 1)
    plt.title(dataset_name, fontsize=16)
    plt.plot(factors, train_accuracy_list, label="Train Accuracy", color='red', marker='o')
    plt.plot(factors, test_accuracy_list, label="Test Accuracy", color='darkred', marker='o')
    plt.ylabel("Accuracy", fontsize=16)
    plt.legend(loc="lower right", fontsize=10)

    plt.subplot(2, 1, 2)
    plt.plot(factors, max_depth_list, label="Max Depth", color='cyan', marker='o')
    plt.plot(factors, max_depth_redundant_list, label="Redundant Max Depth", color='blue', marker='o')
    plt.plot(factors, wapl_list, label="WAPL (Weighted Average Path Length", color='gray', marker='o')
    plt.plot(factors, wapl_redundant_list, label="Redundant WAPL", color='black', marker='o')
    plt.xlabel("Gini Factor", fontsize=16)
    plt.ylabel("Metric", fontsize=16)
    plt.legend(loc="lower right", fontsize=10)
    plt.savefig(filename)


def store_results(result_json, filename):
    with open(filename, 'w') as f:
        json.dump(result_json, f)


def get_opt_factor(results, opt_factor=0.90):
    """
    Get the lower factor that allow a decreasing in test_accuracy not lower than opt_factor.
    Also, get the diff compared to original values (factor=1) in percentage
    """
    # Assume that results are ordered by increasing factors
    reversed_results = sorted(results, key=lambda x: x['factor'], reverse=True)
    max_test_accuracy = reversed_results[0]['test_accuracy']

    max_metrics = reversed_results[0]

    opt_index = -1
    for i in range(1, len(reversed_results)):
        if reversed_results[i]['test_accuracy'] <= opt_factor * max_test_accuracy:
            opt_index = i - 1
            break
    opt_metrics = reversed_results[opt_index]

    diff_results = dict()
    for key in max_metrics:
        if key == "factor":
            diff_results[key] = opt_metrics[key]
        else:
            diff_metric = round(100 * (opt_metrics[key] - max_metrics[key]) / max_metrics[key])
            diff_results[key] = str(diff_metric) + "%"

    return diff_results


def plot_opt_table(bins=False):
    """
    Compares metrics between datasets
    """
    files = [f for f in listdir(RESULTS_FOLDER) if isfile(join(RESULTS_FOLDER, f))]
    json_files = [f for f in files if 'json' in f]

    diff_data = []
    cols = ['factor', 'train_accuracy', 'test_accuracy', 'max_depth', 'max_depth_redundant', 'wapl', 'wapl_redundant']
    rows = []
    for file in json_files:
        with open(f"{RESULTS_FOLDER}/{file}") as json_file:
            file_data = json.load(json_file)
            results_list = file_data['results']
            metric_list = [result['metrics'] for result in results_list if result['bins'] == bins]
            if not metric_list:
                # Wrong file regarding bins
                continue
            else:
                # Get the only result
                metric_list = metric_list[0]
            opt_dict = get_opt_factor(metric_list)
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


def test_dataset(dataset_name, csv_file, column_class_name, columns_to_delete=None, max_depth_tree=5, bins=False):
    print("############################")
    print("Running tests for", dataset_name)
    # Read CSV
    np.set_printoptions(suppress=True)
    data = pd.read_csv(csv_file)
    if columns_to_delete:
        data = data.drop(labels=columns_to_delete, axis=1)
    data = data.astype({column_class_name: str})

    # Read cols as float
    cols = list(data.columns)
    cols.remove(column_class_name)
    for col in cols:
        data[col] = data[col].astype(float)

    data[column_class_name] = data[column_class_name].rank(method='dense', ascending=False).astype(int)

    X = data.drop(labels=column_class_name, axis=1).to_numpy()
    y = data[column_class_name].astype('int').to_numpy() - 1

    unique, counts = np.unique(y, return_counts=True)
    dict(zip(unique, counts))

    # Training Models
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, train_size=0.7)

    results = []
    for factor in np.linspace(0.1, 1.0, num=10):
        dt = DecisionTreeClassifier(max_depth=max_depth_tree)
        score_train, score_test = dt.get_score(X_train, y_train, X_test, y_test, modified_factor=factor, debug=False)
        print(f"Train/test accuracy for factor {factor}: {score_train}, {score_test}")
        max_depth, max_depth_redundant, wapl, wapl_redundant = dt.get_explainability_metrics()
        print(f"max_depth:  {max_depth}, max_depth_redundant: {max_depth_redundant}, wapl: {wapl},"
              f" wapl_redundant: {wapl_redundant}")
        results.append({
            'factor': round(factor, 2),
            'train_accuracy': score_train,
            'test_accuracy': score_test,
            'max_depth': round(max_depth, 3),
            'max_depth_redundant': round(max_depth_redundant, 3),
            'wapl': round(wapl, 3),
            'wapl_redundant': round(wapl_redundant, 3)
        })

    plot_graphic([x['factor'] for x in results],
                 [x['train_accuracy'] for x in results],
                 [x['test_accuracy'] for x in results],
                 [x['max_depth'] for x in results],
                 [x['max_depth_redundant'] for x in results],
                 [x['wapl'] for x in results],
                 [x['wapl_redundant'] for x in results],
                 dataset_name=dataset_name,
                 filename=f'results/{dataset_name}.png')
    store_results({
        'dataset': dataset_name,
        'results': [{'tree_max_depth': max_depth_tree, "bins": bins, 'metrics': results}]},
        filename=f"{RESULTS_FOLDER}/{dataset_name}.json")
    print("Ended testes for", dataset_name)
    print("############################")


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

    plot_opt_table(bins=False)
    plot_opt_table(bins=True)
    # Create bins
    # create_bins("../data/avila/avila-tr.csv", cols_to_remove=['Class'])
    # create_bins("../data/obs_network/obs_network_dataset.csv", cols_to_remove=['NodeStatus', 'id', 'Class'])
    # create_bins("../data/cardiotocography/CTG.csv",
    #             cols_to_remove=['CLASS', 'Tendency', 'A', 'B', 'C', 'D', 'E', 'AD', 'DE', 'LD', 'FS', 'SUSP'])
    # create_bins("../data/default_credit_card/defaults_credit_card.csv", cols_to_remove=['ID', 'default payment next month'])
    # create_bins("../data/eeg_eye_state/eeg_eye_state.csv", cols_to_remove=['eyeDetection'])
    # create_bins("../data/letter-recognition.csv", cols_to_remove=['lettr'])
    # create_bins("../data/online_shoppers_intention.csv", cols_to_remove=['Month', 'Revenue'])
    # create_bins("../data/pendigits.csv", cols_to_remove=['digit'])
    # create_bins("../data/Occupancy_Estimation.csv", cols_to_remove=['Date', 'Time', 'Room_Occupancy_Count'])
