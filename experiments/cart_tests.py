import pandas as pd
import numpy as np
import sklearn.tree as sklearn_tree
from sklearn.model_selection import train_test_split
from cart.cart import DecisionTreeClassifier
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import json


def print_score_sklearn(X_train, y_train, X_test, y_test):
    clf = sklearn_tree.DecisionTreeClassifier(criterion="gini", max_depth=5)
    clf.fit(X_train, y_train)
    print("Train accuracy:", round(clf.score(X_train, y_train), 3))
    print("Test accuracy:", round(clf.score(X_test, y_test), 3))


def plot_graphic(factors, train_accuracy_list, test_accuracy_list, max_depth_list, max_depth_redundant_list, wapl_list,
                 wapl_redundant_list, filename='test.png'):
    figure(figsize=(12, 10), dpi=300)
    plt.subplot(2, 1, 1)
    plt.title('Accuracy versus redundancy metrics', fontsize=16)
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


def test_dataset(dataset_name, csv_file, column_class_name, columns_to_delete=None, max_depth_tree=5):
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
                 filename=f'results/{dataset_name}.png')
    store_results({'tree_max_depth': max_depth_tree, "results": results},
                  filename=f"results/{dataset_name}.txt")


if __name__ == "__main__":
    # test_dataset("dry_bean", "../data/dry_bean/Dry_Bean_Dataset.csv", "Class", max_depth_tree=8)
    # test_dataset("avila", "../data/avila/avila-tr.csv", "Class", max_depth_tree=10)
    # test_dataset("obs_network", "../data/obs_network/obs_network_dataset.csv", "Class",
    #              columns_to_delete=['NodeStatus'], max_depth_tree=10)
    # test_dataset("cardiotocography", "../data/cardiotocography/CTG.csv", "CLASS",
    #              columns_to_delete=['Tendency', 'A', 'B', 'C', 'D', 'E', 'AD', 'DE', 'LD', 'FS', 'SUSP'],
    #              max_depth_tree=5)
    test_dataset("default_credit_card", "../data/default_credit_card/defaults_credit_card.csv",
                 "default payment next month", columns_to_delete=['ID'], max_depth_tree=5)
