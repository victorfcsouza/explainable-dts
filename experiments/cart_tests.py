import pandas as pd
import numpy as np
import sklearn.tree as sklearn_tree
from sklearn.model_selection import train_test_split
from cart.cart import DecisionTreeClassifier
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure


def print_score_sklearn(X_train, y_train):
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


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    data = pd.read_csv("../data/dry_bean/Dry_Bean_Dataset.csv")

    data = data.astype({"Class": str})

    data['Class'] = data['Class'].rank(method='dense', ascending=False).astype(int)

    X = data.drop(labels='Class', axis=1).to_numpy()
    y = data['Class'].astype('int').to_numpy() - 1

    unique, counts = np.unique(y, return_counts=True)
    dict(zip(unique, counts))

    # Training Models
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, train_size=0.7)

    results = []
    for factor in np.linspace(0.1, 1.0, num=10):
        dt = DecisionTreeClassifier(max_depth=8)
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

    # for factor in np.linspace(0.1, 1.0, num=10):
    #     from random import uniform
    #     results.append({
    #         'factor': factor,
    #         'train_accuracy': uniform(0.9, 1.0),
    #         'test_accuracy': uniform(0.9, 1.0),
    #         'max_depth': uniform(7, 8),
    #         'max_depth_redundant': uniform(6, 8),
    #         'wapl': uniform(6, 8),
    #         'wapl_redundant': uniform(6, 8)
    #     })
    plot_graphic([x['factor'] for x in results],
                 [x['train_accuracy'] for x in results],
                 [x['test_accuracy'] for x in results],
                 [x['max_depth'] for x in results],
                 [x['max_depth_redundant'] for x in results],
                 [x['wapl'] for x in results],
                 [x['wapl_redundant'] for x in results],
                 filename='../images/dry_bean.png')
