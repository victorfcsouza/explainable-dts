import pandas as pd
import numpy as np
import sklearn.tree as sklearn_tree
from sklearn.model_selection import train_test_split
from cart.cart import DecisionTreeClassifier


def print_score_sklearn(X_train, y_train):
    clf = sklearn_tree.DecisionTreeClassifier(criterion="gini", max_depth=5)
    clf.fit(X_train, y_train)
    print("Train accuracy:", round(clf.score(X_train, y_train), 3))
    print("Test accuracy:", round(clf.score(X_test, y_test), 3))


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

    for factor in np.linspace(0.2, 1.0, num=5):
        dt = DecisionTreeClassifier(max_depth=8)
        score_train, score_test = dt.get_score(X_train, y_train, X_test, y_test, modified_factor=factor, debug=False)
        print(f"Train/test accuracy for factor {factor}: {score_train}, {score_test}")
        max_depth, max_depth_redundant, wapl, wapl_redundant = dt.get_explainability_metrics()
        print(f"max_depth:  {max_depth}, max_depth_redundant: {max_depth_redundant}, wapl: {wapl},"
              f" wapl_redundant: {wapl_redundant}")
