import pandas as pd
import numpy as np
# import sklearn.tree as tree
from sklearn.model_selection import train_test_split
from cart import DecisionTreeClassifier

np.set_printoptions(suppress=True)
data = pd.read_csv("data/Dry_Bean_Dataset.csv")

data = data.astype({"Class": str})

data['Class'] = data['Class'].rank(method='dense', ascending=False).astype(int)

X = data.drop(labels='Class', axis=1).to_numpy()
y = data['Class'].astype('int').to_numpy() - 1

print("X: ")
print(X)
print("---")
print("y: ")
print(y)
print("classes: ", set(y))

unique, counts = np.unique(y, return_counts=True)
dict(zip(unique, counts))

# Training Models
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, train_size=0.7)


# def print_score_sklearn():
#     clf = tree.DecisionTreeClassifier(criterion="gini", max_depth=5)
#     clf.fit(X_train, y_train)
#     print("Train accuracy:", round(clf.score(X_train, y_train), 3))
#     print("Test accuracy:", round(clf.score(X_test, y_test), 3))


def print_score_scratch():
    clf = DecisionTreeClassifier(max_depth=5)
    clf.fit(X_train, y_train)
    print("Train accuracy:", round(clf.score(X_train, y_train), 3))
    print("Test accuracy:", round(clf.score(X_test, y_test), 3))


# print_score_sklearn()
print_score_scratch()
