import numpy as np
import tree.tree as tree


class DefaultClassifier:
    def __init__(self, max_depth=None, min_samples_stop=0):
        self.max_depth = max_depth
        self.min_samples_stop = min_samples_stop
        self.n_classes_ = None
        self.n_samples = None
        self.n_features_ = None
        self.tree_: tree = None

    def fit(self, X, y, modified_factor=1):
        """Build decision tree classifier."""
        self.n_classes_ = len(set(y))  # classes are assumed to go from 0 to n-1
        self.n_samples = len(y)
        self.n_features_ = X.shape[1]
        feature_index_occurrences = [0] * self.n_features_
        self.tree_ = self._grow_tree(X, y, feature_index_occurrences=feature_index_occurrences,
                                     modified_factor=modified_factor)

    def predict(self, X):
        """Predict class for X."""
        return [self._predict(inputs) for inputs in X]

    def score(self, X, y):
        correct = 0
        predicted_labels = self.predict(X)
        for i in range(len(y)):
            if predicted_labels[i] == y[i]:
                correct += 1
        return correct / len(y)

    def debug(self, feature_names, class_names, show_details=True):
        """Print ASCII visualization of decision tree."""
        self.tree_.debug(feature_names, class_names, show_details)

    def _gini(self, y):
        """Compute Gini impurity of a non-empty node.

        Gini impurity is defined as Σ p(1-p) over all classes, with p the frequency of a
        class within the node. Since Σ p = 1, this is equivalent to 1 - Σ p^2.
        """
        m = y.size
        return 1.0 - sum((np.sum(y == c) / m) ** 2 for c in range(self.n_classes_))

    def _best_split(self, X, y, feature_index_occurrences=None, modified_factor=1):
        raise NotImplementedError()

    def _grow_tree(self, X, y, depth=0, feature_index_occurrences=None, modified_factor=1, calculate_gini=True):
        """Build a decision tree by recursively finding the best split."""
        # Population for each class in current node. The predicted class is the one with
        # largest population.
        num_samples_per_class = [np.sum(y == i) for i in range(self.n_classes_)]
        predicted_class = np.argmax(num_samples_per_class)
        node = tree.Node(
            num_samples=y.size,
            num_samples_per_class=num_samples_per_class,
            predicted_class=predicted_class,
            feature_index_occurrences=feature_index_occurrences.copy()
        )
        if calculate_gini:
            node.gini = self._gini(y)

        # Split recursively until maximum depth is reached.
        if depth < self.max_depth and node.num_samples >= self.min_samples_stop:
            idx, thr = self._best_split(X, y, feature_index_occurrences=feature_index_occurrences,
                                        modified_factor=modified_factor)
            if idx is not None:
                indices_left = X[:, idx] < thr
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]
                node.feature_index = idx
                node.threshold = thr
                node.feature_index_occurrences[idx] += 1
                node.left = self._grow_tree(X_left, y_left, depth + 1,
                                            feature_index_occurrences=node.feature_index_occurrences.copy(),
                                            modified_factor=modified_factor)
                node.right = self._grow_tree(X_right, y_right, depth + 1,
                                             feature_index_occurrences=node.feature_index_occurrences.copy(),
                                             modified_factor=modified_factor)
        return node

    def _predict(self, inputs):
        """Predict class for a single sample."""
        node = self.tree_
        while node.left:
            if inputs[node.feature_index] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.predicted_class

    def get_score(self, X_train, y_train, X_test, y_test, modified_factor=1):
        self.fit(X_train, y_train, modified_factor=modified_factor)
        return round(self.score(X_train, y_train), 3), round(self.score(X_test, y_test), 3)

    def get_explainability_metrics(self):
        # Returns max_depth, max_depth_redundant, wapl, wapl_redundant
        return self.tree_.get_explainability_metrics(self.n_features_)
