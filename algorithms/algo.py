import math
import numpy as np

from algorithms.default_algorithm import DefaultClassifier
from tree import tree


class Algo(DefaultClassifier):
    def __init__(self, max_depth=None, min_samples_stop=0):
        super().__init__(max_depth, min_samples_stop)

    @staticmethod
    def _set_objects(X, y, a, condition, value):
        """
        Calculate the set of objects in :param X that have a condition in attribute index attribute :param a.
        Condition can be: "eq",  "leq" or "gt" compared to :param value

        Return subsets X, y that satisfies condition
        """
        result_X = []
        result_y = []

        for i in range(len(y)):
            if condition == 'eq' and X[i][a] == value:
                result_X.append(X[i])
                result_y.append(y[i])
            elif condition == 'leq' and X[i][a] <= value:
                result_X.append(X[i])
                result_y.append(y[i])
            elif condition == 'gt' and X[i][a] > value:
                result_X.append(X[i])
                result_y.append(y[i])

        return result_X, result_y

    @staticmethod
    def _number_pairs(y):
        """
        Returns the number of pairs that have different classes
        """
        count = 0
        y_sorted = sorted(y)

        i = 1
        j = 0
        n = len(y)
        while i < n:
            while i < n and y_sorted[i] == y_sorted[i - 1]:
                i += 1
            count += (i - j) * (n - i)
            j = i
            i += 1
        return count

    def _get_best_threshold_old(self, X, y, a):
        """
        Get threshold that minimizes _cost for attribute a. Also, returns that cost and all thresholds
        """
        a_values = sorted(list(set(X[:, a])))
        thresholds = [(a_values[i] + a_values[i - 1]) / 2 for i in range(1, len(a_values))]
        # In case of attributes with only one value
        if not thresholds:
            thresholds = [a_values[0]]

        min_cost = math.inf
        t_best = 0
        for t in thresholds:
            cost = self._cost(X, y, a, t)
            if cost < min_cost:
                t_best = t
                min_cost = cost

        return t_best, min_cost, thresholds

    def _get_best_threshold(self, X, y, a):
        m = y.size  # tirar
        classes_parent = [np.sum(y == c) for c in range(self.n_classes_)]  # tirar
        thresholds, classes_thr = zip(*sorted(zip(X[:, a], y)))
        classes_left = [0] * self.n_classes_
        classes_right = classes_parent.copy()
        pairs_left = 0
        pairs_right = self._number_pairs(y)
        node_product = pairs_right * len(y)  # tirar
        cost_min = math.inf
        cost_best_balanced = math.inf
        t_min = None
        t_best_balanced = None
        all_thresholds = []
        for i in range(1, m):
            node_class = classes_thr[i - 1]
            classes_left[node_class] += 1
            classes_right[node_class] -= 1
            pairs_left += sum([classes_left[j] for j in range(self.n_classes_) if j != node_class])
            pairs_right -= sum([classes_right[j] for j in range(self.n_classes_) if j != node_class])
            if thresholds[i] == thresholds[i - 1]:
                continue
            threshold = (thresholds[i] + thresholds[i - 1]) / 2
            all_thresholds.append(threshold)
            prod_left = i * pairs_left
            prod_right = (m - i) * pairs_right
            cost = max(prod_left, prod_right)
            if cost < cost_min:
                cost_min = cost
                t_min = threshold
            if cost <= (2 / 3) * node_product and cost < cost_best_balanced:
                cost_best_balanced = cost
                t_best_balanced = threshold
        return t_best_balanced, cost_best_balanced, t_min, cost_min, all_thresholds

    def _cost(self, X, y, a, t):
        """
        Calculate the cost of a binary split for attribute a at threshold t.
        Formula:
        cost = max{P (S(a, <= t)) · |S(a, <= t)|, P (S(a, > t)) · |S(a, > t)|}.

        a is the index attribute (between 0 and m)
        """

        X_left, y_left = self._set_objects(X, y, a, 'leq', t)
        X_right, y_right = self._set_objects(X, y, a, 'gt', t)
        return max(self._number_pairs(y_left) * len(y_left), self._number_pairs(y_right) * len(y_right))

    def _best_split(self, X, y, feature_index_occurrences=None, modified_factor=1):
        """
        Find the next split for a node.
        Returns:
            best_idx: Index of the feature for best split, or None if no split is found.
            best_thr: Threshold to use for the split, or None if no split is found.
            next_thr: If 3-split is necessary, returns the
        """
        # Need at least two elements to split a node.
        m = y.size
        if m <= 1:
            return None, None, None

        node_product = self._number_pairs(y) * len(y)

        # variables for the 2-step partition
        a_min_balanced = None  # attribute that minimizes cost and satisfies cost <= 2/3 * node_product
        t_min_balanced = None  # attribute relative to previous attribute
        cost_min_balanced = node_product

        # variables for the 3-step partition
        a_min = None  # a* - attribute that minimizes cost
        t_min = None  # t* - threshold relative to previous attribute
        cost_min = math.inf
        all_thresholds = []
        # Loop through all features.
        for a in range(self.n_features_):
            threshold_a_balanced, cost_a_balanced, threshold_a_min, cost_a_min, thresholds_a =\
                self._get_best_threshold(X, y, a)
            all_thresholds.append(thresholds_a)
            if cost_a_min < cost_min:
                a_min = a
                t_min = threshold_a_min
                cost_min = cost_a_min
            if cost_a_balanced < cost_min_balanced:
                a_min_balanced = a
                t_min_balanced = threshold_a_balanced
                cost_min_balanced = cost_a_balanced

        if a_min_balanced is not None:
            return a_min_balanced, t_min_balanced, None

        # Else, Perform a 3-step partition
        if t_min is not None:
            X_left, y_left = self._set_objects(X, y, a_min, 'leq', t_min)
            X_right, y_right = self._set_objects(X, y, a_min, 'gt', t_min)
            thresholds = all_thresholds[a_min]
            t_index = thresholds.index(t_min)

            # TODO: Here we can reutilize these calculation from cost functions and run faster
            if self._number_pairs(y_left) * len(y_left) > (2 / 3) * node_product and t_index > 0:
                return a_min, t_min, thresholds[t_index - 1]
            elif self._number_pairs(y_right) * len(y_right) > (2 / 3) * node_product and t_index < len(thresholds) - 1:
                return a_min, t_min, thresholds[t_index + 1]
            else:
                return a_min, t_min, None
        else:
            return None, None, None

    def _create_node(self, y, feature_index=None, threshold=None, feature_index_occurrences=None, calculate_gini=True):
        num_samples_per_class = [np.sum(y == i) for i in range(self.n_classes_)]
        predicted_class = np.argmax(num_samples_per_class)
        node = tree.Node(
            num_samples=y.size,
            num_samples_per_class=num_samples_per_class,
            predicted_class=predicted_class,
            feature_index=feature_index,
            threshold=threshold,
            feature_index_occurrences=feature_index_occurrences.copy()
        )
        if calculate_gini:
            node.gini = self._gini(y)

        return node

    def _grow_tree(self, X, y, depth=0, feature_index_occurrences=None, modified_factor=1, calculate_gini=False):
        """Build a decision tree by recursively finding the best split."""
        # Population for each class in current node. The predicted class is the one with
        # largest population.
        node = self._create_node(y, feature_index_occurrences=feature_index_occurrences, calculate_gini=calculate_gini)

        # Split recursively until maximum depth is reached.
        if depth < self.max_depth and node.num_samples >= self.min_samples_stop:
            idx, thr, next_thr = self._best_split(X, y, feature_index_occurrences=feature_index_occurrences,
                                                  modified_factor=modified_factor)
            if idx is not None:
                indices_left = X[:, idx] <= thr
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]
                node.feature_index = idx
                node.threshold = thr
                node.feature_index_occurrences[idx] += 1

                # 2-split
                if next_thr is None:
                    node.left = self._grow_tree(X_left, y_left, depth + 1,
                                                feature_index_occurrences=node.feature_index_occurrences.copy(),
                                                modified_factor=modified_factor, calculate_gini=calculate_gini)
                    node.right = self._grow_tree(X_right, y_right, depth + 1,
                                                 feature_index_occurrences=node.feature_index_occurrences.copy(),
                                                 modified_factor=modified_factor, calculate_gini=calculate_gini)
                # 3-split on the left
                elif next_thr < thr:
                    child_features = node.feature_index_occurrences.copy()
                    child_features[idx] += 1
                    left_node = self._create_node(y_left, feature_index=idx, threshold=next_thr,
                                                  feature_index_occurrences=child_features,
                                                  calculate_gini=calculate_gini)
                    indices_left = X_left[:, idx] <= thr
                    X_left_left, y_left_left = X_left[indices_left], y_left[indices_left]
                    X_left_right, y_left_right = X_left[~indices_left], y_left[~indices_left]
                    node.left = left_node
                    if depth + 1 < self.max_depth and left_node.num_samples >= self.min_samples_stop:
                        left_node.left = \
                            self._grow_tree(X_left_left, y_left_left, depth + 2,
                                            feature_index_occurrences=left_node.feature_index_occurrences.copy(),
                                            modified_factor=modified_factor, calculate_gini=calculate_gini)
                        left_node.right = \
                            self._grow_tree(X_left_right, y_left_right, depth + 2,
                                            feature_index_occurrences=left_node.feature_index_occurrences.copy(),
                                            modified_factor=modified_factor, calculate_gini=calculate_gini)
                    node.right = self._grow_tree(X_right, y_right, depth + 1,
                                                 feature_index_occurrences=node.feature_index_occurrences.copy(),
                                                 modified_factor=modified_factor, calculate_gini=calculate_gini)

                # 3-split on the right
                elif next_thr > thr:
                    child_features = node.feature_index_occurrences.copy()
                    child_features[idx] += 1
                    right_node = self._create_node(y_right, feature_index=idx, threshold=next_thr,
                                                   feature_index_occurrences=child_features,
                                                   calculate_gini=calculate_gini)
                    indices_right = X_right[:, idx] <= thr
                    X_right_left, y_right_left = X_right[indices_right], y_right[indices_right]
                    X_right_right, y_right_right = X_right[~indices_right], y_right[~indices_right]
                    node.right = right_node
                    node.left = self._grow_tree(X_left, y_left, depth + 1,
                                                feature_index_occurrences=node.feature_index_occurrences.copy(),
                                                modified_factor=modified_factor, calculate_gini=calculate_gini)
                    if depth + 1 < self.max_depth and right_node.num_samples >= self.min_samples_stop:
                        right_node.left = \
                            self._grow_tree(X_right_left, y_right_left, depth + 2,
                                            feature_index_occurrences=right_node.feature_index_occurrences.copy(),
                                            modified_factor=modified_factor, calculate_gini=calculate_gini)
                        right_node.right = \
                            self._grow_tree(X_right_right, y_right_right, depth + 2,
                                            feature_index_occurrences=right_node.feature_index_occurrences.copy(),
                                            modified_factor=modified_factor, calculate_gini=calculate_gini)
                else:
                    raise ValueError("Threshold not permitted!")
        return node

    def fit(self, X, y, modified_factor=1):
        """Build decision tree classifier."""
        self.n_classes_ = len(set(y))  # classes are assumed to go from 0 to n-1
        self.n_samples = len(y)
        self.n_features_ = X.shape[1]
        feature_index_occurrences = [0] * self.n_features_
        self.tree_ = self._grow_tree(X, y, feature_index_occurrences=feature_index_occurrences,
                                     modified_factor=modified_factor, calculate_gini=False)
