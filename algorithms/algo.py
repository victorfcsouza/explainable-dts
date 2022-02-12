from algorithms.default_algorithm import DefaultClassifier

import math


class AlgoClassifier(DefaultClassifier):
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
        equal_numbers = 0
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

    def _get_best_threshold(self, X, y, a):
        """
        Get threshold that minimizes _cost for attribute a. Also, returns that cost
        """
        a_values = list(set(X[:, a]))
        thresholds = [(a_values[i] + a_values[i - 1]) / 2 for i in range(1, len(a_values))]
        min_cost = math.inf
        t_best = 0
        for t in thresholds:
            cost = self._cost(X, y, a, t)
            if cost < min_cost:
                t_best = t
                min_cost = cost

        return t_best, min_cost

    def _get_best_attr(self, X, y):
        """
        Get attribute and threshold that minimizes cost. Also, return that cost
        """
        min_cost = math.inf
        a_best = 0
        t_best = 0
        for a in range(len(X)):
            t_best_for_a, min_cost_for_a = self._get_best_threshold(X, y, a)
            if min_cost_for_a < min_cost:
                min_cost = min_cost_for_a
                a_best = a
                t_best = t_best_for_a

        return a_best, t_best, min_cost

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
        """
        # Need at least two elements to split a node.
        m = y.size
        if m <= 1:
            return None, None

        node_product = self._number_pairs(y) * len(y)
        cost_min = node_product
        a_best = 0  # For the 2 level split
        t_best = 0  # For the 2 level split

        # Loop through all features.
        for a in range(self.n_features_):
            threshold_a_best, cost_a = self._get_best_threshold(X, y, a)
            if cost_a <= (2 / 3) * node_product:
                return a, threshold_a_best
            if cost_a < cost_min:
                cost_min = cost_a
                a_best = a
                t_best = threshold_a_best

        raise RuntimeError("No split done yet! Perform a 2 level split")

    def fit(self, X, y, modified_factor=1):
        """Build decision tree classifier."""
        self.n_classes_ = len(set(y))  # classes are assumed to go from 0 to n-1
        self.n_samples = len(y)
        self.n_features_ = X.shape[1]
        feature_index_occurrences = [0] * self.n_features_
        self.tree_ = self._grow_tree(X, y, feature_index_occurrences=feature_index_occurrences,
                                     modified_factor=modified_factor, calculate_gini=False)
