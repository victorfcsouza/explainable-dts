import math
import numpy as np

from algorithms.algo import Algo


class AlgoWithGini(Algo):
    def __init__(self, max_depth=None, min_samples_stop=0):
        super().__init__(max_depth, min_samples_stop)

    def _get_best_threshold(self, X, y, a, classes_parent, node_pairs, node_product):
        m = y.size  # tirar
        thresholds, classes_thr = zip(*sorted(zip(X[:, a], y)))
        classes_left = [0] * self.n_classes_
        classes_right = classes_parent.copy()
        pairs_left = 0
        pairs_right = node_pairs
        cost_min = math.inf
        t_min = None
        t_best_gini = None
        best_gini = math.inf
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
            gini_left = 1.0 - sum(
                (classes_left[x] / i) ** 2 for x in range(self.n_classes_)
            )
            gini_right = 1.0 - sum(
                (classes_right[x] / (m - i)) ** 2 for x in range(self.n_classes_)
            )
            gini = (i * gini_left + (m - i) * gini_right) / m

            if cost < cost_min:
                cost_min = cost
                t_min = threshold
            if gini < best_gini and cost <= (2 / 3) * node_product:
                best_gini = gini
                t_best_gini = threshold
        return t_best_gini, best_gini, t_min, cost_min, all_thresholds

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

        node_pairs = self._number_pairs(y)
        node_product = node_pairs * y.size
        classes_parent = [np.sum(y == c) for c in range(self.n_classes_)]  # tirar

        # variables for the 2-step partition
        a_min_gini = None  # attribute that minimizes Gini and satisfies cost <= 2/3 * node_product
        t_min_gini = None  # attribute relative to previous attribute
        min_gini = math.inf

        # variables for the 3-step partition
        a_min = None  # a* - attribute that minimizes cost
        t_min = None  # t* - threshold relative to previous attribute
        cost_min = math.inf
        all_thresholds = []

        # Loop through all features.
        for a in range(self.n_features_):
            threshold_a_gini, gini_a, threshold_a_min, cost_a_min, thresholds_a = \
                self._get_best_threshold(X, y, a, classes_parent, node_pairs, node_product)
            modified_gini_a = gini_a * modified_factor if feature_index_occurrences[a] else gini_a

            all_thresholds.append(thresholds_a)
            if cost_a_min < cost_min:
                a_min = a
                t_min = threshold_a_min
                cost_min = cost_a_min
            if modified_gini_a < min_gini:
                a_min_gini = a
                t_min_gini = threshold_a_gini
                min_gini = modified_gini_a

        if a_min_gini is not None:
            return a_min_gini, t_min_gini, None

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
