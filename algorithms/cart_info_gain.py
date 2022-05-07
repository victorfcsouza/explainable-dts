"""Implementation of the CART algorithm to train decision tree classifiers."""
from algorithms.cart import Cart


class CartInfoGain(Cart):
    def __init__(self, max_depth=None, min_samples_stop=0):
        super().__init__(max_depth, min_samples_stop)

    def _best_split(self, X, y, feature_index_occurrences=None, modified_factor=1,
                    father_feature=None, impurity_metric="gini"):
        return super()._best_split(X=X, y=y, feature_index_occurrences=feature_index_occurrences,
                                   modified_factor=modified_factor, father_feature=father_feature,
                                   impurity_metric="entropy")
