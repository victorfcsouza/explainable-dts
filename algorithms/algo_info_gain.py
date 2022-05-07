from algorithms.algo_with_gini import AlgoWithGini


class AlgoInfoGain(AlgoWithGini):
    def __init__(self, max_depth=None, min_samples_stop=0):
        super().__init__(max_depth, min_samples_stop)

    def _get_best_threshold(self, X, y, a, classes_parent, node_pairs, node_product, gamma_factor,
                            impurity_criterion="gini"):
        return super()._get_best_threshold(X=X, y=y, a=a, classes_parent=classes_parent, node_pairs=node_pairs,
                                           node_product=node_product, gamma_factor=gamma_factor,
                                           impurity_criterion="entropy")

    def _best_split(self, X, y, feature_index_occurrences=None, modified_factor=1, gamma_factor=2 / 3,
                    father_feature=None, impurity_criterion="gini"):
        return super()._best_split(X=X, y=y, feature_index_occurrences=feature_index_occurrences,
                                   modified_factor=modified_factor,
                                   gamma_factor=gamma_factor, father_feature=father_feature,
                                   impurity_criterion="entropy")
