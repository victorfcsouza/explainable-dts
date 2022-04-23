from enum import Enum
import json
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from typing import Dict, List

from utils.file_utils import create_dir


class MetricType(Enum):
    gini_factor = "gini_factor"
    gamma_factor = "gamma"
    train_accuracy = "train_accuracy"
    test_accuracy = "test_accuracy"
    max_depth = "max_depth"
    max_depth_redundant = "max_depth_redundant"
    wad = "wad"
    waes = "waes"
    unbalanced_splits = "unbalanced_splits"
    nodes = "nodes"
    features = "features"


class Metrics:
    def __init__(self, gini_factor, gamma_factor, train_accuracy, test_accuracy, max_depth, max_depth_redundant, wad,
                 waes, unbalanced_splits, nodes, features):
        self.gini_factor = gini_factor
        self.gamma_factor = gamma_factor
        self.train_accuracy = train_accuracy
        self.test_accuracy = test_accuracy
        self.unbalanced_splits = unbalanced_splits
        self.max_depth = max_depth
        self.max_depth_redundant = max_depth_redundant
        self.wad = wad
        self.waes = waes
        self.nodes = nodes
        self.features = features

    def get_metrics(self):
        return {
            MetricType.gini_factor: self.gini_factor,
            MetricType.gamma_factor: self.gamma_factor,
            MetricType.train_accuracy: round(self.train_accuracy, 3),
            MetricType.test_accuracy: round(self.test_accuracy, 3),
            MetricType.max_depth: round(self.max_depth, 3),
            MetricType.max_depth_redundant: round(self.max_depth_redundant, 3),
            MetricType.wad: round(self.wad, 3),
            MetricType.waes: round(self.waes, 3),
            MetricType.unbalanced_splits: self.unbalanced_splits,
            MetricType.nodes: self.nodes,
            MetricType.features: self.features
        }

    @staticmethod
    def get_cols():
        return [MetricType.gini_factor.value,
                MetricType.gamma_factor.value,
                MetricType.train_accuracy.value,
                MetricType.test_accuracy.value,
                MetricType.max_depth.value,
                MetricType.max_depth_redundant.value,
                MetricType.wad.value,
                MetricType.waes.value,
                MetricType.unbalanced_splits.value,
                MetricType.nodes.value,
                MetricType.features.value]


class ResultJson:
    def __init__(self, dataset, algorithm, max_depth_stop, min_samples_stop, results):
        self.dataset = dataset
        self.algorithm = algorithm
        self.max_depth_stop = max_depth_stop
        self.min_samples_stop = min_samples_stop
        self.results = results

    @staticmethod
    def get_cols():
        return ["dataset", "algorithm", "max_depth_stop", "min_samples_stop"]

    def get_result_json(self):
        return {
            'dataset': self.dataset,
            'algorithm': self.algorithm,
            'max_depth_stop': self.max_depth_stop,
            'min_samples_stop': self.min_samples_stop,
            # 'opt_factor': self._get_opt_factor(),
            'results': [{key.value: metric[key] for key in metric} for metric in self.results]  # Convert to string keys
        }


class Test:
    def __init__(self, classifier, dataset_name: str, csv_file: str, max_depth_stop: int,
                 col_class_name: str, cols_to_delete: list = None, min_samples_stop: int = 0,
                 gini_factors=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96,
                               0.97, 0.98, 0.99, 1), gamma_factors=(0.5, 0.6, 0.7, 0.8, 0.9), results_folder="results"):
        self.classifier = classifier
        self.dataset_name = dataset_name
        self.csv_file = csv_file
        self.col_class_name = col_class_name
        self.cols_to_delete = cols_to_delete
        self.max_depth_stop = max_depth_stop
        self.min_samples_stop = min_samples_stop
        self.results_folder = results_folder
        self.gini_factors = gini_factors
        self.gamma_factors = gamma_factors
        self.n_classes = None  # Need to update via run()
        self.n_samples = None  # Need to update via run()
        self.n_features = None  # Need to update via run()
        self.results = None  # Need to update via run()

    def _get_filename(self, extension: str, gini_factor: float = None, gamma_factor: float = None,
                      sub_folder=None) -> str:
        folder = f"{self.results_folder}/{sub_folder}" if sub_folder else f"{self.results_folder}"
        filename = f"{folder}/{self.dataset_name}_{self.classifier.__name__}_depth_{self.max_depth_stop}" \
                   f"_samples_{self.min_samples_stop}_gini-factor_{gini_factor}_gamma_{gamma_factor}"
        filename += f".{extension}"
        return filename

    def _plot_graphic(self):
        factors = [metric[MetricType.gini_factor] for metric in self.results]
        train_accuracy_list = [metric[MetricType.train_accuracy] for metric in self.results]
        test_accuracy_list = [metric[MetricType.test_accuracy] for metric in self.results]
        max_depth_list = [metric[MetricType.max_depth] for metric in self.results]
        max_depth_redundant_list = [metric[MetricType.max_depth_redundant] for metric in self.results]
        wad_list = [metric[MetricType.wad] for metric in self.results]
        waes_list = [metric[MetricType.waes] for metric in self.results]

        figure(figsize=(12, 10), dpi=300)
        plt.subplot(2, 1, 1)
        dataset_name = self.dataset_name
        plt.title(dataset_name, fontsize=16)
        plt.plot(factors, train_accuracy_list, label="Train Accuracy", color='red', marker='o')
        plt.plot(factors, test_accuracy_list, label="Test Accuracy", color='darkred', marker='o')
        plt.ylabel("Accuracy", fontsize=16)
        plt.legend(loc="lower right", fontsize=10)

        plt.subplot(2, 1, 2)
        plt.plot(factors, max_depth_list, label="Max Depth", color='cyan', marker='o')
        plt.plot(factors, max_depth_redundant_list, label="Redundant Max Depth", color='blue', marker='o')
        plt.plot(factors, wad_list, label="WAD (Weighted Average Depth", color='gray', marker='o')
        plt.plot(factors, waes_list, label="WAES (Weighted Average Explanation Size)", color='black', marker='o')
        plt.xlabel("Gini Factor", fontsize=16)
        plt.ylabel("Metric", fontsize=16)
        plt.legend(loc="lower right", fontsize=10)
        filename = self._get_filename("png", sub_folder="img")
        plt.savefig(filename)

    def _store_results(self):
        filename: str = self._get_filename("json", sub_folder="json")
        result_json = {
            'dataset': self.dataset_name,
            'algorithm': self.classifier.__name__,
            'n_samples': self.n_samples,
            'n_features': self.n_features,
            'n_classes': self.n_classes,
            'max_depth_stop': self.max_depth_stop,
            'min_samples_stop': self.min_samples_stop,
            # 'opt_factor': self._get_opt_factor(),
            'results': [{key.value: metric[key] for key in metric} for metric in self.results]  # Convert to string keys
        }
        # Create dir if not exists
        create_dir(filename)
        with open(filename, 'w') as f:
            json.dump(result_json, f, indent=2)

    def _get_opt_factor(self, opt_factor=0.90):
        """
        Get the lower factor that allow a decreasing in test_accuracy not lower than opt_factor.
        Also, get the diff compared to original values (factor=1) in percentage
        """
        # Assume that results are ordered by increasing factors
        reversed_results = sorted(self.results, key=lambda x: x[MetricType.gini_factor], reverse=True)
        max_test_accuracy = reversed_results[0][MetricType.test_accuracy]

        max_metrics = reversed_results[0]

        opt_index = -1
        for i in range(1, len(reversed_results)):
            if reversed_results[i][MetricType.test_accuracy] <= opt_factor * max_test_accuracy:
                opt_index = i - 1
                break
        opt_metrics = reversed_results[opt_index]

        diff_results = dict()
        for key in max_metrics:
            if key == MetricType.gini_factor:
                diff_results[key.value] = opt_metrics[key]
            else:
                diff_metric = round(100 * (opt_metrics[key] - max_metrics[key]) / max_metrics[key])
                diff_results[key.value] = str(diff_metric) + "%"

        return diff_results

    def parse_dataset(self):
        # Read CSV
        np.set_printoptions(suppress=True)
        data = pd.read_csv(self.csv_file)
        if self.cols_to_delete:
            data = data.drop(labels=self.cols_to_delete, axis=1)
        data = data.astype({self.col_class_name: str})

        # Read cols as float
        cols = list(data.columns)
        cols.remove(self.col_class_name)
        for col in cols:
            data[col] = data[col].astype(float)

        data[self.col_class_name] = data[self.col_class_name].rank(method='dense', ascending=False).astype(int)

        X = data.drop(labels=self.col_class_name, axis=1).to_numpy()
        y = data[self.col_class_name].astype('int').to_numpy() - 1

        return X, y

    @staticmethod
    def split_train_test(X, y, random_state=42, train_size=0.7):
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state, train_size=train_size)
        return X_train, X_test, y_train, y_test

    def run(self, store_results=True, plot_graphic=False, debug=False):
        print("\n#######################################################################")
        print(f"### Running tests for {self.dataset_name} with {self.classifier.__name__}, "
              f"max_depth {self.max_depth_stop}, min_samples_stop {self.min_samples_stop}")

        # Read CSV
        X, y = self.parse_dataset()

        # Populate dataset metrics
        self.n_samples, self.n_features = X.shape
        self.n_classes = len(set(y))

        # Training Models
        X_train, X_test, y_train, y_test = self.split_train_test(X, y)

        results = []
        for gini_factor in self.gini_factors:
            for gamma_factor in self.gamma_factors:
                dt = self.classifier(max_depth=self.max_depth_stop, min_samples_stop=self.min_samples_stop)
                score_train, score_test = dt.get_score(X_train, y_train, X_test, y_test, modified_factor=gini_factor,
                                                       gamma_factor=gamma_factor)
                unbalanced_splits, max_depth, max_depth_redundant, wad, waes, nodes, features = \
                    dt.get_explainability_metrics()
                if debug:
                    # dt.tree_.debug(
                    #     feature_names=["Attribute {}".format(i) for i in range(len(X_train))],
                    #     class_names=["Class {}".format(i) for i in range(len(y_train))],
                    #     show_details=True
                    # )
                    tree_img_file = self._get_filename(extension="png", gini_factor=gini_factor, sub_folder="img")
                    dt.tree_.debug_pydot(tree_img_file)
                print(f"\nTrain/test accuracy for gini factor {gini_factor} and gamma {gamma_factor}: "
                      f"{score_train}, {score_test}")
                print(f"max_depth: {max_depth}, max_depth_redundant: {max_depth_redundant}, wad: {wad}, "
                      f"waes: {waes}, nodes: {nodes}, features: {features}")
                print("----------------------------")
                metrics = Metrics(gini_factor=gini_factor,
                                  gamma_factor=gamma_factor,
                                  train_accuracy=score_train,
                                  test_accuracy=score_test,
                                  max_depth=max_depth,
                                  max_depth_redundant=max_depth_redundant,
                                  wad=wad,
                                  waes=waes,
                                  unbalanced_splits=unbalanced_splits,
                                  nodes=nodes,
                                  features=features)

                results.append(metrics.get_metrics())

                # Save Tree objects
                if store_results:
                    pickle_name = self._get_filename(extension="pickle", gini_factor=gini_factor,
                                                     gamma_factor=gamma_factor, sub_folder="pickle")
                    create_dir(pickle_name)
                    with open(pickle_name, 'wb') as handle:
                        pickle.dump(dt.tree_, handle)

        self.results = results
        if plot_graphic:
            self._plot_graphic()
        if store_results:
            self._store_results()

        print(f"\n### Ended tests for {self.dataset_name} with with {self.classifier.__name__}, "
              f"max_depth {self.max_depth_stop}, min_samples_stop {self.min_samples_stop}\n")
