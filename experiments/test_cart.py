from enum import Enum
import json
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from cart.cart import DecisionTreeClassifier


class MetricType(Enum):
    factor = "factor"
    train_accuracy = "train_accuracy"
    test_accuracy = "test_accuracy"
    max_depth = "max_depth"
    max_depth_redundant = "max_depth_redundant"
    wapl = "wapl"
    wapl_redundant = "wapl_redundant"


class Metrics:
    def __init__(self, factor, train_accuracy, test_accuracy, max_depth, max_depth_redundant, wapl, wapl_redundant):
        self.factor = factor
        self.train_accuracy = train_accuracy
        self.test_accuracy = test_accuracy
        self.max_depth = max_depth
        self.max_depth_redundant = max_depth_redundant
        self.wapl = wapl
        self.wapl_redundant = wapl_redundant

    def get_metrics(self):
        return {
            MetricType.factor: self.factor,
            MetricType.train_accuracy: self.train_accuracy,
            MetricType.test_accuracy: self.test_accuracy,
            MetricType.max_depth: self.max_depth,
            MetricType.max_depth_redundant: self.max_depth_redundant,
            MetricType.wapl: self.wapl,
            MetricType.wapl_redundant: self.wapl_redundant
        }


class Test:
    def __init__(self, dataset_name: str, csv_file: str, max_depth_stop: int, bins: bool, col_class_name: str,
                 cols_to_delete: list = None, min_samples_stop: int = 0,
                 factors=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96,
                          0.97, 0.98, 0.99, 1), results_folder="results"):
        self.dataset_name = dataset_name
        self.csv_file = csv_file
        self.col_class_name = col_class_name
        self.cols_to_delete = cols_to_delete
        self.max_depth_stop = max_depth_stop
        self.min_samples_stop = min_samples_stop
        self.bins = bins
        self.results_folder = results_folder
        self.factors = factors
        self.n_classes = None
        self.n_samples = None
        self.n_features = None
        self.results = None  # Need to update via test_dataset

    def _get_filename(self, extension: str) -> str:
        filename = f"{self.results_folder}/{self.dataset_name}_depth_{self.max_depth_stop}" \
                   f"_samples_{self.min_samples_stop}_bins_{self.bins}." + \
                   extension
        return filename

    def _plot_graphic(self):
        factors = [metric[MetricType.factor] for metric in self.results]
        train_accuracy_list = [metric[MetricType.train_accuracy] for metric in self.results]
        test_accuracy_list = [metric[MetricType.test_accuracy] for metric in self.results]
        max_depth_list = [metric[MetricType.max_depth] for metric in self.results]
        max_depth_redundant_list = [metric[MetricType.max_depth_redundant] for metric in self.results]
        wapl_list = [metric[MetricType.wapl] for metric in self.results]
        wapl_redundant_list = [metric[MetricType.wapl_redundant] for metric in self.results]

        figure(figsize=(12, 10), dpi=300)
        plt.subplot(2, 1, 1)
        dataset_name = self.dataset_name if not self.bins else f"{self.dataset_name} with bin"
        plt.title(dataset_name, fontsize=16)
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
        filename = self._get_filename("png")
        plt.savefig(filename)

    def _store_results(self):
        filename: str = self._get_filename("json")
        result_json = {
            'dataset': self.dataset_name,
            'n_samples': self.n_samples,
            'n_features': self.n_features,
            'n_classes': self.n_classes,
            'bins': self.bins,
            'max_depth_stop': self.max_depth_stop,
            'min_samples_stop': self.min_samples_stop,
            'opt_factor': self._get_opt_factor(),
            'results': [{key.value: metric[key] for key in metric} for metric in self.results]  # Convert to string keys
        }

        with open(filename, 'w') as f:
            json.dump(result_json, f, indent=2)

    def _get_opt_factor(self, opt_factor=0.90):
        """
        Get the lower factor that allow a decreasing in test_accuracy not lower than opt_factor.
        Also, get the diff compared to original values (factor=1) in percentage
        """
        # Assume that results are ordered by increasing factors
        reversed_results = sorted(self.results, key=lambda x: x[MetricType.factor], reverse=True)
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
            if key == MetricType.factor:
                diff_results[key.value] = opt_metrics[key]
            else:
                diff_metric = round(100 * (opt_metrics[key] - max_metrics[key]) / max_metrics[key])
                diff_results[key.value] = str(diff_metric) + "%"

        return diff_results

    def run(self, store_results=True, plot_graphic=False, debug=False):
        print("############################")
        print(f"### Running tests for {self.dataset_name} with max_depth {self.max_depth_stop} and bins = {self.bins}")

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

        # Populate dataset metrics
        self.n_samples, self.n_features = X.shape
        self.n_classes = len(set(y))

        unique, counts = np.unique(y, return_counts=True)
        dict(zip(unique, counts))

        # Training Models
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, train_size=0.7)

        results = []
        for factor in self.factors:
            dt = DecisionTreeClassifier(max_depth=self.max_depth_stop)
            score_train, score_test = dt.get_score(X_train, y_train, X_test, y_test, modified_factor=factor,
                                                   debug=debug)

            print(f"Train/test accuracy for factor {factor}: {score_train}, {score_test}")
            max_depth, max_depth_redundant, wapl, wapl_redundant = dt.get_explainability_metrics()
            print(f"max_depth:  {max_depth}, max_depth_redundant: {max_depth_redundant}, wapl: {wapl},"
                  f" wapl_redundant: {wapl_redundant}")
            results.append({
                MetricType.factor: round(factor, 2),
                MetricType.train_accuracy: round(score_train, 3),
                MetricType.test_accuracy: round(score_test, 3),
                MetricType.max_depth: round(max_depth, 3),
                MetricType.max_depth_redundant: round(max_depth_redundant, 3),
                MetricType.wapl: round(wapl, 3),
                MetricType.wapl_redundant: round(wapl_redundant, 3)
            })

        self.results = results
        if plot_graphic:
            self._plot_graphic()
        if store_results:
            self._store_results()

        print(f"### Ended tests for {self.dataset_name} with max_depth {self.max_depth_stop} and bins = {self.bins}")
