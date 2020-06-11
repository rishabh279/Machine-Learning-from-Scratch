from ml_implementations.decision_tree import ClassificationTree
from utils.data_manipulation import get_random_subsets
from utils.misc import bar_widgets

import progressbar
import numpy as np
import math

class RandomForest():
    def __init__(self, n_estimators=100, max_features=None, min_samples_split=2,
                 min_gain=0, max_depth=float("inf")):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.min_gain = min_gain
        self.max_depth = max_depth
        self.progressbar = progressbar.ProgressBar(widgets=bar_widgets)

        self.trees = []
        for _ in range(n_estimators):
            self.trees.append(
                ClassificationTree(
                    min_samples_split=self.min_samples_split,
                    min_impurity=self.min_gain,
                    max_depth=self.max_depth
                )
            )

    def fit(self, x, y):
        n_features = np.shape(x)[1]
        if not self.max_features:
            self.max_features = int(math.sqrt(n_features))

        subsets = get_random_subsets(x, y, self.n_estimators)

        for i in self.progressbar(range(self.n_estimators)):
            x_subset, y_subset = subsets[i]
            idx = np.random.choice(range(n_features), size=self.max_features, replace=True)
            self.trees[i].feature_indicies = idx
            x_subset = x_subset[:, idx]
            self.trees[i].fit(x_subset, y_subset)

    def predict(self, x):
        y_preds = np.empty((x.shape[0], len(self.trees)))
        for i, tree in enumerate(self.trees):
            idx = tree.feature_indicies
            prediction = tree.predict(x[:, idx])
            y_preds[:, i] = prediction
        y_pred = []
        for sample_predictions in y_preds:
            y_pred.append(np.bincount(sample_predictions.astype('int')).argmax())
        return y_pred
