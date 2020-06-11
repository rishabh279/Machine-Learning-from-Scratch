import numpy as np
import math
from sklearn import datasets

from utils.data_manipulation import train_test_split
from utils.data_operation import accuracy_score

class DecisionStump():

    def __init__(self):
        self.polarity = 1
        self.feature_index = None
        self.threshold = None
        self.alpha = None

class Adaboost():
    def __init__(self, n_clf=5):
        self.n_clf = n_clf

    def fit(self, x, y):
        n_samples, n_features = np.shape(x)
        w = np.full(n_samples, (1 / n_samples))

        self.clfs = []
        for _ in range(self.n_clf):
            clf = DecisionStump()
            min_error = float('inf')
            for feature_i in range(n_features):
                feature_values = np.expand_dims(x[:, feature_i], axis=1)
                unique_values = np.unique(feature_values)
                for threshold in unique_values:
                    p = 1
                    prediction = np.ones(np.shape(y))
                    prediction[x[:, feature_i] < threshold] = -1
                    error = sum(w[y != prediction])

                    if error > 0.5:
                        error = 1 - error
                        p = -1

                    if error < min_error:
                        clf.polarity = p
                        clf.threshold = threshold
                        clf.feature_index = feature_i
                        min_error = error
        clf.alpha = 0.5 * math.log((1 - min_error) / (min_error + 1e-10))
        predictions = np.ones(np.shape(y))
        negative_idx = clf.polarity * x[:, clf.feature_index] < clf.polarity * clf.threshold
        predictions[negative_idx] = -1
        w *= np.exp(-clf.alpha * y * predictions)
        w /= np.sum(w)

        self.clfs.append(clf)

    def predict(self, x):
        n_samples = np.shape(x)[0]
        y_pred = np.zeros((n_samples, 1))
        for clf in self.clfs:
            predictions = np.ones(np.shape(y_pred))
            negative_idx = clf.polarity * x[:, clf.feature_index] < clf.polarity * clf.threshold
            predictions[negative_idx] = -1
            y_pred += clf.alpha * predictions

        y_pred = np.sign(y_pred).flatten()
        return y_pred

def main():
    data = datasets.load_digits()
    x = data.data
    y = data.target

    digit1 = 1
    digit2 = 8
    idx = np.append(np.where(y == digit1)[0], np.where(y == digit2)[0])
    y = data.target[idx]
    y[y == digit1] = -1
    y[y == digit2] = 1
    x = data.data[idx]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5)
    clf = Adaboost(n_clf=5)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy:', accuracy)

if __name__ == "__main__":
    main()
