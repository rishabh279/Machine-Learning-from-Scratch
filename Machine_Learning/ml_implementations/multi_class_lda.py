import matplotlib.pyplot as plt
import numpy as np


class LDA():
    def __init__(self):
        pass

    def calculate_scatter_matrices(self, x, y):
        n_features = np.shape(x)[1]
        labels = np.unique(y)

        sw = np.empty((n_features, n_features))
        sb = np.empty((n_features, n_features))
        for label in labels:
            x_label = x[y == label]
            mean = np.mean(x, axis=0)
            sw += (x_label - mean).T.dot(x_label - mean)

            total_mean = np.mean(x_label, axis=0)
            sb += len(x_label) * (mean - total_mean).dot((mean - total_mean).T)

        return sw, sb

    def transform(self, x, y, n_components):
        sw, sb = self.calculate_scatter_matrices(x, y)

        A = np.linalg.inv(sw).dot(sb)

        eigenvalues, eigenvectors = np.linalg.eigh(A)

        idx = eigenvalues.argsort()[::-1]
        eigenvectors = eigenvectors[:, idx][:, :n_components]

        x_transformed = x.dot(eigenvectors)

        return x_transformed

    def plot_in_2d(self, x, y):
        x_transformed = self.transform(x, y, n_components=2)
        x1 = x_transformed[:, 0]
        x2 = x_transformed[:, 1]
        plt.scatter(x1, x2, c=y)
        plt.show()
