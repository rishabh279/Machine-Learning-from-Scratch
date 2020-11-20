from Machine_Learning.ml_implementations.pca import PCA
from sklearn import datasets
import matplotlib.pyplot as plt


def main():
    data = datasets.load_iris()
    x = data.data
    y = data.target

    pca = PCA(2)
    pca.fit(x)
    x_projected = pca.transform(x)

    print(f'Shape of x {x.shape}')
    print(f'Shape of Transformed  x {x_projected.shape}')

    x1 = x_projected[:, 0]
    x2 = x_projected[:, 1]

    plt.scatter(x1, x2,
                c=y, edgecolor='none', alpha=0.8,
                cmap=plt.cm.get_cmap('viridis', 3))
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    main()
