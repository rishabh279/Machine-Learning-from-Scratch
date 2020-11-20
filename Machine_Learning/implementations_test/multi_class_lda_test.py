from sklearn import datasets
from Machine_Learning.utils.data_manipulation import normalize
from Machine_Learning.ml_implementations.multi_class_lda import LDA


def main():
    data = datasets.load_iris()
    x = normalize(data.data)
    y = data.target

    lda = LDA()
    lda.plot_in_2d(x, y)


if __name__ == '__main__':
    main()