from sklearn import datasets
from utils.data_manipulation import train_test_split
from utils.data_operation import accuracy_score
from ml_implementations.random_forest import RandomForest


def main():
    data = datasets.load_iris()
    x = data.data
    y = data.target

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    clf = RandomForest(n_estimators=100)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy:', accuracy)


if __name__=='__main__':
    main()