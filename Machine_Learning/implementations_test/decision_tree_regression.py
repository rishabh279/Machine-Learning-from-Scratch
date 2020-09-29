from utils.data_manipulation import standardize, train_test_split
from utils.data_operation import mean_squared_error
from ml_implementations.decision_tree import RegressionTree
import pandas as pd
import numpy as np


def main():
    data = pd.read_csv('../data/regression_data.txt', sep='\t')
    time = np.atleast_2d(data["time"].values).T
    temp = np.atleast_2d(data["temp"].values).T

    x = standardize(time)
    y = temp[:, 0]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    model = RegressionTree()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    y_pred_line = model.predict(x)
    mse = mean_squared_error(y_test, y_pred)
    print('Mean Squared Error:', mse)

if  __name__ == '__main__':
    main()