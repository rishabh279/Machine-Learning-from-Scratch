import numpy as np


def shuffle_data(x, y, seed=None):
    if seed:
        np.random.seed(seed)
    idx = np.arange(x.shape[0])
    np.random.shuffle(idx)
    return x[idx], y[idx]


def normalize(X, axis=-1, order=2):
    """ Normalize the dataset X """
    l2 = np.atleast_1d(np.linalg.norm(X, order, axis))
    l2[l2 == 0] = 1
    return X / np.expand_dims(l2, axis)


def divide_on_feature(x, feature_i, threshold):
    split_func = None
    if isinstance(threshold, int) or isinstance(threshold, float):
        split_func = lambda sample: sample[feature_i] >= threshold
    else:
        split_func = lambda sample: sample[feature_i] == threshold

    x1 = np.array([sample for sample in x if split_func(sample)])
    x2 = np.array([sample for sample in x if not split_func(sample)])

    return np.array([x1, x2])


def train_test_split(x, y, test_size=0.5, shuffle=True, seed=None):
    if shuffle:
        x, y = shuffle_data(x, y, seed)
    split_i = len(y) - int(len(y) // (1 / test_size))
    x_train, x_test = x[:split_i], x[split_i:]
    y_train, y_test = y[:split_i], y[split_i:]

    return x_train, x_test, y_train, y_test


def standardize(x):
    x_std = x
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    for col in range(np.shape(x)[1]):
        if std[col]:
            x_std[:, col] = (x_std[:, col] - mean[col]) / std[col]

    return x_std


def to_categorical(x, n_col=None):
    if not n_col:
        n_col = np.amax(x) + 1
    one_hot = np.zeros((x.shape[0], n_col))
    one_hot[np.arange(x.shape[0]), x] = 1
    return one_hot


def to_nominal(x):
    return np.argmax(x, axis=1)


def get_random_subsets(x, y, n_subsets, replacements=True):
    n_samples = np.shape(x)[0]
    x_y = np.concatenate((x, y.reshape((1, len(y))).T), axis=1)
    np.random.shuffle(x_y)
    subsets = []

    sunsample_size = int(n_samples // 2)

    if replacements:
        subsample_size = n_samples

    for _ in range(n_subsets):
        idx = np.random.choice(
            range(n_samples),
            size=np.shape(range(subsample_size)),
            replace=replacements)
        x = x_y[idx][:, :-1]
        y = x_y[idx][:, :-1]
        subsets.append([x, y])
    return subsets