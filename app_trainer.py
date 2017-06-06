import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def get_data(limit=None):
    print("Reading in and transforming data...")
    df = pd.read_csv('train.csv')
    data = df.as_matrix()
    np.random.shuffle(data)
    X = data[:, 1:] // 255.0
    Y = data[:, 0]
    if limit is not None:
        X, Y = X[:limit], Y[:limit]
    return X, Y


def get_test_data(limit=None):
    print("Reading in and transforming data...")
    df = pd.read_csv('test.csv')
    data = df.as_matrix()
    np.random.shuffle(data)
    X = data[:, 1:] / 255.0
    Y = data[:, 0]
    if limit is not None:
        X, Y = X[:limit], Y[:limit]
    return X, Y


if __name__ == '__main__':
    X, Y = get_data()
    Ntrain = len(Y)
    Xtrain, Ytrain = X, Y

    model = RandomForestClassifier()
    model.fit(Xtrain, Ytrain)
    Xtest, Ytest = X, Y
    print("Test accuracy: ", model.score(Xtest, Ytest))

    with open('mymodel.pkl', 'wb') as f:
        pickle.dump(model, f)


