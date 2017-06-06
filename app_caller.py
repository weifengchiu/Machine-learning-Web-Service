import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json


def get_data(limit=None):
    print("Reading in and transforming data...")
    df = pd.read_csv('train.csv')
    data = df.as_matrix()
    np.random.shuffle(data)
    X = data[:, 1:] / 255.0
    Y = data[:, 0]
    if limit is not None:
        X, Y = X[:limit], Y[:limit]
    return X, Y


X, Y = get_data()
N = len(Y)
while True:
    i = np.random.choice(N)
    r = requests.post("http://localhost:8888/predict", data={'input': X[i]})
    j = r.json()
    print(j)
    print("Target: ", Y[i])

    plt.imshow(X[i].reshape(28, 28), cmap='gray')
    plt.show()

    response = input("Continue: (Y/n)\n")
    if response in ('n', 'N'):
        break


