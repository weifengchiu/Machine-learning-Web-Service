import pickle
import numpy as np
import json
import os
import tornado.ioloop
import tornado.web
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

if not os.path.exists('mymodel.pkl'):
    exit("Can't run without model!")


# def get_data(limit=None):
#     print("Reading in and transforming data...")
#     df = pd.read_csv('train.csv')
#     data = df.as_matrix()
#     np.random.shuffle(data)
#     X = data[:, 1:] / 255.0
#     Y = data[:, 0]
#     if limit is not None:
#         X, Y = X[:limit], Y[:limit]
#     return X, Y
#
#
# X, Y = get_data()
# Ntrain = len(Y)
# Xtrain, Ytrain = X, Y
#
# model = RandomForestClassifier()
# model.fit(Xtrain, Ytrain)
# Xtest, Ytest = X, Y


with open('mymodel.pkl', mode='rb') as f:
    # a = f.encoding()
    model = pickle.load(f)


class MainHandler(tornado.web.RequestHandler):
    def data_received(self, chunk):
        pass

    def get(self, *args, **kwargs):
        self.write("Hello Tornado!!")


class PredictionHandler(tornado.web.RequestHandler):
    def data_received(self, chunk):
        pass

    def post(self, *args, **kwargs):
        params = self.request.arguments
        x = np.array(params['input'])
        y = model.predict([x])[0]
        self.write(str(np.asscalar(y)))
        self.finish()

if __name__ == '__main__':
    application = tornado.web.Application([
        (r"/", MainHandler),
        (r"/predict", PredictionHandler)
    ])
    application.listen(8888)
    tornado.ioloop.IOLoop.current().start()




