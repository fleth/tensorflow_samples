# -*- coding: utf-8 -*-
import numpy
import random

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM

class Prediction :

  def __init__(self):
    self.length_of_sequences = 4
    self.in_out_neurons = 1
    self.hidden_neurons = 300
    self.batch_size = 10
    self.nb_epoch = 500

  def load_data(self, num, length_of_sequences):
    X, Y = [], []
    for i in range(num * length_of_sequences * 2):
      row = int(random.uniform(1,10.0))
      X.append(row)
    retX = numpy.array(X).reshape(-1, length_of_sequences, 2)
    for data in retX:
      n = 0
      for d in data:
        n += reduce(lambda x, y: x - y, d)
      Y.append(n)
    retY = numpy.array(Y)
    return retX, retY


  def create_model(self) :
    model = Sequential()
    model.add(LSTM(self.hidden_neurons, \
              batch_input_shape=(None, self.length_of_sequences, 2), \
              return_sequences=False))
    model.add(Dense(self.in_out_neurons))
    model.add(Activation("linear"))
    model.compile(loss="mean_squared_error", optimizer="adam")
    return model


  def train(self, X_train, y_train):
    model = self.create_model()
    model.summary()
    # 学習
    model.fit(X_train, y_train, batch_size=self.batch_size, nb_epoch=self.nb_epoch)
    return model

if __name__ == "__main__":

  #
  # (A1-A2...-An) + ...(A1-A2...-An) の結果の学習
  #
  prediction = Prediction()

  # データ準備
  X, Y = prediction.load_data(100, prediction.length_of_sequences)

  # 2割をテストデータへ
  split_pos = int(len(X) * 0.8)
  x_train = X[:split_pos]
  x_test = X[split_pos:]
  y_train = Y[:split_pos]
  y_test = Y[split_pos:]

  model = prediction.train(x_train, y_train)

  predicted = model.predict(x_test)
  print(predicted)
  print(y_test)
