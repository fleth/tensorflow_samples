
import numpy as np
import random
import json
import math

import keras
from keras.models import Sequential
from keras.layers import *
from keras.models import Model
from keras.callbacks import EarlyStopping

# create data
def loaddata(filename):
    f = open(filename, "r", encoding="utf_8_sig")
    lines = f.readlines();
    f.close()
    return lines;

def degrees(data):
    degrees =  math.degrees(data)
    if degrees < 0:
        degrees = 360 + degrees
    return degrees % 360

def make_mini_batch(train_data, index, size_of_mini_batch, size_of_input_data):
    inputs  = np.empty(0)
    outputs = np.empty(0)
    for _ in range(size_of_mini_batch):
        inputs_part    = train_data["inputs"][index]
        outputs_part    = train_data["outputs"][index]
        inputs  = np.append(inputs, inputs_part)
        outputs = np.append(outputs, degrees(outputs_part[0]))
    inputs = inputs.reshape(-1, size_of_input_data, 1)
    return (inputs, outputs)

keys = []
brainwaves = []
mouse = dict()
mouse["move"] = []
mouse["button"] = []

file_list = [
    "../data/20170203_223600.json",
    "../data/20170206_214114.json",
    "../data/20170206_223307.json",
    "../data/20170207_220719.json"]

for filename in file_list:
    inputs = (loaddata(filename))
    for line in inputs:
        data = json.loads(line)
        if int(data["mouse"]["move"][0]) == 0:
            continue
        brainwaves.append(data["brainwaves"])
        keys.append(data["keys"])
        mouse["move"].append(data["mouse"]["move"])
        mouse["button"].append(data["mouse"]["button"])

print("brainwaves: %s" % len(brainwaves))

num_of_input_nodes          = 50
num_of_training_epochs      = 5
size_of_mini_batch          = 100
size_of_batch               = 1500
dropout                     = 0.01

train_data = dict()
train_data["inputs"] = brainwaves
train_data["outputs"] = mouse["move"]

data, labels = make_mini_batch(train_data, 0, size_of_batch, num_of_input_nodes)
tb_cb = keras.callbacks.TensorBoard(log_dir="/tmp/tensorflow_log", histogram_freq=1)
callbacks = [tb_cb]

# 乱数シードを固定する。
random.seed(0)
np.random.seed(0)

model = Sequential()
model.add(Conv1D(64, 3, padding="same", input_shape=(num_of_input_nodes, 1)))
model.add(Conv1D(128, 3, padding="same"))
model.add(MaxPooling1D())
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dense(1, activation="relu"))
model.compile(optimizer="adam", loss="mean_squared_error", metrics=["accuracy"])

model.summary()

model.fit(data, labels, nb_epoch=num_of_training_epochs, batch_size=size_of_mini_batch, callbacks=callbacks)

test_data, test_labels = make_mini_batch(train_data, 300, size_of_mini_batch, num_of_input_nodes)
score = model.evaluate(test_data, test_labels, batch_size=size_of_mini_batch)

print(score)

predict = model.predict(test_data, batch_size=size_of_mini_batch)

print(predict)
print(test_labels)
