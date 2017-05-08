
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

def make_data(train_data, size_of_mini_batch, size_of_input_data):
    inputs  = np.empty(0)
    outputs = np.empty(0)
    base    = 5000
    for index in range(size_of_mini_batch):
        inputs_part    = train_data["inputs"][base + index]
        outputs_part    = train_data["outputs"][base + index]
        inputs  = np.append(inputs, inputs_part)
        tmp = np.zeros(360)
        tmp[int(degrees(outputs_part[0]))] = 1
        outputs = np.append(outputs, tmp)
    inputs = inputs.reshape(-1, size_of_input_data, 1)
    outputs = outputs.reshape(-1, 360)
    return (inputs, outputs)

def make_rand_data(num_of_samples, data, labels):
    rand_data   = np.empty(0)
    rand_labels = np.empty(0)
    for _ in range(num_of_samples):
        index   = random.randint(0, len(data))
        rand_data   = np.append(rand_data, data[index])
        rand_labels = np.append(rand_labels, labels[index])
    rand_data = rand_data.reshape(num_of_samples, -1, 1)
    return (rand_data, rand_labels)

keys = []
brainwaves = []
mouse = dict()
mouse["move"] = []
mouse["button"] = []

file_list = [
    "../data/20170203_223600.json",
    "../data/20170206_214114.json",
    "../data/20170206_223307.json",
    "../data/20170207_220719.json",
    "../data/20170507_152458.json",
    "../data/20170507_161822.json",
    "../data/20170507_204856.json",
    "../data/20170507_214606.json",
    "../data/20170507_222431.json"]

for filename in file_list:
    inputs = (loaddata(filename))
    for line in inputs:
        data = json.loads(line)
        # ignore zero
        if int(data["mouse"]["move"][0]) == 0:
            continue
        brainwaves.append(data["brainwaves"])
        keys.append(data["keys"])
        mouse["move"].append(data["mouse"]["move"])
        mouse["button"].append(data["mouse"]["button"])

print("brainwaves: %s" % len(brainwaves))

num_of_input_nodes          = 50
num_of_output_nodes         = 360
num_of_training_epochs      = 200
size_of_mini_batch          = 100
size_of_batch               = 5000
dropout                     = 0.01

train_data = dict()
train_data["inputs"] = brainwaves
train_data["outputs"] = mouse["move"]

data, labels = make_data(train_data, size_of_batch, num_of_input_nodes)
print(labels)
tb_cb = keras.callbacks.TensorBoard(log_dir="/tmp/tensorflow_log", histogram_freq=1)
callbacks = [tb_cb]

# 乱数シードを固定する。
random.seed(0)
np.random.seed(0)

model = Sequential()
model.add(Conv1D(128, 3, padding="same", input_shape=(num_of_input_nodes, 1)))
model.add(Conv1D(128, 3, padding="same", activation="relu"))
model.add(MaxPooling1D())
model.add(Flatten())
#model.add(LSTM(64))
model.add(Dense(128, activation="relu"))
model.add(Dense(num_of_output_nodes, activation="softmax"))
optimizer   = "adadelta"
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

model.summary()

model.fit(data, labels, nb_epoch=num_of_training_epochs, batch_size=size_of_mini_batch, callbacks=callbacks, validation_split=0.2)

test_data, test_labels = make_rand_data(20, data, labels)
print("%s, %s, %s, %s" % (data.shape, labels.shape, test_data.shape, test_labels.shape))
predict = model.predict(test_data, batch_size=size_of_mini_batch)

print(predict)
print(test_labels)
