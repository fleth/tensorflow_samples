
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
f = open("../data/20170206_214114.json", "r", encoding="utf_8_sig")
inputs = f.readlines();
f.close()

keys = []
brainwaves = []
mouse = dict()
mouse["move"] = []
mouse["button"] = []
for line in inputs:
    data = json.loads(line)
    brainwaves.append(data["brainwaves"])
    keys.append(data["keys"])
    mouse["move"].append(data["mouse"]["move"])
    mouse["button"].append(data["mouse"]["button"])

def make_mini_batch(train_data, size_of_mini_batch, size_of_input_data):
    inputs  = np.empty(0)
    outputs = np.empty(0)
    for _ in range(size_of_mini_batch):
        index   = random.randint(0, len(train_data["inputs"]))
        inputs_part    = train_data["inputs"][index]
        outputs_part    = train_data["outputs"][index]
        inputs  = np.append(inputs, inputs_part)
        degrees =  math.degrees(outputs_part[0])
        if degrees < 0:
            degrees = 360 + degrees
        outputs = np.append(outputs, degrees % 360)
    inputs = inputs.reshape(-1, 1, size_of_input_data)
    return (inputs, outputs)


num_of_input_nodes          = 50
num_of_hidden_nodes         = 150
num_of_output_nodes         = 1
num_of_training_epochs      = 10
size_of_mini_batch          = 10
size_of_batch               = 1000
dropout                     = 0.01

train_data = dict()
train_data["inputs"] = brainwaves
train_data["outputs"] = mouse["move"]

data, labels = make_mini_batch(train_data, size_of_batch, num_of_input_nodes)
tb_cb = keras.callbacks.TensorBoard(log_dir="/tmp/tensorflow_log", histogram_freq=1)
callbacks = [tb_cb]

print(data)

# 乱数シードを固定する。
random.seed(0)
np.random.seed(0)

model = Sequential()
input_layer = Input(batch_shape=(None, 1, num_of_input_nodes))
conv_1d_output_layer = Conv1D(25, 3, padding="same", input_shape=(None, 1, num_of_input_nodes))(input_layer)
flatten_output_layer = Flatten()(conv_1d_output_layer)
prediction_result = Dense(num_of_output_nodes)(flatten_output_layer)
model = Model(input=input_layer, output=prediction_result)
model.compile(optimizer="adam", loss="mean_squared_error")

model.fit(data, labels, nb_epoch=num_of_training_epochs, batch_size=size_of_mini_batch, callbacks=callbacks)

test_data, test_labels = make_mini_batch(train_data, size_of_mini_batch, num_of_input_nodes)
score = model.evaluate(test_data, test_labels, batch_size=size_of_mini_batch)

print(score)

predict = model.predict(test_data, batch_size=size_of_mini_batch)

print(predict)

