
import numpy as np
import random
import json
import math
import time

import keras
from keras.models import Sequential
from keras.layers import *
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

def make_mini_batch(train_data, size_of_mini_batch, length_of_sequences, size_of_input_data):
    inputs  = np.empty(0)
    outputs = np.empty(0)
    for _ in range(size_of_mini_batch):
        index   = random.randint(0, len(train_data["inputs"]) - length_of_sequences)
        inputs_part    = train_data["inputs"][index:index + length_of_sequences]
        outputs_part    = train_data["outputs"][index:index + length_of_sequences]
        inputs  = np.append(inputs, inputs_part)
        degrees =  math.degrees(outputs_part[-1][0])
        if degrees < 0:
            degrees = 360 + degrees
        outputs = np.append(outputs, degrees % 360)
    inputs  = inputs.reshape(-1, length_of_sequences, size_of_input_data)
    outputs = outputs.reshape(-1, 1)
    return (inputs, outputs)

# ハイパーパラメータの生成
random.seed(time.time())

optimizer_list = [
    "sgd",
    "rmsprop",
    "adagrad",
    "adadelta",
    "adam",
    "adamax",
    "nadam",
    "tfoptimizer"
]

loss_list = [
    "mean_squared_error",
    "mean_absolute_error",
    "mean_absolute_percentage_error",
    "mean_squared_logarithmic_error",
    "squared_hinge",
    "hinge",
    "binary_crossentropy",
    "kullback_leibler_divergence",
    "poisson",
    "cosine_proximity"
]

num_of_input_nodes          = 50
num_of_output_nodes         = 1

num_of_hidden_nodes         = random.randint(1, 400)
num_of_dense_nodes          = random.randint(1, 400)
length_of_sequences         = random.randint(1, 40)
dropout                     = random.randint(1, 40) / 100
num_of_training_epochs      = 500
size_of_mini_batch          = random.randint(1, 5) * 100
size_of_batch               = random.randint(1, 5) * 100
#optimizer_id                = random.randint(0, len(optimizer_list) - 1)
optimizer_id                = 4

train_data = dict()
train_data["inputs"] = brainwaves
train_data["outputs"] = mouse["move"]

# 乱数シードを固定する。
random.seed(0)
np.random.seed(0)

model = Sequential()
model.add(LSTM(num_of_hidden_nodes, batch_input_shape=(None, length_of_sequences, num_of_input_nodes)))
model.add(Activation("relu"))
model.add(Dense(num_of_dense_nodes))
model.add(Dropout(dropout))
model.add(Dense(num_of_output_nodes, activation="relu"))
model.compile(optimizer=optimizer_list[optimizer_id],
            loss="cosine_proximity",
            metrics=["accuracy"])

data, labels = make_mini_batch(train_data, size_of_batch, length_of_sequences, num_of_input_nodes)
#tb_cb = keras.callbacks.TensorBoard(log_dir="/tmp/tensorflow_log", histogram_freq=1)
#callbacks = [tb_cb]
callbacks = []

model.fit(data, labels, nb_epoch=num_of_training_epochs, batch_size=size_of_mini_batch, callbacks=callbacks)

test_data, test_labels = make_mini_batch(train_data, size_of_mini_batch, length_of_sequences, num_of_input_nodes)
score = model.evaluate(test_data, test_labels, batch_size=size_of_mini_batch)

log = {
    "loss": score[0],
    "accuracy": score[1],
    "num_of_hidden_nodes": num_of_hidden_nodes,
    "num_of_dense_nodes": num_of_dense_nodes,
    "length_of_sequences": length_of_sequences,
    "dropout": dropout,
    "size_of_mini_batch": size_of_mini_batch,
    "size_of_batch": size_of_batch
}

with open("/tmp/sample2_results.log", "a") as f:
    json.dump(log, f, ensure_ascii=False)
    f.write("\n")
