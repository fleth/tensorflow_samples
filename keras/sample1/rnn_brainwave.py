
import numpy as np
import random
import json
import math

from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM

# create data
f = open("../data/20170203_223600.json", "r", encoding="utf_8_sig")
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



num_of_input_nodes          = 50
num_of_hidden_nodes         = 50
num_of_output_nodes         = 1
length_of_sequences         = 1
num_of_training_epochs      = 50000
size_of_mini_batch          = 100
size_of_batch               = 1000

train_data = dict()
train_data["inputs"] = brainwaves
train_data["outputs"] = mouse["move"]

print("train_data:", train_data)

# 乱数シードを固定する。
random.seed(0)
np.random.seed(0)

model = Sequential()
model.add(LSTM(num_of_hidden_nodes, batch_input_shape=(None, length_of_sequences, num_of_input_nodes)))
model.add(Activation('relu'))
model.add(Dense(num_of_output_nodes))
model.compile(optimizer="adam",
            loss="mean_squared_error",
            metrics=["accuracy"])

data, labels = make_mini_batch(train_data, size_of_batch, length_of_sequences, num_of_input_nodes)
model.fit(data, labels, nb_epoch=num_of_training_epochs, batch_size=size_of_mini_batch)

test_data, test_labels = make_mini_batch(train_data, size_of_mini_batch, length_of_sequences, num_of_input_nodes)
score = model.evaluate(test_data, test_labels, batch_size=size_of_mini_batch)

print(score)

predict = model.predict(test_data, batch_size=size_of_mini_batch)

print(test_labels)
print(predict)
