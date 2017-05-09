
import numpy as np
import random
import json
import math

import keras
from keras.models import Sequential
from keras.layers import *
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras import backend as K

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

def make_data(train_data, size_of_mini_batch, size_of_input_data, concat):
    inputs  = np.empty((0, size_of_input_data * concat))
    outputs = np.empty((0, 360))
    base    = 5000
    for index in range(size_of_mini_batch):
        i = base + index
        inputs_part    = train_data["inputs"][i]
        outputs_part    = train_data["outputs"][i]
        inputs  = np.append(inputs, inputs_part)
        outputs = np.append(outputs, int(degrees(outputs_part[0])))
    inputs = inputs.reshape(-1, size_of_input_data * concat, 1)
    print(outputs[0])
    outputs = keras.utils.to_categorical(outputs, 360)
    print(outputs[0])
    print("inputs: %s, outputs: %s" % (inputs.shape, outputs.shape))
    return (inputs, outputs)

def make_rand_data(num_of_samples, data, labels):
    rand_data   = np.empty((0, 1))
    rand_labels = np.empty((0, len(labels[0])))
    print("%s, %s, %s" % (rand_data.shape, rand_labels.shape, data[0].shape))
    for _ in range(num_of_samples):
        index   = random.randint(0, len(data))
        rand_data   = np.vstack((rand_data, data[index]))
        rand_labels = np.vstack((rand_labels, labels[index]))
    rand_data = rand_data.reshape(num_of_samples, -1, 1)
    print("rand_data: %s, rand_labels: %s" % (rand_data.shape, rand_labels.shape))
    return (rand_data, rand_labels)

concat                      = 2
num_of_input_nodes          = 50
num_of_output_nodes         = 360
num_of_training_epochs      = 50
size_of_mini_batch          = 100
size_of_batch               = 500

keys = []
brainwaves = []
mouse = dict()
mouse["move"] = []
mouse["button"] = []
concat_data = np.empty(0)


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
        concat_data = np.append(concat_data, data["brainwaves"])
        if(len(concat_data) > num_of_input_nodes * concat):
            concat_data = concat_data[num_of_input_nodes:len(concat_data)]
        # ignore zero
        if int(data["mouse"]["move"][0]) == 0 or len(concat_data) < num_of_input_nodes * concat:
            continue
        brainwaves.append(concat_data)
        keys.append(data["keys"])
        mouse["move"].append(data["mouse"]["move"])
        mouse["button"].append(data["mouse"]["button"])

brainwaves      = np.asarray(brainwaves)
keys            = np.asarray(keys)
mouse["move"]   = np.asarray(mouse["move"])
mouse["button"] = np.asarray(mouse["button"])

print("brainwaves: %s" % len(brainwaves))
print("channel first: %s" % K.image_data_format())

train_data = dict()
train_data["inputs"] = brainwaves
train_data["outputs"] = mouse["move"]

data, labels = make_data(train_data, size_of_batch, num_of_input_nodes, concat)
tb_cb = keras.callbacks.TensorBoard(log_dir="/tmp/tensorflow_log", histogram_freq=1)
callbacks = [tb_cb]

# 乱数シードを固定する。
random.seed(0)
np.random.seed(0)

model = Sequential()
model.add(Conv1D(64, 3, padding="same", input_shape=(num_of_input_nodes * concat, 1)))
model.add(Conv1D(128, 3, padding="same", activation="relu"))
model.add(MaxPooling1D())
#model.add(Dropout(0.25))
#model.add(Flatten())
model.add(LSTM(128))
model.add(Dense(256, activation="relu"))
#model.add(Dropout(0.5))
model.add(Dense(num_of_output_nodes, activation="softmax"))
model.compile(optimizer=keras.optimizers.Adadelta(), loss="categorical_crossentropy", metrics=["accuracy"])

model.summary()

model.fit(data, labels, nb_epoch=num_of_training_epochs, batch_size=size_of_mini_batch, callbacks=callbacks, validation_split=0.2)

test_data, test_labels = make_rand_data(20, data, labels)
predict = model.predict(test_data, batch_size=size_of_mini_batch)

print("predict: %s" % predict)
print("test_labels: %s" % test_labels)
