
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM

def make_mini_batch(train_data, size_of_mini_batch, length_of_sequences):
    inputs  = np.empty(0)
    outputs = np.empty(0)
    for _ in range(size_of_mini_batch):
        index   = random.randint(0, len(train_data) - length_of_sequences)
        part    = train_data[index:index + length_of_sequences]
        inputs  = np.append(inputs, part[:, 0])
        outputs = np.append(outputs, part[-1, 1])
    inputs  = inputs.reshape(-1, length_of_sequences, 1)
    outputs = outputs.reshape(-1, 1)
    return (inputs, outputs)

def make_prediction_initial(train_data, index, length_of_sequences):
    return train_data[index:index + length_of_sequences, 0]

train_data_path             = "../data/train_data/normal.npy"
num_of_input_nodes          = 1
num_of_hidden_nodes         = 2
num_of_output_nodes         = 1
length_of_sequences         = 50
num_of_training_epochs      = 500
length_of_initial_sequences = 50
num_of_prediction_epochs    = 100
size_of_mini_batch          = 100
size_of_batch               = 1000
learning_rate               = 0.1
forget_bias                 = 1.0
print("train_data_path             = %s" % train_data_path)
print("num_of_input_nodes          = %d" % num_of_input_nodes)
print("num_of_hidden_nodes         = %d" % num_of_hidden_nodes)
print("num_of_output_nodes         = %d" % num_of_output_nodes)
print("length_of_sequences         = %d" % length_of_sequences)
print("num_of_training_epochs      = %d" % num_of_training_epochs)
print("length_of_initial_sequences = %d" % length_of_initial_sequences)
print("num_of_prediction_epochs    = %d" % num_of_prediction_epochs)
print("size_of_mini_batch          = %d" % size_of_mini_batch)
print("learning_rate               = %f" % learning_rate)
print("forget_bias                 = %f" % forget_bias)

train_data = np.load(train_data_path)
print("train_data:", train_data)

# 乱数シードを固定する。
random.seed(0)
np.random.seed(0)

model = Sequential()
model.add(LSTM(num_of_hidden_nodes, batch_input_shape=(None, length_of_sequences, num_of_input_nodes)))
model.add(Dense(num_of_output_nodes))
model.compile(optimizer="adam",
            loss="mean_squared_error",
            metrics=["accuracy"])

data, labels = make_mini_batch(train_data, size_of_batch, length_of_sequences)

model.fit(data, labels, nb_epoch=num_of_training_epochs, batch_size=size_of_mini_batch)

test_data, test_labels = make_mini_batch(train_data, size_of_mini_batch, length_of_sequences)

score = model.evaluate(test_data, test_labels, batch_size=size_of_mini_batch)

print(score)

inputs = test_data[0]
outputs = np.empty(0)

for epoch in range(num_of_prediction_epochs):
    predict = model.predict(inputs.reshape(1, length_of_sequences, num_of_input_nodes), batch_size=1)
    output = predict[0]
    inputs  = np.delete(inputs, 0)
    inputs  = np.append(inputs, output)
    outputs = np.append(outputs, output)

print(outputs)
