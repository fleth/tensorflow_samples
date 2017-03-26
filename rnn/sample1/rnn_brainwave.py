
import tensorflow as tf
import numpy as np
import random
import json


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
        outputs = np.append(outputs, outputs_part[-1][0])
    inputs  = inputs.reshape(-1, length_of_sequences, size_of_input_data, 1)
    outputs = outputs.reshape(-1, 1)
    return (inputs, outputs)

def make_prediction_initial(train_data, index, length_of_sequences):
    inputs = train_data["inputs"][index:index + length_of_sequences]
    outputs = train_data["outputs"][index:index + length_of_sequences]
    return np.array(inputs), np.array(outputs[-1])




train_data_path             = "../data/train_data/normal.npy"
num_of_input_nodes          = 1
num_of_hidden_nodes         = 2
num_of_output_nodes         = 1
length_of_sequences         = 5
num_of_training_epochs      = 5000
length_of_initial_sequences = 5
num_of_prediction_epochs    = 100
size_of_mini_batch          = 100
size_of_input_data          = 50
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

#train_data = np.load(train_data_path)

train_data = dict()
train_data["inputs"] = brainwaves
train_data["outputs"] = mouse["move"]

print("train_data:", train_data)

# 乱数シードを固定する。
random.seed(0)
np.random.seed(0)
tf.set_random_seed(0)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

with tf.Graph().as_default():
    input_ph      = tf.placeholder(tf.float32, [None, length_of_sequences, size_of_input_data, num_of_input_nodes], name="input")
    supervisor_ph = tf.placeholder(tf.float32, [None, num_of_output_nodes], name="supervisor")
    istate_ph     = tf.placeholder(tf.float32, [None, num_of_hidden_nodes * 2], name="istate") # 1セルあたり2つの値を必要とする。

    with tf.name_scope("inference") as scope:
        weight1_var = tf.Variable(tf.truncated_normal([num_of_input_nodes, num_of_hidden_nodes], stddev=1.0), name="weight1")
        weight2_var = tf.Variable(tf.truncated_normal([num_of_hidden_nodes, num_of_output_nodes], stddev=0.1), name="weight2")
        bias1_var   = tf.Variable(tf.truncated_normal([num_of_hidden_nodes], stddev=1.0), name="bias1")
        bias2_var   = tf.Variable(tf.truncated_normal([num_of_output_nodes], stddev=0.1), name="bias2")

#        weight1_var = tf.Variable(tf.zeros([num_of_input_nodes, num_of_hidden_nodes]), name="weight1")
#        weight2_var = tf.Variable(tf.zeros([num_of_hidden_nodes, num_of_output_nodes]), name="weight2")
#        bias1_var   = tf.Variable(tf.zeros([num_of_hidden_nodes]), name="bias1")
#        bias2_var   = tf.Variable(tf.zeros([num_of_output_nodes]), name="bias2")

        in1 = tf.transpose(input_ph, [1, 0, 2, 3])         # (batch, sequence, size, data) -> (sequence, batch, size, data)
        in2 = tf.reshape(in1, [-1, num_of_input_nodes]) # (sequence, batch, data) -> (sequence * batch * size, data)
        in3 = tf.matmul(in2, weight1_var) + bias1_var
        in4 = tf.split(in3, length_of_sequences * size_of_input_data)     # sequence * (batch, data)

        cell = tf.contrib.rnn.BasicLSTMCell(num_of_hidden_nodes, forget_bias=forget_bias, state_is_tuple=False)
        rnn_output, states_op = tf.contrib.rnn.static_rnn(cell, in4, initial_state=istate_ph)
        output_op = tf.matmul(rnn_output[-1], weight2_var) + bias2_var

    with tf.name_scope("loss") as scope:
        square_error = tf.reduce_mean(tf.square(output_op - supervisor_ph))
        loss_op      = square_error
        tf.summary.scalar("loss", loss_op)

    with tf.name_scope("training") as scope:
        training_op = optimizer.minimize(loss_op)

    summary_op = tf.summary.merge_all()
    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        saver = tf.train.Saver()
        summary_writer = tf.summary.FileWriter("/tmp/tensorflow_log", graph=sess.graph)
        sess.run(init)

        for epoch in range(num_of_training_epochs):
            inputs, supervisors = make_mini_batch(train_data, size_of_mini_batch, length_of_sequences, size_of_input_data)

            train_dict = {
                input_ph:      inputs,
                supervisor_ph: supervisors,
                istate_ph:     np.zeros((size_of_mini_batch, num_of_hidden_nodes * 2)),
            }
            sess.run(training_op, feed_dict=train_dict)

            if (epoch + 1) % 10 == 0:
                summary_str, train_loss = sess.run([summary_op, loss_op], feed_dict=train_dict)
                summary_writer.add_summary(summary_str, epoch)
                print("train#%d, train loss: %e" % (epoch + 1, train_loss))

        #=================================================================================

        inputs, supervisors = make_mini_batch(train_data, num_of_prediction_epochs, length_of_sequences, size_of_input_data)
        outputs = np.empty(0)
        states  = np.zeros((num_of_hidden_nodes * 2)),

        print("initial:", inputs)
        np.save("initial.npy", inputs)

        for epoch in range(num_of_prediction_epochs):
            pred_dict = {
                input_ph:  inputs[epoch].reshape((1, length_of_sequences, size_of_input_data, 1)),
                istate_ph: states,
            }
            output, states = sess.run([output_op, states_op], feed_dict=pred_dict)
            print("prediction#%d, output: %f" % (epoch + 1, output))

            outputs = np.append(outputs, output)

        print("supervisors:", supervisors)
        print("outputs:", outputs)
        np.save("output.npy", outputs)

        saver.save(sess, "data/model")
