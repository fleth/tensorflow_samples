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

num_of_input_nodes = 1
num_of_hidden_nodes = 80
num_of_output_nodes = 2 
length_of_sequences = 20
num_of_training_epochs = 50000 # 学習回数
size_of_input_data = 50
size_of_mini_batch = 100
num_of_prediction_epochs = 200
learning_rate = 0.01
forget_bias = 0.8
num_of_sample = 10000

#def get_batch(batch_size, X, t):
#    rnum = [random.randint(0, len(X) - 1) for x in range(batch_size)]
#    xs = np.array([[[y] for y in list(X[r])] for r in rnum])
#    ts = np.array([[t[r]] for r in rnum])
#    return xs, ts
#
#def create_data(nb_of_samples, sequence_len):
#    X = np.zeros((nb_of_samples, sequence_len))
#    for row_idx in range(nb_of_samples):
#        X[row_idx, :] = np.around(np.random.rand(sequence_len)).astype(int)
#    # Create the targets for each sequence
#    t = np.sum(X, axis=1)
#    return X, t
#
#def make_prediction(nb_of_samples):
#    sequence_len = 10
#    xs, ts = create_data(nb_of_samples, sequence_len)
#    return np.array([[[y] for y in x] for x in xs]), np.array([[x] for x in ts])

def get_batch(batch_size, X, t):
    base = random.randint(0, len(X) - batch_size) 
    rnum = range(base, base + batch_size)
    _xs = np.array([[[y] for y in list(X[r])] for r in rnum])
    xs = np.transpose(_xs, [1, 0, 2])
#    _xs = np.array([X[r] for r in rnum])
    ts = np.array([t[r] for r in rnum])
#    print("get_batch: %s, %s" % (xs.shape, ts.shape))
#    return _xs, ts
    return xs, ts

def create_data(nb_of_samples, inputs, trainings):
    rnum = random.randint(0, len(inputs) - 1 - nb_of_samples)
    X = inputs[rnum: rnum + nb_of_samples]
    t = trainings[rnum: rnum + nb_of_samples]
    print("create_data: %s, %s" % (np.array(X).shape, np.array(t).shape))
    return X, t

def make_prediction(nb_of_samples, inputs, trainings):
    print("make_prediction")
    _xs, _ts = create_data(nb_of_samples, inputs, trainings);
    xs = np.array([[[y] for y in x] for x in _xs])
#    xs = np.array(_xs)
    ts = np.array(_ts)
#    return xs, ts
    return np.transpose(xs, [1, 0, 2]), ts

def inference(input_ph, istate_ph):
    with tf.name_scope("inference") as scope:
        weight1_var = tf.Variable(tf.truncated_normal([num_of_input_nodes, num_of_hidden_nodes], stddev=1.0), name="weight1")
        weight2_var = tf.Variable(tf.truncated_normal([num_of_hidden_nodes, num_of_output_nodes], stddev=1.0), name="weight2")
        bias1_var = tf.Variable(tf.truncated_normal([num_of_hidden_nodes], stddev=1.0), name="bias1")
        bias2_var = tf.Variable(tf.truncated_normal([num_of_output_nodes], stddev=1.0), name="bias2")


        weight1_var = tf.Variable(tf.zeros([num_of_input_nodes, num_of_hidden_nodes]), name="weight1")
        weight2_var = tf.Variable(tf.zeros([num_of_hidden_nodes, num_of_output_nodes]), name="weight2")
        bias1_var = tf.Variable(tf.zeros([num_of_hidden_nodes]), name="bias1")
        bias2_var = tf.Variable(tf.zeros([num_of_output_nodes]), name="bias2")

        print("input_ph: %s" % input_ph)

        in1 = tf.transpose(input_ph, [1, 0, 2])
        in2 = tf.reshape(in1, [-1, num_of_input_nodes])
        in3 = tf.nn.relu(tf.matmul(in2, weight1_var) + bias1_var)
        in4 = tf.split(in3, length_of_sequences, 0)

#        in1 = input_ph 
#        in2 = tf.reshape(in1, [-1, num_of_input_nodes])
#        in3 = tf.nn.relu(tf.matmul(input_ph, weight1_var) + bias1_var)
#        in4 = tf.split(0, size_of_mini_batch, in3)

        cell = tf.contrib.rnn.BasicLSTMCell(
            num_of_hidden_nodes, forget_bias=forget_bias, state_is_tuple=False)
        rnn_outputs, states_op = tf.contrib.rnn.static_rnn(cell, in4, initial_state=istate_ph)
        rnn_output = [ro[0] for ro in rnn_outputs]
        output_op = tf.matmul(rnn_output, weight2_var) + bias2_var

        print("weight1_var: %s" % weight1_var)
        print("weight2_var: %s" % weight2_var)
        print("in1: %s" % in1)
        print("in2: %s" % in2)
        print("in3: %s" % in3)
        print("in4: %s" % in4)
        print("cell: %s" % cell)
        print("output_op: %s" % output_op)
        print("states_op: %s" % states_op)

        w1_hist = tf.summary.histogram("weights1", weight1_var)
        w2_hist = tf.summary.histogram("weights2", weight2_var)
        b1_hist = tf.summary.histogram("biases1", bias1_var)
        b2_hist = tf.summary.histogram("biases2", bias2_var)
        output_hist = tf.summary.histogram("output",  output_op)

        results = [weight1_var, weight2_var, bias1_var,  bias2_var]
        return output_op, states_op, results


def loss(output_op, supervisor_ph):
    with tf.name_scope("loss") as scope:
        square_error = tf.reduce_mean(tf.square(output_op - supervisor_ph))
        loss_op = square_error
        tf.summary.scalar("loss", loss_op)
        return loss_op


def training(loss_op):
    with tf.name_scope("training") as scope:
        training_op = optimizer.minimize(loss_op)
        return training_op


def calc_accuracy(output_op, inputs_data, training_data, prints=False):
    inputs, ts = make_prediction(length_of_sequences, inputs_data, training_data)
    pred_dict = {
        input_ph:  inputs,
        supervisor_ph: ts,
        istate_ph:    np.zeros((size_of_input_data, num_of_hidden_nodes * 2)),
    }
    output = sess.run([output_op], feed_dict=pred_dict)

#    print("output: %s" % output)

    def print_result(i, p, q):
        print("print_result")
#        [print(list(x)[0]) for x in i]
        print("output: %s" % p)
        print("correct: %s" % q)
    if prints:
        print("prints")
        print(output, ts)
#        [print_result(i, p, q) for i, p, q in zip(inputs, output, ts)]

    # ここの評価方法を修正する
    opt = abs(output - ts)
    total = 0
    for x in opt:
        total += sum([1 if y[0] < 0.01 else 0 for y in x])
    print(total)
    print("accuracy %f" % (total / float(len(ts) * len(ts[0]))))
    return output

random.seed(0)
np.random.seed(0)
tf.set_random_seed(0)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

#X, t = create_data(num_of_sample, length_of_sequences)
X, t = create_data(num_of_sample, brainwaves, mouse["move"])

with tf.Graph().as_default():
    input_ph = tf.placeholder(tf.float32, [None, length_of_sequences, num_of_input_nodes], name="input")
    supervisor_ph = tf.placeholder(tf.float32, [None, num_of_output_nodes], name="supervisor")
    istate_ph = tf.placeholder(tf.float32, [None, num_of_hidden_nodes * 2], name="istate")

    output_op, states_op, datas_op = inference(input_ph, istate_ph)
    loss_op = loss(output_op, supervisor_ph)
    training_op = training(loss_op)

    #summary_op = tf.merge_all_summaries()
    summary_op = tf.summary.merge_all()
    #init = tf.initialize_all_variables()
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        saver = tf.train.Saver()
        #summary_writer = tf.train.SummaryWriter("/tmp/tensorflow_log", graph=sess.graph)
        summary_writer = tf.summary.FileWriter("/tmp/tensorflow_log", graph=sess.graph)
        sess.run(init)

        for epoch in range(num_of_training_epochs):
            inputs, supervisors = get_batch(length_of_sequences, X, t)
            train_dict = {
                input_ph:      inputs,
                supervisor_ph: supervisors,
                istate_ph:     np.zeros((size_of_input_data, num_of_hidden_nodes * 2)),
            }
            sess.run(training_op, feed_dict=train_dict)

            if (epoch) % 100 == 0:
                summary_str, train_loss = sess.run([summary_op, loss_op], feed_dict=train_dict)
                print("train#%d, train loss: %e" % (epoch, train_loss))
                summary_writer.add_summary(summary_str, epoch)
                if (epoch) % 500 == 0:
                    calc_accuracy(output_op, brainwaves, mouse["move"])

        calc_accuracy(output_op, brainwaves, mouse["move"], prints=True)
        datas = sess.run(datas_op)
        saver.save(sess, "model.ckpt")
