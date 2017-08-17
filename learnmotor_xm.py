import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


sess = tf.Session()

def read_data(filename):
	lines = []
	with open(filename) as f:
		for line in f:
			line = line.split()
			lines.append(line)
	return lines

def main():
	raw_data = read_data("motor.txt")
	attributes = raw_data.pop(0)
	x_axis = np.array([[item[0]] for item in raw_data]).astype(np.float)
	x_norm = (x_axis - np.mean(x_axis, axis=0)) / float(np.std(x_axis, axis=0))
	y_axis = np.array([[item[1]] for item in raw_data]).astype(np.float)
	y_norm = (y_axis - np.mean(y_axis, axis=0)) / float(np.std(y_axis, axis=0))
	# plot_data(x_axis, y_axis)
	train_nn(x_norm, y_norm)

def split_data(raw_data, raw_value, split_rate = 0.7):
	data_size = raw_data.shape[0]
	train_size = int(data_size * split_rate)
	test_size = data_size - train_size

	train_choice = np.random.choice(data_size, train_size)
	test_choice = np.array([i for i in range(0, data_size) if i not in train_choice])

	train_set, train_value = raw_data[train_choice], raw_data[test_choice]
	test_set, test_value = raw_data[test_choice], raw_data[test_value]

	return train_set, train_value, test_set, test_value




def train_nn(training_data, training_value, regularization_param=0.0):
	# fig = plt.figure()
	# ax = fig.add_subplot(111)
	# ax.scatter(training_data, training_value)
	# plt.ion()
	# plt.show()

	# regularization_param = 0.01
	learning_rate = 0.01

	data_size = training_data.shape[0]
	# batch_size = data_size / 5
	# print data_size, batch_size
	# num_neuron = 10

	x = tf.placeholder(tf.float32, [None, 1], name="input_vector")
	y = tf.placeholder(tf.float32, [None, 1], name="true_value")

	w1 = tf.Variable(tf.truncated_normal([1,10]), name="layer1_weight")
	b1 = tf.Variable(tf.constant(0.1, shape=[10]), name="layer1_bias")

	w2 = tf.Variable(tf.truncated_normal([10,10]), name="layer2_weight")
	b2 = tf.Variable(tf.constant(0.1, shape=[10]), name="layer2_bias")

	w3 = tf.Variable(tf.truncated_normal([10,1]), name="layer2_weight")
	b3 = tf.Variable(tf.constant(0.1, shape=[1]), name="layer2_bias")

	layer1_out = tf.nn.relu(tf.matmul(x, w1) + b1)
	layer2_out = tf.sigmoid(tf.matmul(layer1_out, w2) + b2)
	layer3_out = tf.matmul(layer2_out, w3) + b3


	nr_loss = tf.reduce_mean(tf.reduce_sum(tf.square(y - layer3_out), reduction_indices=[1]))

	l1_loss = tf.reduce_mean(tf.reduce_sum(tf.square(y - layer3_out), reduction_indices=[1]))\
	 + regularization_param * (\
	 	tf.reduce_sum(tf.abs(w1)) + \
	 	tf.reduce_sum(tf.abs(w2)) + \
	 	tf.reduce_sum(tf.abs(w3)))

	l2_loss = tf.reduce_mean(tf.reduce_sum(tf.square(y - layer3_out), reduction_indices=[1]))\
	+ regularization_param * (\
		tf.nn.l2_loss(w1) + \
		tf.nn.l2_loss(w2) + \
		tf.nn.l2_loss(w3))

	optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(l2_loss)

	init = tf.global_variables_initializer()

	# with tf.Session() as sess:
	sess.run(init)

	for i in range(20000):
		# choice = np.random.choice(data_size, batch_size)

		# x_batch, y_batch = training_data[choice], training_value[choice]

		_, l = sess.run([optimizer, l2_loss], feed_dict={x:training_data, y:training_value})
		# if i % 100 == 0:
		# 	try:
		# 		ax.lines.remove(lines[0])
		# 	except Exception:
		# 		pass
		# 	print l
		# 	prediction = sess.run(layer3_out, feed_dict={x:training_data})
		# 	lines = ax.plot(training_data, prediction, 'r-', lw=3)
		# 	plt.pause(0.1)



main()

