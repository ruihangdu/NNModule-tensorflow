import tensorflow as tf
import math
import numpy as np
from GraphModel import Model
import matplotlib.pyplot as plt



init = tf.global_variables_initializer()

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
	x_axis = np.array([[item[0]] for item in raw_data]).astype(np.float32)

	x_norm = (x_axis - np.mean(x_axis, axis=0)) / float(np.std(x_axis, axis=0))
	x_norm += np.random.normal(0, 0.02, x_norm.shape)

	y_axis = np.array([[item[1]] for item in raw_data]).astype(np.float32)

	y_norm = (y_axis - np.mean(y_axis, axis=0)) / float(np.std(y_axis, axis=0))
	y_norm += np.random.normal(0, 0.02, y_norm.shape)

	# train_nn(x_norm, y_norm)
	validation_error = []
	reg_params = np.logspace(-6.0, -3.3, num=14, base=2)
	np.insert(reg_params, 0, 0.0, axis=0)
	for reg_param in reg_params:
		validation_error.append(cross_validation(x_norm, y_norm, reg_param))
	validation_error = np.array(validation_error)

	plt.plot(reg_params, validation_error)
	# plt.ion()
	plt.show()




def cross_validation(x_data, y_data, reg_param):
	errors = []
	for i in range(10):
		train_x, train_y, test_x, test_y = split_data(x_data, y_data)
		placehold_x = tf.placeholder(tf.float32, [None, train_x.shape[1]], name="input_vector")
		placehold_y = tf.placeholder(tf.float32, [None, train_y.shape[1]], name="target_value")

		sess = tf.Session()
		model= train_nn(sess, placehold_x, placehold_y, train_x, train_y, reg_param)
		# print "training done"
		# print "using test data size", test_y.shape[0]
		# errors.append(get_error(model, test_x, test_y))
		errors.append(get_error(sess, placehold_x, placehold_y, model, test_x, test_y))
	errors = np.array(errors)
	return np.mean(errors)


def get_error(sess, x, y, model, test_x, test_y):
	# prediction = test_x
	# for l in model:
	# 	prediction = l.process(prediction)
	# print sess.run(prediction)
	return sess.run(model.error, feed_dict={x:test_x, y:test_y})


def split_data(raw_data, raw_value, split_rate = 0.7):
	data_size = raw_data.shape[0]
	train_size = int(data_size * split_rate)
	test_size = data_size - train_size

	train_choice = np.random.choice(data_size, train_size)
	test_choice = np.array([i for i in range(0, data_size) if i not in train_choice])

	train_set, train_value = raw_data[train_choice], raw_value[train_choice]
	test_set, test_value = raw_data[test_choice], raw_value[test_choice]

	return train_set, train_value, test_set, test_value


def train_nn(sess, x, y, data, target, reg_param):
	# fig = plt.figure()
	# ax = fig.add_subplot(111)
	# ax.scatter(data, target)
	# plt.ion()
	# plt.show()

	# sess = tf.Session()
	# x = tf.placeholder(tf.float32, [None, data.shape[1]], name="input_vector")
	# y = tf.placeholder(tf.float32, [None, target.shape[1]], name="target_value")
	new_model = Model(x, y, sess, reg_param)
	sess.run(init)

	iteration = 0
	old_training_loss = 0

	while True:
		try:
			prediction = sess.run(new_model.prediction, feed_dict={x:data})
		except Exception:
			# print [n.name for n in tf.get_default_graph().as_graph_def().node]
			print [v.name for v in tf.trainable_variables()]
			return


		training_loss = sess.run(new_model.error, feed_dict={x:data, y:target})
		

		# if iteration % 1000 == 0:
		# 	print "round", iteration, training_loss
		# 	try:
		# 		ax.lines.remove(lines[0])
		# 	except Exception:
		# 		pass
		# 	# print l
		# 	# prediction = sess.run(layer3_out, feed_dict={x:training_data})
		# 	lines = ax.plot(data, prediction, 'r-', lw=3)
		# 	plt.pause(0.1)

		if abs(training_loss - old_training_loss) < 0.0000001:
			break

		sess.run(new_model.optimize, feed_dict={x:data, y:target})
		iteration += 1

		old_training_loss = training_loss
	# return new_model.get_layers(), sess
	return new_model
	
if __name__ == "__main__":
	main()