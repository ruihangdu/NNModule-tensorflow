import tensorflow as tf
import numpy as np
import functools

def doublewrap(function):
    """
    A decorator decorator, allowing to use the decorator to be used without
    parentheses if not arguments are provided. All arguments must be optional.
    """
    @functools.wraps(function)
    def decorator(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return function(args[0])
        else:
            return lambda wrapee: function(wrapee, *args, **kwargs)
    return decorator


@doublewrap
def define_scope(function, scope=None, *args, **kwargs):
    """
    A decorator for functions that define TensorFlow operations. The wrapped
    function will only be executed once. Subsequent calls to it will directly
    return the result so that operations are added to the graph only once.

    The operations added by the function live within a tf.variable_scope(). If
    this decorator is used with arguments, they will be forwarded to the
    variable scope. The scope name defaults to the name of the wrapped
    function.
    """
    attribute = '_cache_' + function.__name__
    name = scope or function.__name__
    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(name, *args, **kwargs):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return decorator

# def lazy_property(function):
# 	attribute = '_cache_' + function.__name__

# 	@property
# 	@functools.wraps(function)
# 	def decorator(self):
# 		if not hasattr(self, attribute):
# 			setattr(self, attribute, function(self))
# 		return getattr(self, attribute)
# 	return decorator


class Model:
	class Layer:
		def __init__(self, num_in, num_out, sess, type=None):
			# print num_in, num_out

			self.weight = tf.Variable(tf.random_normal([num_in, num_out]), name="layer_weight")
			# self.bias = tf.Variable(tf.constant(0.1, shape=[num_out]), name="layer_bias")
			self.bias = tf.Variable(tf.zeros([1, num_out]) + 0.1, name="bias")

			init_weights = tf.variables_initializer([self.weight, self.bias], name="init_weights")
			sess.run(init_weights)

			self.activation = type

		def process(self, data):
			# print data
			z = tf.matmul(data, self.weight) + self.bias
			if self.activation == None:
				return z
			else:
				return self.activation(z)


	def __init__(self, data, target, sess, reg_param, loss_func=2):
		self.data = data
		self.target = target
		self.sess = sess

		self.learning_rate = 0.05
		self.reg = loss_func
		self.lbd = reg_param


		self.layers = []
		# print self.data.get_shape()[1]
		layer1 = self.Layer(int(self.data.get_shape()[1]), 10, self.sess, tf.sigmoid)
		layer2 = self.Layer(10, int(self.target.get_shape()[1]), self.sess)
		self.layers.append(layer1)
		self.layers.append(layer2)
		# self._prediction = None
		# self._optimize = None
		# self._error = None
		self.prediction
		self.optimize
		self.error

	def get_layers(self):
		return self.layers
	# def add_layer(self, in_size, out_size, activation_func=None):
	# 	weight = tf.Variable(tf.random_normal([num_in, num_out]), name="layer_weight")
	# 	bias = tf.Variable(tf.zeros([1, num_out]) + 0.1, name="bias")

	# 	return activation_func(tf.add(tf.matmul()))
		# self.layer3

		# data_size = int(data.get_shape()[1])
		# target_size = int(target.get_shape()[1])

		# weight = tf.Variable(tf.truncated_normal([data_size, target_size]))
		# bias = tf.Variable(tf.constant(0.1, shape=[target_size]))

		# z = tf.matmul(data, weight) + bias
		# self._prediction = tf.sigmoid(z)

		# if loss_func == 1:
		# 	loss = tf.reduce_mean(tf.reduce_sum(tf.sqaure(target - self._prediction), reduction_indices=[1])) \
		# 	+ lbd * tf.reduce_sum(tf.abs(weight))
		# else:
		# 	loss = tf.reduce_mean(tf.reduce_sum(tf.sqaure(target - self._prediction), reduction_indices=[1])) \
		# 	+ lbd * tf.reduce_sum(tf.multiply(weight, weight))

		# self._optimize = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
	@define_scope
	def prediction(self):
		# if not self._prediction:
		layer_out = self.data
		for layer in self.layers:
			layer_out = layer.process(layer_out)
			# self._prediction = layer_out
		return layer_out

	@define_scope
	def error(self):
		# if not self._error:
		if self.reg == 1:
			# print "using L1 regularization"
			loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.target - self.prediction), reduction_indices=[1]))\
			+ self.lbd * reduce(lambda x,y: x+y, [tf.reduce_sum(tf.abs(layer.weight)) for layer in self.layers])
		else:
			# print "using L2 regularization"
			loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.target - self.prediction), reduction_indices=[1])) \
			+ self.lbd * reduce(lambda x,y: x+y, [tf.nn.l2_loss(layer.weight) for layer in self.layers])
				# self._error = loss
		return loss

	@define_scope
	def optimize(self):
		# if not self._optimize:
		optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.error)
		return optimizer








