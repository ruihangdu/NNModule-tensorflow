import tensorflow as tf

v1 = tf.Variable(1.0, name="var1")

sess = tf.Session()
writer = tf.summary.FileWriter("logs/", sess.graph)