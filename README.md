# Building neuron network module in TensorFlow
* __Requirements__:
  * Python 2
  * TensorFlow
  * numpy

This implementation is adapted from Danijar Hafner's blog: https://danijar.com/structuring-your-tensorflow-models

* __Modifications__:
  * Nested class "Layer" that allows dynamic layers creation
  * Uses L1 or L2 regularization (By default L2 with learning rate 0.05)
  
The structure of the network is defined in GraphModel.py.

A sigmoid and a linear layer are hard coded into the network.

Two training scripts are presented: learnmotor_m.py and learnmotor_xm.py.
1. learnmotor_m.py uses the structure defined in the GraphModel module. The script trains (using batch gradient descent) and tests the model on the data defined in motor.txt with different lambdas (regularization parameter) and plot log(lambda) vs. MSE on test data. (Data in motor.txt are randomly partitioned into 70% training data and 30% test data at runtime)
2. learnmotor_xm.py defines a network in the train_nn function and only trains the model on training data.

__Sample Result__:
Refer to out5.png. The x axis is log(lambda) and the y axis is MSE on test data

__Future work__:

1. Modify GraphModel.py to allow dynamic layer-adding at runtime
2. Also monitor MSE on training data
