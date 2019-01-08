import tensorflow as tf
from collections import namedtuple

DenseLayer = namedtuple('DenseLayer', ['units', 'activation'])
def create_dense_neural_net(input_tensor, dense_layers):
    current_input_tensor = input_tensor
    for dense_layer in dense_layers:
        current_input_tensor = tf.layers.dense(current_input_tensor, dense_layer.units, activation=dense_layer.activation)

    return current_input_tensor


ConvLayer = namedtuple('ConvLayer', ['filters', 'kernel_size', 'strides'])
def create_convolutional_neural_net(input_tensor, conv_layers):
    current_input_tensor = input_tensor
    for conv_layer in conv_layers:
        current_input_tensor = tf.layers.conv2d(current_input_tensor, conv_layer.filters, conv_layer.kernel_size, strides=conv_layer.strides)

    return current_input_tensor
