from collections import namedtuple

import tensorflow as tf
import gym

DenseLayer = namedtuple('DenseLayer', ['units', 'activation'])

ConvLayer = namedtuple('ConvLayer', ['filters', 'kernel_size', 'strides'])


def create_dense_neural_net(input_tensor, dense_layers, variable_scope=''):
    with tf.variable_scope(variable_scope):
        current_input_tensor = input_tensor
        for dense_layer in dense_layers:
            current_input_tensor = tf.layers.dense(
                current_input_tensor,
                dense_layer.units,
                activation=dense_layer.activation)

        return current_input_tensor


def create_convolutional_neural_net(input_tensor, conv_layers, variable_scope=''):
    with tf.variable_scope(variable_scope):
        current_input_tensor = input_tensor
        for conv_layer in conv_layers:
            current_input_tensor = tf.layers.conv2d(
                current_input_tensor,
                conv_layer.filters,
                conv_layer.kernel_size,
                strides=conv_layer.strides)

        return current_input_tensor


class GenericOpenAIGymEnv:
    def __init__(self, should_render):
        self.should_render = should_render

        self.env = self._create_gym_env()

    def get_number_of_actions(self):
        if isinstance(self.env.action_space, gym.spaces.Box):
            return self.env.action_space.shape[0]
        else:
            return self.env.action_space.n

    def get_random_action(self):
        return self.env.action_space.sample()

    def get_state_shape(self):
        return self.env.observation_space.shape

    def reset(self):
        return self.env.reset()

    def step(self, action):
        if self.should_render:
            self.env.render()

        state, reward, done, _ = self.env.step(action)
        return state, reward, done
