import random

import tensorflow as tf
import numpy as np

from ..library import DenseLayer, create_dense_neural_net


class DQN:
    def __init__(self, environment, save_location, max_items_in_replay_buffer, frames_to_change_epsilon=1e4):
        self.environment = environment
        self.save_location = save_location

        self.state_size = self.environment.get_state_shape()
        self.action_size = self.environment.get_number_of_actions()

        self.input = tf.placeholder(tf.float32, shape=[None, *self.state_size])
        self.actions_taken = tf.placeholder(tf.int32, shape=[None])
        self.y_values = tf.placeholder(tf.float32, shape=[None])

        self.learning_rate = .001
        self.discount_factor = 0.99

        self.max_epsilon = 1.
        self.min_epsilon = .1
        self.frames_to_change_epsilon = frames_to_change_epsilon

        self.frequency_to_copy_into_target = 50

        self.max_items_in_replay_buffer = max_items_in_replay_buffer

        self._create_learning_network()

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver(tf.global_variables())

    def _get_random_action(self):
        return random.randint(0, self.action_size - 1)

    def _create_learning_network(self):
        def get_neural_net(name):
            return create_dense_neural_net(
                self.input,
                [
                    DenseLayer(100, activation=tf.keras.activations.tanh),
                    DenseLayer(50, activation=tf.keras.activations.tanh),
                    DenseLayer(25, activation=tf.keras.activations.tanh),
                    DenseLayer(self.action_size, activation=None),
                ],
                name,
            )

        self.training_nn = get_neural_net('DQNetwork')
        self.target_nn = get_neural_net('TargetDQNetwork')

        self.action_to_take = tf.argmax(self.training_nn, axis=1)
        self.max_target_value = tf.reduce_max(self.target_nn, axis=1)

        self.one_hot_actions = tf.one_hot(self.actions_taken, self.action_size)

        self.Q = tf.reduce_sum(self.one_hot_actions * self.training_nn, axis=1)

        self.loss = tf.reduce_sum(tf.square(self.y_values - self.Q))

        self.train = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def _load_saved_model(self):
        try:
            self.saver.restore(self.session, self.save_location)
            print('successfully loaded old model data')
        except Exception as e:
            print('do not currently have data stored for this model')

    def _save_model(self):
        self.saver.save(self.session, self.save_location)

    def _update_target_graph(self):
        # Get the parameters of our DQNNetwork
        from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "DQNetwork")

        # Get the parameters of our Target_network
        to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "TargetDQNetwork")

        op_holder = []
        # Update our target_network parameters with DQNNetwork parameters
        for from_var, to_var in zip(from_vars, to_vars):
            op_holder.append(to_var.assign(from_var))
        return op_holder

    def train_model(self, number_of_sessions_to_run, max_steps, replay_buffer_batch_size):
        self._load_saved_model()

        current_epsilon = self.max_epsilon
        current_training_index = 0

        replay_buffer = [None]
        replay_buffer_index = 0

        target_copy_index = 0
        session_lengths = []
        loss = None
        for session_index in range(number_of_sessions_to_run):

            state = self.environment.reset()
            for step_index in range(max_steps):
                action_to_take = None
                target_nn_action_values = None

                # Determines whether to take a random action or not
                epsilon_test_random = random.random()
                if epsilon_test_random < current_epsilon:
                    state_actions, target_actions, action_to_take = self.session.run(
                        [self.training_nn, self.target_nn, self.action_to_take],
                        feed_dict={self.input: [state]})
                    action_to_take = action_to_take[0]
                else:
                    action_to_take = self._get_random_action()

                new_state, reward, done = self.environment.step(action_to_take)

                replay_buffer_object = (state, action_to_take, reward, new_state, done)

                if (len(replay_buffer) < self.max_items_in_replay_buffer):
                    replay_buffer[replay_buffer_index] = replay_buffer_object
                    replay_buffer.append(None)
                    replay_buffer_index += 1
                else:
                    replay_buffer[replay_buffer_index] = replay_buffer_object
                    replay_buffer_index = (replay_buffer_index + 1) % \
                        self.max_items_in_replay_buffer

                training_batch = []
                for i in range(replay_buffer_batch_size):
                    index = random.randint(0, len(replay_buffer) - 2)
                    training_batch.append(replay_buffer[index])

                initial_states = [item[0] for item in training_batch]
                actions_taken = [item[1] for item in training_batch]
                next_states = [item[3] for item in training_batch]
                max_next_step_target_values = self.session.run(
                    self.max_target_value,
                    feed_dict={
                        self.input: next_states,
                    })

                y_values = []
                for i in range(len(training_batch)):
                    target_reward = training_batch[i][2]
                    if not training_batch[i][4]:
                        target_reward += self.discount_factor * max_next_step_target_values[i]
                    y_values.append(target_reward)

                loss, _ = self.session.run(
                    [self.loss, self.train],
                    feed_dict={
                        self.input: initial_states,
                        self.y_values: y_values,
                        self.actions_taken: actions_taken,
                    })

                state = new_state

                if done:
                    session_lengths.append(step_index)
                    break

            if target_copy_index == self.frequency_to_copy_into_target:
                self.session.run(self._update_target_graph())
                target_copy_index = 0
            else:
                target_copy_index += 1

            # Only update epsilon after a full run through in the environment
            current_epsilon = self.max_epsilon - (self.max_epsilon - self.min_epsilon) * \
                min(1.0 * session_index / self.frames_to_change_epsilon, 1)

            if session_index % 50 == 0:
                self._save_model()
                print('Session: {} Current Epsilon: {} Avg Length: {} Longest Length: {} Last Loss: {}'.format(
                    session_index,
                    current_epsilon,
                    np.average(session_lengths[-50:]),
                    np.max(session_lengths[-50:]),
                    loss))
        self._save_model()

    def run_model(self, number_of_episodes):
        self._load_saved_model()

        for episode_index in range(number_of_episodes):

            state = self.environment.reset()
            done = False
            while not done:
                action_to_take = self.session.run(
                    self.action_to_take,
                    feed_dict={self.input: [state]})[0]

                state, _, done = self.environment.step(action_to_take)
