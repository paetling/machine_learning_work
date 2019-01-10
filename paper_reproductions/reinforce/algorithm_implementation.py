import sys
import tensorflow as tf
from ..library import DenseLayer, create_dense_neural_net

class Reinforce:
    # environment should have a way to get the state, rewards and say if you are done or not
    def __init__(self, environment):
        self.environment = environment

        self.state_size = self.environment.get_state_shape()
        self.action_size = self.environment.get_number_of_actions()

        self.input = tf.placeholder(tf.float32, shape=[None, *self.state_size])
        self.discounted_episode_reward = tf.placeholder(tf.float32, shape=None)

        self.learning_rate = 2.5e-4
        self.discount_factor = 0.99

        self.softmax = None
        self.loss = None
        self.train = None
        self._create_learning_network(self.action_size)

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())


    def _create_learning_network(self, output_units):
        # the idea is to manually do this update
        # ∆Theta = learning_rate * r * ∆ln(pi)
        self.softmax = create_dense_neural_net(
            self.input,
            [
                DenseLayer(100, activation=tf.keras.activations.tanh),
                DenseLayer(50, activation=tf.keras.activations.tanh),
                DenseLayer(25, activation=None),
                DenseLayer(output_units, activation=tf.keras.activations.softmax),
            ]
        )
        self.log_probailites = tf.math.log(self.softmax)

        self.actions_to_take = tf.distributions.Categorical(probs=self.softmax).sample()

        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        self.gradients_and_vars = optimizer.compute_gradients(self.log_probailites)
        self.old_vars = [pair[1] for pair in self.gradients_and_vars]
        self.new_graidents = [pair[0] * (self.discounted_episode_reward * self.learning_rate) for pair in self.gradients_and_vars]
        self.apply_gradients = optimizer.apply_gradients(zip(self.new_graidents, self.old_vars))


    def train_model(self, number_of_batches, batch_size, max_steps):
        for batch_index in range(number_of_batches):

            discounted_rewards = 0
            number_of_steps_in_batch = []
            for episode_index in range(batch_size):
                number_of_steps_in_batch.append(0)

                state = self.environment.reset()

                for i in range(max_steps):
                    number_of_steps_in_batch[-1] += 1

                    (actions_to_take,) = self.session.run([self.actions_to_take], feed_dict={self.input:[state]})
                    action = actions_to_take[0]
                    old_state = state
                    state, reward, done = self.environment.step(action)

                    discounted_rewards = discounted_rewards * self.discount_factor + reward

                    self.session.run([self.apply_gradients], feed_dict={
                                                                            self.input:[old_state],
                                                                            self.discounted_episode_reward: discounted_rewards
                                                                            })

                    if done:
                        break


            print('Training Batch: ', batch_index)
            print('Average Number of Steps: ', sum(number_of_steps_in_batch)/len(number_of_steps_in_batch))
            print('\n')






