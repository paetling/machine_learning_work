import sys
import statistics
import tensorflow as tf
from ..library import DenseLayer, create_dense_neural_net

class Reinforce:
    # environment should have a way to get the state, rewards and say if you are done or not
    def __init__(self, environment):
        self.environment = environment

        self.state_size = self.environment.get_state_shape()
        self.action_size = self.environment.get_number_of_actions()

        self.input = tf.placeholder(tf.float32, shape=[None, *self.state_size])
        self.actions_taken = tf.placeholder(tf.int64, shape=[None])
        self.discounted_episode_rewards = tf.placeholder(tf.float32, shape=[None])

        self.learning_rate = 2.5e-4
        self.discount_factor = 0.99

        self.softmax = None
        self.loss = None
        self.train = None
        self._create_learning_network(self.action_size)

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())


    def _create_learning_network(self, output_units):

        self.softmax = create_dense_neural_net(
            self.input,
            [
                DenseLayer(100, activation=tf.keras.activations.tanh),
                DenseLayer(50, activation=tf.keras.activations.tanh),
                DenseLayer(25, activation=None),
                DenseLayer(output_units, activation=tf.keras.activations.softmax),
            ]
        )

        self.actions_to_take = tf.distributions.Categorical(probs=self.softmax).sample()

        self.one_hot_actions = tf.one_hot(self.actions_taken, self.action_size)
        self.base_cross_entropy = tf.reduce_sum(-tf.math.multiply(tf.math.log(self.softmax), self.one_hot_actions), axis=1)
        # self.reward_weighted_cross_entropy = tf.math.multiply(self.discounted_episode_rewards, self.base_cross_entropy)

        self.loss = tf.reduce_sum(self.base_cross_entropy)

        self.train = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def train_model(self, number_of_batches, batch_size, max_steps):
        episode_length_threshold = 25

        for batch_index in range(number_of_batches):

            batch_states = []
            batch_actions_taken = []
            batch_rewards = []
            batch_steps = []
            for episode_index in range(batch_size):
                batch_rewards.append([])
                batch_states.append([])
                batch_actions_taken.append([])
                batch_steps.append(0)

                state = self.environment.reset()

                for i in range(max_steps):
                    (actions_to_take,) = self.session.run([self.actions_to_take], feed_dict={self.input:[state]})
                    action = actions_to_take[0]
                    state, reward, done = self.environment.step(action)

                    batch_rewards[-1].append(reward)
                    batch_steps[-1] += 1
                    batch_states[-1].append(state)
                    batch_actions_taken[-1].append(action)

                    if done:
                        break

            average_episode_length = statistics.mean([len(batch_reward) for batch_reward in batch_rewards])
            discounted_rewards = []
            above_threshold_actions_taken = []
            states = []
            actions_taken = []
            for batch_reward_index in range(len(batch_rewards)):
                batch_reward = batch_rewards[batch_reward_index]
                current_reward = 0

                if len(batch_reward) > episode_length_threshold:
                    above_threshold_actions_taken.append(len(batch_reward))

                    for episode_index in range(len(batch_reward)):
                        reward = batch_reward[episode_index]
                        states.append(batch_states[batch_reward_index][episode_index])
                        actions_taken.append(batch_actions_taken[batch_reward_index][episode_index])
                        current_reward = current_reward * self.discount_factor + reward
                        discounted_rewards.append(current_reward)

            # print('actions taken', actions_taken)
            if len(above_threshold_actions_taken) > 0:
                #episode_length_threshold = (5 * episode_length_threshold + statistics.mean(above_threshold_actions_taken)) / 6
                bce, sm, one_a, loss,train = self.session.run([self.base_cross_entropy, self.softmax, self.one_hot_actions,self.loss, self.train], feed_dict={
                                                                                self.input:states,
                                                                                self.actions_taken:actions_taken,
                                                                                self.discounted_episode_rewards: discounted_rewards
                                                                                })
                print('Entropy input: ', sm, one_a)
                print('Training Batch: ', batch_index)
                print('Base Cross Entroy: ', bce)
                print('Average Actions Taken: ', average_episode_length,'   Threshold: ', episode_length_threshold, '  Training Set Actions Taken: ', above_threshold_actions_taken)
                print('Loss: ', loss)
                print('Average Number of Steps: ', sum(batch_steps)/len(batch_steps))
                print('\n')






