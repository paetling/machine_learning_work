import sys
import tensorflow as tf
import numpy as np
from ..library import DenseLayer, create_dense_neural_net

try:
    xrange = xrange
except:
    xrange = range

def discount_rewards(r, gamma):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(xrange(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

class Reinforce:
    # environment should have a way to get the state, rewards and say if you are done or not
    def __init__(self, environment):
        self.environment = environment

        self.state_size = self.environment.get_state_shape()
        self.action_size = self.environment.get_number_of_actions()

        self.input = tf.placeholder(tf.float32, shape=[None, *self.state_size])
        self.actions_taken = tf.placeholder(tf.int64, shape=[None])
        self.softmax_calculated = tf.placeholder(tf.float32, shape=[None, self.action_size])
        self.discounted_episode_rewards = tf.placeholder(tf.float32, shape=[None])

        self.learning_rate = .002
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
                DenseLayer(8, activation=tf.keras.activations.relu),
                DenseLayer(output_units, activation=tf.keras.activations.softmax),
            ]
        )

        self.actions_to_take = tf.distributions.Categorical(probs=self.softmax).sample()

        self.one_hot_actions = tf.one_hot(self.actions_taken, self.action_size)
        # self.base_cross_entropy = tf.reduce_sum(-tf.math.multiply(tf.math.log(self.softmax), one_hot_actions), axis=1)
        # self.reward_weighted_cross_entropy = tf.math.multiply(self.discounted_episode_rewards, self.base_cross_entropy)
        self.action_value_pred = tf.reduce_sum(self.softmax * self.one_hot_actions, 1)

        self.loss = -tf.reduce_mean(tf.log(self.action_value_pred + .00001) * self.discounted_episode_rewards)

        self.train = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=tf.contrib.framework.get_global_step())

    def train_model(self, number_of_batches, batch_size, max_steps):
        for batch_index in range(number_of_batches):

            states = []
            actions_taken = []
            batch_rewards = []
            batch_steps = []

            discounted_rewards = []
            for episode_index in range(batch_size):
                batch_rewards.append([])
                batch_steps.append(0)

                state = self.environment.reset()

                for i in range(max_steps):
                    actions_to_take = self.session.run(self.actions_to_take, feed_dict={self.input:[state]})
                    action = actions_to_take[0]
                    print
                    new_state, reward, done = self.environment.step(action)
                    #print ('Step index', i, 'state: ', state, '  action: ', action)

                    batch_rewards[-1].append(reward)
                    batch_steps[-1] += 1
                    states.append(state)
                    actions_taken.append(action)

                    state = new_state

                    if done:
                        discounted_rewards += discount_rewards(np.array(batch_rewards[-1]), self.discount_factor).tolist()

                        output, one_hot, action_value, loss,train = self.session.run([ self.softmax, self.one_hot_actions, self.action_value_pred,self.loss, self.train], feed_dict={
                                                                            self.input: states,
                                                                            self.actions_taken:actions_taken,
                                                                            self.discounted_episode_rewards: discounted_rewards,
                                                                            })
                        break

            # print('batch rewards: ', batch_rewards)
            # for batch_reward in batch_rewards:
            #     current_reward = 0
            #     reversed_discounted_rewards = []
            #     for reward_index in range(len(batch_reward)):
            #         reward = batch_reward[len(batch_reward) - 1 - reward_index]
            #         # print('reward', reward)
            #         current_reward = self.discount_factor * current_reward + reward
            #         reversed_discounted_rewards.append(current_reward)

            #     discounted_rewards += reversed(reversed_discounted_rewards)

            # print([len(x) for x in batch_rewards], discounted_rewards)
            # print('actions taken', actions_taken)
            # print('Actions and values', [x for x in zip(actions_taken, one_hot, output, action_value)])
            if (batch_index + 1) % 100 == 0:
                print('Training Batch: ', batch_index)
                print('Loss: ', loss)
                print('Average Number of Steps: ', sum(batch_steps)/len(batch_steps))
                print('\n')






