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

        one_hot_actions = tf.one_hot(self.actions_taken, self.action_size)
        self.base_cross_entropy = tf.reduce_sum(-tf.math.multiply(tf.math.log(1 - self.softmax), one_hot_actions), axis=1)
        self.reward_weighted_cross_entropy = tf.math.multiply(self.discounted_episode_rewards, self.base_cross_entropy)

        self.loss = -tf.reduce_sum(self.reward_weighted_cross_entropy)

        self.train = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def train_model(self, number_of_batches, batch_size, max_steps):
        for batch_index in range(number_of_batches):

            states = []
            actions_taken = []
            batch_rewards = []
            batch_steps = []
            for episode_index in range(batch_size):
                batch_rewards.append([])
                batch_steps.append(0)

                state = self.environment.reset()

                for i in range(max_steps):
                    (actions_to_take,) = self.session.run([self.actions_to_take], feed_dict={self.input:[state]})
                    action = actions_to_take[0]
                    state, reward, done = self.environment.step(action)

                    batch_rewards[-1].append(reward)
                    batch_steps[-1] += 1
                    states.append(state)
                    actions_taken.append(action)

                    if done:
                        break


            discounted_rewards = []
            final_rewards = []
            for batch_reward in batch_rewards:
                current_reward = 0

                for i in range(len(batch_reward)):
                    reward = batch_reward[i]
                    current_reward = current_reward * self.discount_factor + reward
                    discounted_rewards.append(current_reward)

                    if i == len(batch_reward) - 1:
                        final_rewards.append(current_reward)

            # print('actions taken', actions_taken)
            bce, loss,train = self.session.run([self.base_cross_entropy, self.loss, self.train], feed_dict={
                                                                            self.input:states,
                                                                            self.actions_taken:actions_taken,
                                                                            self.discounted_episode_rewards: discounted_rewards
                                                                            })
            print('Training Batch: ', batch_index)
            print('Base Cross Entroy: ', bce)
            print('Final Rewards: ', final_rewards)
            print('Loss: ', loss)
            print('Average Number of Steps: ', sum(batch_steps)/len(batch_steps))
            print('\n')






