import sys
import tensorflow as tf
from ..library import DenseLayer, create_dense_neural_net

class Reinforce:
    # environment should have a way to get the state, rewards and say if you are done or not
    def __init__(self, environment, save_location):
        self.environment = environment
        self.save_location = save_location

        self.state_size = self.environment.get_state_shape()
        self.action_size = self.environment.get_number_of_actions()

        self.input = tf.placeholder(tf.float32, shape=[None, *self.state_size])
        self.actions_taken = tf.placeholder(tf.float32, shape=[None, self.action_size])
        self.discounted_episode_rewards = tf.placeholder(tf.float32, shape=[None])

        self.learning_rate = .0001
        self.discount_factor = 0.99

        self.softmax = None
        self.loss = None
        self.train = None
        self._create_learning_network(self.action_size)

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver(tf.global_variables())


    def _create_learning_network(self, output_units):
        self.parameter_neural_nets = []
        self.normal_distributions = []
        self.action_stacks = []
        for action_index in range(self.action_size):
            mu_nn_output = create_dense_neural_net(
                self.input,
                [
                    DenseLayer(100, activation=tf.keras.activations.tanh),
                    DenseLayer(50, activation=tf.keras.activations.tanh),
                    DenseLayer(25, activation=tf.keras.activations.tanh),
                    DenseLayer(1, activation=None),
                ]
            )
            mu_nn_output = tf.squeeze(mu_nn_output, axis=[1])

            sigma_nn_output = create_dense_neural_net(
                self.input,
                [
                    DenseLayer(100, activation=tf.keras.activations.tanh),
                    DenseLayer(50, activation=tf.keras.activations.tanh),
                    DenseLayer(25, activation=tf.keras.activations.tanh),
                    DenseLayer(1, activation=None),
                ]
            )
            sigma_nn_output = tf.squeeze(sigma_nn_output, axis=[1])

            self.parameter_neural_nets.append(tf.stack([mu_nn_output, sigma_nn_output], axis=1))

            normal_distribution = tf.contrib.distributions.Normal(mu_nn_output, sigma_nn_output**2)
            self.normal_distributions.append(normal_distribution)

            self.action_stacks.append(normal_distribution.sample())

        self.actions_to_take = tf.stack(self.action_stacks, axis=1)

        self.pre_stack_log_probabilities = []
        for normal_distribution_index in range(len(self.normal_distributions)):
            normal_distribution = self.normal_distributions[normal_distribution_index]
            self.pre_stack_log_probabilities.append(normal_distribution.log_prob(self.actions_taken[:, normal_distribution_index]))

        self.log_probabilities = tf.stack(self.pre_stack_log_probabilities, axis=1)
        self.sum_of_log_probabilities = tf.reduce_sum(self.log_probabilities, axis=1)

        self.loss = -tf.reduce_sum(self.sum_of_log_probabilities * self.discounted_episode_rewards)
        self.train = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def _discount_rewards(self, non_discounted_rewards):
        """ take 1D float array of rewards and compute discounted reward """
        discounted_rewards = [0.0] * len(non_discounted_rewards)
        total_rewards = 0
        for t in reversed(range(len(non_discounted_rewards))):
            total_rewards = total_rewards * self.discount_factor + non_discounted_rewards[t]
            discounted_rewards[t] = total_rewards
        return discounted_rewards

    def _load_saved_model(self):
        try:
            self.saver.restore(self.session, self.save_location)
            print('successfully loaded old model data')
        except:
            print('do not currently have data stored for this model')

    def _save_model(self):
        self.saver.save(self.session, self.save_location)

    def train_model(self, number_of_batches, batch_size, max_steps):
        self._load_saved_model()

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
                    (actions_to_take,action_stacks, p_nn) = self.session.run([self.actions_to_take, self.action_stacks, self.parameter_neural_nets], feed_dict={self.input:[state]})
                    action = actions_to_take[0]
                    new_state, reward, done = self.environment.step(action)

                    batch_rewards[-1].append(reward)
                    batch_steps[-1] += 1
                    states.append(state)
                    actions_taken.append(action)
                    state = new_state

                    if done:
                        break


            discounted_rewards = []
            final_rewards = []
            for batch_reward in batch_rewards:
                discounted_rewards += self._discount_rewards(batch_reward)

            loss, _ = self.session.run([self.loss, self.train], feed_dict={
                                                                            self.input:states,
                                                                            self.actions_taken:actions_taken,
                                                                            self.discounted_episode_rewards: discounted_rewards})


            if ((batch_index + 1) % 10) == 0:
                self._save_model()
                print('Training Batch: ', batch_index)
                print('Loss: ', loss)
                print('\n')


        self._save_model()


    def run_model(self, number_of_episodes):
        self._load_saved_model()

        for episode_index in range(number_of_episodes):

            state = self.environment.reset()
            done = False
            while not done:
                (actions_to_take,) = self.session.run([self.actions_to_take], feed_dict={self.input:[state]})
                action = actions_to_take[0]

                state, _, done = self.environment.step(action)
