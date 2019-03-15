import sys
import tensorflow as tf
from ..library import DenseLayer, create_dense_neural_net

class A2C:
    def __init__(self, environment, save_location):
        self.environment = environment
        self.save_location = save_location

        self.state_size = self.environment.get_state_shape()
        self.action_size = self.environment.get_number_of_actions()

        self.input = tf.placeholder(tf.float32, shape=[None, *self.state_size])
        self.actions_taken = tf.placeholder(tf.int64, shape=[None])
        self.rewards_for_action = tf.placeholder(tf.float32, shape=[None])
        self.values_of_next_state = tf.placeholder(tf.float32, shape=[None])

        self.policy_learning_rate = .001
        self.value_learning_rate = .001
        self.discount_factor = 0.9999

        self._create_learning_network()

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver(tf.global_variables())

    def _create_learning_network(self):
        self.policy_softmax = create_dense_neural_net(
            self.input,
            [
                DenseLayer(100, activation=tf.keras.activations.tanh),
                DenseLayer(50, activation=tf.keras.activations.tanh),
                DenseLayer(25, activation=tf.keras.activations.tanh),
                DenseLayer(self.action_size, activation=tf.keras.activations.softmax),
            ]
        )

        self.value_output = create_dense_neural_net(
            self.input,
            [
                DenseLayer(100, activation=tf.keras.activations.tanh),
                DenseLayer(50, activation=tf.keras.activations.tanh),
                DenseLayer(25, activation=tf.keras.activations.tanh),
                DenseLayer(1, activation=None),
            ]
        )

        self.actions_to_take = tf.distributions.Categorical(probs=self.policy_softmax).sample()

        self.td_error = self.rewards_for_action + self.discount_factor * self.values_of_next_state - self.value_output  # noqa
        self.advantage = self.td_error
        # transform actions taken into one array where one designates the action to take
        self.one_hot_actions = tf.one_hot(self.actions_taken, self.action_size)
        # get the negative log of the softmax probabilities and then select the one for the action
        # taken as designated by one_hot_actions
        self.base_cross_entropy = -tf.reduce_sum(tf.math.log(self.policy_softmax + 1e-5) * self.one_hot_actions, axis=1)  # noqa
        self.reward_weighted_cross_entropy = self.advantage * self.base_cross_entropy

        self.policy_loss = tf.reduce_sum(self.reward_weighted_cross_entropy)
        self.policy_train = tf.train.AdamOptimizer(self.policy_learning_rate).minimize(self.policy_loss)

        self.value_loss = tf.reduce_sum(tf.square(self.td_error))
        self.value_train = tf.train.AdamOptimizer(self.value_learning_rate).minimize(self.value_loss)

    def _load_saved_model(self):
        try:
            self.saver.restore(self.session, self.save_location)
            print('successfully loaded old model data')
        except:
            print('do not currently have data stored for this model')

    def _save_model(self):
        self.saver.save(self.session, self.save_location)

    def train_model(self, number_of_episodes, max_steps):
        self._load_saved_model()

        loss = None
        batch_steps = []
        for episode_index in range(number_of_episodes):
            state = self.environment.reset()

            for i in range(max_steps):
                (actions_to_take,) = self.session.run([self.actions_to_take], feed_dict={self.input:[state]})
                action = actions_to_take[0]
                new_state, reward, done = self.environment.step(action)

                value_of_next_state = self.session.run(self.value_output, feed_dict={self.input:[new_state]})[0][0]

                loss, advantage, _, _ = self.session.run(
                    [
                        self.policy_loss, self.advantage, self.policy_train, self.value_train,
                    ],
                    feed_dict={
                        self.input: [state],
                        self.actions_taken: [action],
                        self.rewards_for_action: [reward],
                        self.values_of_next_state: [value_of_next_state],
                    },
                )
                print('advantage', advantage, value_of_next_state)

                state = new_state

                if done:
                    break
            batch_steps.append(i + 1)

            if ((episode_index + 1) % 10) == 0:
                self._save_model()
                print('Training Episode: ', episode_index)
                print('Loss: ', loss)
                print('Average Number of Steps: ', sum(batch_steps)/len(batch_steps))
                print('\n')
                batch_steps = []


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
