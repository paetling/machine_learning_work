import tensorflow as tf
from ..library import DenseLayer, create_dense_neural_net

class Reinforce:
    # environment should have a way to get the state, rewards and say if you are done or not
    def __init__(self, environment):
        self.environment = environment

        self.state_size = self.environment.get_state_shape()
        self.action_size = self.environment.get_number_of_actions()

        self.input = tf.placeholder(tf.float32, shape=[None, *self.state_size])
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

        self.actions = tf.distributions.Categorical(probs=self.softmax).sample()

        stacked_discounted_episode_rewards = tf.transpose(tf.stack([self.discounted_episode_rewards for i in range(self.action_size)]))
        print('shapes')
        print(stacked_discounted_episode_rewards.shape)
        print(self.softmax.shape)
        self.element_wise_product = tf.math.multiply(tf.math.log(self.softmax), stacked_discounted_episode_rewards)

        self.loss = -tf.reduce_mean(self.element_wise_product, axis=0)

        self.train = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def train_model(self, number_of_batches, batch_size, max_steps):
        for batch_index in range(number_of_batches):

            states = []
            batch_rewards = []
            batch_steps = []
            for episode_index in range(batch_size):
                batch_rewards.append([])
                batch_steps.append(0)

                state = self.environment.reset()

                for i in range(max_steps):
                    (actions,) = self.session.run([self.actions], feed_dict={self.input:[state]})
                    state, reward, done = self.environment.step(actions[0])

                    batch_rewards[-1].append(reward)
                    batch_steps[-1] += 1
                    states.append(state)

                    if done:
                        break


            discounted_rewards = []
            for batch_reward in batch_rewards:
                current_reward = 0

                for reward in batch_reward:
                    current_reward = current_reward * self.discount_factor + reward
                    discounted_rewards.append(current_reward)

            rewards, element_wise_product,loss,train = self.session.run([self.discounted_episode_rewards, self.element_wise_product,self.loss, self.train], feed_dict={
                                                                            self.input:states,
                                                                            self.discounted_episode_rewards: discounted_rewards
                                                                            })

            print('Training Batch: ', batch_index)
#            print('Element Wise Product: ', element_wise_product)
            print('Discounted Rewards: ', rewards)
            print('Loss: ', loss)
            print('Average Number of Steps: ', sum(batch_steps)/len(batch_steps))
            print('\n')






