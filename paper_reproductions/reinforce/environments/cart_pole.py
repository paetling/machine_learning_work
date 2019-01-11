import gym

class CartPole:
    def __init__(self, should_render):
        self.should_render = should_render
        self.env = gym.make('CartPole-v1')

    def get_number_of_actions(self):
        return self.env.action_space.n

    def get_random_action(self):
        return self.env.action_space.sample

    def get_state_shape(self):
        return self.env.observation_space.shape

    def reset(self):
        return self.env.reset()

    def step(self, action):
        if self.should_render:
            self.env.render()

        state, reward, done, _ = self.env.step(action)
        return state, reward, done

