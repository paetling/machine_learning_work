import gym

class CartPole:
    def __init__(self, should_render):
        self.should_render = should_render
        gym.envs.registration.register(
            id='CartPole-v2',
            entry_point='gym.envs.classic_control:CartPoleEnv',
            tags={'wrapper_config.TimeLimit.max_episode_steps': 5000},
                reward_threshold=4750.0,
        )
        self.env = gym.make('CartPole-v2')

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

