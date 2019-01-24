import gym
from ...library import GenericOpenAIGymEnv

class CartPole(GenericOpenAIGymEnv):
    def __init__(self, should_render):
        GenericOpenAIGymEnv.__init__(self, should_render)

    def _create_gym_env(self):
        gym.envs.registration.register(
            id='CartPole-v2',
            entry_point='gym.envs.classic_control:CartPoleEnv',
            tags={'wrapper_config.TimeLimit.max_episode_steps': 5000},
                reward_threshold=4750.0,
        )
        return gym.make('CartPole-v2')

