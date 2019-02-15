import gym
from .. import GenericOpenAIGymEnv


class HalfCheetah(GenericOpenAIGymEnv):
    def __init__(self, should_render):
        GenericOpenAIGymEnv.__init__(self, should_render)

    def _create_gym_env(self):
        return gym.make('HalfCheetah-v2')
