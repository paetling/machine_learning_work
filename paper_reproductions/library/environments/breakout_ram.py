import gym
from .. import GenericOpenAIGymEnv


class BreakoutRam(GenericOpenAIGymEnv):
    def __init__(self, should_render):
        GenericOpenAIGymEnv.__init__(self, should_render)

    def _create_gym_env(self):
        return gym.make('Breakout-ram-v0')
