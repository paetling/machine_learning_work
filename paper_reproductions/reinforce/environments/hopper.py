import gym
from ...library import GenericOpenAIGymEnv

class Hopper(GenericOpenAIGymEnv):
    def __init__(self, should_render):
        GenericOpenAIGymEnv.__init__(self, should_render)

    def _create_gym_env(self):
        return gym.make('Hopper-v2')


