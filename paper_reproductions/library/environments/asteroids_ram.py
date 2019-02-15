import gym
from .. import GenericOpenAIGymEnv


class AsteroidsRam(GenericOpenAIGymEnv):
    def __init__(self, should_render):
        GenericOpenAIGymEnv.__init__(self, should_render)

    def _create_gym_env(self):
        return gym.make('Asteroids-ram-v0')
