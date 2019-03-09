from .dqn.implementation import DQN
from .library.environments.breakout_ram import BreakoutRam

environment = BreakoutRam(True)

training_sessions = 5000
reinforce = DQN(environment, './.saved_data/dqn/breakout_ram', 10000, training_sessions)

# reinforce.train_model(training_sessions + 2000, 10000, 32)
reinforce.run_model(1)
