from .reinforce.continuous_action_algorithm_implementation import Reinforce
from .reinforce.environments.pendulum import Pendulum

pendulum = Pendulum(True)
reinforce = Reinforce(pendulum, './.saved_data/reinforce/pendulum')

reinforce.train_model(1000, 5, 5000)
#reinforce.run_model(1)
