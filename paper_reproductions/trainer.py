from .reinforce.algorithm_implementation import Reinforce
from .reinforce.environments.cart_pole import CartPole

cart_pole = CartPole(True)
reinforce = Reinforce(cart_pole)

reinforce.train_model(10000, 10, 1000)
