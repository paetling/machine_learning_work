from .reinforce.algorithm_implementation import Reinforce
from .reinforce.environments.cart_pole import CartPole

cart_pole = CartPole(True)
reinforce = Reinforce(cart_pole, './.saved_data/reinforce/cart_pole')

#reinforce.train_model(1000, 5, 5000)
reinforce.run_model(1)
