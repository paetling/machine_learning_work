from .advantage_actor_critic.discrete_action_implementation import A2C
from .library.environments.cart_pole import CartPole

environment = CartPole(False)

reinforce = A2C(environment, './.saved_data/a2c/cart_pole')

reinforce.train_model(100000, 5000)
# reinforce.run_model(1)
