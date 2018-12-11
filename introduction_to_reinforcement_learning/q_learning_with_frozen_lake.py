import numpy as np
import gym
import random

env = gym.make('FrozenLake-v0')
action_size = env.action_space.n
state_size = env.observation_space.n
q_table = np.zeros((state_size, action_size))

def train(q_table, env):
    epsilon = 1.0
    max_epsilon = 1.0
    min_epsilon = .005
    decay_rate = .005

    learning_rate = .2
    max_steps = 99
    gamma = .99

    rewards = []

    total_episodes = 15000

    for episode in range(total_episodes):
        state = env.reset()
        step = 0
        done = False
        total_rewards = 0

        for step in range(max_steps):
            exp_exp_random_num = random.uniform(0, 1)

            action_to_take = None
            should_exploit = exp_exp_random_num > epsilon
            if should_exploit:
                action_to_take = np.argmax(q_table[state, :])
            else:
                action_to_take = env.action_space.sample()

            new_state, reward, done, info = env.step(action_to_take)

            q_table[state, action_to_take] = (1 - learning_rate) * q_table[state, action_to_take] + learning_rate * (reward + gamma * np.max(q_table[new_state, :]))

            total_rewards += reward

            if done:
                break

            state = new_state

        epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)

        rewards.append(total_rewards)

    print("Score over time: " +  str(sum(rewards)/total_episodes))
    print(q_table)

train(q_table, env)

def play_frozen_lake(q_table, env):
    max_steps = 99

    for episode in range(5):
        state = env.reset()
        step = 0
        done = False
        print("****************************************************")
        print("EPISODE ", episode)

        for step in range(max_steps):

            # Take the action (index) that have the maximum expected future reward given that state
            action = np.argmax(q_table[state,:])

            new_state, reward, done, info = env.step(action)

            if done:
                # Here, we decide to only print the last state (to see if our agent is on the goal or fall into an hole)
                env.render()

                # We print the number of step it took.
                print("Number of steps", step)
                break
            state = new_state
    env.close()

play_frozen_lake(q_table, env)
