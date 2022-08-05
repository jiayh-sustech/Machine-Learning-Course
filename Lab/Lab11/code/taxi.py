"""
Reference:
(1) https://www.gymlibrary.ml/environments/toy_text/taxi/
(2) https://www.gymlibrary.ml/content/api/
"""

import gym
import pickle
import os
import numpy as np

# configuration
total_episodes = 10000
total_test_episodes = 100
learning_rate = 0.7
max_steps = 99
default_store_path = os.path.join(os.path.split(__file__)[0], "data", "Qtable.pkl")

# discount rate
gamma = 0.618

env: gym.Env = None
Q_table: np.ndarray = None


def init():
    global Q_table
    global env
    env = gym.make("Taxi-v3", new_step_api=True)
    action_space_n = env.action_space.n
    state_space_n = env.observation_space.n
    print("Number of states:", action_space_n)  # there are 500 states
    print("Number of actions:", state_space_n)  # there are 6 actions: 4 directions, pick up and drop off passenger

    Q_table = np.zeros((state_space_n, action_space_n))  # (500, 6)


def select_action(state):
    global Q_table
    # TODO: Select an action who can lead to a maximum Q value
    action = 0

    return action


def train():
    sample_rewards = []
    print("--> Begin training:")
    for episode in range(total_episodes):
        state = env.reset()  # init an env and return initial state
        sample_reward = 0
        while True:
            # select action
            action = select_action(state)

            # taking the action and then get next state
            new_state, reward, done, _, _ = env.step(action)

            # Calculate the reward of this episode
            sample_reward += reward

            # TODO: Update the Q table
            Q_table[state, action] = 0

            # Update the state
            state = new_state

            # store the episode reward
            if done:
                sample_rewards.append(sample_reward)
                break

        # print the average reward over 1000 episodes
        if episode % 1000 == 0:
            mean_reward = np.mean(sample_rewards)
            sample_rewards = []
            print("average reward:" + str(episode) + ": " + str(mean_reward))


def valid():
    rewards = []
    for episode in range(total_test_episodes):
        state = env.reset()
        total_rewards = 0

        for step in range(max_steps):
            # select action
            action = select_action(state)

            new_state, reward, done, _, _ = env.step(action)

            total_rewards += reward
            if done:
                rewards.append(total_rewards)
                break
            state = new_state
    print("--> Average reward:", np.mean(rewards))


def training_step(save_file=None):
    init()
    train()
    valid()

    # save Q_table
    if save_file is None:
        save_file = default_store_path
    with open(save_file, "wb") as f:
        pickle.dump(Q_table, f)


def display(qtable_path=None):
    global Q_table

    # load Q_table
    if qtable_path is None:
        qtable_path = default_store_path
    with open(qtable_path, "rb") as f:
        Q_table = pickle.load(f)

    env = gym.make("Taxi-v3", render_mode="human", new_step_api=True)
    state = env.reset()
    total_reward = 0
    for step in range(max_steps):
        env.render()

        # select action
        action = select_action(state)

        new_state, reward, done, _, _ = env.step(action)
        total_reward += reward

        if done:
            break
        state = new_state

    print("Reward:", total_reward)


if __name__ == "__main__":
    # firstly training, then Q_table will be stored in ./data
    training_step()

    # then display, you can comment training_step() if you have trained Q_table
    # display()
