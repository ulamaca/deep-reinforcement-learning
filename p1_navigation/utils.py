import random
import numpy as np
from collections import deque
import time
import torch
import os

def dqn(agent, env,
        n_episodes=2000,
        max_t=1000,
        eps_start=1.0, eps_end=0.01, eps_decay=0.995, max_score=13.0, save_dir=""):
    """Deep Q-Learning.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
        save_dir (str): directory to save the trained data
    """
    scores = []  # list containing scores from each episode
    brain_name = env.brain_names[0] # this is specific for the Unity Environment
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start  # initialize epsilon

    t0 = time.time()
    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
        state = env_info.vector_observations[0]  # get the initial state
        score = 0  # initialize the score

        for t in range(max_t):
            action = agent.act(state, eps)
            env_info = env.step(action)[brain_name]  # send the action to the environment
            next_state = env_info.vector_observations[0]  # get the next state
            reward = env_info.rewards[0]  # get the reward
            done = env_info.local_done[0]  # see if episode has finished
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window) >= max_score:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,
                                                                                         np.mean(scores_window)))
            # save the trained agent in a specified dir, if not specified, save it in the root dir
            if save_dir:
                torch.save(agent.qnetwork_local.state_dict(), os.path.join(save_dir, 'checkpoint.pth'))
            else:
                save_dir = os.path.join("./data", "no-name-exp")
                os.makedirs(save_dir, exist_ok=True)
                torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            break
    t1 = time.time()
    print("Total time elapsed: {} seconds".format(t1-t0))
    return scores


def random_color(choice=False):
    "select the color code from "
    lib=['r', 'b', 'g', 'k', 'y', 'c', 'm']
    if isinstance(choice, int):
        if choice<=6:
            return lib[choice]
        else:
            raise ValueError("choice value should be less than/equal to 6")
    else:
        return random.choice(lib)


def play(env, agent, params_path='data/dddqn-1/checkpoint.pth'):

    agent.qnetwork_local.load_state_dict(torch.load(params_path))
    brain_name = env.brain_names[0]  # this is specific for the Unity Environment
    env_info = env.reset(train_mode=False)[brain_name]  # reset the environment
    state = env_info.vector_observations[0]  # get the initial state
    for _ in range(300):
        action = agent.act(state)
        env_info = env.step(action)[brain_name]  # send the action to the environment
        state = env_info.vector_observations[0]  # get the next state
        done = env_info.local_done[0]  # see if episode has finished
        if done:
            break

def get_env_spec(env):
    """
    get and print necessary specifications from a Unity environment
    :param env: a unity environment instance
    :return: a dict with state and action size of the env
    """
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]
    action_size = brain.vector_action_space_size
    state = env_info.vector_observations[0]
    state_size = len(state)
    return {"state_size":state_size,
            "action_size":action_size}