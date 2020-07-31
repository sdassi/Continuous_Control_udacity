import argparse
import gym
import torch
from unityagents import UnityEnvironment
import numpy as np
import matplotlib.pyplot as plt

from ddpg_agent import Agent

def run_episode(env, brain_name, max_t=1000):
    env_info = env.reset(train_mode=False)[brain_name]
    num_agents = len(env_info.agents)
    states = env_info.vector_observations
    agent.reset()
    score_agents = np.zeros(num_agents)
    for _ in range(max_t):
        actions = []
        for i_agent in range(num_agents):
            agent.reset()
            actions.append(agent.act(states[i_agent]))
        env_info = env.step(actions)[brain_name]
        next_states = env_info.vector_observations
        rewards = env_info.rewards
        rewards = [0.1 if rew > 0 else 0 for rew in rewards]
        dones = env_info.local_done
        for i_agent in range(num_agents):
            agent.step(states[i_agent], actions[i_agent], rewards[i_agent], next_states[i_agent], dones[i_agent])
        states = next_states
        score_agents += rewards
        if np.any(dones):
            break 
    return np.mean(score_agents)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_file', '-e', default='', type=str)
    parser.add_argument('--n_episodes', '-n', default=1, type=int)
    args = parser.parse_args()

    #Instantiate the environment
    env = UnityEnvironment(file_name=args.env_file)

    # get the default brain
    brain_name = env.brain_names[0]

    #Instantiate the agent
    state_size=33
    action_size=4
    agent = Agent(state_size, action_size, 0)
    agent.actor_local.load_state_dict(torch.load('checkpoint_actor.pth'))
    agent.critic_local.load_state_dict(torch.load('checkpoint_critic.pth'))

    scores = []
    for i in range(args.n_episodes):
        score = run_episode(env, brain_name)
        scores.append(score)
    print("Average scores :  ", np.mean(scores))