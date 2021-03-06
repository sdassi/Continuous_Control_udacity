import argparse
import gym
import torch
from unityagents import UnityEnvironment
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

from ddpg_agent import Agent

UPDATE_NUM = 10
LEN_DEQUE = 100

def ddpg(n_episodes, max_t, max_len_deque, print_every, threshold, brain_name):
    scores_deque = deque(maxlen=max_len_deque)
    scores = []
    env_info = env.reset(train_mode=True)[brain_name]
    for i_episode in range(1, n_episodes+1):
        num_agents = len(env_info.agents) #number of agents
        states = env_info.vector_observations 
        agent.reset()
        score_agents = np.zeros(num_agents)
        for t in range(max_t):
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
            for _ in range(UPDATE_NUM):
                agent.step_update(t)
            states = next_states
            score_agents += rewards
            if np.any(dones):
                break 
        scores_deque.append(np.mean(score_agents))
        scores.append(np.mean(score_agents))
        if len(scores_deque) == max_len_deque and np.mean(scores_deque) >= threshold:
            print("environment was solved at episode %d" %(i_episode-max_len_deque))
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            return scores
        if i_episode % print_every == 0: 
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
            
    return scores



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_file', '-e', default='', type=str)
    parser.add_argument('--n_episodes', '-n', default=1000, type=int)
    parser.add_argument('--max_t', default=1000, type=int)
    parser.add_argument('--print_every', default=100, type=int)
    args = parser.parse_args()

    #Instantiate the environment
    env = UnityEnvironment(file_name=args.env_file)

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    #Instantiate the agent
    state_size=33
    action_size=4
    agent = Agent(state_size, action_size, 0)

    #Train the agent with ddpg
    threshold = 30.0 #The agent must get an average score > threshold to solve the env 
    scores = ddpg(args.n_episodes, args.max_t, LEN_DEQUE, args.print_every, threshold, brain_name)

    #Plot scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()

    #close env
    env.close()
