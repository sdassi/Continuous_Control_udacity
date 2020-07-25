import argparse
import gym
import torch
from unityagents import UnityEnvironment
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

from ddpg_agent import Agent

def ddpg(n_episodes, max_t, print_every, threshold, brain_name):
    scores_deque = deque(maxlen=print_every)
    scores = []
    env_info = env.reset(train_mode=True)[brain_name]
    for i_episode in range(1, n_episodes+1):
        num_agents = len(env_info.agents) #number of agents
        states = env_info.vector_observations #[0]
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
            dones = env_info.local_done
            for i_agent in range(num_agents):
                agent.step(states[i_agent], actions[i_agent], rewards[i_agent], next_states[i_agent], dones[i_agent])
            for _ in range(10):
                agent.step_update(t)
            states = next_states
            score_agents += rewards
            if np.any(dones):
                #print("finish with done ", t)
                break 
        scores_deque.append(np.mean(score_agents))
        scores.append(np.mean(score_agents))
        if len(scores_deque) == print_every and np.mean(scores_deque) >= threshold:
            print("environment was solved at episode %d" %(i_episode-print_every))
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            return scores
        if i_episode % 10 == 0: #print_every
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
    print("agent instantiated !!!!!!!!!!!!!!")

    #Train the agent with ddpg
    threshold = 30.0 #The agent must get an average score > threshold to solve the env 
    scores = ddpg(args.n_episodes, args.max_t, args.print_every, threshold, brain_name)

    print(scores)

    #Plot scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()

    #close env
    env.close()
