# Continuous_Control_udacity
This is the second project of the deep reinforcement learning specialization in udacity 

## Project details

### Problem definition
This project is about training a double-jointed arm to move to target locations. And the goal is to maintain the arm's position at the target location for as many time steps as possible. <br>

For this project, we will provide you with two separate versions of the Unity environment:
- The first version contains a single agent.
- The second version contains 20 identical agents, each with its own copy of the environment.

I've chosen to work with the second version of the environment : **20 agents**

### State and action spaces
The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

### Reward and score
A reward of +0.1 is provided for each step that the agent's hand is in the goal location. 

### Solving the environment
The environment is considered solved when your 20 agents must get an average score of +30 (over 100 consecutive episodes, and over all agents). Specifically,
- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 20 (potentially different) scores. We then take the average of these 20 scores.
- This yields an average score for each episode (where the average is over all 20 agents).

The environment is considered solved, when the average (over 100 episodes) of those average scores is at least +30.

## Getting started
1. You can start by cloning this project `git clone git@github.com:sdassi/navigation_udacity.git`
2. Install all requirements from the requirements file `pip install -r requirements.txt`
3. Download the environment from the link below (select only the one matching your OS):
    * Linux: [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
    * Mac OSX: [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
    * Windows (32-bit): [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
    * Windows (64-bit): [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

4. Place the downloaded file in the location you want (you can place it in this repo for example), then unzip (or decompress the file).

## Instructions
This code allow you to train the agent or evaluate it. Note that the project already contains a pre-trained weights, if you want to skip the training part and try to evaluate a trained agent that's totally feasible.

### Train the agent
You can start training the agent with this command: `python train.py --env_file <path of the Reacher file>` <br>
`train.py` file has many arguments. But only `env_file` argument is required. It's totally okay if you use default values for the remaining arguments (The way to use default values for aguments is simply not specifying them in the execution command). This is the list of all arguments that can be passed to `train.py` :
- `env_file` : string argument, path of the Reacher environment file (example `Reacher_Linux/Reacher.x86`) 
- `n_episodes` : integer argument, maximal number of training episodes, default: 1000
- `max_t` : int argument, maximal number of timesteps during an episode, default: 1000
- `print_every` : int argument, the frequency of printing the average score during training, default: 100

### Run episode with trained agent
To evaluate the trained agent, you can run: `python eval_agent.py --file_name <path of the Reacher env>` <br>
`eval_agent.py` file has only two arguments:
- `env_file` : The same as defined in the previous section
- `n_episodes`: number of episodes to play, default value 1