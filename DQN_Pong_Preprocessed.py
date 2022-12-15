import gym
import numpy as np
from DQN import DQN
import tensorflow as tf
import matplotlib.pyplot as plt

if __name__ == "__main__":
	print("\nSTART MAIN")
	env = gym.make('PongNoFrameskip-v4',render_mode='human')
	env = gym.wrappers.AtariPreprocessing(env, noop_max=30, frame_skip=4, screen_size=84, terminal_on_life_loss=False, grayscale_obs=True)
	env = gym.wrappers.FrameStack(env,4)
	
	agent_name = "PongSolver"
	
	minibatch_size = 32
	replay_memory_size = 100000
	agent_history_length = None
	target_network_update_frequency = 1000
	gamma = 0.95
	action_repeat = None
	learning_rate = 0.01
	gradient_momentum = 0.95
	initial_epsilon = 1
	epsilon_decay = 0.99
	final_epsilon = 0.1
	update_frequency = 4
	replay_start_size = 100
	
	max_episodes = 10000
	max_timesteps = 200
	
	agent = DQN(agent_name,env,'atari',learning_rate,gradient_momentum)
	
	
	agent.train(max_episodes, max_timesteps, replay_memory_size, gamma, initial_epsilon,final_epsilon,epsilon_decay,minibatch_size,update_frequency,target_network_update_frequency,replay_start_size)
	
	print("END MAIN")
