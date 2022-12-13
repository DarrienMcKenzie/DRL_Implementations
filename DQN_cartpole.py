import gym
import numpy as np
from DQN import DQN
import tensorflow as tf
import matplotlib.pyplot as plt

if __name__ == "__main__":
	print("\nSTART MAIN")
	env = gym.make('CartPole-v1')
	
	minibatch_size = 32
	replay_memory_size = 10000
	agent_history_length = None
	target_network_update_frequency = 1000
	gamma = 0.95
	action_repeat = None
	learning_rate = 0.01
	initial_epsilon = 1
	final_epsilon = 0.1
	
	max_episodes = 10000
	max_timesteps = 200

	agent = DQN(env,'classic',learning_rate=0.1)
	
	agent.train(max_episodes, 
	max_timesteps, 
	replay_memory_size,
	gamma, 
	final_epsilon, 
	minibatch_size,
	target_network_update_frequency)
	
	print("END MAIN")
