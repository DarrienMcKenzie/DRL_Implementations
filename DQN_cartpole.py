import gym
import numpy as np
from DQN import DQN
import tensorflow as tf
import matplotlib.pyplot as plt

if __name__ == "__main__":
	print("\nSTART MAIN")
	env = gym.make('CartPole-v1')
	
	agent_name = "CartpoleSolver"
	
	minibatch_size = 16
	replay_memory_size = 100000
	agent_history_length = None 
	target_network_update_frequency = 200
	gamma = 0.95 #discounting
	action_repeat = None
	update_frequency = 10
	
	learning_rate = 0.04
	gradient_momentum = 0
	
	initial_epsilon = 1
	final_epsilon = 0.1
	epsilon_decay = 0.995
	
	max_episodes = 750
	max_timesteps = 300
	replay_start_size = 5000
	

	agent = DQN(agent_name,env,'classic',learning_rate,gradient_momentum)
	
	agent.train(max_episodes, max_timesteps, replay_memory_size, gamma, initial_epsilon,final_epsilon,epsilon_decay,minibatch_size,update_frequency,target_network_update_frequency,replay_start_size)
	
	print("END MAIN")
