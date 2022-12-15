import gym
import numpy as np
from DQN import DQN
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

if __name__ == "__main__":
	print("\nSTART MAIN")
	env = gym.make('PongNoFrameskip-v4',render_mode='human')
	env = gym.wrappers.AtariPreprocessing(env, noop_max=30, frame_skip=4, screen_size=84, terminal_on_life_loss=False, grayscale_obs=True)
	env = gym.wrappers.FrameStack(env,4)
	
	agent_name = "PongSolver"
	
	minibatch_size = 32
	replay_memory_size = 700000
	agent_history_length = None
	target_network_update_frequency = 10000
	gamma = 0.95
	action_repeat = None
	learning_rate = 0.01
	gradient_momentum = 0.95
	initial_epsilon = 1
	epsilon_decay = 0.99
	final_epsilon = 0.1
	update_frequency = 4
	replay_start_size = 50000
	
	max_episodes = 99
	max_timesteps = 200
	
	agent = DQN(agent_name,env,'atari',learning_rate,gradient_momentum)
	
	agent.Q_network = load_model("PongSolver_2/PongSolverModel.h5")
	agent.Q_network.summary()
	agent.Q_network.load_weights("PongSolver_2/PongSolverWeights.h5")
	agent.Q_target_network = load_model("PongSolver_2/PongSolverModel.h5")
	agent.Q_network.load_weights("PongSolver_2/PongSolverWeights.h5")
	
	
	agent.train(max_episodes, max_timesteps, replay_memory_size, gamma, initial_epsilon,final_epsilon,epsilon_decay,minibatch_size,update_frequency,target_network_update_frequency,replay_start_size)
	
	print("END MAIN")
