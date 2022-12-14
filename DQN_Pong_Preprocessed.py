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
	
	agent = DQN(env,'atari',learning_rate=0.1)
	
	
	agent.train(max_episodes, 
	max_timesteps, 
	replay_memory_size,
	gamma, 
	final_epsilon, 
	minibatch_size,
	target_network_update_frequency)
	
	"""
	env.reset()
	for i in range(2):
		obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
		#print("OBS", obs)
		print("NEW OBS:")
		new_obs = tf.convert_to_tensor(np.array(obs))
		print(new_obs)
		
		print(type(obs))
		print(type(np.array(obs)))
		print(len(obs))
		print(len(obs[0]))
		print(len(obs[0][0]))
		print()
		env.render()
		"""
	print("END MAIN")
