import gym
import numpy as np
import tensorflow as tf
import random
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import Input, Conv2D, Activation, Dense, Flatten
from keras.optimizers import Adam


class DQN():
	def __init__(self,env,env_type,memory_capacity):
		#environment parameters
		self.env = env
		self.env_type = env_type
		
		#memory parameters
		self.memory = deque(memory_capacity)
		
		#creating networks
		self.create_Q_networks(env)
		
	def create_Q_networks(self,env):
		model = Sequential()
		
		if self.env_type == 'classic':
			state_shape  = self.env.observation_space.shape
			model.add(Input(shape=(state_shape)))
			model.add(Dense(50, activation="relu"))
			model.add(Dense(self.env.action_space.n))
			model.compile(loss="mean_squared_error",optimizer=Adam(lr=self.learning_rate))
					
		self.Q_network = model #Q
		self.Q_target_network = model #Q^
		
	def get_action(self,state):
		if np.random.random() < epsilon:
			action = self.env.action_space.sample()
		else:
			state = state.reshape(1,self.env.action_space.n)
			action = np.argmax(self.Q_network.predict(state))
		return action
		
	def experience_replay(self, gamma, memory_batch_size):
		if len(self.memory) < memory_batch_size:
			return
		
		sampled_transitions = random.sample(self.memory, memory_batch_size)
		for transition in sampled_transitions:
			state, action, reward, next_state, terminal_state = transition
			state = state.reshape(1, self.env_action_space.n)
			
			Q_target = self.Q_target_network.predict(state)
			if terminal_state:
				Q_target[action] = reward
			else:
				Q_target[action] = reward + gamma*np.max(self.Q_target_network.predict(next_state))
			
			self.Q_network.fit(state, Q_target, epochs=1)
	
	def train(self, max_episodes, max_timesteps, gamma, epsilon, learning_rate, memory_batch_size, network_reset_interval):
		if self.env_type == 'classic':
			for m in range(max_episodes):
				state, info = self.env.reset()
				time_since_network_reset = 0
				for t in range(max_timesteps):
					action = self.get_action(state)
					next_state, reward, terminated, truncated, info = self.env.step(action)
					terminal_state = bool(terminated) + bool(truncated)
					transition = [state, action, reward, next_state, terminal_state]
					self.memory.append(transition)
					
					state = next_state
					
					self.experience_replay(memory_batch_size)
					
					if time_since_network_reset >= network_reset_interval:
						self.Q_target_network = clone_model(self.Q_network)
						time_since_network_reset = 0
