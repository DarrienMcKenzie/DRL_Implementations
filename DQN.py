import gym
import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import cv2
from collections import deque
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import Input, Conv2D, Activation, Dense, Flatten
from keras.optimizers import RMSprop


class DQN():
	def __init__(self,env,env_type,learning_rate):
		#environment parameters
		self.env = env
		self.env_type = env_type
		
		#creating networks
		self.create_Q_networks(env,learning_rate)
		
	def create_Q_networks(self,env,lr):
		model = Sequential()
		
		if self.env_type == 'classic':
			state_shape  = self.env.observation_space.shape
			model.add(Input(shape=(state_shape)))
			model.add(Dense(24, activation="relu"))
			model.add(Dense(48, activation="relu"))
			model.add(Dense(24, activation="relu"))
			model.add(Dense(self.env.action_space.n))
			model.compile(loss="mean_squared_error",optimizer=RMSprop(learning_rate=lr))
			
		elif self.env_type == 'atari':
			model.add(Input(shape=(84,84,4)))
			model.add(Conv2D(filters=32, kernel_size=(8,8),padding="same", strides=4, activation='relu'))
			model.add(Conv2D(filters=64, kernel_size=(4,4), strides=2, activation='relu'))
			model.add(Conv2D(filters=64, kernel_size=(3,3), strides=1, activation='relu'))
			model.add(Flatten())
			model.add(Dense(512, activation='relu'))
			model.add(Dense(self.env.action_space.n))
			
		model.compile(loss="mean_squared_error",optimizer=RMSprop(learning_rate=lr))
		model.summary()
			
		self.Q_network = model #Q
		self.Q_target_network = model #Q^
		
	def get_action(self,state,epsilon):
		if np.random.random() < epsilon:
			action = self.env.action_space.sample()
		else:
			#state = state.reshape(1,len(state))
			#state = np.array(state)
			#state = tf.convert_to_tensor(state)
			#state = tf.reshape(state,(84,84,4))
			#state = tf.expand_dims(state, axis=0)
			
			#state = state.reshape(4,84,84)
			print("ARG:")
			print(state)
			#print(state)
			print("ARG SHAPE:")
			print(state.shape)
			
			prediction = self.Q_network.predict(state,verbose=0)
			action = np.argmax(prediction)
			print("ACTION SPACE: ", self.env.action_space)
			print("PREDICTION: ", prediction)
			print("ACTION: ", action)
		return action
		
	def experience_replay(self, memory,minibatch_size,gamma):
		if len(memory) < minibatch_size:
			return
		
		sampled_transitions = random.sample(memory, minibatch_size)
		for transition in sampled_transitions:
			state, action, reward, next_state, terminal_state = transition
			#state = state.reshape(1, len(state)) #FORMATTING STATE FOR CLASSIC
			#next_state = state.reshape(1, len(next_state)) #FORMATTING STATE FOR CLASSIC
			
			Q_target = self.Q_target_network.predict(state,verbose=0)
	
			if terminal_state:
				Q_target[0][action] = reward
			else:
				Q_target[0][action] = reward + gamma*np.max(self.Q_target_network.predict(next_state,verbose=0))
			
			self.Q_network.fit(state, Q_target, epochs=1,verbose=0)
			
	def preprocess_frame(self,frame,new_width_dim,new_height_dim,new_frame_size):
		new_frame = frame[new_width_dim[0]:new_width_dim[1], new_height_dim[0]:new_height_dim[1]] #crop image
		new_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY) #convert image to greyscale
		new_frame = cv2.resize(new_frame, (new_frame_size)) #rescale (downsize) image
		new_frame = new_frame.reshape(new_frame_size[0], new_frame_size[1]) / 255 #normalizing frame data
		
		return new_frame
		
	def format_state(self, state):
		if self.env_type == 'classic':
			formatted_state = state.reshape(1, len(state))
			
		elif self.env_type == 'atari':
			formatted_state = tf.convert_to_tensor(state)
			formatted_state = tf.reshape(formatted_state,(84,84,4))
			formatted_state = tf.expand_dims(formatted_state, axis=0)
			formatted_state = formatted_state / 255
		return formatted_state
	
	def train(self, max_episodes, max_timesteps, replay_memory_size, gamma, epsilon,minibatch_size,target_network_update_frequency):
		#print("SELF: ", self)
		#print("MAX EPISODES: ", max_episodes)
		#print("MAX TIMESTEPS: ", max_timesteps)
		#print("REPLAY MEMORY SIZE: ", replay_memory_size)
		#print("GAMMA: ", gamma)
		memory = deque(maxlen=replay_memory_size)
		
		if self.env_type == 'classic':
			print("TRAINING...")
			episode_rewards = []
			average_episode_rewards = []
			for m in range(max_episodes):
				episode_reward = 0
				print()
				print("EPISODE #" + str(m))
				state, info = self.env.reset()
				time_since_network_reset = 0
				for t in range(max_timesteps):
					print("T = ", t)
					action = self.get_action(state,epsilon)
					next_state, reward, terminated, truncated, info = self.env.step(action)
					episode_reward += reward
					terminal_state = bool(terminated) + bool(truncated)
					transition = [state, action, reward, next_state, terminal_state]
					memory.append(transition)
					
					state = next_state
					
					self.experience_replay(memory,minibatch_size,gamma)
					
					if time_since_network_reset >= target_network_update_frequency:
						self.Q_target_network = clone_model(self.Q_network)
						time_since_network_reset = 0
					
					if terminated or truncated:
						print("EPISODE TERMINATED. TOTAL REWARD = " + str(episode_reward))
						episode_rewards.append(episode_reward)
						average_episode_rewards.append(np.mean(np.array(episode_reward)))
						break			
						
		elif self.env_type == 'atari':
			print("TRAINING...")
			episode_rewards = []
			average_episode_rewards = []
			for m in range(max_episodes):
				episode_reward = 0
				print()
				print("EPISODE #" + str(m))
				state, info = self.env.reset()
				state = self.format_state(state)
				time_since_network_reset = 0
				for t in range(max_timesteps):
					print("T = ", t)
					action = self.get_action(state,epsilon)
					next_state, reward, terminated, truncated, info = self.env.step(action)
					next_state = self.format_state(next_state)
					episode_reward += reward
					terminal_state = bool(terminated) + bool(truncated)
					transition = [state, action, reward, next_state, terminal_state]
					memory.append(transition)
					
					state = next_state
					
					self.experience_replay(memory,minibatch_size,gamma)
					
					if time_since_network_reset >= target_network_update_frequency:
						self.Q_target_network = clone_model(self.Q_network)
						time_since_network_reset = 0
					
					if terminated or truncated:
						print("EPISODE TERMINATED. TOTAL REWARD = " + str(episode_reward))
						episode_rewards.append(episode_reward)
						average_episode_rewards.append(np.mean(np.array(episode_reward)))
						break
		
		print("TRAINING COMPLETED")
		
		x = np.arange(0,max_episodes)
		y = np.array(average_episode_rewards)
		plt.title("Average Episode Reward vs. Episodes")
		plt.xlabel("Episodes")
		plt.ylabel("Average Reward")
		plt.plot(x,y)
		plt.savefig('results.png')
