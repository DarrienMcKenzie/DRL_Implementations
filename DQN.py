import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import Input, Conv2D, Activation, Dense, Flatten
from keras.optimizers import Adam


class DQN():
	def __init__(self,env,env_type):
		self.env = env
		self.env_type = env_type
		self.learning_rate = 0.1
		self.create_model(env)

	
	def create_model(self,env):
		model = Sequential()
		
		if self.env_type == 'classic':
			state_shape  = self.env.observation_space.shape
			model.add(Input(shape=(state_shape)))
			model.add(Dense(50, activation="relu"))
			model.add(Dense(self.env.action_space.n))
			#model.compile(loss="mean_squared_error",
			#		optimizer=Adam(lr=self.learning_rate))
					
			model.summary()
		
		self.model = model
		
	def get_action(self,state):
		print("PREDICTION:")
		print(self.model.predict(state.reshape(1,4)))
		action = np.argmax(self.model.predict(state.reshape(1,4)))
		return action
	
	def preprocess_atari_frame(self):
		pass
	
	def train(self):
		pass
		
