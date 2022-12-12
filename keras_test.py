import gym
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

from collections import deque

class DQN:
    def __init__(self, env):
        self.env     = env
        self.memory  = deque(maxlen=2000)
        self.learning_rate = 0.9

        self.model        = self.create_model()
        self.target_model = self.create_model()

    def create_model(self):
        model   = Sequential()
        state_shape  = self.env.observation_space.shape
        model.add(Dense(24, input_dim=state_shape[0], activation="relu"))
        model.add(Dense(48, activation="relu"))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(self.env.action_space.n))
        model.compile(loss="mean_squared_error",
            optimizer=Adam(lr=self.learning_rate))
        return model

    def act(self, state):
        #self.epsilon *= self.epsilon_decay
        #self.epsilon = max(self.epsilon_min, self.epsilon)
        #if np.random.random() < self.epsilon:
        #    return self.env.action_space.sample()
        return np.argmax(self.model.predict(state)[0])


def main():
    env     = gym.make("MountainCar-v0")
    state = env.reset()[0]
    state = state.reshape(1,2)

    dqn_agent = DQN(env=env)

    for trial in range(1000):
    	action = dqn_agent.act(state)
    	new_state, reward, done, _, _ = env.step(action)


if __name__ == "__main__":
    main()
