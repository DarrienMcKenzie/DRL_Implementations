import gym
import numpy as np
from DQN import DQN
import tensorflow as tf

if __name__ == "__main__":
	print("\nSTART MAIN")
	env = gym.make('CartPole-v1')
	policy = DQN(env,'classic')
	env.reset()
	state, reward, terminated, truncated, info = env.step(1)
	
	#print(env.action_space.n)
	#obs_attribrute_count = 4
	#print("STATE = ", state)

	print("ACTION = ", policy.get_action(np.array(state)))
	
	
	
	print("END MAIN")
