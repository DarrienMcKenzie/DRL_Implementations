import gym
env = gym.make('Pong-v4', render_mode='human')
#env = gym.make('CartPole-v1')
env.reset()
done = False

print(env.observation_space)

while not done:
	obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
	env.render()
	#print(len(obs))
	#print(obs)
	#print(len(obs[0]))
env.close()
