import gym

class Environment:
	def __init__(self, env_name, render=False):
		self.env = gym.make(env_name)
		self.render = render
		self.timestep = 0
		self.done = False
		self.reset()

	def reset(self):
		observation = self.env.reset()
		return observation

	def step(self, action):
		if self.render:
			self.env.render()
		observation, reward, done, info = self.env.step(action)
		return observation, reward, done, info

	def close(self):
		self.env.close()