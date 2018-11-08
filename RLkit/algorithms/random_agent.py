class RandomAgent:
	def __init__(self, env_):
		self.env_ = env_
		self.action_space = self.env_.env.action_space
		self.num_actions = self.action_space.n

	def train(self, env):
		pass

	def action(self, state = None):
		return self.action_space.sample()