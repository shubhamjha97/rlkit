from .agent import Agent
from .utils import *

class REINFORCE(Agent):
	def __init__(self, env_, network_specs, value_estimator_specs=None, gamma = 0.95):
		self.env_ = env_
		self.network_specs = network_specs
		self.use_baseline = False
		if value_estimator_specs is not None:
			self.value_estimator_specs = value_estimator_specs
			self.use_baseline = True
		self.gamma = gamma
		self.action_space = self.env_.env.action_space
		self.num_actions = self.action_space.n
		self.state_size = self.env_.env.observation_space.shape[0]
		self.moving_reward = None

		self.layers = []

		self._add_placeholders()
		
		self.policy_final_layer = self._add_model('policy_net', self.state_placeholder, network_specs)
		if self.use_baseline:
			self.value_final_layer = self._add_model('value_estimator', self.state_placeholder, value_estimator_specs)
		
		self.action_logits = tf.layers.dense(self.policy_final_layer, self.num_actions, kernel_initializer = tf.contrib.layers.xavier_initializer(), activation=None, name='action_logits')
		self.action_probs = tf.nn.softmax(self.action_logits, axis=1, name='action_probs')
		self.log_likelihood = tf.log(tf.clip_by_value(self.action_probs, 0.000001, 0.999999, name='clip'), name='log_likelihood')
		if self.use_baseline:
			self.state_values = tf.layers.dense(self.value_final_layer, 1, kernel_initializer = tf.contrib.layers.xavier_initializer(), activation=None, name='state_values')

		self._add_loss()
		self._add_optim()

	def _add_placeholders(self):
		self.state_placeholder = tf.placeholder(shape=[None, self.state_size], dtype=tf.float32, name='state')
		self.returns_placeholder = tf.placeholder(shape=[None, 1], dtype=tf.float32, name='returns')
		self.actions_placeholder = tf.placeholder(shape=[None, self.num_actions], dtype=tf.float32, name='actions')
		self.learning_rate = tf.placeholder(dtype=tf.float32, name='lr')

	def _add_loss(self):
		with tf.name_scope("loss_fn"):
			if self.use_baseline:
				self.loss = -tf.reduce_mean(tf.multiply(tf.subtract(self.returns_placeholder, self.state_values), tf.reshape(tf.reduce_sum(tf.multiply(self.log_likelihood, self.actions_placeholder), axis=1), [-1, 1])), axis=0)
			else:
				self.loss = -tf.reduce_mean(tf.multiply(self.returns_placeholder, tf.reshape(tf.reduce_sum(tf.multiply(self.log_likelihood, self.actions_placeholder), axis=1), [-1, 1])), axis=0)

	def _add_optim(self):
		self.optim_step = tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(self.loss)
	
	def _start_session(self):
		self.sess = tf.Session()
		self.summary_op = tf.summary.merge_all()
		self.writer = tf.summary.FileWriter("tensorboard/pg/", self.sess.graph)
		self.writer.close()
		self.sess.run(tf.global_variables_initializer())

	def train(self, episodes = 10, lr = 0.01, gamma = 0.95, update_steps = 10):
		self.gamma = gamma
		self.lr = lr
		all_moving_rewards=[]

		self._start_session()

		for episode in range(episodes):
			done = False
			obs = self.env_.reset()
			step = 0
			ep_start_time = time()
			self.buffer_ = []
			while not done:
				step+=1
				temp = {}
				action = self.action(obs)
				temp['state'] = obs
				temp['action'] = action
				obs, reward, done, info = self.env_.step(action)
				temp['reward'] = reward
				self.buffer_.append(temp)
			if self.moving_reward is None:
				self.moving_reward = float(sum(x['reward'] for x in self.buffer_))
			else:
				self.moving_reward = 0.99 * self.moving_reward + 0.01 * float(sum(x['reward'] for x in self.buffer_))
			all_moving_rewards.append(self.moving_reward)
			print("Episode:", episode, "Steps:", step, "reward:", self.moving_reward, "lr", self.lr, "Time:", time()-ep_start_time)
			self.update_net(self.lr)

	def action(self, state):
		action_probs = self.sess.run(self.action_probs, feed_dict={self.state_placeholder:np.array(state).reshape(1, -1)})
		action = np.random.choice(list(range(self.num_actions)), p=action_probs[0])
		return action

	def update_net(self, lr = 0.001):
		states = np.array([x['state'] for x in self.buffer_])
		rewards = np.array([x['reward'] for x in self.buffer_])

		discounted_r = np.zeros_like(rewards)
		running_add = 0
		for t in reversed(range(0, rewards.size)):
			running_add = running_add * self.gamma + rewards[t]
			discounted_r[t] = running_add
		returns = discounted_r.reshape([-1, 1])

		actions = np.zeros([len(self.buffer_), self.num_actions])
		for i, x in enumerate(self.buffer_):
			temp_action = x['action']
			actions[i, temp_action] = 1

		__, loss_ = self.sess.run([self.optim_step, self.loss], feed_dict={self.state_placeholder: states, self.returns_placeholder:returns, self.actions_placeholder:actions, self.learning_rate:lr})


	def test():
		pass