from .agent import Agent
import tensorflow as tf
from .utils import *
import pdb


class DQN(Agent):
	def __init__(self, env_, network_specs, buffer_size = 10000, batch_size = 128, gamma = 0.95, eps = 0.01, update_target_every_n = 1000, update_every_n = 300, tau = 0.001, logdir = '.', loss_fn='mse'):
		# TODO(shubham): Add option to disable Tensorboard
		# TODO(shubham): Add logging
		# TODO(shubham): ADD L2 REG
		self.env_ = env_
		self.logdir = logdir
		self.loss_fn = loss_fn
		self.network_specs = network_specs
		self.buffer_size = buffer_size
		self.update_every_n = update_every_n
		self.tau = tau
		self.buffer_ = []
		self.buffer_index = None
		self.update_target_every_n = 100
		self.action_space = self.env_.env.action_space
		self.num_actions = self.action_space.n
		self.state_size = self.env_.env.observation_space.shape[0]
		self.moving_reward = None
		self.gamma = gamma
		self.eps = eps
		self.batch_size = batch_size
		self.moving_reward = None

		self.dqn_scope = 'dqn'
		self.target_dqn_scope = 'target_dqn'

		self._add_placeholders()
		
		self.dqn_final_layer = self._add_model(self.dqn_scope, self.state_placeholder, self.network_specs)
		self.target_dqn_final_layer = self._add_model(self.target_dqn_scope, self.state_placeholder, self.network_specs)

		with tf.variable_scope(self.dqn_scope):
			self.q_values = tf.layers.dense(self.dqn_final_layer, self.num_actions, kernel_initializer = tf.contrib.layers.xavier_initializer(), activation=None, name='q_values')
			self.max_q_values = tf.reshape(tf.reduce_max(self.q_values, axis=1, name='max_q_values'), [-1,1])
			self.selected_q_values = tf.reshape(tf.reduce_sum(tf.multiply(self.q_values, self.actions_placeholder, name='selected_q_values'), axis=1), [-1,1])
		with tf.variable_scope(self.target_dqn_scope):
			self.target_q_values = tf.layers.dense(self.target_dqn_final_layer, self.num_actions, kernel_initializer = tf.contrib.layers.xavier_initializer(), activation=None, name='q_values')
			self.target_max_q_values = tf.reshape(tf.reduce_max(self.target_q_values, axis=1, name='max_q_values'), [-1,1])
			self.target_selected_q_values = tf.reshape(tf.reduce_sum(tf.multiply(self.target_q_values, self.actions_placeholder, name='selected_q_values'), axis=1), [-1,1])

		self._add_target_update(self.tau)
		self._add_loss()
		self._add_optim()

	def _add_placeholders(self):
		self.state_placeholder = tf.placeholder(shape=[None, self.state_size], dtype=tf.float32, name='state')
		self.actions_placeholder = tf.placeholder(shape=[None, self.num_actions], dtype=tf.float32, name='actions')
		self.learning_rate = tf.placeholder(dtype=tf.float32, name='lr')
		self.targets = tf.placeholder(shape=[None, 1], dtype=tf.float32, name='targets')

	def _add_loss(self):
		with tf.name_scope("loss_fn"):
			if self.loss_fn == 'huber':
				self.loss = tf.clip_by_value(tf.losses.huber_loss(self.targets, self.selected_q_values), -1, 1)
			else:
				self.loss = tf.clip_by_value(tf.losses.mean_squared_error(self.targets, self.selected_q_values), -1, 1)

	def _add_optim(self):
		self.optim_step = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

	def _add_target_update(self, tau):
		self.update_op_holder = []
		for var, target_var in zip(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.dqn_scope), tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.target_dqn_scope)):
			self.update_op_holder.append(target_var.assign(var.value() * tau + ((1 - tau) * var.value())))

	def _start_session(self):
		self.sess = tf.Session()
		self.summary_op = tf.summary.merge_all()
		self.writer = tf.summary.FileWriter(os.path.join(self.logdir,"tensorboard/dqn/"), self.sess.graph)
		self.writer.close()
		self.sess.run(tf.global_variables_initializer())

	def train(self, env, episodes = 10, lr = 0.01, gamma = 0.95, eps = 0.01):
		self.gamma = gamma
		self.lr = lr
		self.eps = eps

		self._start_session()

		update_steps = 0
		target_update_steps = 0
		for episode in range(episodes):
			done = False
			obs = self.env_.reset()
			step = 0
			reward_sum = 0
			ep_start_time = time()
			while not done:
				step+=1
				update_steps += 1
				target_update_steps += 1
				experience = {}
				action = self.action(obs)
				experience['state'] = obs
				experience['action'] = action
				obs, reward, done, info = self.env_.step(action)
				reward_sum += reward
				experience['reward'] = reward
				experience['next_state'] = obs
				experience['done'] = done
				self.store_experience(experience)
				if len(self.buffer_) < self.batch_size+1:
					continue
				if update_steps == self.update_every_n:
					self.update_net(self.lr)
					update_steps = 0
				if target_update_steps == self.update_target_every_n:
					target_update_steps = 0
					self.update_target()
			if self.moving_reward is None:
				self.moving_reward = reward_sum
			else:
				self.moving_reward = 0.99 * self.moving_reward + 0.01 * reward_sum

			print("Episode:", episode, "Steps:", step, "reward:", self.moving_reward, "lr", self.lr, "Time:", time()-ep_start_time)

	def test():
		pass

	def update_target(self):
		for op in self.update_op_holder:
			self.sess.run(op)

	def store_experience(self, exp):
		if len(self.buffer_)>=self.buffer_size:
			if self.buffer_index is None:
				self.buffer_index = 0
			if self.buffer_index >= self.buffer_size:
				self.buffer_index = 0
			self.buffer_[self.buffer_index] = exp
			self.buffer_index+=1
		else:
			self.buffer_.append(exp)

	def action(self, state):
		if random.uniform(0,1) < self.eps:
			return random.sample(range(self.num_actions), 1)[0]
		q_values = self.sess.run(self.q_values, feed_dict={self.state_placeholder:np.array(state).reshape(1, -1)})
		action = np.argmax(q_values[0])
		return action

	def update_net(self, lr = 0.001):
		sampled_buffer = random.sample(self.buffer_, min(self.batch_size, len(self.buffer_)))
		states = np.array([x['state'] for x in sampled_buffer])
		next_states = np.array([x['next_state'] for x in sampled_buffer])
		rewards = np.array([x['reward'] for x in sampled_buffer]).reshape([-1, 1])
		done_arr = np.array([x['done'] for x in sampled_buffer]).reshape([-1, 1])

		actions = np.zeros([states.shape[0], self.num_actions])
		for i, x in enumerate(sampled_buffer):
			temp_action = x['action']
			actions[i, temp_action] = 1

		target_q_vals = self.sess.run(self.target_q_values, feed_dict={self.state_placeholder:next_states})
		max_q = np.amax(target_q_vals, axis=1).reshape([-1,1])
		targets = rewards + self.gamma * np.multiply((1-done_arr), max_q)
		__, loss_ = self.sess.run([self.optim_step, self.loss], feed_dict={self.state_placeholder: states, self.actions_placeholder:actions, self.targets:targets, self.learning_rate:lr})
