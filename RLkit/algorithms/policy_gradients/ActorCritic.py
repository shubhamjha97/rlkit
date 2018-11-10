from ..agent import Agent
from ..utils import *

class ActorCritic(Agent):
	def __init__(self, env_, actor_specs, critic_specs=None, gamma = 0.95, logdir = '.', inertia = 0.99):
		self.env_ 			= env_
		self.inertia 		= inertia
		self.actor_specs 	= actor_specs
		self.critic_specs 	= critic_specs
		self.logdir 		= logdir
		self.gamma 			= gamma
		self.action_space 	= self.env_.env.action_space
		self.num_actions 	= self.action_space.n
		self.state_size 	= self.env_.env.observation_space.shape[0]
		self.moving_reward 	= None

		self._add_placeholders()
		
		# Add models
		self.policy_final_layer 	= self._add_model('actor', self.state_placeholder, actor_specs)
		self.value_final_layer 		= self._add_model('critic', self.state_placeholder, critic_specs)
		self.state_values 			= tf.layers.dense(self.value_final_layer, 1, kernel_initializer = tf.contrib.layers.xavier_initializer(), activation=None, name='state_values')
		
		self.action_logits 			= tf.layers.dense(self.policy_final_layer, self.num_actions, kernel_initializer = tf.contrib.layers.xavier_initializer(), activation=None, name='action_logits')
		self.action_probs 			= tf.nn.softmax(self.action_logits, axis=1, name='action_probs')
		self.log_likelihood 		= tf.log(tf.clip_by_value(self.action_probs, 0.000001, 0.999999, name='clip'), name='log_likelihood')

		self._add_loss()
		self._add_optim()

	def _add_placeholders(self):
		self.state_placeholder 		= tf.placeholder(shape=[None, self.state_size], dtype=tf.float32, name='state')
		self.returns_placeholder 	= tf.placeholder(shape=[None, 1], dtype=tf.float32, name='returns')
		self.actions_placeholder 	= tf.placeholder(shape=[None, self.num_actions], dtype=tf.float32, name='actions')
		self.target_state_val 		= tf.placeholder(shape=[None, 1], dtype=tf.float32, name='target_state_val')
		# Learning rates
		self.actor_learning_rate 	= tf.placeholder(dtype=tf.float32, name='actor_lr')
		self.critic_learning_rate 	= tf.placeholder(dtype=tf.float32, name='critic_lr')

	def _add_loss(self):
		with tf.name_scope("loss_fn"):
			self.actor_loss 	= -tf.reduce_mean(tf.multiply(tf.subtract(self.returns_placeholder, self.state_values), tf.reshape(tf.reduce_sum(tf.multiply(self.log_likelihood, self.actions_placeholder), axis=1), [-1, 1])), axis=0)
			self.critic_loss 	= tf.losses.mean_squared_error(self.target_state_val, self.state_values)

	def _add_optim(self):
		self.actor_optim_step 	= tf.train.AdamOptimizer(learning_rate = self.actor_learning_rate).minimize(self.actor_loss)
		self.critic_optim_step 	= tf.train.AdamOptimizer(learning_rate = self.critic_learning_rate).minimize(self.critic_loss)
	
	def _start_session(self):
		self.sess = tf.Session()
		self.summary_op = tf.summary.merge_all()
		self.writer = tf.summary.FileWriter(os.path.join(self.logdir, "tensorboard/AC/"), self.sess.graph)
		self.writer.close()
		self.sess.run(tf.global_variables_initializer())

	def train(self, episodes = 10, actor_lr = 0.01, critic_lr = 0.1, gamma = 0.95, update_steps = 10):
		self.gamma = gamma
		all_moving_rewards = []

		self._start_session()

		for episode in range(episodes):
			done = False
			obs = self.env_.reset()
			step = 0
			ep_start_time = time()
			self.buffer_ = []
			while not done:
				step+=1
				experience = {}
				action = self.action(obs)
				experience['state'] 		= obs
				experience['action'] 		= action
				obs, reward, done, info 	= self.env_.step(action)
				experience['reward'] 		= reward
				experience['next_state']	= obs
				self.buffer_.append(experience)
			if self.moving_reward is None:
				self.moving_reward = float(sum(x['reward'] for x in self.buffer_))
			else:
				self.moving_reward = self.inertia * self.moving_reward + (1-self.inertia) * float(sum(x['reward'] for x in self.buffer_))
			all_moving_rewards.append(self.moving_reward)
			print("Episode:{}\t Steps:{}\t Reward:{}\t Time:{}".format(episode, step, self.moving_reward, time()-ep_start_time))
			self.update_net(actor_lr, critic_lr)

	def action(self, state):
		action_probs 	= self.sess.run(self.action_probs, feed_dict={self.state_placeholder:np.array(state).reshape(1, -1)})
		action 			= np.random.choice(list(range(self.num_actions)), p=action_probs[0])
		return action

	def update_net(self, actor_lr, critic_lr):
		states 		= np.array([x['state'] for x in self.buffer_])
		rewards 	= np.array([x['reward'] for x in self.buffer_])
		next_states = np.array([x['next_state'] for x in self.buffer_])

		next_state_val = self.sess.run(self.state_values, feed_dict={self.state_placeholder:next_states})
		# temp = self.sess.run(self.state_values, feed_dict={self.state_placeholder:states})

		target_state_val = rewards.reshape([-1,1]) + self.gamma*next_state_val

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

		a_, c_, actor_loss_, critic_loss_ = self.sess.run([self.actor_optim_step, self.critic_optim_step, self.actor_loss,
			self.critic_loss], feed_dict={self.state_placeholder: states, 
			self.returns_placeholder:returns, self.actions_placeholder:actions, self.actor_learning_rate:actor_lr,
			self.critic_learning_rate:critic_lr, self.target_state_val:target_state_val})

	def test():
		pass