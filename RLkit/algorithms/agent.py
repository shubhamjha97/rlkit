import tensorflow as tf

class Agent:
	def __init__():
		raise NotImplementedError

	def train():
		raise NotImplementedError

	def test():
		raise NotImplementedError

	def _add_model(self, scope_name = 'model'):
		print(scope_name)
		with tf.name_scope(scope_name):
			for ix, layer in enumerate(self.network_specs):
				if layer['type']=='dense':
					current_layer = tf.layers.dense(inputs = self.layers[-1], units = layer['size'], name = "dense_{}".format(ix))
				elif layer["type"]=="conv":
					pdb.set_trace()
				self.layers.append(current_layer)