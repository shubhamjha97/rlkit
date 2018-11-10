import tensorflow as tf

class Agent:
	def __init__():
		raise NotImplementedError

	def train():
		raise NotImplementedError

	def test():
		raise NotImplementedError

	def _add_model(self, scope_name='model', input_placeholder = None, network_specs=None):
		activations_map = {
		'linear':None,
		'relu':tf.nn.relu,
		'sigmoid':tf.nn.sigmoid,
		'tanh':tf.nn.tanh
		}
		layers = []
		with tf.variable_scope(scope_name):
			for ix, layer in enumerate(network_specs):
				if layer['type']=='dense':
					if ix==0:
						layer = tf.layers.dense(inputs = input_placeholder, units = layer['size'], activation = activations_map[layer['activation']])
						layers.append(layer)
						if ix == len(network_specs)-1:
							return layer
					elif ix == len(network_specs)-1:
						final_layer = tf.layers.dense(inputs = layers[-1], units = layer['size'], activation = activations_map[layer['activation']])
						return final_layer
					else:
						layer = tf.layers.dense(inputs = layers[-1], units = layer['size'], activation = activations_map[layer['activation']])
						layers.append(layer)