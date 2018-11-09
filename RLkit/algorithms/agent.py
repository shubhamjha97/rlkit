import tensorflow as tf
import pdb

class Agent:
	def __init__():
		raise NotImplementedError

	def train():
		raise NotImplementedError

	def test():
		raise NotImplementedError

	def _add_model(self, scope_name='model', input_placeholder = None, network_specs=None):
		layers = []
		with tf.name_scope(scope_name):
			for ix, layer in enumerate(network_specs):
				if layer['type']=='dense':
					if ix==0:
						layer = tf.layers.dense(inputs = input_placeholder, units = layer['size'])
						layers.append(layer)
					elif ix == len(network_specs)-1:
						final_layer = tf.layers.dense(inputs = layers[-1], units = layer['size'])
						return final_layer
					else:
						pdb.set_trace()
						layer = tf.layers.dense(inputs = layers[-1], units = layer['size'])
						layers.append(layer)