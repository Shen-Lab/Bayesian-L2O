# author: Yue Cao
# date  2020-5-12
# Bayesian variables of BL2O

import tensorflow as tf
import numpy as np
import l2_bayes_loss

def create_linear_initializer(input_size, mean=0.0, dtype=tf.float32):
  """Returns a default initializer for weights of a linear module."""
  stddev = 1 / np.sqrt(input_size)
  return tf.truncated_normal_initializer(mean=mean, stddev=stddev, dtype=dtype)

def create_bias_initializer(dtype=tf.float32):
  """Returns a default initializer for the biases of a linear/AddBias module."""
  return tf.zeros_initializer(dtype=dtype)



# bayesianize the gated variables of LSTM network in the optimizer.
def lstm_gate_variables(i, hidden_size, name, stage, num_dims):


	input_shape=num_dims*2
	print (input_shape, "input_shape")
	#exit(0)

	with tf.variable_scope(name) as scope:

		if i==1:
			equiv_input_size = input_shape+hidden_size
		else:
			equiv_input_size = hidden_size*2

		print (equiv_input_size, hidden_size, name)

		if stage==1:
			w = tf.get_variable(name="w",dtype=tf.float32, shape=[equiv_input_size, 4 * hidden_size],trainable=True,\
				initializer=create_linear_initializer(equiv_input_size) )
			b = tf.get_variable(name="b",shape=[4 * hidden_size],dtype=tf.float32, trainable=True, \
				initializer=create_linear_initializer(equiv_input_size))

		else:

			mu_w = tf.get_variable(name="w",dtype=tf.float32, shape=[equiv_input_size, 4 * hidden_size],trainable=True,\
				initializer=create_linear_initializer(equiv_input_size) )

			rho_w = tf.get_variable(name="rho_w",shape=[equiv_input_size, 4 * hidden_size],dtype=tf.float32, trainable=True, \
				initializer=create_linear_initializer(equiv_input_size, mean=-2.5))
			
			# epislon=tf.get_variable(name='epislon_w', shape=mu_w.shape, initializer = tf.random_normal_initializer(), trainable=False)
			epislon = tf.random.normal(shape=mu_w.shape)
			# w = mu_w + tf.math.log(1.0+tf.math.exp(rho_w)) * epislon
			w = mu_w + tf.math.softplus(rho_w) * epislon


			mu_b = tf.get_variable(name="b",shape=[4 * hidden_size],dtype=tf.float32, trainable=True, \
				initializer=create_linear_initializer(equiv_input_size))

			rho_b = tf.get_variable(name="rho_b",shape=[4 * hidden_size],dtype=tf.float32, trainable=True,\
				initializer=create_linear_initializer(equiv_input_size, mean=-2.5))
			
			# epislon=tf.get_variable(name='epislon_b', shape=mu_b.shape, initializer = tf.random_normal_initializer(), trainable=False)
			epislon = tf.random.normal(shape=mu_b.shape)
			# b = mu_b + tf.math.log(1.0+tf.math.exp(rho_b)) * epislon
			b = mu_b + tf.math.softplus(rho_b) * epislon
			#print (epislon, 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
			'''
			l2_bayes_loss.loss.loss=[]
			l2_bayes_loss.loss.loss.append(w)
			'''
			l2_bayes_loss.loss.loss_mu = []
			l2_bayes_loss.loss.loss_rho = []
			l2_bayes_loss.loss.loss_mu += [mu_w, mu_b]
			l2_bayes_loss.loss.loss_rho += [rho_w, rho_b]
	return w,b


# bayesianize the linear variables of the linear layer in the optimizer.
def linear_layer_variables(input_size, output_size, name, stage):

	print (output_size)

	with tf.variable_scope(name) as scope:

		if stage==1:
			w = tf.get_variable(name="w",shape=[input_size, output_size],dtype=tf.float32, trainable=True,\
			initializer=create_linear_initializer(input_size))

			b = tf.get_variable(name="b",shape=[output_size],dtype=tf.float32, trainable=True, \
			initializer=create_bias_initializer())
		else:
			
			mu_w = tf.get_variable(name="w",shape=[input_size, output_size],dtype=tf.float32, trainable=True,\
				initializer=create_linear_initializer(input_size))
			rho_w = tf.get_variable(name="rho_w",shape=[input_size, output_size],dtype=tf.float32, trainable=True, \
				initializer=create_linear_initializer(input_size, mean=-2.5))
			# epislon=tf.get_variable(name='epislon_w', shape=mu_w.shape, initializer = tf.random_normal_initializer(), trainable=False)
			epislon = tf.random.normal(shape=mu_w.shape)

			# w = mu_w + tf.math.log(1.0+tf.math.exp(rho_w)) * epislon
			w = mu_w + tf.math.softplus(rho_w) * epislon


			mu_b = tf.get_variable(name="b",shape=[output_size],dtype=tf.float32, trainable=True, \
				initializer=create_bias_initializer())
			rho_b = tf.get_variable(name="rho_b",shape=[output_size],dtype=tf.float32, trainable=True, \
				initializer=create_linear_initializer(input_size, mean=-2))
			# epislon=tf.get_variable(name='epislon_b', shape=mu_b.shape, initializer = tf.random_normal_initializer(), trainable=False)
			epislon = tf.random.normal(shape=mu_b.shape)

			# b = mu_b + tf.math.log(1.0+tf.math.exp(rho_b)) * epislon
			b = mu_b + tf.math.softplus(rho_b) * epislon

			# l2_bayes_loss.loss.loss.append(w)
			l2_bayes_loss.loss.loss_mu += [mu_w, mu_b]
			l2_bayes_loss.loss.loss_rho += [rho_w, rho_b]
	return w,b
