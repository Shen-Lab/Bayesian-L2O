# Copyright 2016 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Learning to learn (meta) optimizer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import contextlib
import os
import pdb
import pickle

import mock
import sonnet as snt
import tensorflow as tf

from tensorflow.python.framework import ops
from tensorflow.python.util import nest

import networks
import pdb

import l2_bayes_loss


def _nested_assign(ref, value):
  """Returns a nested collection of TensorFlow assign operations.

  Args:
    ref: Nested collection of TensorFlow variables.
    value: Values to be assigned to the variables. Must have the same structure
        as `ref`.

  Returns:
    Nested collection (same structure as `ref`) of TensorFlow assign operations.

  Raises:
    ValueError: If `ref` and `values` have different structures.
  """
  if isinstance(ref, list) or isinstance(ref, tuple):
    if len(ref) != len(value):
      raise ValueError("ref and value have different lengths.")
    result = [_nested_assign(r, v) for r, v in zip(ref, value)]
    if isinstance(ref, tuple):
      return tuple(result)
    return result
  else:
    return tf.assign(ref, value)


def _nested_variable(init, name=None, trainable=False):
  """Returns a nested collection of TensorFlow variables.

  Args:
    init: Nested collection of TensorFlow initializers.
    name: Variable name.
    trainable: Make variables trainable (`False` by default).

  Returns:
    Nested collection (same structure as `init`) of TensorFlow variables.
  """
  if isinstance(init, list) or isinstance(init, tuple):
    result = [_nested_variable(i, name, trainable) for i in init]
    if isinstance(init, tuple):
      return tuple(result)
    return result
  else:
    return tf.Variable(init, name=name, trainable=trainable)


def _wrap_variable_creation(func, custom_getter):
  """Provides a custom getter for all variable creations."""
  original_get_variable = tf.get_variable
  def custom_get_variable(*args, **kwargs):
    if hasattr(kwargs, "custom_getter"):
      raise AttributeError("Custom getters are not supported for optimizee "
                           "variables.")
    return original_get_variable(*args, custom_getter=custom_getter, **kwargs)

  # Mock the get_variable method.
  with mock.patch("tensorflow.get_variable", custom_get_variable):
    return func()


def _get_variables(func):
  """Calls func, returning any variables created, but ignoring its return value.

  Args:
    func: Function to be called.

  Returns:
    A tuple (variables, constants) where the first element is a list of
    trainable variables and the second is the non-trainable variables.
  """
  variables = []
  constants = []

  def custom_getter(getter, name, **kwargs):
    trainable = kwargs["trainable"]
    kwargs["trainable"] = False
    variable = getter(name, **kwargs)
    if trainable:
      variables.append(variable)
    else:
      constants.append(variable)
    return variable

  with tf.name_scope("unused_graph"):
    _wrap_variable_creation(func, custom_getter)

  return variables, constants


def _make_with_custom_variables(func, variables):
  """Calls func and replaces any trainable variables.

  This returns the output of func, but whenever `get_variable` is called it
  will replace any trainable variables with the tensors in `variables`, in the
  same order. Non-trainable variables will re-use any variables already
  created.

  Args:
    func: Function to be called.
    variables: A list of tensors replacing the trainable variables.

  Returns:
    The return value of func is returned.
  """
  variables = collections.deque(variables)

  def custom_getter(getter, name, **kwargs):
    if kwargs["trainable"]:
      return variables.popleft()
    else:
      kwargs["reuse"] = True
      return getter(name, **kwargs)

  return _wrap_variable_creation(func, custom_getter)


MetaLoss = collections.namedtuple("MetaLoss", "loss, loss_kl_1, loss_kl_2, update, reset, fx, x")
MetaStep = collections.namedtuple("MetaStep", "step, loss_kl_1, loss_kl_2, update, reset, fx, x")


def _make_nets(variables, config, net_assignments):
  """Creates the optimizer networks.

  Args:
    variables: A list of variables to be optimized.
    config: A dictionary of network configurations, each of which will be
        passed to networks.Factory to construct a single optimizer net.
    net_assignments: A list of tuples where each tuple is of the form (netid,
        variable_names) and is used to assign variables to networks. netid must
        be a key in config.
    YC add:
    baysianornot:  Whether using bayesian nerual networks  for optimizer networks or not.

  Returns:
    A tuple (nets, keys, subsets) where nets is a dictionary of created
    optimizer nets such that the net with key keys[i] should be applied to the
    subset of variables listed in subsets[i].

  Raises:
    ValueError: If net_assignments is None and the configuration defines more
        than one network.
  """
  # create a dictionary which maps a variable name to its index within the
  # list of variables.
  name_to_index = dict((v.name.split(":")[0], i)
                       for i, v in enumerate(variables))
  print (config)

  if net_assignments is None:
    if len(config) != 1:
      raise ValueError("Default net_assignments can only be used if there is "
                       "a single net config.")

    with tf.variable_scope("vars_optimizer"):
      key = next(iter(config))
      kwargs = config[key]
      net = networks.factory(**kwargs)

    nets = {key: net}
    keys = [key]
    subsets = [range(len(variables))]
  else:
    nets = {}
    keys = []
    subsets = []
    with tf.variable_scope("vars_optimizer"):
      for key, names in net_assignments:
        if key in nets:
          raise ValueError("Repeated netid in net_assigments.")
        nets[key] = networks.factory(**config[key])
        subset = [name_to_index[name] for name in names]
        keys.append(key)
        subsets.append(subset)
        print("Net: {}, Subset: {}".format(key, subset))

  # subsets should be a list of disjoint subsets (as lists!) of the variables
  # and nets should be a list of networks to apply to each subset.
  return nets, keys, subsets


class MetaOptimizer(object):
  """Learning to learn (meta) optimizer.

  Optimizer which has an internal RNN which takes as input, at each iteration,
  the gradient of the function being minimized and returns a step direction.
  This optimizer can then itself be optimized to learn optimization on a set of
  tasks.
  """

  def __init__(self, **kwargs):
    """Creates a MetaOptimizer.

    Args:
      **kwargs: A set of keyword arguments mapping network identifiers (the
          keys) to parameters that will be passed to networks.Factory (see docs
          for more info).  These can be used to assign different optimizee
          parameters to different optimizers (see net_assignments in the
          meta_loss method).
    """
    self._nets = None

    if not kwargs:
      # Use a default coordinatewise network if nothing is given. this allows
      # for no network spec and no assignments.
      self._config = {
          "coordinatewise": {
              "net": "CoordinateWiseDeepLSTM",
              "net_options": {
                  "layers": (20, 20),
                  "preprocess_name": "LogAndSign",
                  "preprocess_options": {"k": 5},
                  "scale": 0.01,
              }}}
    else:
      self._config = kwargs

  def save(self, sess, path):
    """Save meta-optimizer."""
    save_dic={}
    for variables in tf.trainable_variables():
      save_dic[variables.name] = sess.run(variables)

    with open(path, "wb") as f:
      pickle.dump(save_dic,f)


  def restore(self, sess, path):
    
    with open(path, "rb") as f:
      vars_dic = pickle.load(f)

    assign_op=[]
    print (vars_dic)
    for variables in tf.trainable_variables():
      print (variables.name) 
      if variables.name in vars_dic:
        c = tf.constant(vars_dic[variables.name])
        assign_op.append(tf.assign(variables, c))
        print (c, 'dd')
    print ('end')


    sess.run(assign_op)


  def meta_loss(self,
                make_loss,
                len_unroll,
                lambda2=0.1,
                net_assignments=None,
                second_derivatives=False):
    """Returns an operator computing the meta-loss.

    Args:
      make_loss: Callable which returns the optimizee loss; note that this
          should create its ops in the default graph.
      len_unroll: Number of steps to unroll.
      net_assignments: variable to optimizer mapping. If not None, it should be
          a list of (k, names) tuples, where k is a valid key in the kwargs
          passed at at construction time and names is a list of variable names.
      second_derivatives: Use second derivatives (default is false).

    Returns:
      namedtuple containing (loss, update, reset, fx, x)
    """

    # Construct an instance of the problem only to grab the variables. This
    # loss will never be evaluated.
    # pdb.set_trace()
    
    x, constants = _get_variables(make_loss)

    print("Optimizee variables, ", len(x))
    print([op for op in x])
    print("Problem variables", len(constants))
    print([op for op in constants])
    
    # hess
    fx = _make_with_custom_variables(make_loss, x)
    grads = tf.gradients(fx, x)
    #hess_norm_approx = sum([tf.reduce_sum(g*g) for g in grads])

    # create scale placeholder here
    scale = []
    for k in x:
      scale.append(tf.placeholder_with_default(tf.ones(shape=k.shape), shape=k.shape, name=k.name[:-2] + "_scale"))

    # Create the optimizer networks and find the subsets of variables to assign
    # to each optimizer.
    nets, net_keys, subsets = _make_nets(x, self._config, net_assignments)
    print('nets', nets)
    print('subsets', subsets)
    # Store the networks so we can save them later.
    self._nets = nets

    # Create hidden state for each subset of variables.
    state = []
    with tf.name_scope("states"):
      for i, (subset, key) in enumerate(zip(subsets, net_keys)):
        net = nets[key]
        with tf.name_scope("state_{}".format(i)):
          # View this URL: https://github.com/deepmind/learning-to-learn/issues/22
          state.append([net.initial_state_for_inputs(x[j], dtype=tf.float32) for j in subset])
          # state.append(_nested_variable(
          #     [net.initial_state_for_inputs(x[j], dtype=tf.float32)
          #      for j in subset],
          #     name="state", trainable=False))
    def update(net, fx, x, state):
      """Parameter and RNN state update."""
      with tf.name_scope("gradients"):
        gradients = tf.gradients(fx, x)

        # Stopping the gradient here corresponds to what was done in the
        # original L2L NIPS submission. However it looks like things like
        # BatchNorm, etc. don't support second-derivatives so we still need
        # this term.
        if not second_derivatives:
          gradients = [tf.stop_gradient(g) for g in gradients]

      with tf.name_scope("deltas"):
        deltas, state_next = zip(*[net(g, s) for g, s in zip(gradients, state)])
        state_next = list(state_next)

      return deltas, state_next

    def time_step(t, fx_array, x, state):
      """While loop body."""
      x_next = list(x)
      state_next = []

      with tf.name_scope("fx"):
        scaled_x = [x[k] * scale[k] for k in range(len(scale))]
        fx = _make_with_custom_variables(make_loss, scaled_x)
        fx_array = fx_array.write(t, fx)

      with tf.name_scope("dx"):
        for subset, key, s_i in zip(subsets, net_keys, state):
          x_i = [x[j] for j in subset]

          deltas, s_i_next = update(nets[key], fx, x_i, s_i)

          for idx, j in enumerate(subset):
            #print (t,'saaaaaaaaaaaa')
            delta = deltas[idx] #* tf.pow(0.9, tf.cast(t, tf.float32))
            x_next[j] += delta
          state_next.append(s_i_next)

      with tf.name_scope("t_next"):
        t_next = t + 1


      #indices = tf.random_uniform([1,2], 0, 10, tf.int32)
      #hehe = hehe.write(t, indices)

      return t_next, fx_array, x_next, state_next
    # Define the while loop.
    fx_array = tf.TensorArray(tf.float32, size=len_unroll+1,
                              clear_after_read=False)

    # hehe =tf.TensorArray(tf.int32, size=len_unroll,
    #                           clear_after_read=False)

    _, fx_array, x_final, s_final = tf.while_loop(
        cond=lambda t, *_: t < len_unroll,
        body=time_step,
        loop_vars=(0, fx_array, x, state),
        parallel_iterations=1,
        swap_memory=True,
        name="unroll")

    with tf.name_scope("fx"):
      scaled_x_final = [x_final[k] * scale[k] for k in range(len(scale))]
      fx_final = _make_with_custom_variables(make_loss, scaled_x_final)
      fx_array = fx_array.write(len_unroll, fx_final)


    #bayesian.mean_covaraince_variable(tf.trainable_variables())
  
    #gamma= [0.9**i for i in range(21)]
    loss = tf.reduce_sum(fx_array.stack(), name="loss")
    loss_kl_1, loss_kl_2, loss_kl = 0, 0, 0


    lambda1=tf.placeholder_with_default(tf.constant(lambda2, dtype=tf.float32), shape=())
    for key in self._config:
      if self._config[key]['net_options']['stage'] == 2:
        # log p(\theta | \phi)
        # for variables in tf.trainable_variables():
        #   if 'rho' in variables.name:
        #     print (variables)
        #     loss -= lambda1*tf.reduce_sum( tf.math.log( tf.math.log(1.0+tf.math.exp(variables))  ) )

        '''
        for w in l2_bayes_loss.loss.loss:
          print (w)
          loss += lambda1*tf.reduce_sum(w*w)
        '''
        for mu, rho in zip(l2_bayes_loss.loss.loss_mu, l2_bayes_loss.loss.loss_rho):
          # kl = log(1/rho) + (rho^2 + mu^2)/2 - 1/2
          rho = tf.math.softplus(rho)
          loss_kl_1 += ( - tf.reduce_mean( tf.math.log( rho )) ) / len(l2_bayes_loss.loss.loss_mu)
          loss_kl_2 += ( tf.reduce_mean( rho*rho ) + tf.reduce_mean( mu*mu ) ) / 2 / lambda1 / len(l2_bayes_loss.loss.loss_mu)
          loss_kl +=  ( loss_kl_1 + loss_kl_2 )

        loss += loss_kl

      else:
        print ("this is stage 1")
    

    print (fx_array.stack())

    # Reset the state; should be called at the beginning of an epoch.
    with tf.name_scope("reset"):
      # View this URL: https://github.com/deepmind/learning-to-learn/issues/22
      variables = (nest.flatten(_nested_variable(state)) + x + constants)
      # variables = (nest.flatten(state) +
      #              x + constants)
      # Empty array as part of the reset process.
      reset = [tf.variables_initializer(variables), fx_array.close()]
    # Operator to update the parameters and the RNN state after our loop, but
    # during an epoch.
    with tf.name_scope("update"):
      # View this URL: https://github.com/deepmind/learning-to-learn/issues/22
      update = (nest.flatten(_nested_assign(x, x_final)) + 
                 nest.flatten(_nested_assign(_nested_variable(state), s_final)))
      # update = (nest.flatten(_nested_assign(x, x_final)) +
      #           nest.flatten(_nested_assign(state, s_final)))

    # Log internal variables.
    for k, net in nets.items():
      print("Optimizer '{}' variables".format(k))
      print([op for op in tf.trainable_variables()])
    

    # with tf.Session() as sess:
    #   init_optimizer_vars = tf.variables_initializer(tf.trainable_variables())
     
    #   #tf.get_default_graph().finalize()
    #   sess.run(init_optimizer_vars)
    #   sess.run(reset)
    #   print(sess.run(fx_array.stack()))
    # exit(0)

    # pdb.set_trace()
    return MetaLoss(loss, loss, loss, update, reset, fx_final, x_final), scale, x, constants, subsets,\
           lambda1

  def meta_minimize(self, make_loss, len_unroll, learning_rate=0.01, lambda2=0.1, **kwargs):
    """Returns an operator minimizing the meta-loss.

    Args:
      make_loss: Callable which returns the optimizee loss; note that this
          should create its ops in the default graph.
      len_unroll: Number of steps to unroll.
      learning_rate: Learning rate for the Adam optimizer.
      **kwargs: keyword arguments forwarded to meta_loss.

    Returns:
      namedtuple containing (step, update, reset, fx, x)
    """
    info, scale, x, constants, subsets, lambda1 = self.meta_loss(make_loss, len_unroll, lambda2, **kwargs)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    # -------------------------
    total_parameters = 0

    mean_cov_var_list=[]
    for variable in tf.trainable_variables():

      
      print (variable)
      shape = variable.get_shape()
      variable_parameters = 1
      for dim in shape:
        variable_parameters *= dim.value
      # print(variable_parameters)
      total_parameters += variable_parameters
      mean_cov_var_list.append(variable)

    print('The total number of parameters are: ', total_parameters)
   
    # -------------------------
    step = optimizer.minimize(info.loss)
    
    #self.restorer()
    return MetaStep(step, *info[1:]), scale, x, constants, subsets, \
           lambda1, optimizer
