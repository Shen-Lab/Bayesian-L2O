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
"""Learning 2 Learn utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
from timeit import default_timer as timer

import numpy as np
from six.moves import xrange

import problems
import random
import tensorflow as tf
import pdb

def run_epoch(sess, cost_op, ops, reset, num_unrolls, stddev=None, num_rd=10,
              scale=None, rd_scale=False, rd_scale_bound=3.0, assign_func=None, var_x=None,
              step=None, unroll_len=None,
              task_i=-1, data=None, label_pl=None, input_pl=None, num_epoch=None, lambda1=None):
  """Runs one optimization epoch."""
  start = timer()
  # hessian initialization
  if lambda1==None:
    raise ValueError("Must specify a lambda value (tf tensor).")  

  sess.run(reset)


  print (rd_scale, stddev, step)


  if rd_scale:
    assert scale is not None
    r_scale = []
    for k in scale:
      r_scale.append(np.exp(np.random.uniform(-rd_scale_bound, rd_scale_bound,
                        size=k.shape)))
    assert var_x is not None
    #print (sess.run(var_x)[0][0])
    k_value_list = []
    for k_id in range(len(var_x)):
      k_value = sess.run(var_x[k_id])
      k_value = k_value / r_scale[k_id]
      k_value_list.append(k_value)
    assert assign_func is not None
    #assign_func(k_value_list)

    #print (sess.run(var_x)[0][0], 'after')

    feed_rs = {p: v for p, v in zip(scale, r_scale)}
  else:
    feed_rs = {}
  # feed_rs[lambda1]= 2.0**(-num_epoch[0])
  # feed_rs[lambda1]= lambda1
  if stddev is not None:
      unroll_len = stddev.get_shape().as_list()[0]
      assert unroll_len >= num_rd
      feed_dict = {**feed_rs, stddev: np.array([0.1]*num_rd+[0.0]*(unroll_len-num_rd))}
      if step is not None:
          feed_dict[step] = 1
      cost = sess.run([cost_op] + ops, feed_dict=feed_dict)[0]
      feed_dict = feed_rs
      for i in xrange(1, num_unrolls):
        if step is not None:
            feed_dict[step] = i*unroll_len+1
        cost = sess.run([cost_op] + ops, feed_dict=feed_dict)[0]

  else:
      feed_dict = feed_rs
      for i in xrange(num_unrolls):
        if step is not None:
            feed_dict[step] = i*unroll_len+1
        costs = sess.run([cost_op] + ops, feed_dict=feed_dict)
        cost = costs[0]
        if np.isnan(cost) or np.isnan(costs[3]) or np.isnan(costs[4]):
            pdb.set_trace()
        else:
            for aaa in costs[1]:
                if np.isnan(aaa.sum()):
                    pdb.set_trace()

        print (cost, costs[3], costs[4], '|', end=' ')
      print ("num_unrolls:", num_unrolls, '\n')
      #print (feed_dict)

      # print (sess.run(var_x)[0].shape)

      # with open("../data/protein_dock/test_data.pkl", "rb") as f:
      #   data = pickle.load(f)
      # x=sess.run(var_x)[0]
      # x = np.expand_dims(np.expand_dims(x, axis=1), axis=1)
     
      # dx = np.matmul(x, data['basic'].astype(np.float32))
      # dx = np.squeeze(dx, 2)
      # print ( np.mean( np.sqrt(np.sum(dx**2, -1))) )
      #print ( np.mean(np.sqrt(np.sum(sess.run(var_x)[0]**2, axis=1))))
       
  return timer() - start, cost



def run_eval_epoch(sess, cost_op, update, reset, num_unrolls):
  """Runs one optimization epoch."""
  start = timer()
  
  total_cost = []
  for _ in xrange(num_unrolls):
    cost ,_= sess.run([cost_op,update])
    total_cost.append(cost)
  print ("total_cost", total_cost)
  return timer() - start, total_cost


def print_stats(header, total_error, total_time, n):
  """Prints experiment statistics."""
  print(header)
  print("Log Mean Final Error: {:.2f}".format(np.log10(total_error / n)))
  print("Mean epoch time: {:.2f} s".format(total_time / n))


def get_net_path(name, path, idx=0):
  return None if path is None else os.path.join(path, name + ".l2l-{}".format(idx))


def get_default_net_config(name, path, idx=0):
  return {
      "net": "CoordinateWiseDeepLSTM",
      "net_options": {
          "layers": (20, 20),
          "preprocess_name": "LogAndSign",
          "preprocess_options": {"k": 5},
          "scale": 0.01,
      },
      "net_path": get_net_path(name, path, idx)
  }

def get_noncoordinate_net_config(name, path, idx=0):
  return {
      "net": "StandardDeepLSTM",
      "net_options": {
          "layers": (50, 50),
          "preprocess_name": "LogAndSign",
          "preprocess_options": {"k": 5},
          "scale": 0.1,
      },
      "net_path": get_net_path(name, path, idx)
  }


def get_config(problem_name, path=None, model_idx=0, mode='train', num_hidden_layer=None, net_name=None, num_linear_heads=1, init="normal", batch_size=128):
  """Returns problem configuration."""
  if problem_name == "simple":
    problem = problems.simple()
    net_config = {"cw": {
        "net": "CoordinateWiseDeepLSTM",
        "net_options": {"layers": (), "initializer": "zeros"},
        "net_path": get_net_path("cw", path)
    }}
    net_assignments = None
  elif problem_name == "simple-multi":
    problem = problems.simple_multi_optimizer()
    net_config = {
        "cw": {
            "net": "CoordinateWiseDeepLSTM",
            "net_options": {"layers": (), "initializer": "zeros"},
            "net_path": get_net_path("cw", path)
        },
        "adam": {
            "net": "Adam",
            "net_options": {"learning_rate": 0.01}
        }
    }
    net_assignments = [("cw", ["x_0"]), ("adam", ["x_1"])]
  elif problem_name == "quadratic":
    problem = problems.quadratic(batch_size=128, num_dims=10)
    net_config = {"cw": {
        "net": "CoordinateWiseDeepLSTM",
        "net_options": {"layers": (20, 20)},
        "net_path": get_net_path("cw", path)
    }}
    net_assignments = None
  elif problem_name == "mnist":

    problem = problems.mnist(layers=(20,), activation="sigmoid", mode=mode, init="normal")
    net_config = {"cw": get_default_net_config("cw", path, model_idx)}
    net_assignments = None
  elif problem_name == "mnist_odd":
    problem = problems.mnist(layers=(20,), activation="sigmoid", mode=mode, init="normal", odd=True)
    net_config = {"cw": get_default_net_config("cw", path, model_idx)}
    net_assignments = None
  elif "ras" in problem_name:
    problem = problems.rastrigin(mode=mode, num_dims=int(problem_name[-2:]))
    net_config = {"cw": get_default_net_config("cw", path, model_idx)}
    #net_config['cw']['net_options']['output_size']=6
    net_assignments = None
  elif "ackley" in problem_name:
    problem = problems.ackley(mode=mode, num_dims=int(problem_name[-2:]))
    net_config = {"cw": get_default_net_config("cw", path, model_idx)}
    net_assignments = None
  elif "griewank" in problem_name:
    problem = problems.griewank(mode=mode, num_dims=int(problem_name[-2:]))
    net_config = {"cw": get_default_net_config("cw", path, model_idx)}
    net_assignments = None
  elif "privacy_attack" in problem_name:
    problem = problems.privacy_attack(mode=mode)
    net_config = {"cw": get_default_net_config("cw", path, model_idx)}
    net_assignments = None
  elif "protein_dock" in problem_name:
    problem = problems.protein_dock(mode=mode)
    net_config = {"cw": get_default_net_config("cw", path, model_idx)}
    net_assignments = None
  elif problem_name == "mnist_relu":
      mode = "test"
      problem = problems.mnist(layers=(20,), activation="relu", mode=mode)
      net_config = {"cw": get_default_net_config("cw", path, model_idx)}
      net_assignments = None
  elif problem_name == "mnist_deeper":
      mode = "test"
      assert num_hidden_layer is not None
      problem = problems.mnist(layers=(20,) * num_hidden_layer, activation="sigmoid", mode=mode)
      net_config = {"cw": get_default_net_config("cw", path, model_idx)}
      net_assignments = None
  elif problem_name == "mnist_conv":
      mode = "test"
      problem = problems.mnist_conv(mode=mode, batch_norm=True)
      net_config = {"cw": get_default_net_config("cw", path, model_idx)}
      net_assignments = None
  elif problem_name == "cifar10":
    
    problem = problems.cifar10("cifar10", mode=mode)
    net_config = {"cw": get_default_net_config("cw", path, model_idx)}
    net_assignments = None

  elif problem_name == "cifar100":
    
    problem = problems.cifar100("cifar100", mode=mode)
    net_config = {"cw": get_default_net_config("cw", path, model_idx)}
    net_assignments = None

  elif problem_name == "vgg16":
    mode = "train" if path is None else "test"
    problem = problems.vgg16_cifar10("cifar10",
                               mode=mode)
    net_config = {"cw": get_default_net_config("cw", path)}
    net_assignments = None
  elif problem_name == "cifar-multi":
    mode = "train" if path is None else "test"
    problem = problems.cifar10("cifar10",
                               conv_channels=(16, 16, 16),
                               linear_layers=(32,),
                               mode=mode)
    net_config = {
        "conv": get_default_net_config("conv", path),
        "fc": get_default_net_config("fc", path)
    }
    conv_vars = ["conv_net_2d/conv_2d_{}/w".format(i) for i in xrange(3)]
    fc_vars = ["conv_net_2d/conv_2d_{}/b".format(i) for i in xrange(3)]
    fc_vars += ["conv_net_2d/batch_norm_{}/beta".format(i) for i in xrange(3)]
    fc_vars += ["mlp/linear_{}/w".format(i) for i in xrange(2)]
    fc_vars += ["mlp/linear_{}/b".format(i) for i in xrange(2)]
    fc_vars += ["mlp/batch_norm/beta"]
    net_assignments = [("conv", conv_vars), ("fc", fc_vars)]
  else:
    raise ValueError("{} is not a valid problem".format(problem_name))

  if net_name == "RNNprop":
      default_config = {
              "net": "RNNprop",
              "net_options": {
                  "layers": (20, 20),
                  "preprocess_name": "fc",
                  "preprocess_options": {"dim": 20},
                  "scale": 0.01,
                  "tanh_output": True,
                  "num_linear_heads": num_linear_heads
              },
              "net_path": get_net_path("rp", path, model_idx)
          }
      net_config = {"rp": default_config}

  return problem, net_config, net_assignments
