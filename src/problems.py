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
# limitations under the License
# ==============================================================================
"""Learning 2 Learn problems."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tarfile
import sys
import pickle
import numpy as np

from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import sonnet as snt
import tensorflow as tf
import pdb
from tensorflow.contrib.learn.python.learn.datasets import mnist as mnist_dataset
from vgg16 import VGG16

_nn_initializers = {
    "w": tf.random_normal_initializer(mean=0, stddev=0.01),
    "b": tf.random_normal_initializer(mean=0, stddev=0.01),
}

def simple():
  """Simple problem: f(x) = x^2."""

  def build():
    """Builds loss graph."""
    x = tf.get_variable(
        "x",
        shape=[],
        dtype=tf.float32,
        initializer=tf.ones_initializer())
    return tf.square(x, name="x_squared")

  return build


def simple_multi_optimizer(num_dims=2):
  """Multidimensional simple problem."""

  def get_coordinate(i):
    return tf.get_variable("x_{}".format(i),
                           shape=[],
                           dtype=tf.float32,
                           initializer=tf.ones_initializer())

  def build():
    coordinates = [get_coordinate(i) for i in xrange(num_dims)]
    x = tf.concat([tf.expand_dims(c, 0) for c in coordinates], 0)
    return tf.reduce_sum(tf.square(x, name="x_squared"))

  return build


def quadratic(batch_size=128, num_dims=10, stddev=0.01, dtype=tf.float32):
  """Quadratic problem: f(x) = ||Wx - y||."""

  def build():
    """Builds loss graph."""

    # Trainable variable.
    x = tf.get_variable(
        "x",
        shape=[batch_size, num_dims],
        dtype=dtype,
        initializer=tf.random_normal_initializer(stddev=stddev))

    # Non-trainable variables.
    w = tf.get_variable("w",
                        shape=[batch_size, num_dims, num_dims],
                        dtype=dtype,
                        initializer=tf.random_uniform_initializer(),
                        trainable=False)
    y = tf.get_variable("y",
                        shape=[batch_size, num_dims],
                        dtype=dtype,
                        initializer=tf.random_uniform_initializer(),
                        trainable=False)

    product = tf.squeeze(tf.matmul(w, tf.expand_dims(x, -1)))
    return tf.reduce_mean(tf.reduce_sum((product - y) ** 2, 1))

  return build

def indentity_init(batch_size, num_dims, stddev):
  return tf.eye(num_dims, batch_shape=[batch_size])+\
  tf.random.normal(shape=[batch_size, num_dims, num_dims], stddev=stddev)

def ackley(num_dims=None, mode='train'):

  stddev=0.01
  dtype=tf.float32
  def build():
    """Builds loss graph."""

    # Trainable variable.
    if mode=='test':
      x = tf.get_variable(
        "x",
        shape=[1, num_dims],
        dtype=dtype,
        initializer=tf.random_uniform_initializer(-2,2))
      return tf.reduce_mean( -20*tf.math.exp(-0.2*tf.math.sqrt(0.5*tf.reduce_sum(x*x, 1)))\
       - tf.math.exp( 1.0/num_dims*tf.reduce_sum(tf.math.cos(2*3.1415926*x), 1))+2.71828 + 20 )
     
   
    batch_size=128
    x = tf.get_variable(
        "x",
        shape=[batch_size, num_dims],
        dtype=dtype,
        initializer=tf.random_uniform_initializer(-2, 2))

    # Non-trainable variables.
    w = tf.get_variable("w",
                        dtype=dtype,
                        initializer=indentity_init(batch_size, num_dims, stddev),
                        trainable=False)
  
    y = tf.get_variable("y",
                        shape=[batch_size, num_dims],
                        dtype=dtype,
                        initializer=tf.random_normal_initializer(stddev=stddev),
                        trainable=False)

    wcos = tf.get_variable("wcos",
                        shape=[batch_size, num_dims],
                        dtype=dtype,
                        initializer=tf.random_normal_initializer(mean=1.0, stddev=stddev),
                        trainable=False)

    product = tf.squeeze(tf.matmul(w, tf.expand_dims(x, -1)))
    product2 = tf.reduce_sum(wcos*tf.math.cos(2*3.1415926*x), 1) 

    #product3 = tf.reduce_sum((product - y) ** 2, 1) - tf.reduce_sum(product2, 1) + 10*num_dims

        

    return tf.reduce_mean(-20*tf.math.exp(-0.2*tf.math.sqrt(0.5*tf.reduce_sum( (product-y)**2 ,1)))\
       - tf.math.exp( 1.0/num_dims*product2)+2.71828 + 20)#, [x,w,y,wcos]


  return build

def griewank(num_dims=None, mode='train'):


  stddev=0.01
  dtype=tf.float32
  def build():
    """Builds loss graph."""

    # Trainable variable.
    if mode=='test':
      x = tf.get_variable(
        "x",
        shape=[1, num_dims],
        dtype=dtype,
        initializer=tf.random_uniform_initializer(-2, 2))


      return tf.reduce_mean(1 + 1.0/4000 * tf.reduce_sum(x*x, 1)  - tf.reduce_prod(tf.math.cos(x), 1))


    batch_size=128
    x = tf.get_variable(
        "x",
        shape=[batch_size, num_dims],
        dtype=dtype,
        initializer=tf.random_uniform_initializer(-2, 2))

    # Non-trainable variables.
    w = tf.get_variable("w",
                        dtype=dtype,
                        initializer=indentity_init(batch_size, num_dims, stddev),
                        trainable=False)
  
    y = tf.get_variable("y",
                        shape=[batch_size, num_dims],
                        dtype=dtype,
                        initializer=tf.random_normal_initializer(stddev=stddev),
                        trainable=False)

    wcos = tf.get_variable("wcos",
                        shape=[batch_size, num_dims],
                        dtype=dtype,
                        initializer=tf.random_normal_initializer(mean=0.0, stddev=stddev),
                        trainable=False)

    product = tf.squeeze(tf.matmul(w, tf.expand_dims(x, -1)))
    #product2 = tf.reduce_sum(wcos*10*tf.math.cos(2*3.1415926*x), 1) 

    #product3 = tf.reduce_sum((product - y) ** 2, 1) - tf.reduce_sum(product2, 1) + 10*num_dims


    return tf.reduce_mean(1.0 + 1.0/4000 * tf.reduce_sum((product-y)**2, 1)  - tf.reduce_prod(tf.math.cos(x)+wcos , 1))

  return build

def rastrigin(num_dims=None, mode='train'):


  stddev=0.01
  dtype=tf.float32
  def build():
    """Builds loss graph."""

    # Trainable variable.
    if mode=='test':
      x = tf.get_variable(
        "x",
        shape=[1, num_dims],
        dtype=dtype,
        initializer=tf.random_uniform_initializer(-2, 2))

      return tf.reduce_mean ( tf.reduce_sum(x*x - 10*tf.math.cos(2*3.1415926*x), 1)+ 10*num_dims )


    batch_size=128
    x = tf.get_variable(
        "x",
        shape=[batch_size, num_dims],
        dtype=dtype,
        initializer=tf.random_uniform_initializer(-2, 2))

    # Non-trainable variables.
    w = tf.get_variable("w",
                        dtype=dtype,
                        initializer=indentity_init(batch_size, num_dims, stddev),
                        trainable=False)
  
    y = tf.get_variable("y",
                        shape=[batch_size, num_dims],
                        dtype=dtype,
                        initializer=tf.random_normal_initializer(stddev=stddev),
                        trainable=False)

    wcos = tf.get_variable("wcos",
                        shape=[batch_size, num_dims],
                        dtype=dtype,
                        initializer=tf.random_normal_initializer(mean=1.0, stddev=stddev),
                        trainable=False)

    product = tf.squeeze(tf.matmul(w, tf.expand_dims(x, -1)))
    product2 = tf.reduce_sum(wcos*10*tf.math.cos(2*3.1415926*x), 1) 

    #product3 = tf.reduce_sum((product - y) ** 2, 1) - tf.reduce_sum(product2, 1) + 10*num_dims


    return tf.reduce_mean(tf.reduce_sum((product - y) ** 2, 1)) - tf.reduce_mean(product2) + 10*num_dims

  return build


def privacy_attack(mode='train'):

  def build():
    """Builds loss graph."""

    # Trainable variable.
    if mode=='test':

      feature = np.load("../data/privacy_test_feat.npy").astype(np.float32)
      label   = np.load("../data/privacy_test_label.npy").astype(np.float32)
      weight = np.load("../data/privacy_weight.npy").astype(np.float32)

      feature = tf.constant(feature)
      label = tf.constant(label)
      weight = tf.constant(weight)

      x = tf.get_variable(
        "x",
        shape=[feature.shape[0], 4],
        dtype=tf.float32,
        initializer=tf.random_uniform_initializer(0, 0.5))

    else:
      feature = np.load("../data/privacy_train_feat.npy").astype(np.float32)
      label   = np.load("../data/privacy_train_label.npy").astype(np.float32)
      weight = np.load("../data/privacy_weight.npy").astype(np.float32)

      feature = tf.constant(feature)
      label = tf.constant(label)
      weight = tf.constant(weight)

      x = tf.get_variable(
        "x",
        shape=[feature.shape[0], 4],
        dtype=tf.float32,
        initializer=tf.random_uniform_initializer(0, 1))

    feature_all = tf.concat((x, feature), axis=1)

    weight = tf.expand_dims(weight,axis=1)
    return tf.reduce_mean((1./(1.+ tf.exp( - tf.tensordot(feature_all, weight, axes = 1))) - label) **2)

  return build

def protein_dock(mode='train'):

  num_dims=12
  stddev=0.5
  batch_size = 5

  def build():
    x = tf.get_variable(
          "x",
          shape=[batch_size, num_dims],
          dtype=tf.float32,
          initializer=tf.random_normal_initializer(stddev=stddev))

    if mode=='test':
          idx = tf.get_variable("idx",
                      shape=[batch_size],
                      dtype=tf.int64,
                      initializer=tf.constant_initializer(np.arange(0,5)),
                      trainable=False)

          with open("../data/protein_dock/test_data.pkl", "rb") as f:
            data = pickle.load(f)
    else:
          idx = tf.get_variable("idx",
                      shape=[batch_size],
                      dtype=tf.int64,
                      initializer=tf.random_uniform_initializer(0, 125),
                      trainable=False)

          with open("../data/protein_dock/train_data.pkl", "rb") as f:
            data = pickle.load(f)


    coor_init = tf.get_variable("coor_init",
                        shape=data['init'].shape,
                        dtype=tf.float32,
                        initializer=tf.constant_initializer(data['init'], dtype = tf.float32),
                        trainable=False)

    q = tf.get_variable("q",
                        shape=data['q'].shape,
                        dtype=tf.float32,
                        initializer=tf.constant_initializer(data['q'], dtype = tf.float32),
                        trainable=False)
    e = tf.get_variable("e",
                        shape=data['e'].shape,
                        dtype=tf.float32,
                        initializer=tf.constant_initializer(data['e'], dtype = tf.float32),
                        trainable=False)

    r = tf.get_variable("r",
                        shape=data['r'].shape,
                        dtype=tf.float32,
                        initializer=tf.constant_initializer(data['r'], dtype = tf.float32),
                        trainable=False)

    basis = tf.get_variable("basic",
                        shape=data['basic'].shape,
                        dtype=tf.float32,
                        initializer=tf.constant_initializer(data['basic'], dtype = tf.float32),
                        trainable=False)

    coor_init = tf.gather(coor_init, idx)
    q = tf.gather(q, idx)
    e = tf.gather(e, idx)
    r = tf.gather(r, idx)
    basis = tf.gather(basis, idx)


    x = tf.expand_dims(tf.expand_dims(x, axis=1), axis=1)


    dx = tf.matmul(x, basis)
    dx = tf.squeeze(dx, 2)

    new_coor = dx + coor_init

    p2 = tf.reduce_sum(new_coor*new_coor, 2)
    p3 = tf.matmul(new_coor, tf.transpose(new_coor, perm=[0,2,1]))
    p2 = tf.expand_dims(p2, -1)
    pair_dis = tf.sqrt(p2 - 2*p3 + tf.transpose(p2, perm=[0, 2, 1]) + 0.01)

    c7_small = tf.math.less(pair_dis, 7)
    c7 = tf.math.greater(pair_dis, 7)
    c0 = tf.math.greater(pair_dis, 0.1)
    c9 = tf.math.less(pair_dis, 9)

    c7=tf.cast(c7, tf.float32)
    c0=tf.cast(c0,  tf.float32)
    c9=tf.cast(c9,  tf.float32)
    c7_small=tf.cast(c7_small,  tf.float32)

    c79=c7*c9*c0
    c7_small=c7_small*c0

    pair_dis += tf.eye(100, num_columns=100, batch_shape=[batch_size])*100.

    coeff = q/(4.*pair_dis) + tf.sqrt(e) * ( (r/pair_dis)**12 - (r/pair_dis)**6 )

    energy = tf.reduce_mean(tf.reduce_sum( c7_small* coeff*10 + 10*c79 * coeff * ( (9-pair_dis)**2 * (-12 + 2*pair_dis) / 8 ), 1)) - 20.

    return energy

  return build

def ensemble(problems, weights=None):
  """Ensemble of problems.

  Args:
    problems: List of problems. Each problem is specified by a dict containing
        the keys 'name' and 'options'.
    weights: Optional list of weights for each problem.

  Returns:
    Sum of (weighted) losses.

  Raises:
    ValueError: If weights has an incorrect length.
  """
  if weights and len(weights) != len(problems):
    raise ValueError("len(weights) != len(problems)")

  build_fns = [getattr(sys.modules[__name__], p["name"])(**p["options"])
               for p in problems]

  def build():
    loss = 0
    for i, build_fn in enumerate(build_fns):
      with tf.variable_scope("problem_{}".format(i)):
        loss_p = build_fn()
        if weights:
          loss_p *= weights[i]
        loss += loss_p
    return loss

  return build


def _xent_loss(output, labels):
  loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output,
                                                        labels=labels)
  return tf.reduce_mean(loss)

def mnist(layers,  # pylint: disable=invalid-name
          activation="sigmoid",
          batch_size=128,
          mode="train",
          init="normal",
          odd =False):
  """Mnist classification with a multi-layer perceptron."""

  if init=="uniform" or init=="hessian":
      initializers = {
          "w": tf.random_uniform_initializer(minval=-0.0196, maxval=0.0196, seed=None),
          "b": tf.random_uniform_initializer(minval=-0.0196, maxval=0.0196, seed=None),
      }
  else:
      initializers = _nn_initializers


  if activation == "sigmoid":
    activation_op = tf.sigmoid
  elif activation == "relu":
    activation_op = tf.nn.relu
  else:
    raise ValueError("{} activation not supported".format(activation))

  # Data.

  data = mnist_dataset.load_mnist()
  data = getattr(data, mode)
  print (data)
  #exit(0)

  if mode=='train' and odd==True:
    data_indx=[]
    for i in range(len(data.labels)):
      if data.labels[i]<5:
        data_indx.append(i)

    images = tf.constant(data.images[data_indx], dtype=tf.float32, name="MNIST_oddimages")
    images = tf.reshape(images, [-1, 28, 28, 1])
    labels = tf.constant(data.labels[data_indx], dtype=tf.int64, name="MNIST_oddlabels")

  else:

    images = tf.constant(data.images, dtype=tf.float32, name="MNIST_images")
    images = tf.reshape(images, [-1, 28, 28, 1])
    labels = tf.constant(data.labels, dtype=tf.int64, name="MNIST_labels")

  # Network.
  if odd==True:
    last_layer=5
  else:
    last_layer=10
  mlp = snt.nets.MLP(list(layers) + [last_layer],
                       activation=activation_op,
                       initializers=initializers)
  network = snt.Sequential([snt.BatchFlatten(), mlp])

  if mode=='test':
    indices=tf.placeholder(tf.int64, shape=[batch_size])
    batch_images = tf.gather(images, indices)
    #batch_labels = tf.gather(labels, indices)
    output = network(batch_images)
    return [indices, output, data.num_examples, batch_size, data.labels]

  def build():
    if odd==False:
      indices = tf.random_uniform([batch_size], 0, data.num_examples, tf.int64)
    else:
      indices = tf.random_uniform([batch_size], 0, len(data_indx), tf.int64)

    batch_images = tf.gather(images, indices)
    batch_labels = tf.gather(labels, indices)
    output = network(batch_images)
    #return tf.cast(tf.reduce_sum(indices), tf.float32)
    return _xent_loss(output, batch_labels)

  return build

def mnist_conv(activation="relu", # pylint: disable=invalid-name
               batch_norm=True,
               batch_size=128,
               mode="train"):
  """Mnist classification with a multi-layer perceptron."""

  if activation == "sigmoid":
    activation_op = tf.sigmoid
  elif activation == "relu":
    activation_op = tf.nn.relu
  else:
    raise ValueError("{} activation not supported".format(activation))

  # Data.
  data = mnist_dataset.load_mnist()
  data = getattr(data, mode)

  images = tf.constant(data.images, dtype=tf.float32, name="MNIST_images")
  images = tf.reshape(images, [-1, 28, 28, 1])
  labels = tf.constant(data.labels, dtype=tf.int64, name="MNIST_labels")
  if mode == 'train':
      is_training = True
  elif mode == 'test':
      is_training = False
  # Network.
  # mlp = snt.nets.MLP(list(layers) + [10],
  #                    activation=activation_op,
  #                    initializers=_nn_initializers)
  # network = snt.Sequential([snt.BatchFlatten(), mlp])

  def network(inputs, training=is_training):
      # pdb.set_trace()


      def _conv_activation(x):  # pylint: disable=invalid-name
          return tf.nn.max_pool(tf.nn.relu(x),
                                ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1],
                                padding="VALID")
      def conv_layer(inputs, strides, c_h, c_w, output_channels, padding, name):
          n_channels = int(inputs.get_shape()[-1])
          with tf.variable_scope(name) as scope:
              kernel1 = tf.get_variable('weights1',
                                        shape=[c_h, c_w, n_channels, output_channels],
                                        dtype=tf.float32,
                                        initializer=tf.random_normal_initializer(stddev=0.01)
                                        )
              
              biases1 = tf.get_variable('biases1', [output_channels], initializer=tf.constant_initializer(0.0))
          inputs = tf.nn.conv2d(inputs, kernel1, [1, strides, strides, 1], padding)
          inputs = tf.nn.bias_add(inputs, biases1)
          if batch_norm:
              inputs = tf.layers.batch_normalization(inputs, training=training)
          inputs = _conv_activation(inputs)
          return inputs
      if batch_norm:
          linear_activation = lambda x: tf.nn.relu(tf.layers.batch_normalization(x, training=training))
      else:
          linear_activation = tf.nn.relu
      # pdb.set_trace()
      # print(inputs.shape)
      inputs = conv_layer(inputs, 1, 3, 3, 16, "VALID", 'conv_layer1')
      # print(inputs.shape)
      inputs = conv_layer(inputs, 1, 5, 5, 32, "VALID", 'conv_layer2')
      # print(inputs.shape)
      # inputs = linear_activation(inputs)
      inputs = tf.reshape(inputs, [batch_size, -1])
      # print(inputs.shape)
      fc_shape1 = int(inputs.get_shape()[0])
      fc_shape2 = int(inputs.get_shape()[1])
      weights = tf.get_variable("fc_weights",
                                shape=[fc_shape2, 10],
                                dtype=tf.float32,
                                initializer=tf.random_normal_initializer(stddev=0.01))
      # print(weights.shape)
      bias = tf.get_variable("fc_bias",
                             shape=[10, ],
                             dtype=tf.float32,
                             initializer=tf.constant_initializer(0.0))
      # print(bias.shape)
      return tf.nn.relu(tf.nn.bias_add(tf.matmul(inputs, weights), bias))

  def build():
    indices = tf.random_uniform([batch_size], 0, data.num_examples, tf.int64)
    batch_images = tf.gather(images, indices)
    batch_labels = tf.gather(labels, indices)
    output = network(batch_images)
    return _xent_loss(output, batch_labels)

  return build


CIFAR10_URL = "http://www.cs.toronto.edu/~kriz"
CIFAR10_FILE = "cifar-10-binary.tar.gz"
CIFAR10_FOLDER = "cifar-10-batches-bin"


def _maybe_download_cifar10(path):
  """Download and extract the tarball from Alex's website."""
  if not os.path.exists(path):
    os.makedirs(path)
  filepath = os.path.join(path, CIFAR10_FILE)
  if not os.path.exists(filepath):
    print("Downloading CIFAR10 dataset to {}".format(filepath))
    url = os.path.join(CIFAR10_URL, CIFAR10_FILE)
    filepath, _ = urllib.request.urlretrieve(url, filepath)
    statinfo = os.stat(filepath)
    print("Successfully downloaded {} bytes".format(statinfo.st_size))
    tarfile.open(filepath, "r:gz").extractall(path)


def cifar10(path,  # pylint: disable=invalid-name
            batch_norm=False,
            batch_size=128,
            num_threads=4,
            conv_channels=(16, 16, 16),
            linear_layers=None,
            min_queue_examples=1000,
            mode="train",
            layers=(20,), activation='sigmoid', init='normal'
            ):

  """Cifar10 classification with a convolutional network."""

  # Data.
  c = tf.keras.datasets.cifar10.load_data()
  #print (c[0][0].shape, c[0][1].shape, c[1][0].shape, c[1][1].shape, c[0][0][1])

  #_maybe_download_cifar10(path)
  if mode=='train':
    images = tf.constant(c[0][0]/255.0, dtype=tf.float32, name="cifar10_train_images")
    labels = tf.constant(c[0][1], dtype=tf.int64, name="cifar10_train_labels")
    is_training=True

  else:
    images = tf.constant(c[1][0]/255.0, dtype=tf.float32, name="cifar10_test_images")
    labels = tf.constant(c[1][1], dtype=tf.int64, name="cifar10_test_labels")
    is_training=False

  


  # def _conv_activation(x):  # pylint: disable=invalid-name
  #   return tf.nn.max_pool(tf.nn.relu(x),
  #                         ksize=[1, 2, 2, 1],
  #                         strides=[1, 2, 2, 1],
  #                         padding="SAME")

  # conv = snt.nets.ConvNet2D(output_channels=conv_channels,
  #                           kernel_shapes=[5],
  #                           strides=[1],
  #                           paddings=[snt.SAME],
  #                           activation=_conv_activation,
  #                           activate_final=True,
  #                           initializers=_nn_initializers,
  #                           use_batch_norm=batch_norm)

  # if batch_norm:
  #   linear_activation = lambda x: tf.nn.relu(snt.BatchNorm()(x))
  # else:
  #   linear_activation = tf.nn.relu

  # mlp = snt.nets.MLP( [10],
  #                    activation=linear_activation,
  #                    initializers=_nn_initializers)
  # network = snt.Sequential([conv, snt.BatchFlatten(), mlp])
  if init=="uniform" or init=="hessian":
      initializers = {
          "w": tf.random_uniform_initializer(minval=-0.0196, maxval=0.0196, seed=None),
          "b": tf.random_uniform_initializer(minval=-0.0196, maxval=0.0196, seed=None),
      }
  else:
      initializers = _nn_initializers


  if activation == "sigmoid":
    activation_op = tf.sigmoid
  elif activation == "relu":
    activation_op = tf.nn.relu
  else:
    raise ValueError("{} activation not supported".format(activation))

  mlp = snt.nets.MLP(list(layers) + [10],
                     activation=activation_op,
                     initializers=initializers)
  network = snt.Sequential([snt.BatchFlatten(), mlp])


  if mode=='test':
    indices=tf.placeholder(tf.int64, shape=[batch_size])
    batch_images = tf.gather(images, indices)
    #batch_labels = tf.gather(labels, indices)
    output = network(batch_images)
    return [indices, output, len(c[1][1]), batch_size, np.reshape(c[1][1], (len(c[1][1],)))]
  def build():
    indices = tf.random_uniform([batch_size], 0, len(c[0][1]), tf.int64)
    batch_images = tf.gather(images, indices)
    batch_labels = tf.reshape(tf.gather(labels, indices), (batch_size,))

    print (batch_images, batch_labels)
    

    output = network(batch_images)

    print (output, "output")
    #exit(0)
    return _xent_loss(output, batch_labels) 

  return build


def cifar100(path,  # pylint: disable=invalid-name
            batch_norm=False,
            batch_size=128,
            num_threads=4,
            conv_channels=(16, 16, 16),
            linear_layers=None,
            min_queue_examples=1000,
            mode="train"):
  """Cifar10 classification with a convolutional network."""

  # Data.
  c = tf.keras.datasets.cifar100.load_data()
  #print (c[0][0].shape, c[0][1].shape, c[1][0].shape, c[1][1].shape, c[0][0][1])

  #_maybe_download_cifar10(path)
  if mode=='train':
    images = tf.constant(c[0][0]/255.0, dtype=tf.float32, name="cifar10_train_images")
    labels = tf.constant(c[0][1], dtype=tf.int64, name="cifar10_train_labels")
    is_training=True

  else:
    images = tf.constant(c[1][0]/255.0, dtype=tf.float32, name="cifar10_test_images")
    labels = tf.constant(c[1][1], dtype=tf.int64, name="cifar10_test_labels")
    is_training=False

  


  def _conv_activation(x):  # pylint: disable=invalid-name
    return tf.nn.max_pool(tf.nn.relu(x),
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding="SAME")

  conv = snt.nets.ConvNet2D(output_channels=conv_channels,
                            kernel_shapes=[5],
                            strides=[1],
                            paddings=[snt.SAME],
                            activation=_conv_activation,
                            activate_final=True,
                            initializers=_nn_initializers,
                            use_batch_norm=batch_norm)

  if batch_norm:
    linear_activation = lambda x: tf.nn.relu(snt.BatchNorm()(x))
  else:
    linear_activation = tf.nn.relu

  mlp = snt.nets.MLP( [100],
                     activation=linear_activation,
                     initializers=_nn_initializers)
  network = snt.Sequential([conv, snt.BatchFlatten(), mlp])


  if mode=='test':
    indices=tf.placeholder(tf.int64, shape=[batch_size])
    batch_images = tf.gather(images, indices)
    #batch_labels = tf.gather(labels, indices)
    output = network(batch_images)
    return [indices, output, len(c[1][1]), batch_size, np.reshape(c[1][1], (len(c[1][1],)))]
  def build():
    indices = tf.random_uniform([batch_size], 0, len(c[0][1]), tf.int64)
    batch_images = tf.gather(images, indices)
    batch_labels = tf.reshape(tf.gather(labels, indices), (batch_size,))

    print (batch_images, batch_labels)
    

    output = network(batch_images)

    print (output, "output")
    #exit(0)
    return _xent_loss(output, batch_labels) 

  return build

def vgg16_cifar10(path,  # pylint: disable=invalid-name
            batch_norm=False,
            batch_size=128,
            num_threads=4,
            min_queue_examples=1000,
            mode="train"):
    """Cifar10 classification with a convolutional network."""
    
    # Data.
    _maybe_download_cifar10(path)
    # pdb.set_trace()
    # Read images and labels from disk.
    if mode == "train":
        filenames = [os.path.join(path,
                                  CIFAR10_FOLDER,
                                  "data_batch_{}.bin".format(i))
                     for i in xrange(1, 6)]
        is_training = True
    elif mode == "test":
        filenames = [os.path.join(path, CIFAR10_FOLDER, "test_batch.bin")]
        is_training = False
    else:
        raise ValueError("Mode {} not recognised".format(mode))
    
    depth = 3
    height = 32
    width = 32
    label_bytes = 1
    image_bytes = depth * height * width
    record_bytes = label_bytes + image_bytes
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    _, record = reader.read(tf.train.string_input_producer(filenames))
    record_bytes = tf.decode_raw(record, tf.uint8)
    
    label = tf.cast(tf.slice(record_bytes, [0], [label_bytes]), tf.int32)
    raw_image = tf.slice(record_bytes, [label_bytes], [image_bytes])
    image = tf.cast(tf.reshape(raw_image, [depth, height, width]), tf.float32)
    # height x width x depth.
    image = tf.transpose(image, [1, 2, 0])
    image = tf.math.divide(image, 255)

    queue = tf.RandomShuffleQueue(capacity=min_queue_examples + 3 * batch_size,
                                  min_after_dequeue=min_queue_examples,
                                  dtypes=[tf.float32, tf.int32],
                                  shapes=[image.get_shape(), label.get_shape()])
    enqueue_ops = [queue.enqueue([image, label]) for _ in xrange(num_threads)]
    tf.train.add_queue_runner(tf.train.QueueRunner(queue, enqueue_ops))

    vgg = VGG16(0.5, 10)
    def build():
        image_batch, label_batch = queue.dequeue_many(batch_size)
        label_batch = tf.reshape(label_batch, [batch_size])
        # pdb.set_trace()
        output = vgg._build_model(image_batch)
        # print(output.shape)
        return _xent_loss(output, label_batch)
    
    return build
