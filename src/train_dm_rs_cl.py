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
"""Learning 2 Learn training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange
import tensorflow as tf

from tensorflow.contrib.learn.python.learn import monitored_session as ms
import pickle
import meta_rs_cl as meta
import util
import os
import random
from timeit import default_timer as timer
import pdb
# from data_generator import data_loader

flags = tf.flags
logging = tf.logging

gpu_options = tf.GPUOptions(allow_growth=True)
#sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

FLAGS = flags.FLAGS
flags.DEFINE_string("save_path", None, "Path for saving meta-optimizer.")
flags.DEFINE_string("restore_path", None, "Path for restoring stage 1 meta-optimizer.")
flags.DEFINE_integer("num_epochs", 5000, "Number of training epochs.")
flags.DEFINE_integer("log_period", 1, "Log period.")
flags.DEFINE_integer("evaluation_period", 100, "Evaluation period.")
flags.DEFINE_integer("evaluation_epochs", 20, "Number of evaluation epochs.")

flags.DEFINE_string("problem", None, "Type of problem.")
flags.DEFINE_integer("num_steps", 100, "Number of optimization steps per epoch.")
flags.DEFINE_integer("unroll_length", 20, "Meta-optimizer unroll length.")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
flags.DEFINE_boolean("second_derivatives", False, "Use second derivatives.")

flags.DEFINE_float("rd_scale_bound", 3.0, "Bound for random scaling on the main optimizee.")
flags.DEFINE_string("optimizers", "adam", ".")
flags.DEFINE_integer("k", 1, "")

flags.DEFINE_string("if_scale", 'False', "")
flags.DEFINE_string("if_cl", 'False', "")
flags.DEFINE_string("init", "normal", ".")
flags.DEFINE_string("bayesian", 'True', '.')
flags.DEFINE_integer("stage", 2, ".")

flags.DEFINE_float("lambda1", 0.1, ".")

def main(_):


  def training_stage(stage):
    
    if_scale = FLAGS.if_scale
    if_cl = FLAGS.if_cl
  # Configuration for curiculum learning. Only used for training neural networks.
    if if_cl=='True':
      num_steps = [100, 200, 500, 1000, 1500, 2000, 2500, 3000]
      num_unrolls = [int(ns/FLAGS.unroll_length) for ns in num_steps]
      num_unrolls_eval = num_unrolls

      curriculum_idx = 0
    else:
      num_unrolls  = FLAGS.num_steps // FLAGS.unroll_length



    # Problem.
    problem, net_config, net_assignments = util.get_config(FLAGS.problem, init=FLAGS.init)

    
    # YC add:
    if len(net_config)!=1:
      raise ValueError("Default net_assignments can only be used if there is "
                         "a single net config.")
    else:
      for key in net_config:
        net_config[key]['net_options']['bayesianornot'] = FLAGS.bayesian=='True'
        # training stage 1:
        net_config[key]['net_options']['stage'] = stage


    # Optimizer setup.
    optimizer = meta.MetaOptimizer(**net_config)
    minimize, scale, var_x, constants, subsets,\
      lambda1, optimizer_op = optimizer.meta_minimize(
        problem, FLAGS.unroll_length,
        learning_rate=FLAGS.learning_rate, lambda2=FLAGS.lambda1,
        net_assignments=net_assignments,
        second_derivatives=FLAGS.second_derivatives)
    step, loss_kl_1, loss_kl_2, update, reset, cost_op, _ = minimize

    p_val_x = []
    for k in var_x:
      p_val_x.append(tf.placeholder(tf.float32, shape=k.shape))
    assign_ops = [tf.assign(var_x[k_id], p_val_x[k_id]) for k_id in range(len(p_val_x))]

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    start_time = timer()


    with tf.Session() as sess:
      def assign_func(val_x): 
        sess.run(assign_ops, feed_dict={p: v for p, v in zip(p_val_x, val_x)})

      #YC add:  initialize optimizer parameters:
      init_optimizer_vars = tf.variables_initializer(tf.trainable_variables()+optimizer_op.variables())
      sess.run(init_optimizer_vars)
     


      if stage==2:
        optimizer.restore(sess, FLAGS.restore_path)

        bayesian_sampling_vars=[]
        for var in tf.get_collection(tf.GraphKeys.VARIABLES):
          
          if 'epislon' in var.name :
            bayesian_sampling_vars.append(var)
            print (var)

        bayesian_sampling = tf.variables_initializer(bayesian_sampling_vars)
        reset.append(bayesian_sampling)

      for variable in tf.trainable_variables():
          print (variable, sess.run(variable)[0])


      best_evaluation = float("inf")
      train_loss_record = []
      eval_loss_record = []
      num_steps_train = []

      for e in xrange(FLAGS.num_epochs):
        if if_cl=='True':
          num_unrolls_cur = num_unrolls[curriculum_idx]
        else:
          num_unrolls_cur = num_unrolls
        time, cost = util.run_epoch(sess, cost_op, [update, step, loss_kl_1, loss_kl_2], reset,
                                    num_unrolls_cur,
                                    scale=scale,
                                    rd_scale= if_scale=='True',
                                    rd_scale_bound=FLAGS.rd_scale_bound,
                                    assign_func=assign_func,
                                    var_x=var_x,
                                    num_epoch = [e, FLAGS.num_epochs],
                                    lambda1=lambda1)
        train_loss_record.append(cost)
        
        # Evaluation.
        if (e+1) % FLAGS.evaluation_period == 0:
          if if_cl=='True':
            num_unrolls_eval_cur = num_unrolls_eval[curriculum_idx]
          else:
            num_unrolls_eval_cur = num_unrolls
          

          eval_cost = 0
          for _ in xrange(FLAGS.evaluation_epochs):
            sess.run(reset)
            time, cost = util.run_eval_epoch(sess, cost_op, update, reset,
                                        num_unrolls_eval_cur)
            eval_cost += cost[-1]

          print (cost)
          print ("epoch={}, num_steps={}, eval loss={}".format(e, num_unrolls_eval_cur, eval_cost/FLAGS.evaluation_epochs), flush=True)
          eval_loss_record.append(eval_cost/FLAGS.evaluation_epochs)                   


          if eval_cost/FLAGS.evaluation_epochs < best_evaluation:
            best_evaluation = eval_cost/FLAGS.evaluation_epochs
            
            # save model
            print ("update best optmizer")
            optimizer.save(sess, FLAGS.save_path)  

            # update curriculum
            if  if_cl=='True':
              curriculum_idx += 1
              if curriculum_idx >= len(num_unrolls):
                curriculum_idx-= 1
            


      print ("total time = {}s...".format(timer()-start_time))
      # output
      with open(FLAGS.save_path+"_log_.pickle", 'wb') as l_record:
        records = {"eval_loss": eval_loss_record,
                    "train_loss": train_loss_record,
                    'best_eval': best_evaluation,
                    }
        pickle.dump(records, l_record)


  training_stage(FLAGS.stage)


if __name__ == "__main__":
  tf.app.run()
