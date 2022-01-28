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
"""Learning 2 Learn evaluation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle

from six.moves import xrange
import tensorflow as tf

from tensorflow.contrib.learn.python.learn import monitored_session as ms
import numpy as np
import meta_rs_cl
import util
import pdb

flags = tf.flags
logging = tf.logging


FLAGS = flags.FLAGS
flags.DEFINE_string("path", None, "Path to saved meta-optimizer network.")
flags.DEFINE_string("output", None, "Path to the results.")
flags.DEFINE_string("problem", "simple", "Type of problem.")
flags.DEFINE_integer("num_steps", 100,
                     "Number of optimization steps per epoch.")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
flags.DEFINE_string("bayesian", 'True', '.')
flags.DEFINE_integer("run", 10000, "The number of i.i.d runs")
flags.DEFINE_integer("stage", 2, "")
flags.DEFINE_string("mode", 'test', '.')

def main(_):
  # Configuration.
  num_steps = FLAGS.num_steps
  tf.set_random_seed(22)
  if FLAGS.output == None:
    raise ValueError("Must specify an output path.")


  # Problem.
  # here problem is 2d list [indices, outputs]
  problem, net_config, net_assignments = util.get_config(FLAGS.problem, mode=FLAGS.mode)

  for key in net_config:
      net_config[key]['net_options']['bayesianornot'] = FLAGS.bayesian=='True'
      net_config[key]['net_options']['stage'] = FLAGS.stage



  # Optimizer setup.
    
  if FLAGS.path == None:
    raise ValueError("Must specify a path to the BL2O model.")

  optimizer = meta_rs_cl.MetaOptimizer(**net_config)
  meta_loss,_, problem_vars, _, _, _ = optimizer.meta_loss(problem, 1, net_assignments=net_assignments)

  loss, _, _, update, reset, cost_op, x_final = meta_loss
  for v in tf.trainable_variables():
    print (v)   


  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True

  if FLAGS.problem=='privacy_attack':
    x_true = np.load("../data/privacy_test_x.npy")
  elif FLAGS.problem == 'protein_dock':
    with open("../data/protein_dock/test_data.pkl" , "rb") as f:
        data = pickle.load(f)

#  Start session------------------------------------------------
  with tf.Session() as sess:

    # restore trained model
    optimizer.restore(sess, FLAGS.path)
    bayesian_sampling_vars=[]

    # put the epislon variable into the reset operator. The reset operator will be ran before each optimization trajectory.  So before each optimization
    # trajectory, we sample epislon once which means that we sample an optimizer from the posterior.
    for var in tf.get_collection(tf.GraphKeys.VARIABLES):
      
      if 'epislon' in var.name :
        bayesian_sampling_vars.append(var)
        print (var)

    bayesian_sampling = tf.variables_initializer(bayesian_sampling_vars)
    reset.append(bayesian_sampling)

    optimizee_vars = []
    x_best=[]

    if FLAGS.problem == 'protein_dock':
      fmin=1000000
    
    # We will FLAGS.run i.i.d runs. 
    for i in range(FLAGS.run):
      print ('run: ', i)
      sess.run(reset)

      total_time = 0
      total_cost = 0
      loss_record = []
      x_record = []

      for ite in xrange(FLAGS.num_steps):
        cost ,_= sess.run([cost_op,update])
        loss_record.append(cost)

        if FLAGS.problem == "privacy_attack" or FLAGS.problem=='protein_dock':
          x_record.append(sess.run(x_final[0]))
        else:
          x_record.append(sess.run(x_final[0])[0])

      best_idx = np.argmin(loss_record)

      if FLAGS.problem=='protein_dock':
        x = np.expand_dims(np.expand_dims(x_record[best_idx], axis=1), axis=1)
        dx = np.matmul(x, data['basic'].astype(np.float32))
        dx = np.squeeze(dx, 2)
        x = dx + data['init']

      if FLAGS.problem == "privacy_attack" and ( np.any(x_record[best_idx]>1)  or np.any(x_record[best_idx]<-1)):
        i-=1
        continue
      
      if FLAGS.problem == 'protein_dock':
        optimizee_vars.append(x)
        if min(loss_record) < fmin:
          fmin = min(loss_record)
          best_optimizee_vars = x

      else:
        optimizee_vars.append(x_record[best_idx])


      if FLAGS.problem == "privacy_attack":
        x_best.append(np.mean( np.sqrt(np.sum((x_record[best_idx] - x_true)**2, axis=1))))
      elif FLAGS.problem == 'protein_dock':
        x_best.append( np.mean( np.sqrt(np.sum((x - data['bound'])**2, axis=2)), axis=1))
      else:
        x_best.append(np.sqrt( np.sum(x_record[best_idx]**2) ))
      # print ('best x according to min(f(x))', x_record[best_idx])
      print ('dis_to_the_origin: ', x_best[-1])


#------------------------------------------save results
  if FLAGS.problem == 'protein_dock':
    test_list=['1AHW_3', '1AK4_7', '3CPH_7', '1HE8_3', '1JMO_4']
    with open(FLAGS.output, "w") as f:
      for idx,pdb in enumerate(test_list):
        
        dis = []
        for i in range(len(x_best)):
            dis.append( np.mean(np.sqrt( np.sum( ( best_optimizee_vars[idx] - optimizee_vars[i][idx])  **2, axis=1))))
    
        print (pdb+" r90:  %.3f %.3f" %(np.percentile(dis, 5), np.percentile(dis, 95)))
        print (pdb+" ||x-x*||: %.3f"  %(np.mean( np.sqrt(np.sum((best_optimizee_vars[idx] - data['bound'][idx])**2, axis=-1)))))

        f.write(pdb+" r90:  %.3f %.3f\n" %(np.percentile(dis, 5), np.percentile(dis, 95)))
        f.write(pdb+" ||x-x*||: %.3f\n"  %(np.mean( np.sqrt(np.sum((best_optimizee_vars[idx] - data['bound'][idx])**2, axis=-1)))))
        f.write("||x-x*|| samples:\n")
        for i in dis:
          f.write("%.3f " %(i))
        f.write('\n\n\n')

    return 
  # overall best:
  dis = []

  if FLAGS.problem == "privacy_attack":
    for i in range(len(x_best)):
          dis.append( np.mean(np.sqrt( np.sum( ( np.mean(optimizee_vars, axis=0) - optimizee_vars[i])  **2, axis=1))))
  
  else:
    for i in range(len(x_best)):
          dis.append(np.sqrt( np.sum( ( np.mean(optimizee_vars, axis=0) - optimizee_vars[i])  **2)))

  s=0.
  s1=0.
  for d in x_best:
      if d >=np.percentile(dis, 5) and d <=np.percentile(dis, 95):
            s+=1
      if d >=np.percentile(dis, 10) and d <=np.percentile(dis, 90):
            s1+=1


  print("e90: %.3f" %(s/ len(optimizee_vars)))
  print("e80: %.3f" %(s1/len(optimizee_vars)))
  print ("r90:  %.3f" %(np.percentile(dis, 95) - np.percentile(dis, 5)))
  print ("r80:  %.3f" %(np.percentile(dis, 90) - np.percentile(dis, 10)))
  print ("||x-x*||: mean(std) %.3f(%.3f)"  %(np.mean(x_best), np.std(x_best)))


  if FLAGS.stage == 2:
    np.save('./uq_calibration/' + FLAGS.problem + '_opt.npy', np.array(x_best))
    np.save('./uq_calibration/' + FLAGS.problem + '_dis.npy', np.array(dis))
  else:
    np.save('./uq_calibration_dmlstm/' + FLAGS.problem + '_opt.npy', np.array(x_best))
    np.save('./uq_calibration_dmlstm/' + FLAGS.problem + '_dis.npy', np.array(dis))


  with open(FLAGS.output, "w") as f:
    f.write("e90: %.3f\n" %(s/len(optimizee_vars)))
    f.write("e80: %.3f\n" %(s1/len(optimizee_vars)))
    f.write ("r90:  %.3f\n" %(np.percentile(dis, 95) - np.percentile(dis, 5)))
    f.write ("r80:  %.3f\n" %(np.percentile(dis, 90) - np.percentile(dis, 10)))
    f.write ("||x-x*||: mean(std) %.3f(%.3f)\n"  %(np.mean(x_best), np.std(x_best)))

  return 



#########################################################
###  The rest here is for deep learning model evaluation:
#########################################################

  # #   predict on the test set:
  # tf.reset_default_graph()
  # problem, _, _ = util.get_config(FLAGS.problem, mode='test')
  # problem_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

  # input_optimizee_vars=[]
  # assign =[]
  # for var in problem_vars:
  #   c=tf.placeholder(tf.float32, shape=var.shape)
  #   input_optimizee_vars.append(c)
  #   assign.append(tf.assign(var, c))
  #   print (var)

  # indices = problem[0] # tf placeholder tensor shape [batch_size]
  # output = problem[1]  # tf tensor shape [batch_size, #catogory]
  # num_samples = problem[2] #int64
  # batch_size=problem[3]    # int64
  # labels = problem[4]      #  numpy array [num_samples, #catogory]

  # with tf.Session() as sess:

  #   ensemble_outputs=[]
  #   ensemble_loss_record=[]
  #   for one_set_of_optimizee_vars in optimizee_vars:
      
  #     feed_dict={}
  #     for c1, c2 in zip(one_set_of_optimizee_vars, input_optimizee_vars):

  #       feed_dict[c2] =c1


  #     total_ite = int(num_samples/batch_size)
  #     outputs = []
  #     for ite in range(total_ite):
  #         #feed_dict[indices] =  np.arange(ite*batch_size, (ite+1)*batch_size)
  #         sess.run(assign, feed_dict=feed_dict)
  #         out = sess.run(output, feed_dict={indices : np.arange(ite*batch_size, (ite+1)*batch_size)})
  #         outputs+= out.tolist()

  #         print ('pred ite: %d/%d' %(ite+1, total_ite))

  #     ensemble_outputs.append(outputs)
  #     ensemble_loss_record.append(loss_record)



  # final_pred = np.mean(ensemble_outputs, axis=0)
  # final_labels = labels[:total_ite*batch_size]
  # print (final_pred.shape, final_labels.shape)

  # if FLAGS.optimizer != 'L2L':
  #   c=FLAGS.optimizer
  # else:
  #   c=FLAGS.path.split('/')[-1]

  # #print (FLAGS.path.split('/'),c)
  # result_path = 'Result/'+c+'_'+FLAGS.problem
  # with open(result_path, "wb") as f:
  #   pickle.dump({"final_pred": final_pred, "final_labels": final_labels, "ensemble_loss_record": ensemble_loss_record}, f)





if __name__ == "__main__":
  tf.app.run()
