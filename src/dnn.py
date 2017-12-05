import re
import os
import tensorflow as tf
import numpy as np
import argparse
import time
import sys
from dnn_core import DnnCore
import dnn_data_engine as de

def ArgsParser():
  class MyParser(argparse.ArgumentParser):
    def error(self, message):
      sys.stderr.write('error: %s\n' % message)
      self.print_help()
      sys.exit(2)

  parser = MyParser()
  parser.add_argument('-ht', '--height', help='height of the input images', default='28', type=int)
  parser.add_argument('-wt', '--weight', help='weight of the input images', default='28', type=int)
  parser.add_argument('-c', '--channel', help='channel of the input images', default='1', type=int)
  parser.add_argument('-dn', '--dataset_name', help='It should be the first substring before \'.\' of the LIBSVM data set file. For example, mnist, svhn', required=True, type=str)
  parser.add_argument('-lr', '--learning_rate', help='learning rate', default='5e-2', type=float)
  parser.add_argument('-b', '--batch_size', help='batch size', default='100', type=int)
  parser.add_argument('-mmt', '--momentum', help='momentum', default='0.9', type=float)
  parser.add_argument('-train', '--train_path', help='specify train path', default='../data/mnist/mnist.scale', type=str)
  parser.add_argument('-test', '--test_path', help='specify test path', default='../data/mnist/mnist.scale.t', type=str)
  args = parser.parse_args()
  return args

if __name__ == '__main__':
  args = ArgsParser()

  print("SG setting: lr=%lf, batch=%d" % (args.learning_rate, args.batch_size))

  # Hyperparameters
  CHANNEL = args.channel
  HEIGHT = args.height
  WIDTH = args.weight
  NUM_CLASSES = 10

  EPOCH = 360000
  TRAIN_BATCH_SIZE = args.batch_size
  TEST_BATCH_SIZE = 128

  dataset_name = args.dataset_name
  train_path = args.train_path
  test_path = args.test_path

  # Initialization
  counter = 0
  elapsed_time = 0.0
  dataset = de.read_dataset(dataset_name, train_path, test_path, NUM_CLASSES, CHANNEL, HEIGHT, WIDTH, one_hot=True) # Libsvm format dataset

  # Input format
  x = tf.placeholder(tf.float32, shape=[None, HEIGHT * WIDTH * CHANNEL])
  y_ = tf.placeholder(tf.float32, shape=[None, NUM_CLASSES])

  # Specify nn architecture
  inp = CHANNEL * HEIGHT * WIDTH
  clan = 10
  net_struct = [inp, clan, NUM_CLASSES]
  model = DnnCore(dataset_name, net_struct, TRAIN_BATCH_SIZE)
  y = model.inference(x)

  # Specify loss
  loss = model.loss(y, y_)
  batch_test_loss, regularize = model.test_loss(y, y_)
  
  # Specify optimizer
  global_step = tf.contrib.framework.get_or_create_global_step() # Redundant variable
  train_step = model.train(loss, args.learning_rate, args.momentum, global_step)
  
  # Evaluation
  correct_predict = tf.equal( tf.argmax(y, 1), tf.argmax(y_, 1) )
  accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))
  batch_size = tf.size(correct_predict)
  batch_correct_predict = tf.reduce_sum(tf.cast(correct_predict, tf.int32))
  
  # Train
  #with tf.device('/gpu:0'):
  with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    for i in range(EPOCH):

      batch = dataset.train.next_batch(TRAIN_BATCH_SIZE)

      # Do Logging for each 100 times of training steps
      if i % 100 == 0:
        # Evaluate train batch
        [train_loss, train_acc, train_correct_predict, train_batch_size] = sess.run([loss, accuracy, batch_correct_predict, batch_size], feed_dict={x: batch[0], y_: batch[1]})

        # Evaluate test data via pipeline
        correct_test_predict = 0.0; test_batch_size = 0; test_loss = 0.0
        flag = dataset.test.epochs_completed 
        while True:
          test_batch = dataset.test.next_batch(TEST_BATCH_SIZE, withoutMixWithNextEpoch=True, shuffle=False)
          test_predict_result = sess.run([batch_correct_predict, batch_size, batch_test_loss], feed_dict={x: test_batch[0], y_: test_batch[1]})
          correct_test_predict += test_predict_result[0]
          test_batch_size += test_predict_result[1]
          test_loss += test_predict_result[2]
          if dataset.test.epochs_completed - flag != 0:
              break

        # Compute regularization term
        reg = sess.run(regularize, feed_dict={x: test_batch[0], y_: test_batch[1]})

        test_acc = correct_test_predict / test_batch_size
        test_loss = (test_loss + reg) / test_batch_size

        # Timing
        avg_elapsed_time = 0 if i == 0 else elapsed_time / counter

        # Logging
        print("{ 'iter': %d, 'train_loss': %g, 'train accuracy': %g, 'train_correct_predict': %g, 'train_batch_size': %g, 'test_loss': %g, 'test acc': %g, 'correct_test_predict': %d, 'test_batch_size': %d, 'avg_elapsed_time': %g }" % (i, train_loss, train_acc, train_correct_predict, train_batch_size, test_loss, test_acc, int(correct_test_predict), test_batch_size, avg_elapsed_time))

        counter = 0
        elapsed_time = 0.

      # Training
      counter += 1
      t = time.process_time()
      sess.run(train_step, feed_dict={x: batch[0], y_: batch[1]})
      elapsed_time += time.process_time() - t

