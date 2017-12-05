import tensorflow as tf
import numpy as np
import random

class CnnCore:
  def __weight_init(self, shape):
    n = np.prod(shape)
    return tf.Variable(tf.random_normal(shape, stddev=1) * np.sqrt(2/n))
  
  def __read_weight(self, shape, filename):
    """
    Each line should contain one filter. The weight ordering of each filter is R, G, B, R, G, B, ... from left to right, from top to bottom.
    """
    init_weight = np.zeros(shape)
    if (len(shape) == 4):
      # Shape of filter
      row_stride = shape[1] * shape[2]
      with open(filename, 'r') as fp:
        lines = fp.readlines()
        f_idx = 0
        for filt in lines:
          weight = filt.split()
          idx = 0
          for w in weight:
            init_weight[int(idx / row_stride), int(idx / shape[2]) % shape[1], idx % shape[2], f_idx] = w
            idx += 1
          f_idx += 1
    elif (len(shape) == 2):
      # Shape of fully connected layer
      with open(filename, 'r') as fp:
        lines = fp.readlines()
        idx = 0
        for line in lines:
          weight = line.split()
          nxt_idx = 0
          for w in weight:
            init_weight[idx, nxt_idx] = w
            nxt_idx += 1
          idx += 1
    return tf.Variable(tf.convert_to_tensor(init_weight, dtype=tf.float32))

  def __sparse_init(self, shape):
    numPick = int(np.ceil(np.sqrt(shape[0])))  # Sqrt of number of previous neurons

    indices = []
    for i in range(shape[1]):
      candidates = list(range(0, shape[0]))
      indices += list(map(lambda x, y: (x, y), random.sample(candidates, numPick), [i] * numPick))

    values = list(self.rng.randn(len(indices)))

    W = tf.SparseTensor(indices, values, shape)

    return tf.Variable(tf.sparse_tensor_to_dense(W, default_value=0, validate_indices=False))


  def __bias_init(self, shape):
    return tf.Variable(tf.constant(0.0, shape=shape))


  def __conv2d(self, x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='VALID') # No zero padding


  def __init__(self, dataset_name, channel, height, num_classes, trainBatchSize, seed=528):
    self.seed = seed
    self.CHANNEL = channel
    self.HT = height
    self.NUM_CLASSES = num_classes
    self.TRAIN_BATCH_SIZE = trainBatchSize
    self.rng = np.random.RandomState(self.seed)

    self.W1 = self.__weight_init([4, 4, self.CHANNEL, 32])
    self.W2 = self.__weight_init([4, 4, 32, 32])
    self.W3 = self.__weight_init([4, 4, 32, 64])
    self.W_fc1 = self.__sparse_init([int((self.HT - 4)/2 - 5)**2 * 64, self.NUM_CLASSES])


  def inference(self, x_imgs):
    # NN model
    h1_conv = tf.nn.relu(tf.nn.conv2d(x_imgs, self.W1, strides=[1,2,2,1], padding='VALID')) # No zero padding
    h2_conv = tf.nn.relu(self.__conv2d(h1_conv, self.W2))
    h3_conv = tf.nn.relu(self.__conv2d(h2_conv, self.W3))
    h3_conv_flat = tf.reshape(h3_conv, [-1, int((self.HT - 4)/2 - 5)**2 * 64])
    y = tf.matmul(h3_conv_flat, self.W_fc1)
    return y


  def loss(self, y, y_):
    """
    Arg:
      y: activation
      y_: one hot label vector
  
    """
    self.y = y
    self.y_ = y_

    C = self.TRAIN_BATCH_SIZE
    loss = (tf.reduce_sum(
        tf.square(y - y_)) + 
            tf.nn.l2_loss(self.W1)
            + tf.nn.l2_loss(self.W2) 
            + tf.nn.l2_loss(self.W3)
            + tf.nn.l2_loss(self.W_fc1) ) / C
    return loss


  def test_loss(self, y, y_):
    self.y = y
    self.y_ = y_
    batch_test_loss = tf.reduce_sum(tf.square(y - y_))
    regularize = tf.reduce_sum( tf.nn.l2_loss(self.W1)
            + tf.nn.l2_loss(self.W2) 
            + tf.nn.l2_loss(self.W3)
            + tf.nn.l2_loss(self.W_fc1) ) 
    return batch_test_loss, regularize


  def train(self, loss, lr, mmt, global_step):
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
    #optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=mmt, use_nesterov=True)
    optimizer = tf.train.AdagradOptimizer(learning_rate=lr)
    #optimizer = tf.train.RMSPropOptimizer(learning_rate=lr)
    train_step = optimizer.minimize(loss)
    return train_step

