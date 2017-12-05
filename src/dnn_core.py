import tensorflow as tf
import numpy as np
import random

class DnnCore:
  def __weight_init(self, shape):
    n = np.prod(shape)
    return tf.Variable(tf.random_normal(shape, stddev=1) * np.sqrt(2/n))
  
  def __read_weight(self, shape, filename):
    init_weight = np.zeros(shape)
    if (len(shape) == 4):
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


  def __init__(self, dataset_name, net_struct, trainBatchSize, seed=528):
    self.net_struct = net_struct
    self.seed = seed
    self.TRAIN_BATCH_SIZE = trainBatchSize
    self.rng = np.random.RandomState(self.seed)
    self.W = [0] * len(net_struct)

    try:
        for i in range(len(net_struct)-1):
            self.W[i] = self.__sparse_init([net_struct[i], net_struct[i+1]])
    except Exception as e:
      print(e)
      exit()


  def inference(self, x):
    # NN model
    z = x
    for i in range(len(self.net_struct)-2):
        s = tf.matmul(z, self.W[i])
        z = tf.nn.sigmoid(s)
    y = tf.matmul(z, self.W[len(self.net_struct) - 2])
    return y


  def loss(self, y, y_):
    """
    Arg:
      y: activation
      y_: one hot label vector
  
    """
    self.y = y
    self.y_ = y_
    net_struct = self.net_struct
    C = self.TRAIN_BATCH_SIZE

    loss = tf.reduce_sum(tf.square(y - y_))
    for i in range(len(net_struct)-1):
        loss += tf.nn.l2_loss(self.W[i])
    loss /= C
    return loss


  def test_loss(self, y, y_):
    self.y = y
    self.y_ = y_
    net_struct = self.net_struct
    loss = 0

    batch_test_loss = tf.reduce_sum(tf.square(y - y_))
    for i in range(len(net_struct)-1):
        loss += tf.nn.l2_loss(self.W[i])
    regularize = tf.reduce_sum( loss )
    return batch_test_loss, regularize


  def train(self, loss, lr, mmt, global_step):
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
    #optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=mmt, use_nesterov=True)
    optimizer = tf.train.AdagradOptimizer(learning_rate=lr)
    #optimizer = tf.train.RMSPropOptimizer(learning_rate=lr)
    train_step = optimizer.minimize(loss)
    return train_step

