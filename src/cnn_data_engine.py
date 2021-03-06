# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
#
# Modifications copyright (C) 2017 Kent Loong Tan
# ==============================================================================

import time
import numpy
from tensorflow.python.framework import random_seed
from tensorflow.contrib.learn.python.learn.datasets import base

try:
    import cPickle as pickle
except:
    import pickle as pickle

class DataSet(object):

  def __init__(self,
               images,
               labels,
               fake_data=False,
               one_hot=False,
               scaling=False,
               reshape=True,
               seed=None):
    """Construct a DataSet.
    one_hot arg is used only if fake_data is true.  When `scaling` is true,
    it scales the input from `[0, 255]` into `[0, 1]`.
    """
    seed1, seed2 = random_seed.get_seed(seed)
    # If op level seed is not set, use whatever graph level seed is returned
    numpy.random.seed(seed1 if seed is None else seed2)
    if fake_data:
      self._num_examples = 10000
      self.one_hot = one_hot
    else:
      assert images.shape[0] == labels.shape[0], (
          'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
      self._num_examples = images.shape[0]

      # Convert shape from [num examples, rows, columns, depth]
      # to [num examples, rows*columns*depth] (assuming depth == 1)
      if reshape:
        #assert images.shape[3] == 3
        images = images.reshape(images.shape[0],
                                images.shape[1] * images.shape[2] * images.shape[3])
      if scaling:
        # Convert from [0, 255] -> [0.0, 1.0].
        images = images.astype(numpy.float32)
        images = numpy.multiply(images, 1.0 / 255.0)
    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size, withoutMixWithNextEpoch=False, shuffle=True):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    # Shuffle for the first epoch
    if self._epochs_completed == 0 and start == 0 and shuffle:
      perm0 = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm0)
      self._images = self.images[perm0]
      self._labels = self.labels[perm0]
    # Go to the next epoch
    if start + batch_size > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Get the rest examples in this epoch
      rest_num_examples = self._num_examples - start
      images_rest_part = self._images[start:self._num_examples]
      labels_rest_part = self._labels[start:self._num_examples]
      # Shuffle the data
      if shuffle:
        perm = numpy.arange(self._num_examples)
        numpy.random.shuffle(perm)
        self._images = self.images[perm]
        self._labels = self.labels[perm]
      if withoutMixWithNextEpoch:
        self._index_in_epoch = 0
        return images_rest_part, labels_rest_part
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size - rest_num_examples
      end = self._index_in_epoch
      images_new_part = self._images[start:end]
      labels_new_part = self._labels[start:end]
      return numpy.concatenate((images_rest_part, images_new_part), axis=0) , numpy.concatenate((labels_rest_part, labels_new_part), axis=0)
    else:
      self._index_in_epoch += batch_size
      end = self._index_in_epoch
      return self._images[start:end], self._labels[start:end]

def svm_read_problem(data_file_name, num_features, return_scipy=False):
    """
    svm_read_problem(data_file_name, return_scipy=False) -> [y, x], y: list, x: list of dictionary
    svm_read_problem(data_file_name, return_scipy=True)  -> [y, x], y: ndarray, x: csr_matrix

    Read LIBSVM-format data from data_file_name and return labels y
    and data instances x.
    """
    prob_y = [] 
    prob_x = []
    row_ptr = [0]
    col_idx = []
    i = 0
    with open(data_file_name) as fp:
        i = 0
        lines = fp.readlines()
        for line in lines:
            line = line.split(None, 1)
            # In case an instance with all zero features
            if len(line) == 1: line += ['']
            label, features = line

            idx = 1
            xi = [0.0] * num_features
            for e in features.split():
                ind, val = e.split(":")
                if int(ind) == idx:
                    xi[idx-1] = float(val)
                else:
                    while (idx < int(ind)):
                        idx += 1
                    xi[idx-1] = float(val)
                idx += 1
            prob_x += [xi]
            prob_y += [float(label)]
            i += 1
    return (prob_y, prob_x)

def read_dataset(dataset_name,
                   train_path,
                   test_path,
                   num_classes,
                   channel,
                   height,
                   width,
                   one_hot=False,
                   y_label_offset=0,
                   scaling=False,
                   reshape=True,
                   validation_size=0,
                   seed=None):
    # Read data in LIBSVM format
    num_features = channel * height * width

    # Read libsvm data
    try:
        print('Read data from `./data/' + dataset_name + '/train_data.pkl`...')
        with open('data/' + dataset_name + '/train_data.pkl', 'rb') as filehandler:
            (y_train, x_train) = pickle.load(filehandler)
    except:
        print('(No such file or directory: `./data/' + dataset_name + '/train_data.pkl`)')
        print('Read data from ' + train_path + '...')
        y_train, x_train = svm_read_problem(train_path, num_features)
        x_train = numpy.array(x_train); y_train = numpy.array(y_train)

    try:
        print('Read data from `./data/' + dataset_name + '/test_data.pkl`...')
        with open('data/' + dataset_name + '/test_data.pkl', 'rb') as filehandler:
            (y_test, x_test) = pickle.load(filehandler)
    except:
        print('(No such file or directory: `./data/' + dataset_name + '/test_data.pkl`)')
        print('Read data from ' + test_path + '...')
        y_test, x_test = svm_read_problem(test_path, num_features)
        x_test = numpy.array(x_test); y_test = numpy.array(y_test)

    train_images = data_reformat(x_train, channel, height, width)
    test_images = data_reformat(x_test, channel, height, width)

    if y_label_offset != 0:
        y_test -= y_label_offset

    if one_hot:
        train_labels = dense_to_one_hot(y_train, num_classes)
        test_labels = dense_to_one_hot(y_test, num_classes)
    else:
        train_labels = y_train
        test_labels = y_test

    if not 0 <= validation_size <= len(train_images):
        raise ValueError(
            'Validation size should be between 0 and {}. Received: {}.'
            .format(len(train_images), validation_size))

    validation_images = train_images[:validation_size]
    validation_labels = train_labels[:validation_size]
    train_images = train_images[validation_size:]
    train_labels = train_labels[validation_size:]

    options = dict(scaling=scaling, reshape=reshape, seed=seed)

    train = DataSet(train_images, train_labels, **options)
    validation = DataSet(validation_images, validation_labels, **options)
    test = DataSet(test_images, test_labels, **options)
  
    return base.Datasets(train=train, validation=validation, test=test)

def data_reformat(images, channel, height, width):
    """ Convert images from [image_index, channel * height * width] to [image_index, height, width, channel]"""
    num_images = images.shape[0]
    _images = images.reshape(num_images, channel, height, width)
    permutation = [0, 2, 3, 1]  # permute to [num_examples, height, width, channel]
    return numpy.transpose(_images, permutation)

def dense_to_one_hot(labels_dense, num_classes):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = numpy.arange(num_labels) * num_classes
  labels_one_hot = numpy.zeros((num_labels, num_classes))
  labels_one_hot.flat[list(index_offset + labels_dense.ravel())] = 1
  return labels_one_hot

if __name__=='__main__':
    num_inst = 10000
    CHANNEL = 1
    HEIGHT = 28
    WIDTH = 28
    NUM_CLASSES = 10
    TEST_SIZE = 10000.0
    train_path = '/home/loong/data/mnist.scale'
    test_path = '/home/loong/data/mnist.scale.t'
    dataset = read_dataset(train_path, test_path, NUM_CLASSES, CHANNEL, HEIGHT, WIDTH, one_hot=True)
